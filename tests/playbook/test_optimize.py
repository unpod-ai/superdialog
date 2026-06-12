"""Tests for the optimize loop: scoring, reflection, paired rounds."""

import json
import textwrap

import yaml as _yaml

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.editable import FullDoc, SimpleDoc
from superdialog.playbook.eval_bridge import EvalReport, PersonaSpec, SessionMetrics
from superdialog.playbook.models import Playbook
from superdialog.playbook.optimize import (
    ObjectiveBreakdown,
    OptimizeReport,
    ParetoFrontier,
    RoundTrace,
    optimize,
    propose_edits,
    score_report,
)
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_eval_bridge import ScriptedUser
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_GUIDANCE = "journeys.booking.checkpoints.collect.guidance"


def _session(**kw) -> SessionMetrics:
    base = dict(
        persona="p",
        completed=True,
        outcome="confirmed",
        turns=4,
        turns_per_checkpoint={"booking.collect": 2, "booking.confirm": 2},
        slot_accuracy=1.0,
        slot_diffs={},
        repair_count=0,
        degraded_count=0,
        event_log_jsonl="",
    )
    base.update(kw)
    return SessionMetrics(**base)


def test_breakdown_dimensions_match_metrics() -> None:
    report = EvalReport(sessions=[_session(), _session(completed=False, outcome=None)])
    b = score_report(report)
    assert isinstance(b, ObjectiveBreakdown)
    assert b.completion_rate == 0.5
    assert b.slot_accuracy == 1.0
    # smoothness proxy: mean turns/checkpoint over COMPLETED sessions only
    assert b.mean_turns_per_checkpoint == 2.0
    assert b.repair_rate == 0.0


def test_scalar_objective_is_weighted_sum_in_unit_range() -> None:
    good = score_report(EvalReport(sessions=[_session()]))
    bad = score_report(
        EvalReport(
            sessions=[
                _session(
                    completed=False,
                    outcome=None,
                    slot_accuracy=0.0,
                    repair_count=3,
                    turns_per_checkpoint={"a": 8},
                ),
            ]
        )
    )
    assert 0.0 <= bad.objective < good.objective <= 1.0


def test_empty_report_scores_zero() -> None:
    b = score_report(EvalReport(sessions=[]))
    assert b.objective == 0.0
    assert b.completion_rate == 0.0


def test_smoothness_rewards_fewer_turns_per_checkpoint() -> None:
    smooth = score_report(
        EvalReport(sessions=[_session(turns_per_checkpoint={"a": 1, "b": 1})])
    )
    bumpy = score_report(
        EvalReport(sessions=[_session(turns_per_checkpoint={"a": 6, "b": 6})])
    )
    assert smooth.objective > bumpy.objective


def test_incomplete_sessions_earn_no_smoothness_credit() -> None:
    # A fail-fast incomplete session must not raise the smoothness term.
    failing = score_report(
        EvalReport(
            sessions=[
                _session(
                    completed=False,
                    outcome=None,
                    slot_accuracy=0.0,
                    turns_per_checkpoint={"a": 1},
                )
            ]
        )
    )
    assert failing.mean_turns_per_checkpoint == 0.0  # nothing completed


class CannedEditsLLM:
    """Candidate LLM: pops scripted outputs, repeating the last one forever."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


def _edit_json(
    address: str = _GUIDANCE, new_text: str = "Ask for the city first, warmly."
) -> str:
    return json.dumps([{"address": address, "new_text": new_text}])


def _report(**kw) -> EvalReport:
    base = dict(
        persona="p",
        completed=False,
        outcome=None,
        turns=6,
        turns_per_checkpoint={"booking.collect": 6},
        slot_accuracy=0.0,
        slot_diffs={"city": ("Pune", None)},
        repair_count=2,
        degraded_count=0,
        event_log_jsonl='{"type":"utterance","version":1,"role":"user","text":"uh"}',
    )
    base.update(kw)
    return EvalReport(sessions=[SessionMetrics(**base)])


async def test_propose_returns_doc_and_edits() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM([_edit_json()])
    proposal = await propose_edits(doc, _report(), llm, max_attempts=3)
    assert proposal is not None
    cand, edits = proposal
    assert (
        cand.compile().checkpoint("booking.collect").guidance
        == "Ask for the city first, warmly."
    )
    assert edits[0].address == _GUIDANCE
    # the prompt showed current prose, the editable address, and the evidence
    prompt = " ".join(m["content"] for m in llm.calls[0])
    assert "Collect naturally." in prompt
    assert _GUIDANCE in prompt
    assert "city" in prompt


async def test_fenced_json_is_accepted() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM(["```json\n" + _edit_json() + "\n```"])
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is not None


async def test_invalid_json_retries_then_falls_back() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM(["not json at all", '{"also": "not a list"}'])
    assert await propose_edits(doc, _report(), llm, max_attempts=2) is None
    assert len(llm.calls) == 2


async def test_frozen_address_is_rejected() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM(
        [
            _edit_json(
                address="journeys.booking.checkpoints.confirm.gate", new_text="soft"
            )
        ]
    )
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is None


async def test_broken_jinja_is_rejected() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM([_edit_json(new_text="Hello {{ slots.city ")])
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is None


async def test_empty_edit_list_is_rejected() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM(["[]"])
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is None


def _breakdown(completion: float, slot: float, turns: float) -> ObjectiveBreakdown:
    return ObjectiveBreakdown(
        objective=completion,
        completion_rate=completion,
        slot_accuracy=slot,
        mean_turns_per_checkpoint=turns,
        repair_rate=0.0,
    )


def _trace(round_no: int, completion: float, slot: float, turns: float) -> RoundTrace:
    return RoundTrace(
        round_no=round_no,
        accepted=True,
        incumbent_breakdown=_breakdown(0.1, 0.1, 9.0),
        candidate_breakdown=_breakdown(completion, slot, turns),
    )


def test_frontier_keeps_non_dominated() -> None:
    f = ParetoFrontier()
    f.consider(_trace(1, completion=0.9, slot=0.5, turns=2.0))
    f.consider(_trace(2, completion=0.5, slot=0.9, turns=2.0))  # trades off
    f.consider(_trace(3, completion=0.4, slot=0.4, turns=3.0))  # dominated
    assert sorted(t.round_no for t in f.members) == [1, 2]


def test_frontier_drops_newly_dominated_member() -> None:
    f = ParetoFrontier()
    f.consider(_trace(1, completion=0.6, slot=0.6, turns=2.0))
    f.consider(_trace(2, completion=0.9, slot=0.9, turns=1.0))  # dominates #1
    assert [t.round_no for t in f.members] == [2]


def test_frontier_ignores_rounds_without_a_candidate() -> None:
    f = ParetoFrontier()
    f.consider(
        RoundTrace(
            round_no=1,
            accepted=False,
            incumbent_breakdown=_breakdown(0.5, 0.5, 2.0),
            detail="no valid candidate",
        )
    )
    assert f.members == []


_IDLE = {"slots": {}, "advance": None, "note": None}
_ADVANCE = {
    "slots": {"city": "Pune", "date": "2026-06-12"},
    "advance": "booking.confirm",
    "note": None,
}
_HOLD_OK = (200, {"data": {"hold_id": "h1"}})

_PERSONAS = [
    PersonaSpec(
        name="closer",
        traits="direct",
        goal="book in Pune",
        ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
    )
]


def _improving_agent_factory(playbook: Playbook) -> PlaybookAgent:
    """Director completes the booking only after the 'warmly' mutation."""
    improved = "warmly" in playbook.checkpoint("booking.collect").guidance
    return PlaybookAgent(
        playbook=playbook,
        talker_llm=StreamLLM(["Which", " city?"]),
        director_llm=CannedLLM(_ADVANCE if improved else _IDLE),
        http=FakeHttp([_HOLD_OK] * 4),
    )


async def test_optimize_improves_and_emits_final_incumbent() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM([_edit_json(new_text="Collect warmly.")])
    report = await optimize(
        doc,
        personas=_PERSONAS,
        candidate_llm=llm,
        user_llm=ScriptedUser(["Pune on 2026-06-12 please", "ok"]),
        agent_factory=_improving_agent_factory,
        rounds=2,
        n=1,
    )
    assert isinstance(report, OptimizeReport)
    assert "Collect warmly." in report.final_yaml
    assert report.final_breakdown.objective > report.initial_breakdown.objective
    accepted = [t for t in report.trace if t.accepted]
    assert accepted and accepted[0].edits[0].address == _GUIDANCE
    assert accepted[0].candidate_breakdown is not None


async def test_no_valid_candidate_keeps_the_input() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    report = await optimize(
        doc,
        personas=_PERSONAS,
        candidate_llm=CannedEditsLLM(["not json"]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory,
        rounds=1,
        n=1,
    )
    assert report.final_yaml == doc.emit()
    assert report.trace[0].accepted is False
    assert report.trace[0].detail == "no valid candidate"


async def test_noop_edit_is_never_accepted_and_round_cap_holds() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    noop = _edit_json(new_text="Collect naturally.")  # identical prose
    report = await optimize(
        doc,
        personas=_PERSONAS,
        candidate_llm=CannedEditsLLM([noop]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory,
        rounds=3,
        n=1,
        patience=99,
    )
    assert len(report.trace) == 3
    assert not any(t.accepted for t in report.trace)


async def test_patience_stops_early() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    noop = _edit_json(new_text="Collect naturally.")
    report = await optimize(
        doc,
        personas=_PERSONAS,
        candidate_llm=CannedEditsLLM([noop]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory,
        rounds=5,
        n=1,
        patience=2,
    )
    assert len(report.trace) == 2  # stopped after `patience` stale rounds


_SIMPLE_TWO_STEP = textwrap.dedent("""
    name: "Mini"
    goal: "Say hello and close."
    persona:
      identity: "You are a tiny demo agent."
    opening: "Greet the caller."
    playbook:
      - id: hello
        purpose: "Open the call."
        say: "Greet and ask how to help."
        done_when: "Caller responded."
      - id: done
        purpose: "Close."
        say: "Wrap up."
        done_when: "Closed."
""")


def _simple_improving_factory(playbook: Playbook) -> PlaybookAgent:
    improved = "warmly" in playbook.checkpoint("main.hello").guidance
    verdict = {"slots": {}, "advance": "main.done", "note": None} if improved else _IDLE
    return PlaybookAgent(
        playbook=playbook,
        talker_llm=StreamLLM(["Hi", " there"]),
        director_llm=CannedLLM(verdict),
        http=FakeHttp([]),
    )


async def test_simple_doc_optimizes_and_stays_simple() -> None:
    doc = SimpleDoc.from_text(_SIMPLE_TWO_STEP)
    edit = json.dumps(
        [
            {
                "address": "steps.hello.say",
                "new_text": "Greet warmly and ask how to help.",
            }
        ]
    )
    report = await optimize(
        doc,
        personas=[PersonaSpec(name="p", traits="brief", goal="say hi")],
        candidate_llm=CannedEditsLLM([edit]),
        user_llm=ScriptedUser(["hello", "bye"]),
        agent_factory=_simple_improving_factory,
        rounds=1,
        n=1,
    )
    assert any(t.accepted for t in report.trace)
    out = _yaml.safe_load(report.final_yaml)
    assert "playbook" in out and "journeys" not in out  # still simple format
    assert "warmly" in out["playbook"][0]["say"]
