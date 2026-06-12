"""Tests for the optimize loop: scoring, reflection, paired rounds."""

import json

from superdialog.playbook.editable import FullDoc
from superdialog.playbook.eval_bridge import EvalReport, SessionMetrics
from superdialog.playbook.optimize import (
    ObjectiveBreakdown,
    propose_edits,
    score_report,
)
from tests.playbook.test_models import MINIMAL_YAML

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
