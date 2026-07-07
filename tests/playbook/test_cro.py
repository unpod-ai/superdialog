"""CRO suffix-replay: first_affected_version, from_version, and the guard gate."""

import json
import textwrap

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.editable import Edit, FullDoc
from superdialog.playbook.eval_bridge import EvalReport, PersonaSpec, SessionMetrics
from superdialog.playbook.events import AdvanceEvent, EventLog, UtteranceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.optimize import cro_guard, optimize
from superdialog.playbook.replay import first_affected_version, replay
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_eval_bridge import ScriptedUser
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_optimize import CannedEditsLLM
from tests.playbook.test_replay import SequencedLLM
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

CRO_YAML = textwrap.dedent("""
    journeys:
      flow:
        checkpoints:
          - id: ask
            goal: "Route the caller"
            advance_when:
              - {when: "wants to book", judge: llm, to: flow.book}
              - {when: "wants to cancel", judge: llm, to: flow.cancel}
          - id: book
            goal: "Confirm the booking"
            advance_when:
              - {when: "changes mind", judge: llm, to: flow.cancel}
          - id: cancel
            goal: "Close out"
            terminal: true
""")

_PB = Playbook.from_yaml(CRO_YAML)
_IDLE = {"slots": {}, "advance": None, "note": None}
_BOOK_EDITS = [
    Edit(
        address="journeys.flow.checkpoints.book.guidance",
        new_text="Confirm warmly.",
    )
]


def _booked_log() -> EventLog:
    """v1 init→ask, v2 user, v3 advance→book, v4 assistant, v5 user (idle)."""
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="flow.ask", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="I want to book a slot"))
    log.append(
        AdvanceEvent(
            from_checkpoint="flow.ask", to_checkpoint="flow.book", rule="llm:flow.book"
        )
    )
    log.append(UtteranceEvent(role="assistant", text="Booking it now."))
    log.append(UtteranceEvent(role="user", text="great, thanks"))
    return log


# -- first_affected_version ---------------------------------------------------


def test_global_edit_affects_from_the_start() -> None:
    edits = [Edit(address="persona", new_text="x")]
    assert first_affected_version(_booked_log(), _PB, edits) == 0


def test_checkpoint_edit_starts_at_the_advance_into_it() -> None:
    assert first_affected_version(_booked_log(), _PB, _BOOK_EDITS) == 3


def test_initial_checkpoint_edit_affects_from_the_start() -> None:
    edits = [Edit(address="journeys.flow.checkpoints.ask.goal", new_text="Route.")]
    assert first_affected_version(_booked_log(), _PB, edits) == 0


def test_unvisited_checkpoint_edit_affects_nothing() -> None:
    edits = [Edit(address="journeys.flow.checkpoints.cancel.goal", new_text="Bye.")]
    assert first_affected_version(_booked_log(), _PB, edits) is None


def test_simple_step_address_matches_by_short_id() -> None:
    edits = [Edit(address="steps.book.say", new_text="Confirm warmly.")]
    assert first_affected_version(_booked_log(), _PB, edits) == 3


def test_mixed_edits_take_the_earliest_boundary() -> None:
    edits = _BOOK_EDITS + [Edit(address="persona", new_text="x")]
    assert first_affected_version(_booked_log(), _PB, edits) == 0


# -- replay from_version ------------------------------------------------------


async def test_replay_from_version_holds_prefix_turns() -> None:
    llm = SequencedLLM([_IDLE])
    report = await replay(_booked_log(), _PB, llm, from_version=3)
    assert report.turns == 1 and report.turns_held == 1
    assert llm.calls == 1  # only the affected suffix paid a Director call
    assert report.stable


async def test_replay_default_covers_all_turns() -> None:
    llm = SequencedLLM([{"slots": {}, "advance": "flow.book", "note": None}, _IDLE])
    report = await replay(_booked_log(), _PB, llm)
    assert report.turns == 2 and report.turns_held == 0
    assert report.stable


# -- cro_guard ----------------------------------------------------------------


def _guard_report(log: EventLog) -> EvalReport:
    return EvalReport(
        sessions=[
            SessionMetrics(
                persona="happy",
                completed=True,
                outcome="confirmed",
                turns=2,
                turns_per_checkpoint={"flow.ask": 1, "flow.book": 1},
                slot_accuracy=1.0,
                slot_diffs={},
                repair_count=0,
                degraded_count=0,
                event_log_jsonl=log.to_jsonl(),
            )
        ]
    )


def _candidate(edits: list[Edit]) -> FullDoc:
    return FullDoc.from_text(CRO_YAML).apply(edits)


async def test_cro_guard_passes_a_stable_candidate() -> None:
    llm = SequencedLLM([_IDLE])  # matches recorded: no advance on the book turn
    ok, reason = await cro_guard(
        _candidate(_BOOK_EDITS), _BOOK_EDITS, _guard_report(_booked_log()), llm
    )
    assert ok and reason == ""
    assert llm.calls == 1  # suffix-only: one turn re-judged, prefix held


async def test_cro_guard_rejects_a_destabilizing_candidate() -> None:
    llm = SequencedLLM([{"slots": {}, "advance": "flow.cancel", "note": None}])
    ok, reason = await cro_guard(
        _candidate(_BOOK_EDITS), _BOOK_EDITS, _guard_report(_booked_log()), llm
    )
    assert not ok
    assert reason.startswith("cro-guard:") and "happy" in reason


async def test_cro_guard_skips_sessions_the_edits_cannot_touch() -> None:
    edits = [Edit(address="journeys.flow.checkpoints.cancel.goal", new_text="Bye.")]
    llm = SequencedLLM([_IDLE])
    ok, reason = await cro_guard(
        _candidate(edits), edits, _guard_report(_booked_log()), llm
    )
    assert ok and reason == ""
    assert llm.calls == 0  # never visited: provably unaffected, zero LLM spend


async def test_cro_guard_is_open_without_completed_guard_sessions() -> None:
    report = _guard_report(_booked_log())
    report.sessions[0].completed = False
    llm = SequencedLLM([_IDLE])
    ok, _ = await cro_guard(_candidate(_BOOK_EDITS), _BOOK_EDITS, report, llm)
    assert ok and llm.calls == 0


# -- optimize integration -----------------------------------------------------

_ADVANCE = {
    "slots": {"city": "Pune", "date": "2026-06-12"},
    "advance": "booking.confirm",
    "note": None,
}
_HOLD_OK = (200, {"data": {"hold_id": "h1"}})


async def test_optimize_gate_rejects_before_paying_for_evals() -> None:
    factory_calls = 0

    def factory(playbook: Playbook) -> PlaybookAgent:
        nonlocal factory_calls
        factory_calls += 1
        return PlaybookAgent(
            playbook=playbook,
            talker_llm=StreamLLM(["Which", " city?"]),
            director_llm=CannedLLM(_ADVANCE),
            http=FakeHttp([_HOLD_OK] * 4),
        )

    doc = FullDoc.from_text(MINIMAL_YAML)
    edit = json.dumps(
        [
            {
                "address": "journeys.booking.checkpoints.collect.guidance",
                "new_text": "Collect warmly.",
            }
        ]
    )
    report = await optimize(
        doc,
        personas=[
            PersonaSpec(
                name="closer",
                traits="direct",
                goal="book in Pune",
                ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
            )
        ],
        candidate_llm=CannedEditsLLM([edit]),
        user_llm=ScriptedUser(["Pune on 2026-06-12 please", "ok"]),
        agent_factory=factory,
        rounds=1,
        n=1,
        # Replay diverges from the recorded advance -> guard must reject.
        director_llm=SequencedLLM([_IDLE]),
    )
    assert report.trace[0].accepted is False
    assert report.trace[0].detail.startswith("cro-guard:")
    assert factory_calls == 1  # initial eval only — no paired eval was paid
    assert report.final_yaml == doc.emit()
