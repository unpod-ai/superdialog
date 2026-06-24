import json
import textwrap

from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SlotWriteEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.replay import replay
from tests.playbook.test_models import MINIMAL_YAML


class SequencedLLM:
    """Director LLM: one canned verdict per call, then repeats the last."""

    def __init__(self, verdicts: list[dict]) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    async def complete(self, messages: list[dict], **kwargs: object) -> str:
        i = min(self.calls, len(self.verdicts) - 1)
        self.calls += 1
        return json.dumps(self.verdicts[i])


def _recorded_log() -> EventLog:
    """Two-turn booking log: slots + advance on turn 1, slot write on turn 2."""
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="Pune on 2026-06-12 please"))
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    log.append(
        SlotWriteEvent(
            key="date", value="2026-06-12", status="confirmed", by="director"
        )
    )
    log.append(
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="llm:booking.confirm",
        )
    )
    log.append(UtteranceEvent(role="assistant", text="Your booking is held."))
    log.append(UtteranceEvent(role="user", text="yes, Pune is right"))
    # city is scoped to booking.collect — no slot write at booking.confirm
    return log


_STABLE_VERDICTS = [
    {
        "slots": {"city": "Pune", "date": "2026-06-12"},
        "advance": "booking.confirm",
        "note": None,
    },
    {"slots": {}, "advance": None, "note": None},
]


async def test_stable_replay() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    report = await replay(_recorded_log(), pb, SequencedLLM(_STABLE_VERDICTS))
    assert report.turns == 2
    assert report.advance_matches == 1
    assert report.slot_matches == 2  # city+date on turn 1; city scoped to collect, not confirm
    assert report.diffs == []
    assert report.stable


# Routing playbook with two llm rules from one checkpoint, so a replayed
# verdict can legitimately pick a different target than the recorded one.
ROUTE_YAML = textwrap.dedent("""
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
            terminal: true
""")


def _route_log(turns: int = 1) -> EventLog:
    """init -> flow.ask, user turn 1 + recorded advance, optional turn 2."""
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="flow.ask", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="I want to book a slot"))
    log.append(
        AdvanceEvent(
            from_checkpoint="flow.ask",
            to_checkpoint="flow.book",
            rule="llm:flow.book",
        )
    )
    if turns > 1:
        log.append(UtteranceEvent(role="assistant", text="Booking it now."))
        log.append(UtteranceEvent(role="user", text="great, thanks"))
    return log


async def test_diverging_advance_detected() -> None:
    pb = Playbook.from_yaml(ROUTE_YAML)
    llm = SequencedLLM([{"slots": {}, "advance": "flow.cancel", "note": None}])
    report = await replay(_route_log(), pb, llm)
    assert not report.stable and report.advance_matches == 0
    assert len(report.diffs) == 1
    diff = report.diffs[0]
    assert diff.kind == "advance"
    assert diff.recorded == "flow.book" and diff.replayed == "flow.cancel"
    assert diff.at_version == 2


async def test_missing_and_extra() -> None:
    pb = Playbook.from_yaml(ROUTE_YAML)
    llm = SequencedLLM(
        [
            {"slots": {}, "advance": None, "note": None},  # recorded had one
            {"slots": {}, "advance": "flow.cancel", "note": None},  # none rec.
        ]
    )
    report = await replay(_route_log(turns=2), pb, llm)
    assert report.turns == 2 and report.advance_matches == 0
    assert [d.kind for d in report.diffs] == ["missing_advance", "extra_advance"]
    missing, extra = report.diffs
    assert missing.recorded == "flow.book" and missing.replayed is None
    assert extra.recorded is None and extra.replayed == "flow.cancel"


async def test_replay_is_pure() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = _recorded_log()
    version_before = log.version
    events_before = list(log.events)
    await replay(log, pb, SequencedLLM(_STABLE_VERDICTS))
    assert log.version == version_before
    assert log.events == events_before


async def test_runtime_made_advances_excluded() -> None:
    """Pipeline/policy advances in the window are not Director decisions."""
    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="hello"))
    log.append(
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="pipeline",
            by="director",
        )
    )
    log.append(
        AdvanceEvent(
            from_checkpoint="booking.confirm",
            to_checkpoint="booking.collect",
            rule="policy:silence",
            by="policy",
        )
    )
    llm = SequencedLLM([{"slots": {}, "advance": None, "note": None}])
    report = await replay(log, pb, llm)
    assert report.stable
    assert report.advance_matches == 0 and report.diffs == []


def test_anchor_is_stable_across_folds() -> None:
    from superdialog.playbook.events import EventLog, SessionStartEvent
    from superdialog.playbook.state import ConversationState

    log = EventLog()
    log.append(SessionStartEvent(started_at="2026-06-24T09:00:00+05:30",
                                 timezone="Asia/Kolkata"))
    a = ConversationState.fold(log).now
    b = ConversationState.fold(log).now
    assert a == b and a is not None  # folded from the log, never re-clocked
