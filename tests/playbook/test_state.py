import textwrap

from superdialog.playbook.events import (
    AdvanceEvent,
    EnvWriteEvent,
    EventLog,
    ExternalEvent,
    SessionEndEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
    SummaryEvent,
    ToolCallEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.state import ConversationState
from tests.playbook.test_models import MINIMAL_YAML


def _log() -> EventLog:
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(
        UtteranceEvent(role="assistant", text="Hi! Where would you like to play?")
    )
    log.append(UtteranceEvent(role="user", text="Pune, tomorrow"))
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="provisional", by="talker")
    )
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    log.append(
        SlotWriteEvent(key="course_id", value="c-9", status="confirmed", by="tool")
    )
    return log


def test_fold_basics() -> None:
    state = ConversationState.fold(_log())
    assert state.checkpoint_id == "booking.collect"
    assert state.slots["city"].status == "confirmed"
    assert state.version == 6
    assert [m.role for m in state.transcript] == ["assistant", "user"]


def test_confirmed_not_downgraded_by_provisional() -> None:
    log = _log()
    log.append(
        SlotWriteEvent(key="city", value="Pune?", status="provisional", by="talker")
    )
    state = ConversationState.fold(log)
    assert state.slots["city"].value == "Pune"
    assert state.slots["city"].status == "confirmed"


def test_invalidates_clears_dependents() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)  # city invalidates course_id
    log = _log()
    log.append(
        SlotWriteEvent(key="city", value="Mumbai", status="confirmed", by="director")
    )
    state = ConversationState.fold(log, playbook=pb)
    assert "course_id" not in state.slots


def test_authoritative_slot_rejects_talker_writes() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)  # price is authoritative
    log = _log()
    log.append(
        SlotWriteEvent(key="price", value=999, status="provisional", by="talker")
    )
    state = ConversationState.fold(log, playbook=pb)
    assert "price" not in state.slots
    log.append(
        SlotWriteEvent(key="price", value=1200, status="confirmed", by="director")
    )
    state = ConversationState.fold(log, playbook=pb)
    assert state.slots["price"].value == 1200


def test_lanes_and_end() -> None:
    log = _log()
    log.append(EnvWriteEvent(key="ACCESS_TOKEN", value="t-1"))
    log.append(
        ToolResultEvent(
            tool="hold_slot", store_as="hold_result", ok=True, data={"hold_id": "h1"}
        )
    )
    log.append(SteeringNoteEvent(text="don't re-ask city", kind="steer"))
    log.append(SummaryEvent(text="Caller wants Pune tomorrow."))
    log.append(SessionEndEvent(outcome="confirmed"))
    state = ConversationState.fold(log)
    assert state.env["ACCESS_TOKEN"] == "t-1"
    assert state.tool_results["hold_result"].data == {"hold_id": "h1"}
    assert state.steering_note == "don't re-ask city"
    assert state.summary == "Caller wants Pune tomorrow."
    assert state.ended and state.outcome == "confirmed"


def test_same_value_reconfirmation_keeps_dependents() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)  # city invalidates course_id
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="provisional", by="talker")
    )
    log.append(
        SlotWriteEvent(key="course_id", value="c-9", status="confirmed", by="tool")
    )
    # Director settles city with the SAME value: dependents must survive.
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    state = ConversationState.fold(log, playbook=pb)
    assert state.slots["course_id"].value == "c-9"
    assert state.slots["city"].status == "confirmed"
    # A genuinely changed value still clears dependents.
    log.append(
        SlotWriteEvent(key="city", value="Mumbai", status="confirmed", by="director")
    )
    state = ConversationState.fold(log, playbook=pb)
    assert "course_id" not in state.slots


def test_self_invalidation_guard() -> None:
    pb = Playbook.from_yaml(
        textwrap.dedent("""
            journeys:
              j:
                checkpoints:
                  - id: only
                    slots:
                      x: {type: str, invalidates: [x]}
                    terminal: true
        """)
    )
    log = EventLog()
    log.append(SlotWriteEvent(key="x", value="v1", status="confirmed", by="director"))
    state = ConversationState.fold(log, playbook=pb)
    assert state.slots["x"].value == "v1"
    # A changed value re-runs invalidation but must not erase its own write.
    log.append(SlotWriteEvent(key="x", value="v2", status="confirmed", by="director"))
    state = ConversationState.fold(log, playbook=pb)
    assert state.slots["x"].value == "v2"


def test_fold_output_does_not_alias_log() -> None:
    log = EventLog()
    log.append(
        SlotWriteEvent(
            key="prefs", value={"k": "original"}, status="confirmed", by="director"
        )
    )
    log.append(ToolResultEvent(tool="t", store_as="res", ok=True, data={"n": 1}))
    state = ConversationState.fold(log)
    state.slots["prefs"].value["k"] = "mutated"
    state.tool_results["res"].data["n"] = 99
    fresh = ConversationState.fold(log)
    assert fresh.slots["prefs"].value == {"k": "original"}
    assert fresh.tool_results["res"].data == {"n": 1}


def test_counters_and_resets() -> None:
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="hello"))
    log.append(ToolCallEvent(tool="hold_slot"))
    log.append(ToolCallEvent(tool="hold_slot"))
    log.append(ExternalEvent(kind="silence", name="s1"))
    log.append(ExternalEvent(kind="silence", name="s2"))
    state = ConversationState.fold(log)
    assert state.tool_call_counts["hold_slot"] == 2
    assert state.silence_count == 2
    assert state.user_turns_in_checkpoint == 1
    log.append(
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="llm:booking.confirm",
        )
    )
    state = ConversationState.fold(log)
    assert state.silence_count == 0
    assert state.user_turns_in_checkpoint == 0
    assert state.completed == ["booking.collect"]


def test_steering_note_cleared_on_advance() -> None:
    """A steer note is advice for the current step; advancing clears it."""
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(SteeringNoteEvent(text="wrap this step up", kind="steer"))
    assert ConversationState.fold(log).steering_note == "wrap this step up"
    log.append(
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="llm:booking.confirm",
        )
    )
    state = ConversationState.fold(log)
    assert state.steering_note is None
    assert state.steering_kind == "steer"


def test_fold_sets_now_from_session_start() -> None:
    from datetime import datetime
    from superdialog.playbook.events import EventLog, SessionStartEvent
    from superdialog.playbook.state import ConversationState

    log = EventLog()
    log.append(
        SessionStartEvent(
            started_at="2026-06-24T10:30:00+05:30", timezone="Asia/Kolkata"
        )
    )
    state = ConversationState.fold(log)
    assert isinstance(state.now, datetime)
    assert state.now.year == 2026 and state.now.month == 6 and state.now.day == 24


def test_fold_now_defaults_none_without_session_start() -> None:
    from superdialog.playbook.events import EventLog
    from superdialog.playbook.state import ConversationState

    assert ConversationState.fold(EventLog()).now is None


def test_gating_helpers() -> None:
    log = EventLog()
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="provisional", by="talker")
    )
    state = ConversationState.fold(log)
    assert state.confirmed([]) is True
    assert state.filled([]) is True
    assert state.confirmed(["missing"]) is False
    assert state.filled(["city"]) is True
    assert state.confirmed(["city"]) is False
    assert state.slot_value("city") == "Pune"
    assert state.slot_value("missing") is None


def _fold(*events: UtteranceEvent) -> ConversationState:
    log = EventLog()
    for e in events:
        log.append(e)
    return ConversationState.fold(log)


def test_language_follows_bridge_detection() -> None:
    s = _fold(UtteranceEvent(role="user", text="namaste", language="hi"))
    assert s.language == "hi"


def test_language_sticky_against_missing_signal() -> None:
    # turn 2 reports no language -> keep the last known one (no flip-flop)
    s = _fold(
        UtteranceEvent(role="user", text="namaste", language="hi"),
        UtteranceEvent(role="user", text="batao", language=None),
    )
    assert s.language == "hi"


def test_language_updates_on_genuine_switch() -> None:
    s = _fold(
        UtteranceEvent(role="user", text="namaste", language="hi"),
        UtteranceEvent(role="user", text="english is fine", language="en"),
    )
    assert s.language == "en"


def test_language_none_when_never_reported() -> None:
    s = _fold(UtteranceEvent(role="user", text="hello"))
    assert s.language is None  # backward compatible: no directive rendered


def test_slot_value_carries_entity_default_caller() -> None:
    from superdialog.playbook.state import SlotValue

    sv = SlotValue(value="x", status="confirmed", by="director", version=1)
    assert sv.entity == "caller"  # default, backward compatible
    assert (
        SlotValue(
            value="x", status="confirmed", by="director", version=1, entity="partner"
        ).entity
        == "partner"
    )


def test_fold_propagates_entity_from_event_to_slot_value() -> None:
    log = EventLog()
    log.append(
        SlotWriteEvent(
            key="dob",
            value="1986-07-12",
            status="confirmed",
            by="director",
            entity="partner",
        )
    )
    state = ConversationState.fold(log)
    assert state.slots["dob"].entity == "partner"  # carried through fold


def test_fold_entity_defaults_caller_backward_compatible() -> None:
    log = EventLog()
    log.append(SlotWriteEvent(key="dob", value="x", status="confirmed", by="director"))
    state = ConversationState.fold(log)
    assert state.slots["dob"].entity == "caller"  # unchanged: single-entity
