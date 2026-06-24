import pytest
from pydantic import ValidationError

from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    ExternalEvent,
    SlotWriteEvent,
    UtteranceEvent,
)


def test_append_stamps_monotonic_versions() -> None:
    log = EventLog()
    e1 = log.append(UtteranceEvent(role="user", text="hi"))
    e2 = log.append(
        SlotWriteEvent(key="city", value="Pune", status="provisional", by="talker")
    )
    assert (e1.version, e2.version) == (1, 2)
    assert log.version == 2


def test_jsonl_round_trip() -> None:
    log = EventLog()
    log.append(UtteranceEvent(role="user", text="hi"))
    log.append(
        AdvanceEvent(
            from_checkpoint=None,
            to_checkpoint="booking.collect",
            rule="r0",
            by="director",
        )
    )
    log.append(
        ExternalEvent(kind="silence", name="user_silence", payload={"elapsed_ms": 6000})
    )
    restored = EventLog.from_jsonl(log.to_jsonl())
    assert [type(e).__name__ for e in restored.events] == [
        "UtteranceEvent",
        "AdvanceEvent",
        "ExternalEvent",
    ]
    assert restored.version == 3
    assert restored.events[2].payload["elapsed_ms"] == 6000


def test_append_rejects_prestamped_version() -> None:
    log = EventLog()
    with pytest.raises(ValueError):
        log.append(UtteranceEvent(role="user", text="hi", version=99))


def test_non_contiguous_versions_rejected() -> None:
    events = [
        UtteranceEvent(role="user", text="hi").model_copy(update={"version": 5}),
        UtteranceEvent(role="user", text="again").model_copy(update={"version": 2}),
    ]
    with pytest.raises(ValueError):
        EventLog(events=events)


def test_non_contiguous_versions_rejected_from_jsonl() -> None:
    lines = "\n".join(
        [
            '{"type": "utterance", "role": "user", "text": "hi", "version": 5}',
            '{"type": "utterance", "role": "user", "text": "yo", "version": 2}',
        ]
    )
    with pytest.raises(ValueError):
        EventLog.from_jsonl(lines)


def test_logged_events_are_immutable() -> None:
    log = EventLog()
    log.append(UtteranceEvent(role="user", text="hi"))
    with pytest.raises(ValidationError):
        log.events[0].text = "TAMPERED"


def test_append_leaves_input_unstamped() -> None:
    log = EventLog()
    e = UtteranceEvent(role="user", text="hi")
    log.append(e)
    assert e.version == 0


def test_empty_round_trip() -> None:
    restored = EventLog.from_jsonl("")
    assert restored.version == 0
    assert not restored.events


def test_session_start_event_round_trips() -> None:
    from superdialog.playbook.events import EventLog, SessionStartEvent

    log = EventLog()
    log.append(SessionStartEvent(started_at="2026-06-24T10:30:00+05:30",
                                 timezone="Asia/Kolkata"))
    restored = EventLog.from_jsonl(log.to_jsonl())
    e = restored.events[0]
    assert e.type == "session_start"
    assert e.started_at == "2026-06-24T10:30:00+05:30"
    assert e.timezone == "Asia/Kolkata"
