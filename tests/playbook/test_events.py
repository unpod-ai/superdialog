import pytest

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


def test_append_rejects_prestamped_version() -> None:
    log = EventLog()
    with pytest.raises(ValueError):
        log.append(UtteranceEvent(role="user", text="hi", version=99))
