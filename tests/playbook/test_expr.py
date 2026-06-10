import pytest

from superdialog.playbook.expr import ExprError, evaluate
from superdialog.playbook.events import EventLog, SlotWriteEvent, ToolResultEvent
from superdialog.playbook.state import ConversationState


def _state() -> ConversationState:
    log = EventLog()
    log.append(
        SlotWriteEvent(key="players", value=4, status="confirmed", by="director")
    )
    log.append(
        ToolResultEvent(
            tool="availability",
            store_as="availability_result",
            ok=True,
            status=200,
            data={
                "slots": [
                    {"id": "s1", "time": "09:00", "price": 1200},
                    {"id": "s2", "time": "10:00", "price": 1500},
                ]
            },
        )
    )
    log.append(
        ToolResultEvent(
            tool="confirm",
            store_as="confirm_result",
            ok=False,
            status=503,
            error="upstream",
        )
    )
    return ConversationState.fold(log)


def test_result_predicates() -> None:
    s = _state()
    assert evaluate("results.availability_result.ok", s) is True
    assert evaluate("results.confirm_result.ok", s) is False
    assert evaluate("results.confirm_result.status == 503", s) is True
    assert evaluate("len(results.availability_result.data.slots) > 0", s) is True


def test_slot_predicates_and_boolean_ops() -> None:
    s = _state()
    assert evaluate("slots.players == 4 and results.availability_result.ok", s) is True
    assert evaluate("slots.missing", s) is None  # missing -> None, falsy
    assert evaluate("not slots.missing", s) is True


def test_views_pipe_helpers() -> None:
    s = _state()
    out = evaluate("pluck(results.availability_result.data.slots, 'time')", s)
    assert out == ["09:00", "10:00"]
    assert evaluate("first(results.availability_result.data.slots).id", s) == "s1"


def test_unsafe_constructs_rejected() -> None:
    s = _state()
    for bad in (
        "__import__('os')",
        "slots.__class__",
        "(lambda: 1)()",
        "[x for x in []]",
    ):
        with pytest.raises(ExprError):
            evaluate(bad, s)
