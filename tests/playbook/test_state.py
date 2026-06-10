from superdialog.playbook.events import (
    AdvanceEvent,
    EnvWriteEvent,
    EventLog,
    SessionEndEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
    SummaryEvent,
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
