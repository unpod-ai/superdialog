import json

from superdialog.playbook.director import Director
from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SlotWriteEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.state import ConversationState
from tests.playbook.test_models import MINIMAL_YAML


class CannedLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return json.dumps(self.payload)


def _state(extra_events=()) -> tuple[Playbook, ConversationState]:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="Pune tomorrow please"))
    for e in extra_events:
        log.append(e)
    return pb, ConversationState.fold(log, playbook=pb)


async def test_extracts_slots_and_advances_when_requires_met() -> None:
    pb, state = _state()
    llm = CannedLLM(
        {
            "slots": {"city": "Pune", "date": "2026-06-11"},
            "advance": "booking.confirm",
            "note": "Confirm date format.",
        }
    )
    decision = await Director(pb, llm).evaluate(state)
    slot_events = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert {e.key for e in slot_events} == {"city", "date"}
    assert all(e.status == "confirmed" for e in slot_events)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.confirm"


async def test_requires_blocks_advance_and_steers() -> None:
    pb, state = _state()
    llm = CannedLLM(
        {"slots": {"city": "Pune"}, "advance": "booking.confirm", "note": None}
    )
    decision = await Director(pb, llm).evaluate(state)  # date missing
    assert not [e for e in decision.events if isinstance(e, AdvanceEvent)]
    notes = [e for e in decision.events if e.type == "steering_note"]
    assert notes and "date" in notes[0].text


async def test_expr_rules_fire_without_llm() -> None:
    pb, state = _state(
        extra_events=[
            AdvanceEvent(
                from_checkpoint="booking.collect",
                to_checkpoint="booking.confirm",
                rule="t",
            ),
            ToolResultEvent(
                tool="confirm_and_hold", store_as="pipeline", ok=True, data={}
            ),
        ]
    )
    llm = CannedLLM({"slots": {}, "advance": None, "note": None})
    decision = await Director(pb, llm).evaluate(state, expr_only=True)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.close" and adv[0].by == "expr"
    assert llm.calls == []  # no LLM round-trip


async def test_expr_rule_applies_set_writes() -> None:
    pb, state = _state(
        extra_events=[
            AdvanceEvent(
                from_checkpoint="booking.collect",
                to_checkpoint="booking.confirm",
                rule="t",
            ),
            ToolResultEvent(
                tool="confirm_and_hold", store_as="pipeline", ok=False, error="503"
            ),
        ]
    )
    llm = CannedLLM({"slots": {}, "advance": None, "note": None})
    decision = await Director(pb, llm).evaluate(state, expr_only=True)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.collect"
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert any(
        e.key == "error_context" and e.value == "booking_confirm_failed" for e in writes
    )


async def test_interrupt_overrides_rules() -> None:
    pb, state = _state()
    llm = CannedLLM(
        {"slots": {}, "advance": None, "note": None, "interrupt": "goodbye"}
    )
    decision = await Director(pb, llm).evaluate(state)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.close"
    assert adv[0].rule == "interrupt:goodbye"


async def test_malformed_llm_json_yields_degraded() -> None:
    pb, state = _state()

    class BadLLM:
        async def complete(self, messages, **kwargs) -> str:
            return "not json {"

    decision = await Director(pb, BadLLM()).evaluate(state)
    assert decision.events == [] and decision.degraded
