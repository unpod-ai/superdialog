import json
import textwrap

from superdialog.playbook.director import Director
from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SessionStartEvent,
    SlotWriteEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.state import ConversationState, SlotValue
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
    assert decision.detail == "json_parse_error"


async def test_degraded_detail_distinguishes_failure_modes() -> None:
    pb, state = _state()

    class RaisingLLM:
        async def complete(self, messages, **kwargs) -> str:
            raise RuntimeError("boom")

    class ListLLM:
        async def complete(self, messages, **kwargs) -> str:
            return "[1, 2]"

    assert (await Director(pb, RaisingLLM()).evaluate(state)).detail == "llm_error"
    assert (await Director(pb, ListLLM()).evaluate(state)).detail == "non_dict_verdict"


HARD_GATE_YAML = textwrap.dedent("""
    persona: "You verify payments."
    journeys:
      pay:
        checkpoints:
          - id: verify
            gate: hard
            goal: "Verify the one-time code"
            slots:
              otp: {type: str}
            advance_when:
              - {when: "user gave the code", judge: llm, to: pay.done,
                 requires: [otp]}
          - id: done
            terminal: true
""")


def _hard_state(pb: Playbook) -> ConversationState:
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="pay.verify", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="my code is 4242"))
    return ConversationState.fold(log, playbook=pb)


async def test_hard_gate_not_self_attesting() -> None:
    """A single verdict must not confirm its own requires through a hard gate."""
    pb = Playbook.from_yaml(HARD_GATE_YAML)
    state = _hard_state(pb)
    llm = CannedLLM({"slots": {"otp": "4242"}, "advance": "pay.done", "note": None})
    decision = await Director(pb, llm).evaluate(state)
    assert not [e for e in decision.events if isinstance(e, AdvanceEvent)]
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert writes and writes[0].key == "otp"
    assert writes[0].status == "provisional"
    notes = [e for e in decision.events if e.type == "steering_note"]
    assert notes and "confirmation" in notes[0].text and "otp" in notes[0].text


async def test_hard_gate_advances_once_slot_confirmed() -> None:
    """Pre-confirmed requires (e.g. by a tool) do let the verdict advance."""
    pb = Playbook.from_yaml(HARD_GATE_YAML)
    state = _hard_state(pb)
    state.slots["otp"] = SlotValue(
        value="4242", status="confirmed", by="tool", version=1
    )
    llm = CannedLLM({"slots": {}, "advance": "pay.done", "note": None})
    decision = await Director(pb, llm).evaluate(state)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "pay.done"


async def test_authoritative_slots_never_written_by_verdict() -> None:
    pb, state = _state()
    llm = CannedLLM(
        {"slots": {"price": 1200.0, "city": "Pune"}, "advance": None, "note": None}
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert not [e for e in writes if e.key == "price"]  # authoritative: tool-only
    assert [e for e in writes if e.key == "city"]


TYPED_SLOTS_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: c
            slots:
              mode: {type: enum, values: [a, b]}
              count: {type: int}
          - id: end
            terminal: true
""")


async def test_enum_and_type_validation() -> None:
    pb = Playbook.from_yaml(TYPED_SLOTS_YAML)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.c", rule="init"))
    log.append(UtteranceEvent(role="user", text="seven, mode c"))
    state = ConversationState.fold(log, playbook=pb)
    llm = CannedLLM(
        {"slots": {"mode": "c", "count": "7"}, "advance": None, "note": None}
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = {e.key: e for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert "mode" not in writes  # "c" is not an enum member of [a, b]
    assert writes["count"].value == 7 and isinstance(writes["count"].value, int)


async def test_verdict_prompt_warns_against_injection() -> None:
    pb, state = _state()
    llm = CannedLLM({"slots": {}, "advance": None, "note": None})
    await Director(pb, llm).evaluate(state)
    system = llm.calls[0][0]["content"]
    assert "untrusted" in system


async def test_verdict_prompt_includes_tool_results() -> None:
    """Result-dependent llm rules must see what the tools actually did."""
    pb, state = _state(
        extra_events=[
            ToolResultEvent(
                tool="hold_slot",
                store_as="hold_result",
                ok=True,
                status=200,
                data={"secret": "never-dumped"},
            )
        ]
    )
    llm = CannedLLM({"slots": {}, "advance": None, "note": None})
    await Director(pb, llm).evaluate(state)
    system = llm.calls[0][0]["content"]
    assert "Tool results:" in system
    assert "hold_result" in system and "ok=True" in system and "status=200" in system
    assert "never-dumped" not in system  # compact summary, no data dump


async def test_verdict_prompt_tool_results_empty_state() -> None:
    pb, state = _state()  # no tool results yet
    llm = CannedLLM({"slots": {}, "advance": None, "note": None})
    await Director(pb, llm).evaluate(state)
    system = llm.calls[0][0]["content"]
    assert "Tool results:\n(none)" in system


def test_verdict_prompt_injects_date_discipline_when_date_slot() -> None:
    from datetime import datetime, timezone  # noqa: F401
    from superdialog.playbook.director import _verdict_prompt
    from superdialog.playbook.events import (
        EventLog,
        AdvanceEvent,
        SessionStartEvent,
        UtteranceEvent,
    )
    from superdialog.playbook.state import ConversationState

    pb = Playbook.from_yaml(MINIMAL_YAML)  # booking.collect has a `date` slot
    log = EventLog()
    log.append(
        SessionStartEvent(started_at="2026-06-24T00:00:00+00:00", timezone="UTC")
    )
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="tomorrow"))
    state = ConversationState.fold(log, playbook=pb)
    cp = pb.checkpoint("booking.collect")
    system = _verdict_prompt(pb, cp, state)[0]["content"]
    assert "CURRENT DATE & TIME" in system
    assert "absolute" in system.lower()


async def test_date_slot_normalized_to_absolute() -> None:
    from superdialog.playbook.events import (
        EventLog,
        AdvanceEvent,
        SessionStartEvent,
        UtteranceEvent,
        SlotWriteEvent,
    )
    from superdialog.playbook.state import ConversationState

    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = EventLog()
    log.append(
        SessionStartEvent(started_at="2026-06-24T00:00:00+00:00", timezone="UTC")
    )
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="Pune tomorrow"))
    state = ConversationState.fold(log, playbook=pb)
    llm = CannedLLM(
        {"slots": {"city": "Pune", "date": "tomorrow"}, "advance": None, "note": None}
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert writes["date"] == "2026-06-25"  # normalized against the anchor


def test_verdict_prompt_no_date_block_when_no_date_slot() -> None:
    # A checkpoint with no date slot must not get the date block (and the
    # _VERDICT_PREAMBLE prefix is unaffected).
    import textwrap
    from superdialog.playbook.director import _verdict_prompt, _VERDICT_PREAMBLE
    from superdialog.playbook.events import (
        EventLog,
        AdvanceEvent,
        SessionStartEvent,
        UtteranceEvent,
    )
    from superdialog.playbook.state import ConversationState

    pb = Playbook.from_yaml(
        textwrap.dedent("""
        persona: "A."
        journeys:
          j:
            checkpoints:
              - id: ask
                slots: {name: {type: str}}
              - id: done
                terminal: true
    """)
    )
    log = EventLog()
    log.append(
        SessionStartEvent(started_at="2026-06-24T00:00:00+00:00", timezone="UTC")
    )
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.ask", rule="init"))
    log.append(UtteranceEvent(role="user", text="hi"))
    state = ConversationState.fold(log, playbook=pb)
    msg = _verdict_prompt(pb, pb.checkpoint("j.ask"), state)[0]
    assert "CURRENT DATE & TIME" not in msg["content"]
    assert msg["content"].startswith(_VERDICT_PREAMBLE)  # cache prefix intact


_PARTNER_YAML = textwrap.dedent("""
    multi_entity: true
    persona: "You collect details."
    journeys:
      main:
        checkpoints:
          - id: collect_partner
            entity: partner
            gate: soft
            slots:
              date_of_birth: {type: date, required: true, gate: soft}
            advance_when:
              - {when: "dob given", judge: llm, to: main.done,
                 requires: [date_of_birth]}
          - id: done
            terminal: true
            outcome: confirmed
""")


def _partner_state(text: str) -> tuple[Playbook, EventLog, ConversationState]:
    """At the partner checkpoint with the caller's DOB already confirmed."""
    pb = Playbook.from_yaml(_PARTNER_YAML)
    log = EventLog()
    log.append(
        SessionStartEvent(started_at="2026-06-24T00:00:00+00:00", timezone="UTC")
    )
    log.append(
        AdvanceEvent(
            from_checkpoint=None, to_checkpoint="main.collect_partner", rule="init"
        )
    )
    log.append(
        SlotWriteEvent(
            key="date_of_birth",
            value="1986-06-04",
            status="confirmed",
            by="director",
            entity="caller",
        )
    )
    log.append(UtteranceEvent(role="user", text=text))
    return pb, log, ConversationState.fold(log, playbook=pb)


async def test_director_tags_extraction_with_checkpoint_entity() -> None:
    pb, log, state = _partner_state("12 July 1986")
    llm = CannedLLM(
        {
            "slots": {"date_of_birth": "12 July 1986"},
            "advance": "main.done",
            "note": None,
        }
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert writes and all(e.entity == "partner" for e in writes)
    # Fold the Director's events onto the seeded log: partner DOB stored,
    # caller DOB intact (no clobber).
    for e in decision.events:
        log.append(e)
    final = ConversationState.fold(log, playbook=pb)
    assert final.slot_value("date_of_birth", entity="partner") == "1986-07-12"
    assert final.slot_value("date_of_birth", entity="caller") == "1986-06-04"
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "main.done"


async def test_single_entity_director_unchanged() -> None:
    # Backward compat: a non-multi-entity flow writes bare 'caller' keys.
    pb, state = _state()
    llm = CannedLLM(
        {
            "slots": {"city": "Pune", "date": "2026-06-11"},
            "advance": "booking.confirm",
            "note": None,
        }
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert {e.key for e in writes} == {"city", "date"}
    assert all(e.entity == "caller" for e in writes)


def test_verdict_prompt_states_entity_when_multi_entity() -> None:
    from superdialog.playbook.director import _verdict_prompt

    pb, _log, state = _partner_state("12 July 1986")
    cp = pb.checkpoint("main.collect_partner")
    system = _verdict_prompt(pb, cp, state)[0]["content"]
    # (a) the LLM is told whose details this checkpoint collects
    assert "You are collecting details for: partner" in system
    # (b) the known view groups by entity so it never asks "whose?": the
    # caller's already-confirmed DOB is shown labeled, distinct from partner's.
    assert "caller" in system and "1986-06-04" in system


def test_verdict_prompt_flat_when_not_multi_entity() -> None:
    # multi_entity off ⇒ byte-identical: no entity line, flat "Already known".
    from superdialog.playbook.director import _verdict_prompt

    pb, state = _state()
    cp = pb.checkpoint("booking.collect")
    system = _verdict_prompt(pb, cp, state)[0]["content"]
    assert "You are collecting details for:" not in system
    assert '"Already known"' not in system  # grouping not introduced
    assert "Already known: " in system  # today's flat shape
