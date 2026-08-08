import json
import textwrap

from superdialog.playbook.director import Director
from superdialog.playbook.events import (
    AdvanceEvent,
    DegradedEvent,
    EventLog,
    SessionStartEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
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
    # A goodbye at a step with REQUIRED slots unfilled first gets ONE steer to
    # collect them (terminal-interrupt slot guard); interrupts still override
    # advance rules — proven with the required slots filled in the verdict.
    pb, state = _state()
    llm = CannedLLM(
        {
            "slots": {"city": "Pune", "date": "2026-07-05"},
            "advance": "booking.confirm",  # interrupt must still win over this
            "note": None,
            "interrupt": "goodbye",
        }
    )
    decision = await Director(pb, llm).evaluate(state)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.close"
    assert adv[0].rule == "interrupt:goodbye"


class _RecordingLLM:
    """Records the kwargs the Director passes to ``complete``."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    async def complete(self, messages, **kwargs) -> str:
        self.kwargs = kwargs
        return json.dumps({"slots": {}, "advance": None, "note": None})


async def test_director_requests_json_mode_by_default() -> None:
    pb, state = _state()
    llm = _RecordingLLM()
    await Director(pb, llm).evaluate(state)
    assert llm.kwargs.get("json_mode") is True


async def test_director_structured_output_can_be_disabled() -> None:
    pb, state = _state()
    llm = _RecordingLLM()
    await Director(pb, llm, structured_output=False).evaluate(state)
    assert "json_mode" not in llm.kwargs


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


async def test_llm_error_recovers_on_retry() -> None:
    # A single transient failure (timeout, rate limit, gateway blip) must not
    # degrade the turn -- the Talker would barrier-wait on a Director that
    # never speaks, then improvise once its own timeout fired (a hallucinated
    # "(Wait for tool result)" turn was observed in production from exactly
    # this path). One retry recovers the common transient case.
    pb, state = _state()

    class FailsOnceLLM:
        def __init__(self) -> None:
            self.attempts = 0

        async def complete(self, messages, **kwargs) -> str:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient")
            return json.dumps({"slots": {}, "advance": None, "note": None})

    llm = FailsOnceLLM()
    decision = await Director(pb, llm).evaluate(state)
    assert not decision.degraded
    assert llm.attempts == 2


async def test_llm_error_still_degrades_after_retry_fails() -> None:
    pb, state = _state()

    class AlwaysRaisingLLM:
        def __init__(self) -> None:
            self.attempts = 0

        async def complete(self, messages, **kwargs) -> str:
            self.attempts += 1
            raise RuntimeError("boom")

    llm = AlwaysRaisingLLM()
    decision = await Director(pb, llm).evaluate(state)
    assert decision.degraded and decision.detail == "llm_error"
    assert llm.attempts == 2


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


async def test_terminal_advance_blocked_while_repair_active() -> None:
    """A repair note in flight means caller frustration, not closure: a verdict
    advancing to a terminal checkpoint is suppressed in favour of a recovery note."""
    pb = Playbook.from_yaml(HARD_GATE_YAML)
    state = _hard_state(pb)
    state.slots["otp"] = SlotValue(
        value="4242", status="confirmed", by="tool", version=1
    )
    state.steering_note = "you re-asked the code"
    state.steering_kind = "repair"
    llm = CannedLLM({"slots": {}, "advance": "pay.done", "note": None})
    decision = await Director(pb, llm).evaluate(state)
    assert not [e for e in decision.events if isinstance(e, AdvanceEvent)]
    recover = [
        e
        for e in decision.events
        if isinstance(e, SteeringNoteEvent) and e.kind == "repair"
    ]
    assert recover and "do NOT end" in recover[0].text


async def test_terminal_advance_proceeds_without_repair() -> None:
    """Backward compat: with no repair note, a normal close still ends the call."""
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


RESOLVE_FROM_YAML = textwrap.dedent("""
    journeys:
      main:
        checkpoints:
          - id: collect
            goal: "collect a course"
            slots:
              course_name:
                description: "Course name the caller said"
              course_id:
                resolve_from:
                  result: city_courses_result
                  list_field: courses
                  name_field: name
                  id_field: course_id
                description: "Confirmed course_id if already identified"
            advance_when:
              - {when: "course confirmed", judge: llm, to: main.done}
          - id: done
            terminal: true
            outcome: completed
""")


def _resolve_from_state(with_candidates: bool) -> tuple[Playbook, ConversationState]:
    pb = Playbook.from_yaml(RESOLVE_FROM_YAML)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="main.collect", rule="init"))
    log.append(UtteranceEvent(role="user", text="Book for DLF Golf Course"))
    if with_candidates:
        log.append(
            ToolResultEvent(
                tool="courses_by_city",
                store_as="city_courses_result",
                ok=True,
                data={"courses": [{"name": "DLF Golf and Country Club", "course_id": "course_ddfd8225"}]},
            )
        )
    return pb, ConversationState.fold(log, playbook=pb)


async def test_resolve_from_slot_rejects_raw_name_when_candidate_list_missing() -> None:
    # The real production failure: city_courses_result was never fetched
    # (caller went straight from greeting to booking collection), yet the
    # Director verdict still tried to write the caller's spoken name
    # straight into course_id -- an opaque id field, never something the
    # caller utters verbatim. Must be rejected, not stored as a fake id.
    pb, state = _resolve_from_state(with_candidates=False)
    llm = CannedLLM(
        {
            "slots": {"course_id": "DLF Golf Course", "course_name": "DLF Golf Course"},
            "advance": None,
            "note": None,
        }
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert "course_id" not in writes  # rejected: not a live candidate id
    assert writes.get("course_name") == "DLF Golf Course"  # plain slot: unaffected


async def test_resolve_from_slot_accepts_a_real_candidate_id() -> None:
    pb, state = _resolve_from_state(with_candidates=True)
    llm = CannedLLM(
        {"slots": {"course_id": "course_ddfd8225"}, "advance": None, "note": None}
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert writes.get("course_id") == "course_ddfd8225"


async def test_resolve_from_slot_rejects_non_candidate_even_when_list_present() -> None:
    # A live candidate list existing this turn doesn't excuse a value that
    # isn't actually one of the listed ids (model hallucination, stale id).
    pb, state = _resolve_from_state(with_candidates=True)
    llm = CannedLLM(
        {"slots": {"course_id": "DLF Golf Course"}, "advance": None, "note": None}
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert "course_id" not in writes


async def test_resolve_block_instructs_bare_affirmation_uses_assistant_offer() -> None:
    # A caller confirming ("yes", "hold that") never restates the name/time
    # the assistant just offered -- CANDIDATE RESOLUTION must tell the
    # Director to match the ASSISTANT's own preceding turn against the
    # candidate list in that case, not just the caller's current utterance
    # (which is empty of any matchable name). Prompt-content regression
    # guard, mirrors test_verdict_prompt_lists_enum_members' pattern: the
    # LLM's actual reasoning can't be unit tested, but the instruction it's
    # given can be.
    pb, state = _resolve_from_state(with_candidates=True)
    llm = CannedLLM({"slots": {}, "advance": None, "note": None})
    await Director(pb, llm).evaluate(state)
    system = llm.calls[0][0]["content"]
    assert "BARE AFFIRMATION" in system
    assert "ASSISTANT" in system


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


async def test_verdict_prompt_lists_enum_members() -> None:
    # Root cause of a production bug: _coerce_slot validates enum extractions
    # by strict membership (value in spec.values), but the prompt previously
    # showed only the literal word "enum" for an enum slot's type -- never
    # the actual allowed member strings. A model extracting a perfectly
    # sensible free-text form ("one" for a caller who said "one player")
    # silently failed that membership check and the slot was dropped with
    # no error anywhere. Asserting the members appear in the prompt is the
    # regression guard for that gap.
    pb = Playbook.from_yaml(TYPED_SLOTS_YAML)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.c", rule="init"))
    log.append(UtteranceEvent(role="user", text="mode a"))
    state = ConversationState.fold(log, playbook=pb)
    llm = CannedLLM({"slots": {}, "advance": None, "note": None})
    await Director(pb, llm).evaluate(state)
    system = llm.calls[0][0]["content"]
    assert "values=a|b" in system


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


# STRUCTURAL non-values: artifacts no caller utters. Semantic strings
# ('none', 'unknown', 'not specified') are the LLM's judgment — the verdict
# prompt's decline-convention makes them deliberate answers, and blindly
# filtering them killed a production call (ENDSESSION: caller's "No." junked
# into a re-ask loop).
_JUNK_TABLE = [
    "",
    "NULL",
    "null",
    " ",
    None,  # JSON null: str coercion would mint the literal 'None' string
]

_TRUSTED_SEMANTIC_VALUES = ["None", "none", "n/a", "Not Specified", "unknown"]


async def test_structural_junk_rejected_under_v2() -> None:
    for junk in _JUNK_TABLE:
        pb, state = _state()
        llm = CannedLLM({"slots": {"city": junk}, "advance": None, "note": None})
        decision = await Director(pb, llm).evaluate(state)
        writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
        assert not [e for e in writes if e.key == "city"], f"wrote junk {junk!r}"
        degraded = [e for e in decision.events if isinstance(e, DegradedEvent)]
        assert any(e.detail == "junk_rejected:city" for e in degraded), repr(junk)


async def test_semantic_values_trusted_to_the_llm_under_v2() -> None:
    # 'none'/'unknown'/'n/a' are STATED answers under the prompt's
    # decline-convention — the engine must not override the LLM's judgment.
    for value in _TRUSTED_SEMANTIC_VALUES + ["Noneville"]:
        pb, state = _state()
        llm = CannedLLM({"slots": {"city": value}, "advance": None, "note": None})
        decision = await Director(pb, llm).evaluate(state)
        writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
        assert [e for e in writes if e.key == "city"], f"filtered {value!r}"
        assert not [e for e in decision.events if isinstance(e, DegradedEvent)]


async def test_junk_rejected_detail_is_entity_namespaced_off_caller() -> None:
    # Off-caller checkpoints namespace the junk detail as
    # junk_rejected:<entity>:<key>, mirroring slot_churn's {entity}:{key}
    # keying (caller stays bare for backward compat).
    pb, _log, state = _partner_state("umm")
    llm = CannedLLM({"slots": {"date_of_birth": ""}, "advance": None, "note": None})
    decision = await Director(pb, llm).evaluate(state)
    degraded = [e for e in decision.events if isinstance(e, DegradedEvent)]
    assert any(e.detail == "junk_rejected:partner:date_of_birth" for e in degraded)


async def test_allow_empty_slot_accepts_empty_string_as_a_real_answer() -> None:
    # Real production defect: special_requests='' on a caller's genuine
    # decline ("no add-ons") was rejected as junk every time, so the expr
    # rule gating the booking hold on `special_requests != None` never
    # fired -- the caller repeated "no" three times and the flow eventually
    # narrated a fabricated success with no real hold ever placed.
    # allow_empty: true declares "" as THIS slot's own legitimate value.
    yaml_text = textwrap.dedent("""
        persona: "Assistant."
        journeys:
          j:
            checkpoints:
              - id: ask
                goal: "collect special requests"
                gate: soft
                slots:
                  special_requests: {type: str, allow_empty: true}
                advance_when:
                  - {when: "done", judge: llm, to: j.done}
              - id: done
                terminal: true
    """)
    pb = Playbook.from_yaml(yaml_text)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.ask", rule="init"))
    log.append(UtteranceEvent(role="user", text="no add-ons"))
    state = ConversationState.fold(log, playbook=pb)
    llm = CannedLLM(
        {"slots": {"special_requests": ""}, "advance": None, "note": None}
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert any(e.key == "special_requests" and e.value == "" for e in writes)
    assert not [e for e in decision.events if isinstance(e, DegradedEvent)]
    # "none" is a SEMANTIC value, not structural junk: the verdict prompt's
    # decline-convention makes it a stated answer (G4 redesign — a lexical
    # filter junking a caller's "No." re-ask-looped a production call).
    # Structural junk (the literal "null") is still rejected.
    llm2 = CannedLLM(
        {"slots": {"special_requests": "none"}, "advance": None, "note": None}
    )
    decision2 = await Director(pb, llm2).evaluate(state)
    writes2 = [e for e in decision2.events if isinstance(e, SlotWriteEvent)]
    assert any(e.key == "special_requests" and e.value == "none" for e in writes2)
    llm3 = CannedLLM(
        {"slots": {"special_requests": "null"}, "advance": None, "note": None}
    )
    decision3 = await Director(pb, llm3).evaluate(state)
    writes3 = [e for e in decision3.events if isinstance(e, SlotWriteEvent)]
    assert not [e for e in writes3 if e.key == "special_requests"]
    degraded3 = [e for e in decision3.events if isinstance(e, DegradedEvent)]
    assert any(e.detail == "junk_rejected:special_requests" for e in degraded3)


async def test_junk_slot_values_pass_under_legacy() -> None:
    # legacy_continuity: true preserves the old accept-anything behavior
    # byte-for-byte: 'None' is written, no junk_rejected audit event.
    pb = Playbook.from_yaml(MINIMAL_YAML).model_copy(
        update={"legacy_continuity": True}
    )
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="Pune tomorrow please"))
    state = ConversationState.fold(log, playbook=pb)
    for raw in ["None", None]:  # legacy also keeps null -> str-coerced "None"
        llm = CannedLLM({"slots": {"city": raw}, "advance": None, "note": None})
        decision = await Director(pb, llm).evaluate(state)
        writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
        assert [e for e in writes if e.key == "city" and e.value == "None"], repr(raw)
        assert not [e for e in decision.events if isinstance(e, DegradedEvent)]


ENUM_JUNK_MEMBER_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: c
            slots:
              tier: {type: enum, values: ["unknown", "basic"]}
          - id: end
            terminal: true
""")


async def test_declared_enum_member_is_never_junk_under_v2() -> None:
    # A junk-table token that is authored enum vocabulary must still fill:
    # membership trumps the junk table.
    pb = Playbook.from_yaml(ENUM_JUNK_MEMBER_YAML)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.c", rule="init"))
    log.append(UtteranceEvent(role="user", text="not sure, unknown I guess"))
    state = ConversationState.fold(log, playbook=pb)
    llm = CannedLLM({"slots": {"tier": "unknown"}, "advance": None, "note": None})
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert [e for e in writes if e.key == "tier" and e.value == "unknown"]
    assert not [e for e in decision.events if isinstance(e, DegradedEvent)]


async def test_identical_confirmed_rewrite_is_dropped_under_v2() -> None:
    # westgate2 wrote `staying` 4x with the identical value: re-extracting
    # an already-confirmed value must produce no event (no version churn).
    for raw in ["Pune", "  Pune  "]:  # comparison is on the COERCED value
        pb, state = _state(
            extra_events=[
                SlotWriteEvent(
                    key="city", value="Pune", status="confirmed", by="director"
                )
            ]
        )
        llm = CannedLLM({"slots": {"city": raw}, "advance": None, "note": None})
        decision = await Director(pb, llm).evaluate(state)
        writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
        assert not [e for e in writes if e.key == "city"], repr(raw)
        assert not [e for e in decision.events if isinstance(e, DegradedEvent)]


async def test_identical_provisional_rewrite_still_written_under_v2() -> None:
    # Re-extraction of a provisional value is meaningful signal
    # (fast-release/confirmation flows) — the rewrite IS emitted.
    pb, state = _state(
        extra_events=[
            SlotWriteEvent(
                key="city", value="Pune", status="provisional", by="director"
            )
        ]
    )
    llm = CannedLLM({"slots": {"city": "Pune"}, "advance": None, "note": None})
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert [e for e in writes if e.key == "city" and e.value == "Pune"]


async def test_changed_value_rewrite_still_written_under_v2() -> None:
    # A correction must flow: confirmed "Pune" -> verdict "Mumbai" is written.
    pb, state = _state(
        extra_events=[
            SlotWriteEvent(
                key="city", value="Pune", status="confirmed", by="director"
            )
        ]
    )
    llm = CannedLLM({"slots": {"city": "Mumbai"}, "advance": None, "note": None})
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert [e for e in writes if e.key == "city" and e.value == "Mumbai"]


async def test_identical_confirmed_rewrite_kept_under_legacy() -> None:
    # legacy_continuity: true preserves the old rewrite-everything behavior.
    pb = Playbook.from_yaml(MINIMAL_YAML).model_copy(
        update={"legacy_continuity": True}
    )
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="Pune tomorrow please"))
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    state = ConversationState.fold(log, playbook=pb)
    llm = CannedLLM({"slots": {"city": "Pune"}, "advance": None, "note": None})
    decision = await Director(pb, llm).evaluate(state)
    writes = [e for e in decision.events if isinstance(e, SlotWriteEvent)]
    assert [e for e in writes if e.key == "city" and e.value == "Pune"]
