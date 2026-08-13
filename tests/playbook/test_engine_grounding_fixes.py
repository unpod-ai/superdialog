"""Engine-level grounding fixes, each pinned to a real production incident.

These cover three framework defects found by replaying five live golfai calls
against the real playbook. Every one had previously been worked around inside
the YAML -- a hand-written Jinja guard at each of 14 tool injection points, an
ISO-shape gate at 5 more, extra advance rules at 3 checkpoints -- because the
engine gave the author no hook. These tests exist so the engine keeps the
guarantee instead, for every playbook rather than the two that were patched.

  #2  director._coerce_slot: `type: date` accepted unresolvable prose
  #3  director unknown_advance_target dropped the whole TURN, not just the edge
  #4  no way to ask "did this tool actually run" -- new `calls` namespace

(The fourth fix, sentence-scoped never_say on the token stream, lives with the
other stream-filter tests in test_audit_guards.py.)
"""

import textwrap
from datetime import date, datetime

from superdialog.playbook.director import (
    _INVALID,
    _ISO_DATE_RE,
    Director,
    _coerce_slot,
)
from superdialog.playbook.events import (
    AdvanceEvent,
    DegradedEvent,
    EventLog,
    SessionStartEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from superdialog.playbook.expr import evaluate
from superdialog.playbook.models import Playbook, SlotSpec
from superdialog.playbook.render import render_template, template_namespace
from superdialog.playbook.state import ConversationState
from superdialog.playbook.toolexec import _template_ns
from tests.playbook.test_director import CannedLLM, _state

NOW = datetime(2026, 8, 13, 12, 0)


# --- #2  type: date must actually enforce -------------------------------------
#
# normalize_date is pass-through on failure by contract, so an unresolvable
# phrase came back verbatim and was STORED as the date. It then reached tool
# templates as-is and shipped as ?date=this weekend, which the availability API
# answers with HTTP 500 -- so the caller is told nothing is available when
# nothing was ever checked. Observed live on two of five replayed calls.


def _date(value):
    return _coerce_slot(value, SlotSpec(type="date"), NOW)


def test_unresolvable_date_prose_is_rejected_not_stored():
    for prose in (
        "this weekend",  # the live value that produced ?date=this weekend
        "weekend",
        "soon",
        "whenever",
        "sometime next quarter",
        "flexible",
        "none",  # director keeps 'none' as a real value; it is still not a date
        "",
    ):
        assert _date(prose) is _INVALID, f"{prose!r} was accepted as a date"


def test_resolvable_dates_still_normalize_to_iso():
    # Relative resolution is the whole point of the date type and must survive.
    assert _date("tomorrow") == "2026-08-14"
    assert _date("15 August") == "2026-08-15"
    assert _date("2026-08-20") == "2026-08-20"
    for v in ("tomorrow", "next friday", "15 August", "2026-08-20"):
        assert _ISO_DATE_RE.fullmatch(_date(v)), v


def _future(value):
    return _coerce_slot(value, SlotSpec(type="date", future_only=True), NOW)


def test_future_only_rejects_a_well_formed_past_date():
    """Shape is not bookability.

    Live: the caller said "this Saturday, the 10th of June" (a contradiction --
    10 June is not a Saturday) and the Director emitted "2024-06-10", two years
    before the call. An ISO-shape check waves that through and availability was
    requested for 2024.
    """
    assert _future("2024-06-10") is _INVALID
    assert _future("2026-08-12") is _INVALID  # yesterday relative to NOW
    assert _future("2026-08-13") == "2026-08-13"  # today is still bookable
    assert _future("2026-08-14") == "2026-08-14"


def test_future_only_is_opt_in_so_past_dates_stay_legal_by_default():
    """Rejecting the past by DEFAULT was tried and is wrong: date_of_birth is a
    date slot whose whole purpose is a past value, and defaulting to reject
    stopped it advancing (test_two_entity_dob_collision_regression). Only the
    author knows which date slots are forward-looking."""
    assert _date("1986-06-04") == "1986-06-04"
    assert _date("2024-06-10") == "2024-06-10"


def test_future_only_still_rejects_prose():
    """future_only relaxes nothing about the shape check."""
    assert _future("this weekend") is _INVALID
    assert _future("soon") is _INVALID


def test_impossible_calendar_date_is_rejected_even_when_iso_shaped():
    assert _date("2026-02-30") is _INVALID
    assert _date("2026-13-01") is _INVALID


def test_future_check_is_skipped_without_a_call_anchor():
    """No anchor means no "today" to compare against; the shape check still
    applies, so this cannot regress to storing prose."""
    spec = SlotSpec(type="date", future_only=True)
    assert _coerce_slot("2024-06-10", spec, None) == "2024-06-10"
    assert _coerce_slot("this weekend", spec, None) is _INVALID


def test_date_objects_are_accepted_and_isoformatted():
    assert _date(date(2026, 8, 20)) == "2026-08-20"
    assert _date(datetime(2026, 8, 20, 9, 30)) == "2026-08-20"


def test_date_rejection_now_matches_time_and_enum_parity():
    """date was the lone type that did not reject; the others already did."""
    assert _coerce_slot("half past nowhere", SlotSpec(type="time")) is _INVALID
    assert _coerce_slot("five", SlotSpec(type="enum", values=["1", "2"])) is _INVALID
    assert _date("half past nowhere") is _INVALID


async def test_director_does_not_write_a_prose_date_end_to_end():
    """Full path: a verdict claiming a prose date writes no date slot, so a
    requires-gated advance cannot fire on garbage."""
    pb, state = _state()
    llm = CannedLLM(
        {"slots": {"city": "Pune", "date": "this weekend"}, "advance": None}
    )
    decision = await Director(pb, llm).evaluate(state)
    written = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert written.get("city") == "Pune"
    assert "date" not in written, f"prose date reached state: {written}"


async def test_director_still_writes_a_resolvable_date_end_to_end():
    """With the per-call anchor present -- the production path -- a relative
    date the caller actually said still resolves and is still written."""
    pb, state = _state(
        [SessionStartEvent(started_at="2026-08-13T12:00:00+00:00", timezone="UTC")]
    )
    assert state.now is not None, "anchor missing; this test would prove nothing"
    llm = CannedLLM({"slots": {"date": "tomorrow"}, "advance": None})
    decision = await Director(pb, llm).evaluate(state)
    written = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert written.get("date") == "2026-08-14", written


async def test_relative_date_without_a_call_anchor_is_dropped_not_stored():
    """Documents the one behaviour cliff this fix introduces.

    normalize_date needs ``state.now`` (from SessionStartEvent) to do weekday
    arithmetic; without it "tomorrow" is unresolvable. Previously it was stored
    verbatim and shipped as ?date=tomorrow -- the HTTP 500 path. Now the slot
    stays unwritten and the checkpoint re-asks: degraded, but a caller being
    asked again beats a caller being told nothing is available when nothing was
    checked, on a slot that drives an irreversible booking.
    """
    pb, state = _state()  # no SessionStartEvent -> state.now is None
    assert state.now is None
    llm = CannedLLM({"slots": {"date": "tomorrow"}, "advance": None})
    decision = await Director(pb, llm).evaluate(state)
    written = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert "date" not in written
    # An absolute date needs no anchor and is unaffected.
    llm2 = CannedLLM({"slots": {"date": "2026-08-20"}, "advance": None})
    decision2 = await Director(pb, llm2).evaluate(_state()[1])
    written2 = {
        e.key: e.value for e in decision2.events if isinstance(e, SlotWriteEvent)
    }
    assert written2.get("date") == "2026-08-20"


# --- #3  an unknown advance target must not silently eat the turn -------------
#
# The verdict named checkpoints that do not exist (main.payment,
# main.check_availability, mainTEE_TIME_SEARCH...). The edge was dropped and
# logged, but so was the turn: the Talker re-asked the same question, giving
# camps of 6-9 turns on one checkpoint -- and a camped Talker eventually runs
# out of legitimate things to say and invents a slot, a price, a payment link.


async def _unknown_target_decision():
    pb, state = _state()
    llm = CannedLLM({"slots": {}, "advance": "booking.does_not_exist"})
    return await Director(pb, llm).evaluate(state)


async def test_unknown_advance_target_still_logs_degraded():
    decision = await _unknown_target_decision()
    degraded = [e for e in decision.events if isinstance(e, DegradedEvent)]
    assert any("unknown_advance_target" in e.detail for e in degraded), decision.events
    assert not [e for e in decision.events if isinstance(e, AdvanceEvent)]


async def test_unknown_advance_target_steers_instead_of_dead_turn():
    """The fix: the turn now carries a correction, so the Talker does not just
    repeat itself into a camp."""
    decision = await _unknown_target_decision()
    steers = [e for e in decision.events if isinstance(e, SteeringNoteEvent)]
    assert steers, "unknown target produced no steer -- the turn is still dead"
    assert any("does not exist" in e.text for e in steers)


async def test_unknown_advance_target_steer_leaks_no_checkpoint_ids():
    """The steer renders into the Talker's SYSTEM prompt, so it must not name
    internal ids -- those would be spoken aloud."""
    decision = await _unknown_target_decision()
    for e in decision.events:
        if isinstance(e, SteeringNoteEvent):
            assert "booking." not in e.text
            assert "does_not_exist" not in e.text


async def test_self_target_is_not_treated_as_unknown():
    """A verdict naming the checkpoint it is already on means "stay here".

    That correctly produces no advance, so there is nothing to correct and no
    camp to break. Steering on it told the Talker its own step did not exist on
    an ordinary stay-put turn -- seen twice in one live session. Still logged,
    because it is worth auditing, but no steer.
    """
    pb, state = _state()
    assert state.checkpoint_id == "booking.collect"
    llm = CannedLLM({"slots": {}, "advance": "booking.collect"})
    decision = await Director(pb, llm).evaluate(state)
    assert not [
        e
        for e in decision.events
        if isinstance(e, SteeringNoteEvent) and "does not exist" in e.text
    ], "self-target produced a bogus 'step does not exist' steer"
    assert not [e for e in decision.events if isinstance(e, AdvanceEvent)]


async def test_a_real_advance_target_is_unaffected():
    pb, state = _state()
    llm = CannedLLM(
        {"slots": {"city": "Pune", "date": "2026-08-20"}, "advance": "booking.confirm"}
    )
    decision = await Director(pb, llm).evaluate(state)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.confirm"
    assert not [
        e
        for e in decision.events
        if isinstance(e, DegradedEvent) and "unknown_advance_target" in e.detail
    ]


# --- anchor: no-op re-writes must not dilute the signal -----------------------
#
# The identical-confirmed-value check now runs BEFORE the anchor check. Before
# that, every carry-forward re-write logged anchor_miss (city='Gurugram' on a
# turn where the caller never repeated the city), so nearly every write in a
# session was flagged -- correct ones included -- and anchor="enforce" looked
# unusable. That left the only guard capable of catching an invented slot
# (preferred_time='afternoon' from a caller who named only a day) stuck in
# audit-only shadow mode.


async def test_identical_confirmed_rewrite_logs_no_anchor_miss():
    """The no-op case: the slot already holds this exact confirmed value, so it
    needs no fresh evidence and must not be flagged."""
    pb, state = _state(
        [SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")]
    )
    llm = CannedLLM({"slots": {"city": "Pune"}, "advance": None})
    decision = await Director(pb, llm, anchor="shadow").evaluate(state)
    misses = [
        e
        for e in decision.events
        if isinstance(e, DegradedEvent) and "anchor_miss" in e.detail
    ]
    assert not misses, f"no-op re-write still flagged: {[e.detail for e in misses]}"


async def test_a_new_unanchored_value_is_still_flagged():
    """The signal must survive: a NEW value absent from the caller's turn is
    exactly what anchor_miss is for."""
    pb, state = _state()  # last user text: "Pune tomorrow please"
    llm = CannedLLM({"slots": {"city": "Bengaluru"}, "advance": None})
    decision = await Director(pb, llm, anchor="shadow").evaluate(state)
    assert [
        e
        for e in decision.events
        if isinstance(e, DegradedEvent) and "anchor_miss" in e.detail
    ], "an unanchored new value was not flagged"


async def test_enforce_drops_a_new_unanchored_value():
    pb, state = _state()
    llm = CannedLLM({"slots": {"city": "Bengaluru"}, "advance": None})
    decision = await Director(pb, llm, anchor="enforce").evaluate(state)
    written = {e.key for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert "city" not in written, "enforce let an unanchored value through"


async def test_enforce_keeps_a_value_the_caller_actually_said():
    """The risk of enforce is dropping CORRECT writes. A value present in the
    caller's turn must survive it."""
    pb, state = _state()  # "Pune tomorrow please"
    llm = CannedLLM({"slots": {"city": "Pune"}, "advance": None})
    decision = await Director(pb, llm, anchor="enforce").evaluate(state)
    written = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert written.get("city") == "Pune", written


# --- per-slot anchor: enforce where the caller speaks, not everywhere ---------
#
# Session-wide anchor="enforce" was tried live and was actively harmful: 4/5,
# 3/5, 4/5 against 5/5, 4/5, 5/5 on shadow, and zero bookings completed in 15
# sessions against 3 of 5. It rejects DERIVED slots -- "8 AM" legitimately
# becomes time_from=07:00 / time_to=09:00 and "07:00" is not in "8 am" -- so
# correct writes were dropped, availability never fired, and the Talker,
# starved of data, invented MORE prices. Per-slot is the fix: enforce on
# caller-stated slots only.


async def test_slot_level_enforce_rejects_a_value_with_no_span():
    """The S5 bug: preferred_time silently set to a value the caller never said.

    Legal value, nothing spoken wrongly, so neither never_say nor type: time
    can see it. Only the anchor can.
    """
    pb, state = _state()  # caller said: "Pune tomorrow please"
    cp = pb.checkpoint("booking.collect")
    cp.slots["city"].anchor = "enforce"
    llm = CannedLLM({"slots": {"city": "Bengaluru"}, "advance": None})
    # Session default stays shadow; the slot's own override does the work.
    decision = await Director(pb, llm, anchor="shadow").evaluate(state)
    written = {e.key for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert "city" not in written, "slot-level enforce did not reject an unanchored write"


async def test_slot_level_inherit_leaves_other_slots_alone():
    """Marking one slot must not enforce the rest -- that is the whole point."""
    pb, state = _state()
    cp = pb.checkpoint("booking.collect")
    cp.slots["city"].anchor = "enforce"
    assert cp.slots["date"].anchor == "inherit"
    llm = CannedLLM(
        {"slots": {"city": "Bengaluru", "date": "2026-08-20"}, "advance": None}
    )
    decision = await Director(pb, llm, anchor="shadow").evaluate(state)
    written = {e.key for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert "city" not in written, "enforced slot leaked through"
    assert "date" in written, "an inherit slot was wrongly enforced"


async def test_slot_level_off_exempts_a_derived_slot_under_global_enforce():
    """The inverse: a derived slot can opt OUT even when the session enforces,
    so a session-wide policy cannot starve the tool path."""
    pb, state = _state()
    cp = pb.checkpoint("booking.collect")
    cp.slots["city"].anchor = "off"
    llm = CannedLLM({"slots": {"city": "Bengaluru"}, "advance": None})
    decision = await Director(pb, llm, anchor="enforce").evaluate(state)
    written = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert written.get("city") == "Bengaluru", written


async def test_enforced_slot_still_accepts_a_value_the_caller_said():
    """The risk of enforce is dropping CORRECT writes. A value present in the
    caller's turn must survive it."""
    pb, state = _state()  # "Pune tomorrow please"
    cp = pb.checkpoint("booking.collect")
    cp.slots["city"].anchor = "enforce"
    llm = CannedLLM({"slots": {"city": "Pune"}, "advance": None})
    decision = await Director(pb, llm, anchor="shadow").evaluate(state)
    written = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert written.get("city") == "Pune", written


async def test_enforced_slot_accepts_a_normalized_value_via_its_span():
    """Why per-slot enforce is safe on preferred_time but not on time_from.

    A non-date/time slot with a valid span anchors whatever its normalized form
    (director.py's _anchor_ok returns True on a present span), so "8 AM" -> a
    reshaped value still lands. A derived slot has no span by construction and
    would be rejected -- which is why only caller-stated slots are marked.
    """
    pb, state = _state()  # utterance contains "Pune"
    cp = pb.checkpoint("booking.collect")
    cp.slots["city"].anchor = "enforce"
    llm = CannedLLM(
        {
            "slots": {"city": "PUNE-WEST"},  # normalized, not literally in the turn
            "spans": {"city": "Pune"},  # but the evidence is
            "advance": None,
        }
    )
    decision = await Director(pb, llm, anchor="shadow").evaluate(state)
    written = {e.key: e.value for e in decision.events if isinstance(e, SlotWriteEvent)}
    assert written.get("city") == "PUNE-WEST", written


# --- #4  `calls`: distinguish "returned nothing" from "never ran" -------------
#
# A tool skipped by its own `when:` guard returns [] from ToolExecutor.execute
# -- no ToolResultEvent at all -- so results.X is absent, which reads exactly
# like a call that ran and found nothing. A checkpoint told callers it had just
# checked availability, four turns running, in a session with zero availability
# calls. `calls` is the signal that was missing.

CALLS_YAML = textwrap.dedent(
    """
    persona: "p"
    journeys:
      main:
        checkpoints:
          - id: step
            guidance: "checked={{ 'YES' if calls['hold_slot'] else 'NO' }}"
            advance_when:
              - {when: "calls['hold_slot'] == 0", judge: expr, to: main.done}
          - id: done
            terminal: true
    tools:
      - id: hold_slot
        url: "https://api.test/x?ran={{ calls['hold_slot'] }}"
        store_response_as: hold_result
    """
)


def _calls_state(*tool_ids):
    pb = Playbook.from_yaml(CALLS_YAML)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="main.step", rule="init"))
    for tid in tool_ids:
        log.append(ToolCallEvent(tool=tid, args={}))
    return pb, ConversationState.fold(log, playbook=pb)


def test_calls_is_zero_when_a_tool_never_ran():
    pb, state = _calls_state()
    assert state.tool_call_counts.get("hold_slot", 0) == 0
    assert render_template("{{ calls['hold_slot'] }}", pb, state) == "0"
    assert "checked=NO" in render_template(pb.checkpoint("main.step").guidance, pb, state)


def test_calls_counts_attempts_once_the_tool_ran():
    pb, state = _calls_state("hold_slot", "hold_slot")
    assert render_template("{{ calls['hold_slot'] }}", pb, state) == "2"
    assert "checked=YES" in render_template(
        pb.checkpoint("main.step").guidance, pb, state
    )


def test_calls_is_available_to_expr_gates():
    _pb, state = _calls_state()
    assert evaluate("calls['hold_slot'] == 0", state) is True
    _pb2, state2 = _calls_state("hold_slot")
    assert evaluate("calls['hold_slot'] == 0", state2) is False
    # An id never seen is 0, not an error that collapses the whole expression.
    assert evaluate("calls['no-such-tool'] == 0", state) is True


def test_calls_is_available_to_tool_templates():
    _pb, state = _calls_state("hold_slot")
    assert _template_ns(state)["calls"]["hold_slot"] == 1
    assert _template_ns(state)["calls"]["never-seen"] == 0


def test_calls_namespace_is_additive_only():
    """Nothing existing changed name or meaning."""
    pb, state = _calls_state()
    ns = template_namespace(pb, state)
    assert {"slots", "views", "results", "calls"} <= set(ns)
    assert {"slots", "env", "results", "calls"} <= set(_template_ns(state))


def test_calls_distinguishes_skipped_from_returned_nothing():
    """The exact confusion that produced the spoken lie.

    A tool that ran and returned an empty payload counts 1 in `calls`; a tool
    that never ran counts 0. Reading `results` alone, both look identical.
    """
    pb = Playbook.from_yaml(CALLS_YAML)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="main.step", rule="init"))
    log.append(ToolCallEvent(tool="hold_slot", args={}))
    log.append(
        ToolResultEvent(tool="hold_slot", store_as="hold_result", ok=True, data={})
    )
    ran_empty = ConversationState.fold(log, playbook=pb)
    _pb, never_ran = _calls_state()

    # results renders empty in BOTH cases -- the ambiguity that caused the bug.
    assert not render_template("{{ results.hold_result.data }}", pb, never_ran).strip()
    # calls separates them.
    assert render_template("{{ calls['hold_slot'] }}", pb, ran_empty) == "1"
    assert render_template("{{ calls['hold_slot'] }}", pb, never_ran) == "0"
