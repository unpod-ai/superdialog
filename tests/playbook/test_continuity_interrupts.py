from superdialog.playbook.events import AdvanceEvent, DegradedEvent
from tests.playbook.continuity_fixtures import make_runtime

INTERRUPT = {
    "slots": {},
    "advance": None,
    "note": None,
    "interrupt": "price_guardrail",
}
PLAIN = {"slots": {}, "advance": None, "note": None}

_rt = make_runtime


async def test_self_interrupt_holds_detour_instead_of_self_looping() -> None:
    """westgate2 steps 10-12: a second price question while already inside
    pricing_faq must NOT re-fire the interrupt onto itself."""
    rt = _rt([INTERRUPT, INTERRUPT, PLAIN])
    await rt.start()
    await rt.on_user_text("what's the price?")          # detour in
    assert rt.state.checkpoint_id == "main.pricing_faq"
    assert rt.state.resume_stack == ["main.ask_location"]

    await rt.on_user_text("and any discounts?")          # self-interrupt turn
    # stays in the detour: no self-loop advance, stack unchanged
    assert rt.state.checkpoint_id == "main.pricing_faq"
    assert rt.state.resume_stack == ["main.ask_location"]
    self_loops = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent)
        and e.from_checkpoint == e.to_checkpoint == "main.pricing_faq"
    ]
    assert self_loops == []
    assert rt.state.steering_note  # the hold-open steer is live

    await rt.on_user_text("ok thanks")                   # detour done
    assert rt.state.checkpoint_id == "main.ask_location"  # resumed correctly
    assert rt.state.resume_stack == []


async def test_interrupt_can_refire_after_detour_completes() -> None:
    """E2: a completed detour must not kill the interrupt for the whole call."""
    rt = _rt([INTERRUPT, PLAIN, INTERRUPT])
    await rt.start()
    await rt.on_user_text("price?")                      # detour in
    await rt.on_user_text("thanks")                      # resume out
    assert rt.state.checkpoint_id == "main.ask_location"

    await rt.on_user_text("wait, price again?")          # must fire AGAIN
    assert rt.state.checkpoint_id == "main.pricing_faq"


async def test_unknown_advance_target_is_logged_not_silent() -> None:
    rt = _rt([{"slots": {}, "advance": "main.nonexistent", "note": None}])
    await rt.start()
    await rt.on_user_text("hello")
    assert rt.state.checkpoint_id == "main.ask_location"  # no advance
    assert any(
        isinstance(e, DegradedEvent)
        and e.detail == "unknown_advance_target:main.nonexistent"
        for e in rt.log.events
    )


async def test_unknown_interrupt_id_is_logged_and_falls_through() -> None:
    """Sibling of unknown_advance_target: a verdict naming an interrupt id no
    spec declares must be auditable, and must NOT swallow the rest of the
    verdict — a combined advance still lands."""
    rt = _rt(
        [
            {"slots": {}, "advance": None, "note": None, "interrupt": "no_such_id"},
            {
                "slots": {"location": "Pune"},
                "advance": "main.pitch",
                "interrupt": "no_such_id",
            },
        ]
    )
    await rt.start()
    await rt.on_user_text("hello")
    assert rt.state.checkpoint_id == "main.ask_location"  # no phantom detour
    degraded = [
        e
        for e in rt.log.events
        if isinstance(e, DegradedEvent)
        and e.detail == "unknown_interrupt_id:no_such_id"
    ]
    assert len(degraded) == 1

    # Fall-through: same-verdict advance (corroborated by the location slot
    # write) is still honored despite the bogus interrupt id.
    await rt.on_user_text("I'm in Pune")
    assert rt.state.checkpoint_id == "main.pitch"
    degraded = [
        e
        for e in rt.log.events
        if isinstance(e, DegradedEvent)
        and e.detail == "unknown_interrupt_id:no_such_id"
    ]
    assert len(degraded) == 2


async def test_goodbye_interrupt_to_terminal_carries_closing_steer() -> None:
    """A goodbye-class interrupt into a TERMINAL checkpoint must steer the
    Talker to a brief close: simple-format playbooks have no authored
    verbatim on the closing step, and an unguided Talker free-wheels into
    offers/pitches on the goodbye turn (Rohan-1 eval, ts 0.3-0.4 every run)."""
    rt = _rt([{"slots": {}, "advance": None, "note": None,
               "interrupt": "global_goodbye"}])
    await rt.start()
    await rt.on_user_text("Then I'm good. Goodbye.")
    assert rt.state.ended
    assert rt.state.steering_note is not None
    assert "no questions, no offers" in rt.state.steering_note
