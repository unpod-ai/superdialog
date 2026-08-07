"""End-to-end continuity scenario regressions (Task 16).

Group A replays the two production failure sequences (westgate2 self-interrupt,
golfai conveyor bounce). Groups B-C are seam tests carried over from the Task
1/8 reviews: nested detours held via the resume-stack guard branch, slot writes
surviving held turns, and the corroboration stamps on interrupt/junk/churn
paths. All run PlaybookRuntime + SeqLLM end to end — no engine mocks.
"""

from superdialog.playbook.events import AdvanceEvent, DegradedEvent, SlotWriteEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from superdialog.playbook.supervisor import detect_triggers
from tests.playbook.continuity_fixtures import CONTINUITY_YAML, make_runtime

PRICE = {"slots": {}, "advance": None, "note": None, "interrupt": "price_guardrail"}
AVAIL = {
    "slots": {},
    "advance": None,
    "note": None,
    "interrupt": "availability_guardrail",
}
PLAIN = {"slots": {}, "advance": None, "note": None}

_rt = make_runtime

# ask_budget's rule with requires stripped so its hop can be prose-only
# (same idiom as test_continuity_advances).
BUDGET_STRIPPED_YAML = CONTINUITY_YAML.replace(
    "to: main.close,\n             requires: [budget]}",
    "to: main.close}",
)
assert BUDGET_STRIPPED_YAML != CONTINUITY_YAML  # guard: the surgery landed


def _trajectory(rt: PlaybookRuntime) -> list[tuple[str | None, str, str]]:
    return [
        (e.from_checkpoint, e.to_checkpoint, e.rule)
        for e in rt.log.events
        if isinstance(e, AdvanceEvent)
    ]


# -- Group A: production sequences -------------------------------------------


async def test_westgate2_self_interrupt_sequence_full_trajectory() -> None:
    """westgate2 steps 10-13: price interrupt, price again inside the detour
    (held, no self-loop), then a plain turn resumes — the FULL ordered edge
    list is pinned, so any spurious advance anywhere breaks this."""
    rt = _rt([PRICE, PRICE, PLAIN, PRICE])
    await rt.start()
    await rt.on_user_text("what's the price?")
    await rt.on_user_text("and any discounts?")  # self-interrupt: held
    await rt.on_user_text("ok thanks")  # detour done: resume
    assert _trajectory(rt) == [
        (None, "main.ask_location", "init"),
        ("main.ask_location", "main.pricing_faq", "interrupt:price_guardrail"),
        ("main.pricing_faq", "main.ask_location", "resume"),
    ]
    assert rt.state.resume_stack == []
    # E2: a LATER price question must still fire the interrupt.
    await rt.on_user_text("sorry, the price again?")
    assert rt.state.checkpoint_id == "main.pricing_faq"
    assert _trajectory(rt)[-1] == (
        "main.ask_location",
        "main.pricing_faq",
        "interrupt:price_guardrail",
    )


async def test_golfai_conveyor_bounce_reads_uncorroborated() -> None:
    """golfai conveyor: one real answer, then prose-only advances. The second
    advance is uncorroborated + steered; a third prose-only advance makes the
    supervisor's uncorroborated_streak trigger fire."""
    pb = Playbook.from_yaml(BUDGET_STRIPPED_YAML)
    rt = _rt(
        [
            {"slots": {"location": "Pune"}, "advance": "main.pitch"},
            {"slots": {}, "advance": "main.ask_budget"},  # conveyor hop 1
            {"slots": {}, "advance": "main.close"},  # conveyor hop 2
        ],
        pb=pb,
    )
    await rt.start()
    await rt.on_user_text("I'm in Pune")
    await rt.on_user_text("hmm what?")
    adv = [e for e in rt.log.events if isinstance(e, AdvanceEvent)]
    assert adv[-1].to_checkpoint == "main.ask_budget"
    assert adv[-1].corroborated is False
    assert rt.state.steering_note is not None  # steer live at ask_budget
    assert "without confirmed input" in rt.state.steering_note
    await rt.on_user_text("uh")
    adv = [e for e in rt.log.events if isinstance(e, AdvanceEvent)]
    assert adv[-1].corroborated is False  # third hop: direct plan-letter pin
    assert "uncorroborated_streak" in detect_triggers(rt.state, rt.log, pb)


# -- Group B: nested detour (Task 1 review carry-over) ------------------------


async def test_nested_detour_hold_and_unwind() -> None:
    rt = _rt([PRICE, AVAIL, PRICE, PLAIN, PLAIN])
    await rt.start()
    await rt.on_user_text("price?")  # detour A
    assert rt.state.checkpoint_id == "main.pricing_faq"
    assert rt.state.resume_stack == ["main.ask_location"]

    await rt.on_user_text("when are you open?")  # nested detour B
    assert rt.state.checkpoint_id == "main.availability_faq"
    assert rt.state.resume_stack == ["main.ask_location", "main.pricing_faq"]

    # Interrupt A fires AGAIN inside B's detour: its target pricing_faq is ON
    # the resume stack, so the guard branch holds the detour instead of
    # pushing availability_faq onto its own return path.
    await rt.on_user_text("wait, the price?")
    assert rt.state.checkpoint_id == "main.availability_faq"  # held, no advance
    assert rt.state.resume_stack == ["main.ask_location", "main.pricing_faq"]
    assert rt.state.steering_note  # hold-open steer is live

    await rt.on_user_text("ok")  # unwind one level
    assert rt.state.checkpoint_id == "main.pricing_faq"
    assert rt.state.resume_stack == ["main.ask_location"]

    await rt.on_user_text("got it")  # unwind to the original step
    assert rt.state.checkpoint_id == "main.ask_location"
    assert rt.state.resume_stack == []


async def test_slot_write_survives_hold_turn() -> None:
    """A held self-interrupt turn suppresses the advance, never the extraction:
    the verdict's slot write on the detour checkpoint must land."""
    rt = _rt(
        [
            AVAIL,
            {
                "slots": {"callback_time": "tomorrow 5pm"},
                "interrupt": "availability_guardrail",  # self-interrupt: held
            },
        ]
    )
    await rt.start()
    await rt.on_user_text("when are you open?")
    assert rt.state.checkpoint_id == "main.availability_faq"
    before = len([e for e in rt.log.events if isinstance(e, AdvanceEvent)])

    await rt.on_user_text("tomorrow at 5 works — but when exactly?")
    assert rt.state.checkpoint_id == "main.availability_faq"  # still held
    after = len([e for e in rt.log.events if isinstance(e, AdvanceEvent)])
    assert after == before  # the advance was suppressed...
    assert rt.state.slots["callback_time"].value == "tomorrow 5pm"  # ...not the write


# -- Group C: corroboration seams (Task 8 review carry-over) ------------------


async def test_interrupt_advances_stamp_corroborated_none() -> None:
    rt = _rt([PRICE])
    await rt.start()
    await rt.on_user_text("price?")
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.rule == "interrupt:price_guardrail"
    ]
    assert adv and adv[-1].corroborated is None  # unclassified, not False


async def test_junk_only_turn_prose_advance_is_uncorroborated() -> None:
    """golfai budget='None': the junk write is rejected (DegradedEvent), so the
    same verdict's requires-less advance has no evidence — uncorroborated."""
    rt = _rt(
        [
            {"slots": {"location": "Pune"}, "advance": "main.pitch"},
            {"slots": {}, "advance": "main.ask_budget"},
            {"slots": {"budget": "None"}, "advance": "main.close"},
        ],
        pb=Playbook.from_yaml(BUDGET_STRIPPED_YAML),
    )
    await rt.start()
    await rt.on_user_text("Pune")
    await rt.on_user_text("ok")
    await rt.on_user_text("whatever you think")
    assert "budget" not in rt.state.slots  # junk never landed
    assert any(
        isinstance(e, DegradedEvent) and e.detail == "junk_rejected:budget"
        for e in rt.log.events
    )
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.to_checkpoint == "main.close"
    ]
    assert adv and adv[-1].corroborated is False


async def test_churn_dropped_repeat_prose_advance_is_uncorroborated() -> None:
    """Re-extracting an identical confirmed slot is dropped by the churn
    dampener, so it cannot vouch for a requires-less advance."""
    yaml = CONTINUITY_YAML.replace(  # strip ONLY ask_location's requires (the
        '- {when: "location given", judge: llm, to: main.pitch,\n'  # same
        "             requires: [location]}",  # pattern also exists at pricing_faq)
        '- {when: "location given", judge: llm, to: main.pitch}',
    )
    assert yaml != CONTINUITY_YAML
    rt = _rt(
        [
            {"slots": {"location": "Pune"}, "advance": None},
            {"slots": {"location": "Pune"}, "advance": "main.pitch"},
        ],
        pb=Playbook.from_yaml(yaml),
    )
    await rt.start()
    await rt.on_user_text("I'm in Pune")
    assert rt.state.slots["location"].status == "confirmed"  # soft gate
    await rt.on_user_text("Pune, like I said")
    writes = [
        e
        for e in rt.log.events
        if isinstance(e, SlotWriteEvent) and e.key == "location"
    ]
    assert len(writes) == 1  # the identical re-extraction was dropped
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.to_checkpoint == "main.pitch"
    ]
    assert adv and adv[-1].corroborated is False
    assert rt.state.steering_note is not None
    assert "without confirmed input" in rt.state.steering_note
