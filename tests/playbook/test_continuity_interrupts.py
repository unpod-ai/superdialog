from superdialog.playbook.events import AdvanceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.continuity_fixtures import CONTINUITY_YAML, SeqLLM
from tests.playbook.test_toolexec import FakeHttp

INTERRUPT = {
    "slots": {},
    "advance": None,
    "note": None,
    "interrupt": "price_guardrail",
}
PLAIN = {"slots": {}, "advance": None, "note": None}


def _rt(payloads: list[dict]) -> PlaybookRuntime:
    return PlaybookRuntime(
        Playbook.from_yaml(CONTINUITY_YAML),
        director_llm=SeqLLM(payloads),
        http=FakeHttp([]),
    )


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
