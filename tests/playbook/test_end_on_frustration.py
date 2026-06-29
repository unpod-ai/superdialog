"""End-to-end regression for the premature-goodbye on caller frustration.

Transcript shape that broke in production: after the agent re-asked an
already-answered slot (an active `repair` steering note — what
``check_repairs`` raises), the caller's frustrated "बता तो दिया" ("I already
told you") was read as closure and the Director advanced to the terminal
goodbye checkpoint. The repair-aware guard (director.py) now suppresses a
terminal advance while a repair signal is active and recovers instead — while a
genuine close with no repair note still ends the call normally.
"""

import textwrap

from superdialog.playbook.events import SessionEndEvent, SteeringNoteEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_toolexec import FakeHttp

_PB = textwrap.dedent("""
    persona: "You collect a date of birth for a numerology reading."
    journeys:
      main:
        checkpoints:
          - id: collect
            goal: "collect the caller's date of birth"
            slots:
              date_of_birth: {type: date, gate: soft}
            advance_when:
              - {when: "the caller is finished and wants to end",
                 judge: llm, to: main.done}
          - id: done
            terminal: true
            outcome: completed
            say_verbatim: "Thank you, take care. Goodbye."
""")


def _runtime() -> PlaybookRuntime:
    # The Director always "decides" to end (advance to the terminal checkpoint);
    # whether the call actually ends is what the guard governs.
    return PlaybookRuntime(
        Playbook.from_yaml(_PB),
        director_llm=CannedLLM({"slots": {}, "advance": "main.done", "note": None}),
        http=FakeHttp([]),
    )


async def test_frustration_does_not_end_the_call() -> None:
    rt = _runtime()
    await rt.start()
    assert rt.state.checkpoint_id == "main.collect"
    # The engine just flagged a re-ask: a repair note is active.
    rt.log.append(
        SteeringNoteEvent(
            kind="repair",
            text="You already have date_of_birth; acknowledge it, don't re-ask.",
        )
    )
    await rt.on_user_text("बता तो दिया")  # "I already told you" — frustration
    assert not rt.state.ended
    assert not any(isinstance(e, SessionEndEvent) for e in rt.log.events)
    assert rt.state.checkpoint_id == "main.collect"  # stayed, recovered
    assert rt.state.steering_kind == "repair"  # recovery note in place


async def test_genuine_close_still_ends_the_call() -> None:
    rt = _runtime()
    await rt.start()
    # No repair note in flight: a real close should end the call as before.
    await rt.on_user_text("nothing else, thanks")
    assert rt.state.ended and rt.state.outcome == "completed"
    assert any(isinstance(e, SessionEndEvent) for e in rt.log.events)
