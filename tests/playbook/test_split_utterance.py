"""Split-utterance onset streaming (task 5.3, capability ``dialogue-gate-policy``).

Asserts the Talker emits a commitment-free onset as its first token(s) before
the Director settles on a gated turn, that the onset never carries a slot value,
and that the committal payload waits for confirmation.
"""

from __future__ import annotations

import textwrap

import anyio

from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SlotWriteEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.state import ConversationState
from superdialog.playbook.talker import ONSET_TEMPLATES, Talker
from tests.playbook.test_models import MINIMAL_YAML


class StreamLLM:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls = 0

    async def stream(self, messages, **kwargs):
        self.calls += 1
        for c in self.chunks:
            yield c


# A SOFT checkpoint carrying one per-slot-hard (risky) slot.
PER_SLOT_HARD_YAML = textwrap.dedent("""
    journeys:
      intake:
        checkpoints:
          - id: collect
            slots:
              phone: {type: str, gate: hard}
            advance_when:
              - {when: "given", judge: llm, to: intake.done, requires: [phone]}
          - id: done
            terminal: true
""")


def _state(pb: Playbook, checkpoint: str, extra=()) -> ConversationState:
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint=checkpoint, rule="init")
    )
    log.append(UtteranceEvent(role="user", text="hello"))
    for e in extra:
        log.append(e)
    return ConversationState.fold(log, playbook=pb)


async def test_onset_precedes_director_on_gated_turn() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    state = _state(pb, "booking.confirm")  # gate: hard, say_verbatim payload
    event = anyio.Event()

    async def wait_director() -> ConversationState:
        await event.wait()
        return state

    talker = Talker(pb, StreamLLM([]), barrier_timeout=10.0, hold_timeout=10.0)
    received: list[str] = []

    async def consume() -> None:
        async for c in talker.speak(state, director_done=wait_director):
            received.append(c.text)

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await anyio.sleep(0.1)  # Director still pending (event not set)
        onset = talker._select_onset(state)
        assert received and received[0].strip() == onset  # first token is the onset
        # The committal payload has NOT been spoken before the Director settles.
        assert "Your booking is held." not in "".join(received)
        event.set()
    # After confirmation the committal payload is appended.
    assert "".join(received).strip().endswith("Your booking is held.")


async def test_per_slot_hard_slot_gates_a_soft_checkpoint() -> None:
    """A soft checkpoint with a hard slot still barriers (onset-first)."""
    pb = Playbook.from_yaml(PER_SLOT_HARD_YAML)
    state = _state(pb, "intake.collect")
    event = anyio.Event()

    async def wait_director() -> ConversationState:
        await event.wait()
        return state

    talker = Talker(pb, StreamLLM(["PAYLOAD"]), barrier_timeout=10.0, hold_timeout=10.0)
    received: list[str] = []

    async def consume() -> None:
        async for c in talker.speak(state, director_done=wait_director):
            received.append(c.text)

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await anyio.sleep(0.1)
        assert received and received[0].strip() == talker._select_onset(state)
        assert "PAYLOAD" not in "".join(received)  # payload still barriered
        event.set()
    assert "".join(received).strip().endswith("PAYLOAD")


async def test_onset_never_contains_a_slot_value() -> None:
    pb = Playbook.from_yaml(PER_SLOT_HARD_YAML)
    # A provisional, unconfirmed phone value is present in state.
    state = _state(
        pb,
        "intake.collect",
        extra=[
            SlotWriteEvent(
                key="phone", value="555-0100", status="provisional", by="director"
            )
        ],
    )
    talker = Talker(pb, StreamLLM([]))
    onset = talker._select_onset(state)
    assert onset in ONSET_TEMPLATES  # static template, never interpolated
    assert "555-0100" not in onset


async def test_onset_selection_is_deterministic_and_value_independent() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    state = _state(pb, "booking.confirm")
    talker = Talker(pb, StreamLLM([]))
    # Same checkpoint -> same onset across calls; always a member of the set.
    assert talker._select_onset(state) == talker._select_onset(state)
    assert talker._select_onset(state) in ONSET_TEMPLATES


async def test_soft_turn_has_no_onset() -> None:
    """A fully-soft checkpoint streams directly — no onset, no barrier."""
    pb = Playbook.from_yaml(MINIMAL_YAML)
    state = _state(pb, "booking.collect")  # soft, no hard slots

    async def never() -> ConversationState:
        await anyio.sleep(3600)
        return state

    talker = Talker(pb, StreamLLM(["hi"]))
    with anyio.fail_after(1):
        chunks = [c async for c in talker.speak(state, director_done=never)]
    text = "".join(c.text for c in chunks)
    assert text == "hi"
    assert not any(o in text for o in ONSET_TEMPLATES)
