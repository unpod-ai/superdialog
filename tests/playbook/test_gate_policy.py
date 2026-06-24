"""Per-slot gate policy + fast-classifier release (capability ``dialogue-gate-policy``).

Covers the Director side (tasks 4.3, 6.3): per-slot confirm-vs-fill decisions,
default-soft backward compatibility, and the confidence-driven fast verdict with
its deny list for known hard gates. Talker-side onset/split tests live in
``test_talker.py``.
"""

from __future__ import annotations

import json
import textwrap

from superdialog.playbook.director import Director, _is_known_hard_gate
from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SlotWriteEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.state import ConversationState, SlotValue


class CannedLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return json.dumps(self.payload)


# A soft checkpoint with one risky slot (phone) annotated hard per-slot.
PER_SLOT_YAML = textwrap.dedent("""
    journeys:
      intake:
        checkpoints:
          - id: collect
            goal: "Collect name and phone"
            gate: soft
            slots:
              name: {type: str}
              phone: {type: str, gate: hard}
            advance_when:
              - {when: "both given", judge: llm, to: intake.done,
                 requires: [name, phone]}
          - id: done
            terminal: true
""")

# A hard checkpoint with one low-risk slot (nickname) annotated soft per-slot.
SOFT_SLOT_YAML = textwrap.dedent("""
    journeys:
      pay:
        checkpoints:
          - id: verify
            gate: hard
            slots:
              otp: {type: str}
              nickname: {type: str, gate: soft}
            advance_when:
              - {when: "nickname given", judge: llm, to: pay.done,
                 requires: [nickname]}
          - id: done
            terminal: true
""")

# A non-sensitive hard slot eligible for confidence-based fast release.
QUANTITY_YAML = textwrap.dedent("""
    journeys:
      order:
        checkpoints:
          - id: ask
            gate: hard
            slots:
              quantity: {type: int}
            advance_when:
              - {when: "quantity given", judge: llm, to: order.done,
                 requires: [quantity]}
          - id: done
            terminal: true
""")


def _state(pb: Playbook, checkpoint: str, user: str) -> ConversationState:
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint=checkpoint, rule="init")
    )
    log.append(UtteranceEvent(role="user", text=user))
    return ConversationState.fold(log, playbook=pb)


def _writes(decision) -> dict[str, SlotWriteEvent]:
    return {e.key: e for e in decision.events if isinstance(e, SlotWriteEvent)}


def _advances(decision) -> list[AdvanceEvent]:
    return [e for e in decision.events if isinstance(e, AdvanceEvent)]


# ---------------------------------------------------------------------------
# Task 4.3 — per-slot gate decisions
# ---------------------------------------------------------------------------


async def test_risky_slot_not_confirmed_blocks_advance() -> None:
    """A per-slot-hard phone is provisional and gates advance until confirmed."""
    pb = Playbook.from_yaml(PER_SLOT_YAML)
    state = _state(pb, "intake.collect", "I'm Sam, 555-0100")
    llm = CannedLLM(
        {"slots": {"name": "Sam", "phone": "555-0100"}, "advance": "intake.done"}
    )
    decision = await Director(pb, llm).evaluate(state)
    writes = _writes(decision)
    assert writes["name"].status == "confirmed"  # low-risk slot
    assert writes["phone"].status == "provisional"  # risky slot, not yet confirmed
    assert not _advances(decision)  # phone unconfirmed -> blocked
    notes = [e for e in decision.events if e.type == "steering_note"]
    assert notes and "confirmation" in notes[0].text and "phone" in notes[0].text
    assert "name" not in notes[0].text  # the filled low-risk slot is not flagged


async def test_risky_slot_advances_once_confirmed() -> None:
    """With the phone pre-confirmed, the verdict may advance."""
    pb = Playbook.from_yaml(PER_SLOT_YAML)
    state = _state(pb, "intake.collect", "yes that's right")
    state.slots["phone"] = SlotValue(
        value="555-0100", status="confirmed", by="tool", version=1
    )
    llm = CannedLLM({"slots": {"name": "Sam"}, "advance": "intake.done"})
    decision = await Director(pb, llm).evaluate(state)
    adv = _advances(decision)
    assert adv and adv[0].to_checkpoint == "intake.done"


async def test_low_risk_slot_advances_in_hard_checkpoint() -> None:
    """A per-slot-soft slot advances on a provisional fill even in a hard cp."""
    pb = Playbook.from_yaml(SOFT_SLOT_YAML)
    state = _state(pb, "pay.verify", "call me Sammy")
    llm = CannedLLM({"slots": {"nickname": "Sammy"}, "advance": "pay.done"})
    decision = await Director(pb, llm).evaluate(state)
    writes = _writes(decision)
    assert writes["nickname"].status == "confirmed"  # soft override beats hard cp
    adv = _advances(decision)
    assert adv and adv[0].to_checkpoint == "pay.done"


async def test_unannotated_slot_inherits_checkpoint_gate() -> None:
    """No per-slot annotation -> the slot follows the checkpoint gate (default)."""
    director = Director(Playbook.from_yaml(SOFT_SLOT_YAML), CannedLLM({}))
    cp = director._pb.checkpoint("pay.verify")
    assert director._slot_gate("otp", cp) == "hard"  # inherits cp gate
    assert director._slot_gate("nickname", cp) == "soft"  # per-slot override


# ---------------------------------------------------------------------------
# Task 6.3 — fast-classifier barrier release
# ---------------------------------------------------------------------------


async def test_high_confidence_releases_without_full_model() -> None:
    """A confident non-sensitive hard slot is confirmed in one verdict."""
    pb = Playbook.from_yaml(QUANTITY_YAML)
    state = _state(pb, "order.ask", "three please")
    llm = CannedLLM(
        {
            "slots": {"quantity": "3"},
            "confidence": {"quantity": 0.95},
            "advance": "order.done",
        }
    )
    director = Director(pb, llm, fast_release=True, fast_release_threshold=0.85)
    decision = await director.evaluate(state)
    writes = _writes(decision)
    assert writes["quantity"].status == "confirmed"  # fast release
    adv = _advances(decision)
    assert adv and adv[0].to_checkpoint == "order.done"


async def test_low_confidence_escalates() -> None:
    """Below threshold -> provisional, no advance (falls through to full loop)."""
    pb = Playbook.from_yaml(QUANTITY_YAML)
    state = _state(pb, "order.ask", "maybe three?")
    llm = CannedLLM(
        {
            "slots": {"quantity": "3"},
            "confidence": {"quantity": 0.4},
            "advance": "order.done",
        }
    )
    director = Director(pb, llm, fast_release=True, fast_release_threshold=0.85)
    decision = await director.evaluate(state)
    assert _writes(decision)["quantity"].status == "provisional"
    assert not _advances(decision)


async def test_known_hard_gate_never_fast_released() -> None:
    """A sensitive slot (phone) never releases on confidence alone."""
    pb = Playbook.from_yaml(PER_SLOT_YAML)
    state = _state(pb, "intake.collect", "Sam, 555-0100")
    llm = CannedLLM(
        {
            "slots": {"name": "Sam", "phone": "555-0100"},
            "confidence": {"name": 0.99, "phone": 0.99},
            "advance": "intake.done",
        }
    )
    director = Director(pb, llm, fast_release=True)
    decision = await director.evaluate(state)
    writes = _writes(decision)
    assert writes["phone"].status == "provisional"  # denied: known hard gate
    assert not _advances(decision)


async def test_fast_release_off_by_default() -> None:
    """Default Director never fast-releases even with a confidence signal."""
    pb = Playbook.from_yaml(QUANTITY_YAML)
    state = _state(pb, "order.ask", "three")
    llm = CannedLLM(
        {
            "slots": {"quantity": "3"},
            "confidence": {"quantity": 0.99},
            "advance": "order.done",
        }
    )
    decision = await Director(pb, llm).evaluate(state)  # fast_release defaults off
    assert _writes(decision)["quantity"].status == "provisional"
    assert not _advances(decision)


async def test_deny_marker_matching() -> None:
    assert _is_known_hard_gate("user_phone")
    assert _is_known_hard_gate("emailAddress")
    assert _is_known_hard_gate("payment_card")
    assert not _is_known_hard_gate("city")
    assert not _is_known_hard_gate("quantity")


async def test_confidence_requested_only_when_fast_release() -> None:
    pb = Playbook.from_yaml(QUANTITY_YAML)
    state = _state(pb, "order.ask", "three")
    plain = CannedLLM({"slots": {}, "advance": None})
    await Director(pb, plain).evaluate(state)
    assert "confidence" not in plain.calls[0][0]["content"]

    fast = CannedLLM({"slots": {}, "advance": None})
    await Director(pb, fast, fast_release=True).evaluate(state)
    assert "confidence" in fast.calls[0][0]["content"]
