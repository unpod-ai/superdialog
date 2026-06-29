"""End-to-end regression for the "whose date of birth?" collision.

Drives a two-checkpoint playbook (collect_self → caller, collect_partner →
partner, both with `date_of_birth`) through the transcript shape that broke
in production: the caller gives a DOB, the partner gives a different DOB, then
the user re-utters "date of birth" ambiguously. With entity-scoped slots both
DOBs are retained and distinct, the partner checkpoint's `requires` is met by
the *partner's* DOB, and the rendered Known-information labels both — so the
Talker never has to ask "whose date of birth?".
"""

import json
import textwrap

from superdialog.playbook.director import Director
from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SessionStartEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.render import render_view
from superdialog.playbook.state import ConversationState

_PB = textwrap.dedent("""
    multi_entity: true
    persona: "You collect dates of birth for a numerology reading."
    journeys:
      main:
        checkpoints:
          - id: collect_self
            entity: caller
            gate: soft
            slots:
              date_of_birth: {type: date, required: true, gate: soft}
            advance_when:
              - {when: "caller gave their dob", judge: llm,
                 to: main.collect_partner, requires: [date_of_birth]}
          - id: collect_partner
            entity: partner
            gate: soft
            slots:
              date_of_birth: {type: date, required: true, gate: soft}
            advance_when:
              - {when: "partner dob given", judge: llm, to: main.done,
                 requires: [date_of_birth]}
          - id: done
            terminal: true
            outcome: confirmed
""")


class CannedLLM:
    """Stub Director LLM that returns a fixed verdict per turn."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def complete(self, messages, **kwargs) -> str:
        return json.dumps(self.payload)


async def _turn(pb: Playbook, log: EventLog, payload: dict) -> ConversationState:
    """Fold the log, run one Director turn, append its events, re-fold."""
    state = ConversationState.fold(log, playbook=pb)
    decision = await Director(pb, CannedLLM(payload)).evaluate(state)
    for e in decision.events:
        log.append(e)
    return ConversationState.fold(log, playbook=pb)


async def test_two_entity_dob_collision_regression() -> None:
    pb = Playbook.from_yaml(_PB)
    log = EventLog()
    log.append(
        SessionStartEvent(started_at="2026-06-24T00:00:00+00:00", timezone="UTC")
    )
    log.append(
        AdvanceEvent(
            from_checkpoint=None, to_checkpoint="main.collect_self", rule="init"
        )
    )

    # Turn 1 — caller's DOB; verdict confirms it and advances to the partner
    # checkpoint. (ISO in the verdict keeps the date deterministic; the
    # natural utterance text is what production sees.)
    log.append(UtteranceEvent(role="user", text="4 June 1986"))
    state = await _turn(
        pb,
        log,
        {
            "slots": {"date_of_birth": "1986-06-04"},
            "advance": "main.collect_partner",
            "note": None,
        },
    )
    assert state.checkpoint_id == "main.collect_partner"

    # Turn 2 — partner's DOB at the partner checkpoint.
    log.append(UtteranceEvent(role="user", text="12 July 1986"))
    state = await _turn(
        pb,
        log,
        {
            "slots": {"date_of_birth": "1986-07-12"},
            "advance": "main.done",
            "note": None,
        },
    )

    # Both DOBs retained and DISTINCT — partner did NOT clobber caller.
    assert state.slot_value("date_of_birth", entity="caller") == "1986-06-04"
    assert state.slot_value("date_of_birth", entity="partner") == "1986-07-12"
    assert "date_of_birth" in state.slots  # caller stays on the bare key
    assert "partner:date_of_birth" in state.slots  # partner namespaced

    # The partner checkpoint's `requires` is met by the PARTNER's DOB.
    assert state.confirmed(["date_of_birth"], entity="partner") is True

    # Known-information labels both people; the Talker can answer without
    # asking "whose date of birth?".
    system = render_view(pb, state).messages[0]["content"]
    assert "caller:" in system and "partner:" in system
    assert "1986-06-04" in system and "1986-07-12" in system

    # Turn 3 — the ambiguous re-utterance that triggered the bug. Even if the
    # LLM re-extracts a DOB at the partner checkpoint, it tags partner; the
    # caller's value is untouched and both remain distinct.
    log.append(UtteranceEvent(role="user", text="my date of birth"))
    state = await _turn(
        pb,
        log,
        {
            "slots": {"date_of_birth": "1986-07-12"},
            "advance": None,
            "note": None,
        },
    )
    assert state.slot_value("date_of_birth", entity="caller") == "1986-06-04"
    assert state.slot_value("date_of_birth", entity="partner") == "1986-07-12"


async def test_single_entity_flow_byte_identical() -> None:
    """Backward-compat guard: multi_entity off ⇒ flat caller storage + a flat
    Known-information block, exactly as before entity scoping."""
    pb = Playbook.from_yaml(
        textwrap.dedent("""
        persona: "You collect a date of birth."
        journeys:
          main:
            checkpoints:
              - id: collect
                slots:
                  date_of_birth: {type: date, required: true}
              - id: done
                terminal: true
    """)
    )
    log = EventLog()
    log.append(
        SessionStartEvent(started_at="2026-06-24T00:00:00+00:00", timezone="UTC")
    )
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="main.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="4 June 1986"))
    state = await _turn(
        pb, log, {"slots": {"date_of_birth": "1986-06-04"}, "advance": None}
    )
    assert pb.multi_entity is False
    assert "date_of_birth" in state.slots  # bare key
    assert "caller:date_of_birth" not in state.slots
    system = render_view(pb, state).messages[0]["content"]
    assert "## Known information\n- " in system  # today's flat shape
    assert "caller:" not in system and "partner:" not in system
