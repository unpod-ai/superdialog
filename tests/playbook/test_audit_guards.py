"""Wave-3 audit guards: never_say stream filter, KB flag, terminal-interrupt."""

import textwrap

from superdialog.playbook.director import _WRAP_MARKER, Director
from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    SteeringNoteEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.render import render_view
from superdialog.playbook.state import ConversationState
from superdialog.playbook.talker import _filter_never_say
from tests.playbook.test_director import CannedLLM, _state


# --- never_say: deterministic stream enforcement ------------------------------


async def _tokens(*chunks: str):
    for c in chunks:
        yield c


async def _run_filter(phrases: list[str], *chunks: str) -> str:
    out = []
    async for piece in _filter_never_say(_tokens(*chunks), phrases):
        out.append(piece)
    return "".join(out)


async def test_never_say_excises_phrase_within_one_chunk():
    got = await _run_filter(["free upgrade"], "I can offer a free upgrade today!")
    assert got == "I can offer a  today!"


async def test_never_say_catches_phrase_split_across_chunks():
    got = await _run_filter(["free upgrade"], "a free upg", "rade for you")
    assert "free upgrade" not in got
    assert "for you" in got


async def test_never_say_is_case_insensitive():
    got = await _run_filter(["Free Upgrade"], "a FREE upgrade here")
    assert "free upgrade" not in got.casefold()


async def test_never_say_passthrough_when_no_phrases():
    got = await _run_filter([], "hello ", "world")
    assert got == "hello world"


async def test_never_say_clean_text_unchanged():
    got = await _run_filter(["forbidden"], "perfectly ", "normal ", "reply")
    assert got == "perfectly normal reply"


# --- KB gating: explicit uses_kb beats the substring heuristic -----------------

_KB_YAML = textwrap.dedent("""
    persona: "You are a helper."
    knowledge_base: "Opening hours: 9-5."
    journeys:
      main:
        checkpoints:
          - id: legacy
            guidance: "Answer from the knowledge_base."
            advance_when:
              - {when: "done", judge: llm, to: main.annotated_off}
          - id: annotated_off
            uses_kb: false
            guidance: "Mentions knowledge_base but must stay lean."
            advance_when:
              - {when: "done", judge: llm, to: main.annotated_on}
          - id: annotated_on
            uses_kb: true
            guidance: "No mention at all."
            terminal: true
            outcome: closed
""")


def _kb_state(cp_id: str) -> tuple[Playbook, ConversationState]:
    pb = Playbook.from_yaml(_KB_YAML)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint=cp_id, rule="init"))
    log.append(UtteranceEvent(role="user", text="hi"))
    return pb, ConversationState.fold(log, playbook=pb)


def _system(pb: Playbook, state: ConversationState) -> str:
    return render_view(pb, state, token_budget=4000).messages[0]["content"]


def test_kb_legacy_substring_still_injects():
    pb, state = _kb_state("main.legacy")
    assert "## Knowledge base" in _system(pb, state)


def test_kb_uses_kb_false_overrides_substring_mention():
    pb, state = _kb_state("main.annotated_off")
    assert "## Knowledge base" not in _system(pb, state)


def test_kb_uses_kb_true_injects_without_mention():
    pb, state = _kb_state("main.annotated_on")
    assert "## Knowledge base" in _system(pb, state)


# --- terminal-interrupt slot guard ---------------------------------------------


async def test_goodbye_interrupt_steers_once_when_required_slots_missing():
    pb, state = _state()  # at booking.collect; city+date required, unfilled
    llm = CannedLLM({"slots": {}, "interrupt": "goodbye"})
    decision = await Director(pb, llm).evaluate(state)
    assert not [e for e in decision.events if isinstance(e, AdvanceEvent)]
    notes = [e for e in decision.events if isinstance(e, SteeringNoteEvent)]
    assert notes and notes[0].text.startswith(_WRAP_MARKER)
    assert "city" in notes[0].text and "date" in notes[0].text


async def test_goodbye_interrupt_passes_on_second_ask():
    # The wrap marker is already the live steering note -> honor the goodbye.
    pb, state = _state(
        extra_events=[SteeringNoteEvent(text=f"{_WRAP_MARKER} — ask", kind="steer")]
    )
    llm = CannedLLM({"slots": {}, "interrupt": "goodbye"})
    decision = await Director(pb, llm).evaluate(state)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.close"


async def test_goodbye_interrupt_passes_when_slots_filled():
    llm = CannedLLM(
        {"slots": {"city": "Pune", "date": "2026-07-05"}, "interrupt": "goodbye"}
    )
    pb, state = _state()
    decision = await Director(pb, llm).evaluate(state)
    adv = [e for e in decision.events if isinstance(e, AdvanceEvent)]
    assert adv and adv[0].to_checkpoint == "booking.close"
