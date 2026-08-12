"""Deterministic goodbye backstop (director.py).

Real transcript that broke: a caller said "Then I'm good. Goodbye, VP
Chandigarh" and the agent — instead of closing — launched into pricing. The
global_goodbye interrupt is LLM-judged, and the verdict missed it (ASR noise /
mid-pitch barge-in), so the goodbye was invisible. The backstop fires the
goodbye interrupt on a clear spoken bye token when the verdict set none, while
staying silent on frustration ('I already told you') and embedded byes.
"""

import textwrap

import pytest

from superdialog.playbook.director import _clear_goodbye, _confirmed_goodbye
from superdialog.playbook.events import SessionEndEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_toolexec import FakeHttp

_PB = textwrap.dedent("""
    journeys:
      main:
        checkpoints:
          - id: collect
            goal: "collect the enquiry"
            advance_when:
              - {when: "caller is done", judge: llm, to: main.done}
          - id: done
            terminal: true
            outcome: completed
            say_verbatim: "Thank you, take care. Goodbye."
    interrupts:
      - id: global_goodbye
        when: "Caller says goodbye/bye or wants to end the call."
        judge: llm
        to: main.done
        resume: false
""")


def _runtime() -> PlaybookRuntime:
    # The Director NEVER fires the interrupt (advance null, no interrupt key) —
    # exactly the miss from the transcript. The backstop is what must catch it.
    return PlaybookRuntime(
        Playbook.from_yaml(_PB),
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=FakeHttp([]),
    )


# -- unit: the classifier --------------------------------------------------------


def test_clear_goodbye_recognizes_explicit_close() -> None:
    assert _clear_goodbye("Goodbye")
    assert _clear_goodbye("Then I'm good. Goodbye, VP Chandigarh")  # the real one
    assert _clear_goodbye("ok bye")
    assert _clear_goodbye("bye bye")


def test_clear_goodbye_recognizes_casual_elongation() -> None:
    # A typed "byeee"/"byee" (common in real chat, not just voice ASR) must
    # still route to close -- the word-boundary-only match previously missed
    # these, leaving a finished conversation resumable indefinitely.
    assert _clear_goodbye("byeee")
    assert _clear_goodbye("byee")
    assert _clear_goodbye("goodbyeee")
    assert _clear_goodbye("byes")


def test_clear_goodbye_ignores_frustration_and_embedded_bye() -> None:
    assert not _clear_goodbye("बता तो दिया")  # "I already told you" — no bye token
    assert not _clear_goodbye("maybe later")  # 'maybe' is not a bye
    assert not _clear_goodbye("")
    # a bare 'bye' inside a long, continuing utterance must NOT end the call
    assert not _clear_goodbye(
        "bye for now but first tell me all about the amenities and pricing please"
    )


def test_confirmed_goodbye_rejects_bare_negation_and_affirmation() -> None:
    # Real production cases: "any add-ons or special requests?" -> "No."
    # and, on the opposite polarity, "did you want to change something?" ->
    # "Yes." both routed straight to call_end mid-booking because the
    # Director LLM claimed the goodbye interrupt. Neither carries a bye
    # token or an explicit close phrase, so neither clears this check.
    assert not _confirmed_goodbye("No.")
    assert not _confirmed_goodbye("Nope.")
    assert not _confirmed_goodbye("No thanks.")
    assert not _confirmed_goodbye("Nothing, thanks.")
    assert not _confirmed_goodbye("Nahi")
    assert not _confirmed_goodbye("Bas")
    assert not _confirmed_goodbye("Yes.")
    assert not _confirmed_goodbye("yes please")


def test_confirmed_goodbye_rejects_phrasing_variety_and_filler_prefixes() -> None:
    # A prior negation-start regex kept missing real phrasing variety: first
    # a decline with no enumerated trailing phrase ("No, I don't require."),
    # then a filler word before the negation ("Uh, no, no time.") defeating
    # the anchored ^(no|nope|...) match entirely. Positive evidence (no bye
    # token, no explicit close phrase) rejects all of these at once instead
    # of requiring one more shape to be enumerated after each live miss.
    assert not _confirmed_goodbye("No, I don't require.")
    assert not _confirmed_goodbye("No, I don't need that.")
    assert not _confirmed_goodbye("Not needed.")
    assert not _confirmed_goodbye("No, we're good.")
    assert not _confirmed_goodbye("Uh, no, no time.")
    assert not _confirmed_goodbye("no, I also wanted to ask about pricing")


def test_confirmed_goodbye_recognizes_actual_goodbyes() -> None:
    assert _confirmed_goodbye("No, that's all, goodbye")  # has a bye token
    assert _confirmed_goodbye("Goodbye")
    assert _confirmed_goodbye("Sorry, I have to go now")
    assert _confirmed_goodbye("Can you call me back later")
    assert _confirmed_goodbye("I'm driving, can we do this another time")


def test_confirmed_goodbye_rejects_empty_text() -> None:
    assert not _confirmed_goodbye("")


# -- the full utterance matrix ----------------------------------------------------
#
# One table, every category of caller reply this guard has to get right,
# each tagged with which real golfai incident (if any) it documents. This
# is the living regression suite the classifier's docstrings keep
# referencing "the next live miss" against -- when a new false positive or
# false negative turns up in production, it gets a row here, not just a
# one-off assertion.
#
# category tags:
#   negation        bare decline of an in-flow question -- must NOT confirm
#   affirmation     bare confirmation of an in-flow question -- must NOT confirm
#   filler          negation/affirmation with a leading filler word
#   frustration     caller is annoyed, not leaving -- must NOT confirm
#   continuation    longer reply that keeps the conversation going
#   embedded_bye    contains "bye" mid-sentence but isn't a close
#   explicit_bye    an actual spoken goodbye -- MUST confirm
#   explicit_close  a non-"bye" closing phrase (has to go/driving/etc) -- MUST confirm
#   hindi           Hindi/Hinglish variant of the above, either script
#   ambiguous       genuinely context-dependent; documents the conservative
#                   default this guard takes (does not confirm) rather than
#                   claiming a single "correct" answer
_GOODBYE_MATRIX = [
    # -- negation: real golfai incident (any add-ons? -> No.) --------------
    ("No.", False, "negation"),
    ("Nope.", False, "negation"),
    ("Nah.", False, "negation"),
    ("Nahi", False, "negation"),
    ("Bas", False, "negation"),
    ("Nothing.", False, "negation"),
    ("Not needed.", False, "negation"),
    ("No thanks.", False, "negation"),
    ("No, I don't require.", False, "negation"),
    ("No, I don't need that.", False, "negation"),
    ("No, we're good.", False, "negation"),
    # -- affirmation: real golfai incident (did you want to change something? -> Yes.) --
    ("Yes.", False, "affirmation"),
    ("Yeah.", False, "affirmation"),
    ("Sure.", False, "affirmation"),
    ("Okay.", False, "affirmation"),
    ("Ok.", False, "affirmation"),
    ("Haan", False, "affirmation"),
    ("Theek hai", False, "affirmation"),
    ("Please proceed.", False, "affirmation"),
    ("Go ahead.", False, "affirmation"),
    # -- filler-prefixed: real golfai incident (Uh, no, no time.) -----------
    ("Uh, no, no time.", False, "filler"),
    ("Um, yeah.", False, "filler"),
    ("Uh, no.", False, "filler"),
    ("Well, no.", False, "filler"),
    ("Uh, okay.", False, "filler"),
    # -- frustration: caller is annoyed, still on the call ------------------
    ("I already told you that.", False, "frustration"),
    ("बता तो दिया", False, "frustration"),
    ("This is so annoying, just fix it.", False, "frustration"),
    # -- continuation: longer replies that keep the conversation going ------
    ("no, I also wanted to ask about pricing", False, "continuation"),
    ("No, I also wanted to know about the buggy rental please.", False, "continuation"),
    ("Actually can we check a different date instead.", False, "continuation"),
    ("What about Sunday morning, is that available.", False, "continuation"),
    # -- embedded bye: contains "bye" but is not a close (finding from the ---
    # explore-mode matrix design -- _confirmed_goodbye must match _clear_
    # goodbye's own length-bounded bye check, not a raw substring search)
    ("bye for now but first tell me all about the amenities and pricing please", False, "embedded_bye"),
    ("I'll say bye once we sort out the booking, but first what's the price.", False, "embedded_bye"),
    # -- explicit goodbye: MUST confirm --------------------------------------
    ("Goodbye", True, "explicit_bye"),
    ("Goodbye.", True, "explicit_bye"),
    ("bye bye", True, "explicit_bye"),
    ("ok bye", True, "explicit_bye"),
    ("Thanks, bye.", True, "explicit_bye"),
    ("No, that's all, goodbye", True, "explicit_bye"),
    ("byeee", True, "explicit_bye"),
    ("Then I'm good. Goodbye, VP Chandigarh", True, "explicit_bye"),
    # -- explicit non-bye close phrases: MUST confirm ------------------------
    ("I have to go now.", True, "explicit_close"),
    ("Sorry, gotta go.", True, "explicit_close"),
    ("Can you call me back later.", True, "explicit_close"),
    ("Please call me back tomorrow instead.", True, "explicit_close"),
    ("I'm driving right now, can we do this another time.", True, "explicit_close"),
    ("I'm busy right now, call back another time.", True, "explicit_close"),
    ("Please stop calling me.", True, "explicit_close"),
    ("Can you just end the call please.", True, "explicit_close"),
    ("This is not a good time for me.", True, "explicit_close"),
    # -- Hindi/Hinglish variants: MUST confirm -------------------------------
    ("फोन रखती हूँ", True, "hindi"),
    ("phone rakhti hoon", True, "hindi"),
    ("मुझे जाना है", True, "hindi"),
    ("mujhe jaana hai", True, "hindi"),
    ("baad mein baat karte hain, abhi mujhe jaana hai", True, "hindi"),
    # -- ambiguous: conservative default is "do not confirm" -- documents ---
    # the choice, not a claim that this is the one correct reading
    ("That's all.", False, "ambiguous"),
    ("Okay, thanks.", False, "ambiguous"),
    ("Alright.", False, "ambiguous"),
]


@pytest.mark.parametrize(
    ("text", "expected", "category"),
    _GOODBYE_MATRIX,
    ids=[f"{cat}:{text[:30]!r}" for text, _, cat in _GOODBYE_MATRIX],
)
def test_confirmed_goodbye_matrix(text: str, expected: bool, category: str) -> None:
    assert _confirmed_goodbye(text) is expected, (
        f"category={category!r} text={text!r} expected confirmed={expected}"
    )


# -- integration: the backstop through the runtime -------------------------------


async def test_missed_goodbye_is_caught_and_closes_the_call() -> None:
    rt = _runtime()
    await rt.start()
    assert rt.state.checkpoint_id == "main.collect"
    await rt.on_user_text("Then I'm good. Goodbye, VP Chandigarh")
    assert rt.state.ended and rt.state.outcome == "completed"
    assert any(isinstance(e, SessionEndEvent) for e in rt.log.events)


async def test_frustration_does_not_trigger_the_backstop() -> None:
    rt = _runtime()
    await rt.start()
    await rt.on_user_text("बता तो दिया")  # frustration, no bye token
    assert not rt.state.ended
    assert rt.state.checkpoint_id == "main.collect"


async def test_embedded_bye_in_a_continuing_utterance_does_not_close() -> None:
    rt = _runtime()
    await rt.start()
    await rt.on_user_text(
        "bye for now but first tell me all about the amenities and pricing please"
    )
    assert not rt.state.ended


async def test_llm_claimed_goodbye_on_bare_negation_is_suppressed() -> None:
    # The real production case, reproduced end-to-end: the Director LLM
    # itself (not just the backstop) wrongly claims interrupt=global_goodbye
    # on a bare "No." answering an in-flow question. The false-goodbye guard
    # must drop the interrupt so the conversation continues instead of
    # ending the call.
    rt = PlaybookRuntime(
        Playbook.from_yaml(_PB),
        director_llm=CannedLLM(
            {"slots": {}, "advance": None, "note": None, "interrupt": "global_goodbye"}
        ),
        http=FakeHttp([]),
    )
    await rt.start()
    await rt.on_user_text("No.")
    assert not rt.state.ended


@pytest.mark.parametrize(
    ("text", "expected_ended", "category"),
    _GOODBYE_MATRIX,
    ids=[f"{cat}:{text[:30]!r}" for text, _, cat in _GOODBYE_MATRIX],
)
async def test_llm_claimed_goodbye_end_to_end_matrix(
    text: str, expected_ended: bool, category: str
) -> None:
    """Full-pipeline version of test_confirmed_goodbye_matrix: for each of
    the same 58 caller replies, the Director LLM is forced (via CannedLLM)
    to claim interrupt=global_goodbye no matter what -- the worst case,
    where the LLM decided on its own to fire the interrupt on this exact
    answer to the in-flow "collect the enquiry" question. Asserts what
    actually happens to the call: does it really end, or does the
    deterministic guard drop the LLM's claim and keep the conversation
    going. This is the question-and-answer framing directly -- not "does
    this string look like a goodbye" in isolation, but "the agent asked
    something, the caller answered this, and the LLM (wrongly, in every
    row here) called for the interrupt anyway -- does the call actually
    hang up or not."
    """
    rt = PlaybookRuntime(
        Playbook.from_yaml(_PB),
        director_llm=CannedLLM(
            {"slots": {}, "advance": None, "note": None, "interrupt": "global_goodbye"}
        ),
        http=FakeHttp([]),
    )
    await rt.start()
    assert rt.state.checkpoint_id == "main.collect"
    await rt.on_user_text(text)
    assert rt.state.ended is expected_ended, (
        f"category={category!r} text={text!r} expected call-ended={expected_ended}"
    )


# -- post-terminal silence (Westgate resurrection fix) ---------------------------


class _CountingTalker:
    """Streams a re-engagement line; counts how many times it is asked to speak."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, **kw):
        self.calls += 1
        yield "Hello! How can I help you with Westgate today?"


class _CountingDirector:
    """Canned verdict; counts completions (each = one post-turn LLM call)."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        import json as _json

        return _json.dumps({"slots": {}, "advance": "main.done", "note": None})


_END_PB = textwrap.dedent("""
    journeys:
      main:
        checkpoints:
          - id: ask
            goal: "ask"
            advance_when: [{when: "done", judge: llm, to: main.done}]
          - id: done
            terminal: true
            outcome: completed
            say_verbatim: "Thank you. Goodbye."
""")


async def _ended_agent():
    from superdialog.playbook.agent import PlaybookAgent

    talker, director = _CountingTalker(), _CountingDirector()
    agent = PlaybookAgent(
        playbook=Playbook.from_yaml(_END_PB),
        talker_llm=talker,
        director_llm=director,
        http=FakeHttp([]),
    )
    async for _ in agent.greet():
        pass
    async for _ in agent.stream_turn("ok done"):  # -> terminal
        pass
    assert agent.runtime.state.ended
    return agent, talker, director


async def test_post_terminal_turn_is_silent() -> None:
    agent, talker, director = await _ended_agent()
    t_before, d_before = talker.calls, director.calls
    spoken = [c.text async for c in agent.stream_turn("Hello?") if c.text]
    assert spoken == []  # no resurrection speech
    # neither the Talker nor the Director ran on the post-terminal turn
    assert talker.calls == t_before
    assert director.calls == d_before
    assert agent.runtime.state.ended  # still ended


async def test_post_terminal_turn_is_recorded_for_audit() -> None:
    agent, _, _ = await _ended_agent()
    n_before = sum(1 for e in agent.runtime.log.events if e.type == "utterance")
    async for _ in agent.stream_turn("मेरे पास टाइम है, बात करोगी?"):
        pass
    utts = [e for e in agent.runtime.log.events if e.type == "utterance"]
    assert len(utts) == n_before + 1
    assert utts[-1].text == "मेरे पास टाइम है, बात करोगी?"


async def test_repeated_post_terminal_turns_never_resurrect() -> None:
    agent, talker, _ = await _ended_agent()
    for probe in ("Hello?", "Hello? Hello?", "I have time now"):
        spoken = [c.text async for c in agent.stream_turn(probe) if c.text]
        assert spoken == []
    assert talker.calls == 1  # only the pre-terminal turn ever spoke
