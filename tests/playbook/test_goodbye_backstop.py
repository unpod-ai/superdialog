"""Deterministic goodbye backstop (director.py).

Real transcript that broke: a caller said "Then I'm good. Goodbye, VP
Chandigarh" and the agent — instead of closing — launched into pricing. The
global_goodbye interrupt is LLM-judged, and the verdict missed it (ASR noise /
mid-pitch barge-in), so the goodbye was invisible. The backstop fires the
goodbye interrupt on a clear spoken bye token when the verdict set none, while
staying silent on frustration ('I already told you') and embedded byes.
"""

import textwrap

from superdialog.playbook.director import _clear_goodbye, _false_goodbye
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


def test_false_goodbye_recognizes_bare_negation() -> None:
    # The real production case: "any add-ons or special requests?" -> "No."
    # routed straight to call_end mid-booking, before payment link or
    # confirmation, because the Director LLM claimed the goodbye interrupt.
    assert _false_goodbye("No.")
    assert _false_goodbye("Nope.")
    assert _false_goodbye("No thanks.")
    assert _false_goodbye("Nothing, thanks.")
    assert _false_goodbye("Nahi")
    assert _false_goodbye("Bas")


def test_false_goodbye_covers_phrasing_the_first_cut_missed() -> None:
    # A second real production case: the first version of this guard only
    # allowed a fixed set of trailing phrases (thanks/that's all/else) and
    # missed this exact live call -- "No, I don't require." isn't on any
    # enumerated list but is an ordinary decline of an in-flow offer, not a
    # goodbye. The length+negation-start rewrite must catch it.
    assert _false_goodbye("No, I don't require.")
    assert _false_goodbye("No, I don't need that.")
    assert _false_goodbye("Not needed.")
    assert _false_goodbye("No, we're good.")


def test_false_goodbye_ignores_actual_goodbyes_and_longer_replies() -> None:
    assert not _false_goodbye("No, that's all, goodbye")  # has a bye token
    assert not _false_goodbye("no, I also wanted to ask about pricing")
    assert not _false_goodbye("Goodbye")
    assert not _false_goodbye("")
    assert not _false_goodbye("yes please")


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
    assert rt.state.checkpoint_id == "main.collect"
    assert rt.state.checkpoint_id == "main.collect"


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
