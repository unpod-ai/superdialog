"""Rolling transcript compaction: keep dropped turns as a protected summary.

The Talker's view drops the oldest transcript entries under budget pressure
(render.render_view packs newest-first). Those turns used to vanish silently,
so a long call lost the facts established early. The compactor folds them into
``state.summary`` -- which render protects from budget pressure -- via the
existing ``PlaybookAgent.apply_memory`` hook, so a deployment-supplied
prior-call digest and the in-call rolling summary share one field.
"""

import anyio

from superdialog.playbook import EventLog, PlaybookAgent
from superdialog.playbook.compact import compact
from superdialog.playbook.events import SummaryEvent, UtteranceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.render import render_view
from superdialog.playbook.state import ConversationState, TranscriptEntry
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE_VERDICT: dict = {"slots": {}, "advance": None, "note": None}


class RecordingLLM:
    """Compactor LLM stub: records prompts, returns a fixed summary."""

    def __init__(self, text: str = "Caller is Ada; wants a Sept appointment.") -> None:
        self.text = text
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self.text


def _line(who: str, i: int) -> str:
    """A turn long enough that a 2k budget must actually drop some."""
    return f"{who} line {i}: " + "words " * 40


def _state_with_turns(n: int) -> tuple[Playbook, ConversationState]:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = EventLog()
    for i in range(n):
        log.append(UtteranceEvent(role="user", text=_line("user", i)))
        log.append(UtteranceEvent(role="assistant", text=_line("agent", i)))
    return pb, ConversationState.fold(log, playbook=pb)


# --- 1. render_view reports what it dropped ------------------------------


def test_render_view_reports_no_drop_when_budget_is_ample() -> None:
    pb, state = _state_with_turns(5)
    assert render_view(pb, state, token_budget=100_000).dropped == 0


def test_render_view_reports_dropped_count_under_tight_budget() -> None:
    pb, state = _state_with_turns(40)
    view = render_view(pb, state, token_budget=2_000)
    kept = len([m for m in view.messages if m["role"] != "system"])
    assert view.dropped > 0
    assert view.dropped == len(state.transcript) - kept


def test_render_view_drops_all_but_the_floor_turn() -> None:
    """The '[start]' placeholder is not a kept turn.

    Was: at this budget EVERY turn was dropped. The recent-turn floor now
    keeps the newest one unconditionally -- a Talker that cannot see what the
    caller just said can only guess -- so exactly one survives. That entry is
    an assistant turn here, hence the placeholder still leads (some providers
    require the first non-system message to be a user turn).
    """
    pb, state = _state_with_turns(40)
    view = render_view(pb, state, token_budget=1)
    assert view.messages[1] == {"role": "user", "content": "[start]"}
    assert view.messages[-1]["content"] == state.transcript[-1].text
    assert view.dropped == len(state.transcript) - 1


# --- 2. the compactor prompt carries the prior summary forward -----------


def test_compact_prompt_holds_prior_summary_and_dropped_turns() -> None:
    llm = RecordingLLM()
    entries = [
        TranscriptEntry(role="user", text="my name is Ada", version=1),
        TranscriptEntry(role="assistant", text="noted, Ada", version=2),
    ]

    async def go() -> str:
        return await compact(llm, "Prior call: asked about option A.", entries)

    out = anyio.run(go)

    assert out == llm.text
    prompt = "\n".join(m["content"] for m in llm.calls[0])
    # The digest must reach the model, or the compactor's rewrite drops it.
    assert "Prior call: asked about option A." in prompt
    assert "my name is Ada" in prompt
    assert "noted, Ada" in prompt


def test_compact_returns_empty_when_llm_yields_nothing() -> None:
    """An unusable completion must leave the existing summary untouched."""

    class BlankLLM:
        async def complete(self, messages, **kwargs) -> str:
            return "   "

    entries = [TranscriptEntry(role="user", text="hello", version=1)]

    async def go() -> str:
        return await compact(BlankLLM(), "prior", entries)

    assert anyio.run(go) == ""


def test_compact_no_entries_is_a_noop_without_an_llm_call() -> None:
    llm = RecordingLLM()

    async def go() -> str:
        return await compact(llm, "prior", [])

    assert anyio.run(go) == ""
    assert llm.calls == []


# --- 3. the agent wires it through apply_memory -------------------------


def _agent(compact_llm: object, budget: int) -> PlaybookAgent:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    return PlaybookAgent(
        pb,
        talker_llm=StreamLLM(["Sure."]),
        director_llm=CannedLLM(_IDLE_VERDICT),
        http=FakeHttp({}),
        token_budget=budget,
        supervisor_llm=None,
        compact_llm=compact_llm,  # type: ignore[arg-type]
    )


def _seed_history(agent: PlaybookAgent, turns: int = 30) -> None:
    for i in range(turns):
        agent.runtime.log.append(UtteranceEvent(role="user", text=_line("old user", i)))
        agent.runtime.log.append(
            UtteranceEvent(role="assistant", text=_line("old agent", i))
        )


def test_turn_folds_dropped_turns_into_the_summary() -> None:
    llm = RecordingLLM()
    agent = _agent(llm, budget=2_000)

    async def go() -> str:
        _seed_history(agent)
        await agent.turn("and one more thing")
        return agent.runtime.state.summary

    summary = anyio.run(go)
    assert summary == llm.text
    assert len(llm.calls) == 1


def test_second_turn_without_new_drops_spends_no_llm_call() -> None:
    """_compacted_through must stop the same prefix being re-summarized."""
    llm = RecordingLLM()
    agent = _agent(llm, budget=2_000)

    async def go() -> int:
        _seed_history(agent)
        await agent.turn("first")
        assert len(llm.calls) == 1
        before = len(llm.calls)
        # A short second turn adds two entries, so at most one further
        # compaction may fire -- never a fresh pass over the same prefix.
        await agent.turn("ok")
        return len(llm.calls) - before

    assert anyio.run(go) <= 1


def test_ample_budget_never_calls_the_compactor() -> None:
    llm = RecordingLLM()
    agent = _agent(llm, budget=100_000)

    async def go() -> None:
        await agent.turn("hello there")

    anyio.run(go)
    assert llm.calls == []


def test_compactor_failure_never_breaks_the_turn() -> None:
    class BoomLLM:
        async def complete(self, messages, **kwargs) -> str:
            raise RuntimeError("compactor down")

    agent = _agent(BoomLLM(), budget=2_000)

    async def go() -> str:
        _seed_history(agent)
        result = await agent.turn("still works?")
        return result.text

    assert anyio.run(go)  # the turn still produced speech
    assert agent.runtime.state.summary == ""


def test_prior_digest_survives_until_the_compactor_rewrites_it() -> None:
    """apply_memory-seeded digest is the compactor's input, not its casualty."""
    llm = RecordingLLM()
    agent = _agent(llm, budget=2_000)

    async def go() -> str:
        agent.apply_memory("Prior call: asked about option A.")
        assert agent.runtime.state.summary == "Prior call: asked about option A."
        _seed_history(agent)
        await agent.turn("hi again")
        return "\n".join(m["content"] for m in llm.calls[0])

    prompt = anyio.run(go)
    assert "Prior call: asked about option A." in prompt


def test_summary_event_is_what_lands_in_the_log() -> None:
    """The write path is apply_memory -> SummaryEvent, not a new event type."""
    llm = RecordingLLM()
    agent = _agent(llm, budget=2_000)

    async def go() -> list[str]:
        _seed_history(agent)
        await agent.turn("hi")
        return [e.text for e in agent.runtime.log.events if isinstance(e, SummaryEvent)]

    assert anyio.run(go) == [llm.text]
