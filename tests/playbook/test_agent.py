"""PlaybookAgent: the Playbook engine behind the public Agent protocol."""

from typing import AsyncIterator

import anyio

from superdialog.agent import Agent, TurnResult
from superdialog.playbook import EventLog, PlaybookAgent
from superdialog.playbook.events import UtteranceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.talker import StreamsLLM
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE_VERDICT: dict = {"slots": {}, "advance": None, "note": None}


class SpyStreamLLM(StreamLLM):
    """StreamLLM that records the messages passed to each stream() call."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(chunks)
        self.prompts: list[list[dict]] = []

    async def stream(self, messages, **kwargs):
        self.prompts.append(messages)
        async for c in super().stream(messages, **kwargs):
            yield c


class SlowStreamLLM:
    """Talker LLM that sleeps between tokens so a barge-in lands mid-stream."""

    def __init__(self, chunks: list[str], delay: float = 0.05) -> None:
        self.chunks = chunks
        self.delay = delay

    async def stream(self, messages: list[dict], **kwargs: object) -> AsyncIterator:
        for c in self.chunks:
            yield c
            await anyio.sleep(self.delay)


def _agent(
    verdict: dict | None = None,
    http_responses: list[tuple[int, dict]] | None = None,
    talker_llm: StreamsLLM | None = None,
) -> PlaybookAgent:
    return PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=talker_llm or StreamLLM(["Which", " city?"]),
        director_llm=CannedLLM(verdict or _IDLE_VERDICT),
        http=FakeHttp(http_responses or []),
    )


def test_satisfies_agent_protocol() -> None:
    assert isinstance(_agent(), Agent)


async def test_turn_returns_text_and_metadata() -> None:
    agent = _agent()
    result = await agent.turn("hello")
    assert isinstance(result, TurnResult)
    assert result.text == "Which city?"
    assert result.metadata["checkpoint"] == "booking.collect"


async def test_streaming_turn_yields_real_chunks() -> None:
    agent = _agent()
    result = await agent.turn("hello", stream=True)
    assert not isinstance(result, TurnResult)
    chunks = [c async for c in result]
    assert [c.text for c in chunks if c.text] == ["Which", " city?"]
    assert chunks[-1].done


async def test_chat_ctx_round_trip() -> None:
    agent = _agent()
    await agent.turn("hello")
    ctx = agent.chat_ctx
    agent2 = _agent()
    agent2.load_chat_ctx(ctx)
    assert agent2.chat_ctx.items == ctx.items


def test_assist_logs_system_message() -> None:
    agent = _agent()
    agent.assist("note")
    assert any(
        e.role == "system" and e.text == "note" for e in agent.runtime.state.transcript
    )


async def test_talker_speech_logged() -> None:
    agent = _agent()
    await agent.turn("hello")
    spoken = [
        e
        for e in agent.runtime.log.events
        if isinstance(e, UtteranceEvent)
        and e.role == "assistant"
        and e.text == "Which city?"
    ]
    assert len(spoken) == 1  # exactly once — never duplicated
    assert spoken[0].spoke_from_version is not None


async def test_event_log_round_trip() -> None:
    agent = _agent()
    await agent.turn("hello")
    restored = EventLog.from_jsonl(agent.event_log.to_jsonl())
    assert restored.version == agent.event_log.version
    agent2 = _agent()
    agent2.load_event_log(restored)
    assert agent2.runtime.state.checkpoint_id == agent.runtime.state.checkpoint_id


async def test_barge_in_aborts_cleanly() -> None:
    """aclose() mid-stream: no leak, partial speech logged, agent usable."""
    agent = _agent(
        verdict={"slots": {"city": "Pune"}, "advance": None, "note": None},
        talker_llm=SlowStreamLLM(["Where", " to", " today?"]),
    )
    gen = await agent.turn("hello", stream=True)
    assert not isinstance(gen, TurnResult)
    first = await gen.__anext__()  # consume exactly one live chunk
    assert first.text == "Where"
    await gen.aclose()  # barge-in: must return normally, nothing escapes
    partial = [
        e
        for e in agent.runtime.log.events
        if isinstance(e, UtteranceEvent) and e.role == "assistant" and e.text == "Where"
    ]
    assert len(partial) == 1  # partial talker speech logged exactly once
    assert partial[0].spoke_from_version is not None
    # The Director's decision from the aborted turn landed (shielded).
    assert agent.runtime.state.slots["city"].value == "Pune"
    version_after_abort = agent.runtime.log.version
    # The runtime is still usable: a follow-up non-streaming turn works.
    result = await agent.turn("Pune please")
    assert isinstance(result, TurnResult)
    assert agent.runtime.log.version > version_after_abort


async def test_load_event_log_via_public_seam() -> None:
    """load_log swaps equal-length distinct logs without cache poison."""
    agent = _agent()
    log_a = EventLog()
    log_a.append(UtteranceEvent(role="user", text="alpha"))
    log_b = EventLog()
    log_b.append(UtteranceEvent(role="user", text="beta"))
    agent.load_event_log(log_a)
    assert agent.runtime.state.transcript[-1].text == "alpha"  # cache primed
    # Equal-length swap: the version check alone cannot tell A from B.
    agent.load_event_log(log_b)
    assert agent.runtime.state.transcript[-1].text == "beta"


async def test_turn_includes_pass_through() -> None:
    agent = _agent(
        verdict={
            "slots": {"city": "Pune", "date": "2026-06-12"},
            "advance": "booking.confirm",
            "note": None,
        },
        http_responses=[(200, {"data": {"hold_id": "h1"}})],
    )
    result = await agent.turn("Pune tomorrow")
    assert isinstance(result, TurnResult)
    # The Director advanced this turn, so the Talker's stale "Which city?"
    # is suppressed from the final text; the verbatim pass-through stands.
    assert "Which city?" not in result.text
    assert "held" in result.text
    assert result.metadata["ended"] is True
    assert result.metadata["outcome"] == "confirmed"


async def test_talker_sees_current_user_turn() -> None:
    """The Talker's prompt must contain the CURRENT user utterance, not lag."""
    spy = SpyStreamLLM(["Which", " city?"])
    agent = _agent(talker_llm=spy)  # idle verdict, soft collect gate
    await agent.turn("I want a massage")
    assert spy.prompts, "Talker stream was never called"
    user_msgs = [m["content"] for m in spy.prompts[-1] if m["role"] == "user"]
    assert any("I want a massage" in c for c in user_msgs)


async def test_user_utterance_logged_once() -> None:
    """The user utterance is appended exactly once across a streaming turn."""
    agent = _agent()
    await agent.turn("hello")
    users = [
        e
        for e in agent.runtime.log.events
        if isinstance(e, UtteranceEvent) and e.role == "user" and e.text == "hello"
    ]
    assert len(users) == 1


async def test_advance_suppresses_stale_reply() -> None:
    """An advancing turn drops the stale Talker question from Turn.text."""
    agent = _agent(
        verdict={
            "slots": {"city": "Pune", "date": "2026-06-12"},
            "advance": "booking.confirm",
            "note": None,
        },
        http_responses=[(200, {"data": {"hold_id": "h1"}})],
    )
    result = await agent.turn("Pune tomorrow")
    assert isinstance(result, TurnResult)
    assert "held" in result.text  # pass-through verbatim present
    assert "city?" not in result.text  # stale collect question suppressed
