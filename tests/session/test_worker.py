"""Group 6 — SessionWorker behaviour."""

from __future__ import annotations

import asyncio

import pytest

from superdialog.chat_context import ChatContext, ChatMessage
from superdialog.session import InMemorySessionStore, SessionWorker


class _CountingAgent:
    """Minimal Agent stub that records assists and turns; tracks instance id."""

    _instance_count = 0

    def __init__(self) -> None:
        _CountingAgent._instance_count += 1
        self.instance_id = _CountingAgent._instance_count
        self._chat = ChatContext()

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat

    def load_chat_ctx(self, ctx: ChatContext) -> None:
        self._chat = ctx

    def assist(self, text: str) -> None:
        self._chat.items.append(ChatMessage("system", text))

    async def turn(self, text: str, *, stream: bool = False):
        self._chat.items.append(ChatMessage("user", text))
        self._chat.items.append(ChatMessage("assistant", f"echo: {text}"))
        return {"text": f"echo: {text}"}


@pytest.fixture(autouse=True)
def _reset_counter():
    _CountingAgent._instance_count = 0
    yield


@pytest.mark.asyncio
async def test_acquire_creates_fresh_session_when_no_record() -> None:
    worker = SessionWorker(agent_factory=_CountingAgent)
    async with worker.acquire("new") as h:
        result = await h.turn("hi")
    assert result == {"text": "echo: hi"}
    assert worker.cache_size == 1


@pytest.mark.asyncio
async def test_persists_state_on_release() -> None:
    store = InMemorySessionStore()
    worker = SessionWorker(agent_factory=_CountingAgent, store=store)
    async with worker.acquire("X") as h:
        await h.turn("hello")

    record = await store.load("X")
    assert record is not None
    contents = [m.content for m in record.chat_ctx.items]
    assert "hello" in contents
    assert "echo: hello" in contents


@pytest.mark.asyncio
async def test_resume_from_store_rehydrates_chat_ctx() -> None:
    store = InMemorySessionStore()
    worker = SessionWorker(agent_factory=_CountingAgent, store=store)

    async with worker.acquire("Y") as h:
        await h.turn("first")
    # Drop the cache to force a fresh agent on next acquire
    await worker.close_session("Y")
    assert worker.cache_size == 0

    async with worker.acquire("Y") as h:
        await h.turn("second")
        roles = [m.role for m in h.agent.chat_ctx.items]
    assert roles == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_different_ids_use_different_agents() -> None:
    worker = SessionWorker(agent_factory=_CountingAgent)
    async with worker.acquire("A") as ha, worker.acquire("B") as hb:
        assert ha.agent is not hb.agent


@pytest.mark.asyncio
async def test_same_id_concurrent_serialises() -> None:
    worker = SessionWorker(agent_factory=_CountingAgent)
    timeline: list[tuple[str, str]] = []

    async def task(label: str, delay: float) -> None:
        async with worker.acquire("same") as h:
            timeline.append(("enter", label))
            await asyncio.sleep(delay)
            await h.turn(f"msg-{label}")
            timeline.append(("exit", label))

    await asyncio.gather(task("a", 0.02), task("b", 0.01))
    # Every "enter" must be paired with the matching "exit" before the next
    # "enter" — i.e. serial execution under the per-session lock.
    assert timeline[0][0] == "enter"
    assert timeline[1][0] == "exit"
    assert timeline[1][1] == timeline[0][1]  # same label closed
    assert timeline[2][0] == "enter"
    assert timeline[2][1] != timeline[0][1]  # different label opens next
    assert timeline[3][0] == "exit"


@pytest.mark.asyncio
async def test_lru_eviction_persists_before_drop() -> None:
    store = InMemorySessionStore()
    worker = SessionWorker(agent_factory=_CountingAgent, store=store, max_sessions=2)

    async with worker.acquire("A") as h:
        await h.turn("a-msg")
    async with worker.acquire("B") as h:
        await h.turn("b-msg")
    async with worker.acquire("C") as h:
        await h.turn("c-msg")

    # A should have been evicted, B and C should remain.
    assert "A" not in worker._cache
    # A's state should be in the store.
    record = await store.load("A")
    assert record is not None
    assert any(m.content == "a-msg" for m in record.chat_ctx.items)


@pytest.mark.asyncio
async def test_close_session_evicts_and_persists() -> None:
    store = InMemorySessionStore()
    worker = SessionWorker(agent_factory=_CountingAgent, store=store)
    async with worker.acquire("Z") as h:
        await h.turn("hello")
    await worker.close_session("Z")
    assert "Z" not in worker._cache
    record = await store.load("Z")
    assert record is not None


@pytest.mark.asyncio
async def test_close_session_persist_false_deletes_store_record() -> None:
    """A session closed with persist=False must not be resumable.

    Regression: close_session used to always persist, so a later acquire()
    for the same id reloaded the dead transcript into a fresh Agent instance
    -- indistinguishable from silently restarting the conversation. The next
    acquire() here must build a genuinely new, empty-history Agent.
    """
    store = InMemorySessionStore()
    worker = SessionWorker(agent_factory=_CountingAgent, store=store)
    async with worker.acquire("Z") as h:
        await h.turn("hello")
    first_instance_id = worker._cache["Z"][1].instance_id

    await worker.close_session("Z", persist=False)
    assert "Z" not in worker._cache
    assert await store.load("Z") is None

    async with worker.acquire("Z") as h:
        second_instance_id = h.agent.instance_id
        assert h.agent.chat_ctx.items == []
    assert second_instance_id != first_instance_id
