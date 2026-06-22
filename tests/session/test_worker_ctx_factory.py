"""Group 6 — SessionWorker per-session ``agent_factory_ctx`` (playbook hot-load).

These exercise the additive context-aware factory that lets one worker process
multiplex *different* playbooks, keyed by ``playbook_id`` supplied per call via
``acquire(session_id, init=...)``. The existing zero-arg ``agent_factory`` form
must keep working unchanged.
"""

from __future__ import annotations

import pytest

from superdialog.chat_context import ChatContext
from superdialog.session import SessionInit, SessionWorker


class _PlaybookAgentStub:
    """Agent stub that records the playbook_id it was built for."""

    def __init__(self, playbook_id: str) -> None:
        self.playbook_id = playbook_id
        self._chat = ChatContext()

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat

    def load_chat_ctx(self, ctx: ChatContext) -> None:
        self._chat = ctx

    async def turn(self, text: str, *, stream: bool = False):
        return {"text": f"[{self.playbook_id}] {text}"}


@pytest.mark.asyncio
async def test_ctx_factory_receives_session_init_with_metadata() -> None:
    seen: dict[str, SessionInit] = {}

    def factory(init: SessionInit) -> _PlaybookAgentStub:
        seen["init"] = init
        return _PlaybookAgentStub(init.metadata["playbook_id"])

    worker = SessionWorker(agent_factory_ctx=factory)
    init = SessionInit(session_id="s1", metadata={"playbook_id": "PB_42"})
    async with worker.acquire("s1", init=init) as h:
        result = await h.turn("hi")
        # init metadata is bound onto the session for persistence / metering
        assert h.session.metadata["playbook_id"] == "PB_42"

    assert seen["init"].session_id == "s1"
    assert seen["init"].metadata["playbook_id"] == "PB_42"
    assert result == {"text": "[PB_42] hi"}


@pytest.mark.asyncio
async def test_distinct_playbooks_multiplex_in_one_worker() -> None:
    def factory(init: SessionInit) -> _PlaybookAgentStub:
        return _PlaybookAgentStub(init.metadata["playbook_id"])

    worker = SessionWorker(agent_factory_ctx=factory)
    async with worker.acquire(
        "a", init=SessionInit("a", {"playbook_id": "PB_a"})
    ) as ha:
        ra = await ha.turn("x")
    async with worker.acquire(
        "b", init=SessionInit("b", {"playbook_id": "PB_b"})
    ) as hb:
        rb = await hb.turn("x")

    assert ra == {"text": "[PB_a] x"}
    assert rb == {"text": "[PB_b] x"}


@pytest.mark.asyncio
async def test_ctx_factory_called_once_per_cached_session() -> None:
    calls: list[str] = []

    def factory(init: SessionInit) -> _PlaybookAgentStub:
        calls.append(init.session_id)
        return _PlaybookAgentStub(init.metadata["playbook_id"])

    worker = SessionWorker(agent_factory_ctx=factory)
    init = SessionInit("s", {"playbook_id": "PB_1"})
    async with worker.acquire("s", init=init) as h:
        await h.turn("1")
    async with worker.acquire("s", init=init) as h:  # cached → no rebuild
        await h.turn("2")

    assert calls == ["s"]  # bound at creation; factory invoked exactly once


def test_requires_exactly_one_factory_form() -> None:
    with pytest.raises(ValueError):
        SessionWorker()  # neither form
    with pytest.raises(ValueError):
        SessionWorker(
            agent_factory=lambda: None,  # type: ignore[arg-type,return-value]
            agent_factory_ctx=lambda init: None,  # type: ignore[arg-type,return-value]
        )  # both forms


@pytest.mark.asyncio
async def test_zero_arg_factory_still_supported() -> None:
    class _Z:
        def __init__(self) -> None:
            self._chat = ChatContext()

        @property
        def chat_ctx(self) -> ChatContext:
            return self._chat

        def load_chat_ctx(self, ctx: ChatContext) -> None:
            self._chat = ctx

        async def turn(self, text: str, *, stream: bool = False):
            return {"text": text}

    worker = SessionWorker(agent_factory=_Z)
    async with worker.acquire("z") as h:
        assert await h.turn("ok") == {"text": "ok"}
