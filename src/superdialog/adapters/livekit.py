"""LiveKit Agents adapter for any superdialog :class:`Agent`.

Mirrors the ``livekit-plugins-langchain`` pattern: expose a class that
quacks like ``livekit.agents.llm.LLM`` so a LiveKit ``Agent`` can use any
superdialog Agent (``DialogMachine``, ``LLMAgent``, ``LangChainAgent``)
as its turn engine.

PORT NOTE: confidence-driven barge-in interop (VAD end-of-speech signals
piped into a richer streaming protocol) is a v0.4 follow-up. The adapter
currently consumes ``Agent.turn(text, stream=True)`` and surfaces tokens
as LiveKit ``ChatChunk`` frames.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from superdialog.stream import StreamChunk

if TYPE_CHECKING:  # pragma: no cover - only for static type checkers
    from superdialog.agent import Agent
else:
    Agent = Any  # runtime alias so annotations don't import the protocol

logger = logging.getLogger(__name__)


def _require_livekit() -> Any:
    """Return the ``livekit.agents.llm`` module or raise a friendly error."""
    try:
        from livekit.agents import llm as lk_llm  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "DialogMachineLLM requires the livekit extra: "
            "`pip install superdialog[livekit]`"
        ) from e
    return lk_llm


class DialogMachineLLM:
    """LiveKit ``Agent(llm=...)`` adapter backed by any superdialog :class:`Agent`.

    Usage::

        from livekit.agents import Agent as LKAgent, AgentSession
        from superdialog import DialogMachine
        from superdialog.adapters.livekit import DialogMachineLLM

        dm = DialogMachine(flow=flow, llm="openai/gpt-5.1")
        lk_agent = LKAgent(llm=DialogMachineLLM(dm))
        await session.start(agent=lk_agent)

    Any superdialog ``Agent`` (``DialogMachine``, ``LLMAgent``,
    ``LangChainAgent``, or a custom Protocol implementation) works in
    place of ``dm``. The adapter duck-types LiveKit's LLM protocol; the
    precise method shape depends on the ``livekit-agents`` version
    pinned in the ``[livekit]`` extra. Construction is lazy: importing
    this module without livekit installed is allowed, only instantiation
    fails.
    """

    def __init__(self, agent: Agent) -> None:
        _require_livekit()  # fail-fast with a clear error message
        self.agent = agent

    def chat(
        self,
        *,
        chat_ctx: Any,
        fnc_ctx: Any = None,
        conn_options: Any = None,
        **kwargs: Any,
    ) -> "DialogMachineStream":
        """Return a streaming response object LiveKit can iterate over."""
        return DialogMachineStream(
            agent=self.agent,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
        )


class DialogMachineStream:
    """Async iterator that drives a single :class:`Agent` turn.

    Pulls the latest user text out of LiveKit's ``ChatContext``, runs
    ``agent.turn(text, stream=True)`` and yields LiveKit-shaped
    ``ChatChunk`` frames. Falls back to plain dict frames when the
    livekit-agents version does not expose a public ``ChatChunk``
    constructor we can rely on.
    """

    def __init__(
        self,
        agent: Agent,
        chat_ctx: Any,
        fnc_ctx: Any = None,
    ) -> None:
        self._agent = agent
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx
        self._iter: AsyncIterator[StreamChunk] | None = None
        self.final_metadata: dict[str, Any] = {}

    async def _ensure_iter(self) -> AsyncIterator[StreamChunk]:
        if self._iter is not None:
            return self._iter
        user_text = _extract_latest_user_text(self._chat_ctx)
        stream = await self._agent.turn(user_text, stream=True)
        assert hasattr(
            stream, "__aiter__"
        ), "Agent.turn(stream=True) must yield an async iterator"
        self._iter = stream  # type: ignore[assignment]
        return self._iter

    def __aiter__(self) -> "DialogMachineStream":
        return self

    async def __anext__(self) -> Any:
        iterator = await self._ensure_iter()
        try:
            chunk: StreamChunk = await iterator.__anext__()
        except StopAsyncIteration:
            raise
        if chunk.done and chunk.turn is not None:
            self.final_metadata = chunk.turn.metadata or {}
        return _to_chat_chunk(chunk)

    async def aclose(self) -> None:  # pragma: no cover - tested via stop
        iterator = self._iter
        self._iter = None
        if iterator is not None and hasattr(iterator, "aclose"):
            await iterator.aclose()


def _extract_latest_user_text(chat_ctx: Any) -> str:
    """Pull the most recent user message out of a LiveKit ChatContext.

    Tolerates the small API shifts between livekit-agents minor versions
    by trying a sequence of attribute paths and falling back to ``str``.
    """
    if chat_ctx is None:
        return ""
    messages = (
        getattr(chat_ctx, "messages", None) or getattr(chat_ctx, "items", None) or []
    )
    for msg in reversed(list(messages)):
        role = getattr(msg, "role", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        if role != "user":
            continue
        content = (
            getattr(msg, "content", None)
            if not isinstance(msg, dict)
            else msg.get("content")
        )
        if isinstance(content, list):
            parts = [
                p if isinstance(p, str) else getattr(p, "text", "") for p in content
            ]
            return "".join(parts)
        if isinstance(content, str):
            return content
    return ""


def _to_chat_chunk(chunk: StreamChunk) -> Any:
    """Render a :class:`StreamChunk` as a LiveKit ``ChatChunk``.

    Older livekit-agents releases expose ``ChatChunk(request_id, delta)``;
    newer ones use keyword-only constructors. When the symbol is missing
    we fall back to a plain dict — LiveKit's runtime accepts dicts in
    several call sites and tests can assert on shape directly.
    """
    lk_llm = _require_livekit()
    chat_chunk_cls = getattr(lk_llm, "ChatChunk", None)
    delta_cls = getattr(lk_llm, "ChoiceDelta", None)
    if chat_chunk_cls is None:
        return {"text": chunk.text, "done": chunk.done}
    # Try a sequence of constructor shapes that have shipped across
    # livekit-agents versions. Fall back to a plain dict if every shape
    # raises (ValidationError, TypeError, etc.).
    candidates: list[dict[str, Any]] = []
    if delta_cls is not None:
        delta_obj = delta_cls(role="assistant", content=chunk.text)
        candidates.append({"id": "", "delta": delta_obj})
        candidates.append({"request_id": "", "delta": delta_obj})
    candidates.append({"id": "", "delta": {"content": chunk.text}})
    candidates.append({"request_id": "", "delta": {"content": chunk.text}})
    candidates.append({"content": chunk.text})
    for kwargs in candidates:
        try:
            return chat_chunk_cls(**kwargs)
        except Exception:  # noqa: BLE001 - probe across LK versions
            continue
    return {"text": chunk.text, "done": chunk.done}


__all__ = ["DialogMachineLLM", "DialogMachineStream"]
