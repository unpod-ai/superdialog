"""LLMAgent — raw chat brain implementing the :class:`Agent` Protocol.

No state machine, no slots, no flow position. Just chat history + an
LLMProvider resolved from a model URI. Useful when you want sessions /
persistence / concurrency from SuperDialog but no flow opinion on top.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from ..agent import TurnResult
from ..chat_context import ChatContext, ChatMessage
from ..llm.provider import LLMProvider
from ..llm.resolver import resolve_llm
from ..observability.observer import NullObserver, Observer
from ..stream import StreamChunk


class LLMAgent:
    """Bare-bones chat agent backed by an :class:`LLMProvider`."""

    def __init__(
        self,
        llm: str | LLMProvider,
        *,
        system_prompt: str = "",
        chat_ctx: ChatContext | None = None,
        observer: Observer | None = None,
        trace_id: str = "",
        **provider_opts: Any,
    ) -> None:
        self._provider: LLMProvider = resolve_llm(llm) if isinstance(llm, str) else llm
        self._system_prompt = system_prompt
        self._chat: ChatContext = chat_ctx or ChatContext()
        self._observer: Observer = observer or NullObserver()
        self._trace_id: str = trace_id
        self._provider_opts = provider_opts

    # ---- Agent Protocol ---------------------------------------------------

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat

    def load_chat_ctx(self, ctx: ChatContext) -> None:
        self._chat = ctx

    def set_observer(self, observer: Observer, trace_id: str) -> None:
        """Inject an observer + session trace_id after construction."""
        self._observer = observer
        self._trace_id = trace_id

    def assist(self, text: str) -> None:
        if not text:
            return
        self._chat.items.append(ChatMessage(role="system", content=text))

    async def turn(
        self,
        text: str,
        *,
        stream: bool = False,
    ) -> TurnResult | AsyncIterator[StreamChunk]:
        self._chat.items.append(ChatMessage(role="user", content=text))
        messages = self._build_messages()

        if stream:
            return self._stream_turn(messages)

        obs_id = self._observer.on_generation_start(self._trace_id, "turn", messages)
        t0 = time.perf_counter()
        result = await self._provider.complete(messages, **self._provider_opts)
        latency_ms = (time.perf_counter() - t0) * 1000
        self._chat.items.append(ChatMessage(role="assistant", content=result.text))
        full_metadata = {**result.metadata, "latency_ms": latency_ms}
        self._observer.on_generation_end(
            obs_id, result.text, result.tool_calls, full_metadata
        )
        return TurnResult(
            text=result.text,
            tool_calls=[],
            metadata=full_metadata,
        )

    async def _stream_turn(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[StreamChunk]:
        buffer: list[str] = []
        async for chunk in self._provider.stream(messages, **self._provider_opts):
            if chunk.text:
                buffer.append(chunk.text)
            yield StreamChunk(
                text=chunk.text or "",
                done=chunk.done,
                turn=None,
            )
        final_text = "".join(buffer)
        self._chat.items.append(ChatMessage(role="assistant", content=final_text))

    # ---- internals --------------------------------------------------------

    def _build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        for m in self._chat.items:
            messages.append({"role": m.role, "content": m.content})
        return messages


__all__ = ["LLMAgent"]
