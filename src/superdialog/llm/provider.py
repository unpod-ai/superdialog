"""LLMProvider protocol + result/stream types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass
class CompletionResult:
    text: str
    tool_calls: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class StreamChunk:
    text: str | None
    tool_call_delta: dict[str, Any] | None
    done: bool
    usage: dict[str, int] | None = None  # prompt_tokens + completion_tokens when available


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]: ...
