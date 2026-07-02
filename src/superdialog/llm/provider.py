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


def apply_json_mode(opts: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ``json_mode`` flag into the backend's ``response_format`` param.

    Returns a new opts dict with ``json_mode`` removed. When it was truthy and
    the caller did not already set ``response_format``, request JSON-object
    output (``{"type": "json_object"}``) — the OpenAI-compatible primitive every
    supported backend (OpenAI SDK, LiteLLM, any-llm) understands, so the verdict
    string is reliably parseable. No-op when the flag is absent/false, so
    non-verdict calls are unchanged.
    """
    out = dict(opts)
    if out.pop("json_mode", False):
        out.setdefault("response_format", {"type": "json_object"})
    return out
