"""AnyLlmProvider — LLMProvider backed by any-llm (official provider SDKs).

`any-llm <https://github.com/mozilla-ai/any-llm>`_ delegates to each provider's
official SDK (openai, anthropic, …) rather than reimplementing them, giving
native tool-calling fidelity plus multi-provider breadth behind one call. It
returns OpenAI-compatible ``ChatCompletion`` objects, so this mirrors
:class:`LitellmProvider`'s contract.

The package is an optional dependency (``any-llm-sdk``); import is deferred to
call time so the rest of superdialog runs without it. ``resolve_llm`` falls back
to LiteLLM when it is absent.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from .provider import CompletionResult, StreamChunk


def _split_uri(uri: str) -> tuple[str | None, str]:
    """``'anthropic/claude-haiku-4-5'`` -> ``('anthropic', 'claude-haiku-4-5')``.

    A bare model with no scheme returns ``(None, uri)`` so any-llm infers it.
    """
    if "/" in uri:
        provider, model = uri.split("/", 1)
        return provider, model
    return None, uri


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Normalize any-llm tool-call objects to plain dicts (OpenAI shape)."""
    return [
        tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
        for tc in (raw_calls or [])
    ]


class AnyLlmProvider:
    """``LLMProvider`` backed by ``any_llm.acompletion``."""

    def __init__(self, model: str, **default_opts: Any) -> None:
        self._provider, self._model = _split_uri(model)
        self.model = model
        self.default_opts: dict[str, Any] = default_opts

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        """One completion; returns text + normalized tool calls + usage metadata."""
        import any_llm

        merged = {**self.default_opts, **opts}
        t0 = time.perf_counter()
        resp = await any_llm.acompletion(
            model=self._model,
            provider=self._provider,
            messages=messages,
            tools=tools,
            **merged,
        )
        msg = resp.choices[0].message
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            text=getattr(msg, "content", None) or "",
            tool_calls=_normalize_tool_calls(getattr(msg, "tool_calls", None)),
            metadata={
                "latency_ms": latency_ms,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "model": self.model,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield streamed text/tool-call deltas; tolerates usage-only chunks."""
        import any_llm

        merged = {**self.default_opts, **opts, "stream": True}
        resp = await any_llm.acompletion(
            model=self._model,
            provider=self._provider,
            messages=messages,
            tools=tools,
            **merged,
        )
        async for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue  # usage-only chunk
            delta = chunk.choices[0].delta
            tcs = getattr(delta, "tool_calls", None)
            tc_delta: dict[str, Any] | None = None
            if tcs:
                first = tcs[0]
                tc_delta = (
                    first.model_dump() if hasattr(first, "model_dump") else dict(first)
                )
            yield StreamChunk(
                text=getattr(delta, "content", None),
                tool_call_delta=tc_delta,
                done=chunk.choices[0].finish_reason is not None,
            )
