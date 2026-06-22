"""LitellmProvider — LLMProvider impl backed by litellm.acompletion."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import litellm

from .anyllm_provider import _extract_usage
from .provider import CompletionResult, StreamChunk


class LitellmProvider:
    def __init__(self, model: str, **default_opts: Any) -> None:
        self.model = model
        self.default_opts: dict[str, Any] = default_opts

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        merged = {**self.default_opts, **opts}
        t0 = time.perf_counter()
        resp = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, **merged
        )
        msg = resp.choices[0].message
        raw_calls = msg.tool_calls or []
        tool_calls = [
            tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            for tc in raw_calls
        ]
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        usage_dict = _extract_usage(usage) if usage else {}
        return CompletionResult(
            text=msg.content or "",
            tool_calls=tool_calls,
            metadata={
                "latency_ms": latency_ms,
                **usage_dict,
                "model": self.model,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        merged = {**self.default_opts, **opts, "stream": True}
        # Request usage on the trailing stream chunk. Without this, some providers
        # (verified: Anthropic) stream NO usage at all, so token + cache accounting
        # silently reports zero for streamed turns — notably the playbook Talker.
        # litellm then yields a usage-bearing chunk (often choices=[] or a
        # post-done chunk); the loop below captures it on every chunk, before the
        # choices-based branching. Callers may override.
        merged.setdefault("stream_options", {"include_usage": True})
        resp = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, **merged
        )
        usage_meta: dict[str, int] = {}
        pending_done: StreamChunk | None = None
        async for chunk in resp:
            chunk_choices = getattr(chunk, "choices", None)
            u = getattr(chunk, "usage", None)
            if u and not usage_meta:
                usage_meta = _extract_usage(u)
            if not chunk_choices:
                continue
            delta = chunk.choices[0].delta
            is_done = chunk.choices[0].finish_reason is not None
            tcs = getattr(delta, "tool_calls", None)
            tc_delta: dict[str, Any] | None = None
            if tcs:
                first = tcs[0]
                tc_delta = (
                    first.model_dump() if hasattr(first, "model_dump") else dict(first)
                )
            sc = StreamChunk(
                text=getattr(delta, "content", None),
                tool_call_delta=tc_delta,
                done=is_done,
                usage=None,
            )
            if is_done:
                pending_done = sc
            else:
                yield sc
        if pending_done is not None:
            yield StreamChunk(
                text=pending_done.text,
                tool_call_delta=pending_done.tool_call_delta,
                done=True,
                usage=usage_meta if usage_meta else None,
            )
