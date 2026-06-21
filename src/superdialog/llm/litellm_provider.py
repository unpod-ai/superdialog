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
        # Do NOT pass stream_options. With stream_options=None, litellm auto-computes
        # usage from OpenAI's trailing usage chunk and stores it in the done chunk's
        # _hidden_params["usage"]. With stream_options set, litellm tries to yield
        # a separate usage chunk — but swallows it internally, making it unreachable.
        print(f"[LITELLM-DBG] stream model={self.model!r}", flush=True)
        resp = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, **merged
        )
        usage_meta: dict[str, int] = {}
        pending_done: StreamChunk | None = None
        async for chunk in resp:
            chunk_choices = getattr(chunk, "choices", None)
            # Capture usage from any chunk that carries it — litellm v1.88 yields
            # it on a chunk with choices=[StreamingChoices(finish_reason=None)] AFTER
            # the done chunk, so we can't key on choices alone.
            u = getattr(chunk, "usage", None)
            if u and not usage_meta:
                usage_meta = _extract_usage(u)
                print(f"[LITELLM-DBG] captured usage from chunk usage_meta={usage_meta}", flush=True)
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
        print(f"[LITELLM-DBG] stream exhausted usage_meta={usage_meta}", flush=True)
        if pending_done is not None:
            yield StreamChunk(
                text=pending_done.text,
                tool_call_delta=pending_done.tool_call_delta,
                done=True,
                usage=usage_meta if usage_meta else None,
            )
