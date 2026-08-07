"""LitellmProvider — LLMProvider impl backed by litellm.acompletion."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import litellm

from .anyllm_provider import _extract_usage
from .provider import CompletionResult, StreamChunk, apply_json_mode


def _resolve_dynamic_credentials(opts: dict[str, Any]) -> None:
    """Replace a callable ``api_key`` with a freshly sourced token, in place.

    Lets a caller register a refreshing credential (the LiveKit gateway's
    short-lived JWT) that is re-read per request instead of captured once. A
    plain string key passes through untouched.
    """
    key = opts.get("api_key")
    if callable(key):
        opts["api_key"] = key()


class LitellmProvider:
    def __init__(self, model: str, **default_opts: Any) -> None:
        self.model = model
        self.default_opts: dict[str, Any] = default_opts

    async def warmup(self) -> None:
        """No-op: litellm keeps a process-global in-memory client cache, so the
        first real request warms every subsequent one; there is no per-instance
        pool to pre-open. Boot-time warm (a real request) covers the import cost.
        """
        return None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        # print(
        #     f"[LITELLM-DBG] complete model={self.model!r} msgs={len(messages)}",
        #     flush=True,
        # )
        merged = {**self.default_opts, **apply_json_mode(opts)}
        _resolve_dynamic_credentials(merged)
        t0 = time.perf_counter()
        try:
            resp = await litellm.acompletion(
                model=self.model, messages=messages, tools=tools, **merged
            )
        except Exception:
            # print(
            #     f"[LITELLM-DBG] complete FAILED model={self.model!r} "
            #     f"exc={type(_e).__name__}: {_e}",
            #     flush=True,
            # )
            raise
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
        # print(
        #     f"[LITELLM-DBG] stream model={self.model!r} msgs={len(messages)}",
        #     flush=True,
        # )
        merged = {**self.default_opts, **opts, "stream": True}
        _resolve_dynamic_credentials(merged)
        # Request usage on the trailing stream chunk. Without this, some providers
        # (verified: Anthropic) stream NO usage at all, so token + cache accounting
        # silently reports zero for streamed turns — notably the playbook Talker.
        # litellm then yields a usage-bearing chunk (often choices=[] or a
        # post-done chunk); the loop below captures it on every chunk, before the
        # choices-based branching. Callers may override.
        merged.setdefault("stream_options", {"include_usage": True})
        try:
            resp = await litellm.acompletion(
                model=self.model, messages=messages, tools=tools, **merged
            )
        except Exception:
            # print(
            #     f"[LITELLM-DBG] stream FAILED model={self.model!r} "
            #     f"exc={type(_e).__name__}: {_e}",
            #     flush=True,
            # )
            raise
        usage_meta: dict[str, int] = {}
        pending_done: StreamChunk | None = None
        produced_output = False
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
            if sc.text or sc.tool_call_delta:
                produced_output = True
            if is_done:
                pending_done = sc
            else:
                yield sc
        # Some gateway relays (seen in production) return a stream whose final
        # usage is real (completion_tokens > 0 — the model demonstrably
        # generated output server-side) but never send a single content delta.
        # That text was never actually sent, so no amount of chunk-parsing can
        # recover it from *this* stream. Retry once as a plain non-streaming
        # call and surface that text instead of silently yielding nothing.
        if not produced_output and usage_meta.get("completion_tokens"):
            result = await self.complete(messages, tools, **opts)
            if result.text or result.tool_calls:
                yield StreamChunk(
                    text=result.text or None,
                    tool_call_delta=None,
                    done=True,
                    usage={
                        k: v
                        for k, v in result.metadata.items()
                        if k
                        in (
                            "prompt_tokens",
                            "completion_tokens",
                            "cache_read_tokens",
                            "cache_write_tokens",
                        )
                    },
                )
                return
        if pending_done is not None:
            yield StreamChunk(
                text=pending_done.text,
                tool_call_delta=pending_done.tool_call_delta,
                done=True,
                usage=usage_meta if usage_meta else None,
            )
