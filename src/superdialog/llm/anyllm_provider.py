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


def _usage_get(u: Any, name: str) -> Any:
    """Read ``name`` off a usage object or dict."""
    v = getattr(u, name, None)
    if v is None and isinstance(u, dict):
        v = u.get(name)
    return v


def _extract_cache_usage(u: Any) -> dict[str, int]:
    """Normalize provider-specific prompt-cache token counts, when present.

    Explicit (Anthropic/Bedrock/Vertex/Gemini): ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``. Automatic (OpenAI/xAI):
    ``prompt_tokens_details.cached_tokens``. Deepseek:
    ``prompt_cache_hit_tokens``. All collapse to a unified
    ``cache_read_tokens`` / ``cache_write_tokens`` so savings are loggable
    regardless of backend. Absent fields are simply omitted.
    """
    out: dict[str, int] = {}

    read = _usage_get(u, "cache_read_input_tokens")
    if read is not None:
        out["cache_read_tokens"] = int(read)

    if "cache_read_tokens" not in out:
        details = _usage_get(u, "prompt_tokens_details")
        cached = None
        if details is not None:
            cached = getattr(details, "cached_tokens", None)
            if cached is None and isinstance(details, dict):
                cached = details.get("cached_tokens")
        if cached is None:
            cached = _usage_get(u, "prompt_cache_hit_tokens")  # deepseek
        if cached:
            out["cache_read_tokens"] = int(cached)

    write = _usage_get(u, "cache_creation_input_tokens")
    if write is not None:
        out["cache_write_tokens"] = int(write)

    return out


def _extract_usage(u: Any) -> dict[str, int]:
    """Normalize provider-specific token field names to prompt_tokens/completion_tokens.

    OpenAI:    prompt_tokens / completion_tokens
    Anthropic: input_tokens  / output_tokens

    Also surfaces unified prompt-cache counts (``cache_read_tokens`` /
    ``cache_write_tokens``) when the provider reports them.
    """
    prompt = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None) or 0
    completion = (
        getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None) or 0
    )
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        **_extract_cache_usage(u),
    }


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
        self._client: Any | None = None  # reused AnyLLM instance (persistent client)

    def _ensure_client(self) -> Any:
        """Return a cached ``AnyLLM`` instance, building it once on first use.

        ``any_llm.acompletion`` rebuilds the provider's SDK client — and thus a
        fresh httpx connection pool — on every call, so each turn pays a new
        TCP+TLS handshake. Caching the ``AnyLLM`` instance keeps the keep-alive
        pool alive across turns, removing that per-call setup (this is what
        LiteLLM already does via its in-memory client cache). Import stays
        deferred so superdialog runs without the optional ``any-llm-sdk``.
        """
        if self._client is None:
            from any_llm import AnyLLM

            provider = self._provider
            if provider is None:  # bare model uri -> let any-llm infer the provider
                provider, self._model = AnyLLM.split_model_provider(self.model)
            self._client = AnyLLM.create(provider, api_key=None, api_base=None)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        """One completion; returns text + normalized tool calls + usage metadata."""
        client = self._ensure_client()
        merged = {**self.default_opts, **opts}
        t0 = time.perf_counter()
        resp = await client.acompletion(
            model=self._model,
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
                **(_extract_usage(usage) if usage else {}),
                "model": self.model,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield streamed text/tool-call deltas; captures usage from usage-only chunks."""
        client = self._ensure_client()
        merged = {**self.default_opts, **opts, "stream": True}
        # OpenAI streaming suppresses usage by default; request it explicitly.
        if self._provider in ("openai", None):
            merged.setdefault("stream_options", {"include_usage": True})
        print(
            f"[ANYLLM-DBG] stream provider={self._provider!r} model={self._model!r} stream_options={merged.get('stream_options')}",
            flush=True,
        )
        resp = await client.acompletion(
            model=self._model,
            messages=messages,
            tools=tools,
            **merged,
        )
        usage_meta: dict[str, int] = {}
        pending_done: StreamChunk | None = None
        async for chunk in resp:
            if not getattr(chunk, "choices", None):
                u = getattr(chunk, "usage", None)
                print(
                    f"[ANYLLM-DBG] usage-only chunk u={u} choices={getattr(chunk, 'choices', None)!r}",
                    flush=True,
                )
                if u:
                    usage_meta = _extract_usage(u)
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
                # Buffer the done chunk — OpenAI's usage-only chunk follows after
                # this, so we emit done only once the full stream is exhausted.
                pending_done = sc
            else:
                yield sc
        # Stream exhausted: usage_meta now contains the final token counts.
        print(
            f"[ANYLLM-DBG] stream done. usage_meta={usage_meta} pending_done={pending_done is not None}",
            flush=True,
        )
        if pending_done is not None:
            yield StreamChunk(
                text=pending_done.text,
                tool_call_delta=pending_done.tool_call_delta,
                done=True,
                usage=usage_meta if usage_meta else None,
            )
