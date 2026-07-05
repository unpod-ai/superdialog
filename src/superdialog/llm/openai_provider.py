"""OpenAIProvider — LLMProvider backed by the official ``openai`` SDK.

For OpenAI-compatible endpoints: the OpenAI API, the LiveKit inference gateway
(JWT + gateway base URL), or any openai-style ``base_url``. Native tool-calling
fidelity comes from the SDK; multi-provider breadth lives in
:class:`AnyLlmProvider` / :class:`LitellmProvider`.

Mirrors :class:`LitellmProvider`'s ``complete``/``stream`` contract so it is a
drop-in :class:`~superdialog.llm.provider.LLMProvider`.
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator

from .anyllm_provider import _extract_usage
from .provider import CompletionResult, StreamChunk, apply_json_mode

# LiveKit inference gateway JWT lifetime. ``create_access_token`` mints a 600s
# token upstream; we request a longer one and rebuild the client before it
# lapses so a long voice call never 401s mid-stream. STT/TTS re-mint per WS
# connect (gateway_stt/gateway_tts); this HTTP path caches an AsyncOpenAI
# client, so it must refresh the token itself.
_LK_TOKEN_TTL_S = 1800.0
_LK_REFRESH_MARGIN_S = 120.0


def strip_provider_prefix(model: str) -> str:
    """Strip a leading ``openai/`` (or ``livekit/``) scheme to a bare model id."""
    for prefix in ("openai/", "livekit/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def make_openai_client() -> Any:
    """Build an ``AsyncOpenAI`` client (OpenAI default, or LiveKit gateway).

    ``LLM_BACKEND=livekit`` (or LiveKit credentials present with no explicit
    backend) routes through the LiveKit inference gateway via a generated
    access token; otherwise the default OpenAI endpoint with ``OPENAI_API_KEY``.
    """
    from openai import AsyncOpenAI

    lk_api_key = os.environ.get("LIVEKIT_API_KEY") or os.environ.get(
        "LIVEKIT_INFERENCE_API_KEY"
    )
    lk_api_secret = os.environ.get("LIVEKIT_API_SECRET") or os.environ.get(
        "LIVEKIT_INFERENCE_API_SECRET"
    )
    backend = os.environ.get(
        "LLM_BACKEND",
        "livekit" if (lk_api_key and lk_api_secret) else "openai",
    )
    if backend == "livekit" and lk_api_key and lk_api_secret:
        try:
            from livekit.agents.inference.llm import (
                create_access_token,
                get_default_inference_url,
            )

            token = create_access_token(lk_api_key, lk_api_secret)
            return AsyncOpenAI(api_key=token, base_url=get_default_inference_url())
        except ImportError:
            pass
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def make_livekit_client(ttl: float = _LK_TOKEN_TTL_S) -> tuple[Any, float]:
    """Build an ``AsyncOpenAI`` bound to the LiveKit inference gateway.

    Returns ``(client, expiry_monotonic)`` — the monotonic clock value at which
    the minted access token lapses, so the caller can refresh before then.
    Requires ``LIVEKIT_API_KEY`` + ``LIVEKIT_API_SECRET`` (or their
    ``LIVEKIT_INFERENCE_*`` aliases); raises ``ValueError`` when absent so a
    ``livekit/`` model fails loudly instead of silently hitting OpenAI.
    """
    from openai import AsyncOpenAI

    from .livekit_gateway import livekit_gateway_url, mint_livekit_token

    token = mint_livekit_token(ttl)
    return (
        AsyncOpenAI(api_key=token, base_url=livekit_gateway_url()),
        time.monotonic() + ttl,
    )


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Normalize SDK tool-call objects to plain dicts (OpenAI tool-call shape)."""
    return [
        tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
        for tc in (raw_calls or [])
    ]


class OpenAIProvider:
    """``LLMProvider`` backed by ``openai.AsyncOpenAI.chat.completions``."""

    def __init__(
        self, model: str, *, livekit: bool = False, **default_opts: Any
    ) -> None:
        self.model = model
        self.default_opts: dict[str, Any] = default_opts
        # ``livekit`` forces the LiveKit inference gateway (JWT + gateway URL)
        # regardless of the ambient LLM_BACKEND, and enables token refresh.
        self._livekit = livekit
        self._client: Any = None
        self._client_expiry = 0.0

    def _ensure_client(self) -> Any:
        if self._livekit:
            now = time.monotonic()
            if (
                self._client is None
                or now >= self._client_expiry - _LK_REFRESH_MARGIN_S
            ):
                self._client, self._client_expiry = make_livekit_client()
            return self._client
        if self._client is None:
            self._client = make_openai_client()
        return self._client

    def _build_kwargs(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        # The LiveKit gateway routes by <provider>/<model> (openai/gpt-4o-mini,
        # google/gemma-4-31b-it), so keep the prefix there. The direct OpenAI
        # API wants a bare id, so strip it only off that path.
        model = self.model if self._livekit else strip_provider_prefix(self.model)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        return kwargs

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        """One completion; returns text + normalized tool calls + usage metadata."""
        client = self._ensure_client()
        opts = apply_json_mode(opts)
        kwargs = {**self._build_kwargs(messages, tools), **self.default_opts, **opts}
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            text=msg.content or "",
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
        """Yield streamed text/tool-call deltas; emit usage on the final chunk.

        Requests ``stream_options.include_usage`` and captures the trailing
        usage-only chunk so streamed turns report tokens — parity with
        LitellmProvider. Without this the LiveKit gateway (livekit/) path
        reports zero tokens for streamed turns, undercounting cost/billing.
        """
        client = self._ensure_client()
        kwargs = {
            **self._build_kwargs(messages, tools),
            **self.default_opts,
            **opts,
            "stream": True,
        }
        kwargs.setdefault("stream_options", {"include_usage": True})
        resp = await client.chat.completions.create(**kwargs)
        usage_meta: dict[str, int] = {}
        pending_done: StreamChunk | None = None
        async for chunk in resp:
            u = getattr(chunk, "usage", None)
            if u and not usage_meta:
                usage_meta = _extract_usage(u)
            if not getattr(chunk, "choices", None):
                continue  # usage-only chunk (stream_options include_usage)
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
            )
            # Defer the terminal chunk so usage (a later usage-only chunk) rides
            # out on it — matches LitellmProvider's contract.
            if is_done:
                pending_done = sc
            else:
                yield sc
        if pending_done is not None:
            yield StreamChunk(
                text=pending_done.text,
                tool_call_delta=pending_done.tool_call_delta,
                done=True,
                usage=usage_meta or None,
            )
