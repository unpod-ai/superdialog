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

from .provider import CompletionResult, StreamChunk, apply_json_mode


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


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Normalize SDK tool-call objects to plain dicts (OpenAI tool-call shape)."""
    return [
        tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
        for tc in (raw_calls or [])
    ]


class OpenAIProvider:
    """``LLMProvider`` backed by ``openai.AsyncOpenAI.chat.completions``."""

    def __init__(self, model: str, **default_opts: Any) -> None:
        self.model = model
        self.default_opts: dict[str, Any] = default_opts
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = make_openai_client()
        return self._client

    def _build_kwargs(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": strip_provider_prefix(self.model),
            "messages": messages,
        }
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
        """Yield streamed text/tool-call deltas; tolerates usage-only chunks."""
        client = self._ensure_client()
        kwargs = {
            **self._build_kwargs(messages, tools),
            **self.default_opts,
            **opts,
            "stream": True,
        }
        resp = await client.chat.completions.create(**kwargs)
        async for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue  # usage-only chunk (stream_options include_usage)
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
