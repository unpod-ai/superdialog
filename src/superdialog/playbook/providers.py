"""Reusable Director/Talker adapters over an :class:`LLMProvider`.

A Playbook's Director needs a plain-text completion per verdict
(:class:`~superdialog.playbook.director.CompletesLLM`) and its Talker needs
a token stream (:class:`~superdialog.playbook.talker.StreamsLLM`). Both shapes
are trivially backed by the project's :class:`LLMProvider`; these adapters
do that wiring once so CLIs, examples, and hosts share it instead of
re-declaring inline classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMUsageEvent:
    """Per-call LLM usage emitted by the playbook adapters for billing.

    Duck-typed to the unpod-sdk usage callback (``tokens_in`` / ``tokens_out`` /
    ``cached`` / ``cache_write`` / ``model``), so the playbook engine reports
    tokens to the same sink the graph engine's ``LLMCallData`` feeds.
    """

    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: int = 0
    cache_write: int = 0
    role: str = ""  # "director" | "talker" | "" (unknown emitter)
    #: True when tokens_in/out are a char/4 estimate because the backend
    #: streamed/returned no usage object (e.g. Cerebras / some gateways). The
    #: billing sink can flag or discount estimated rows; a conservative-upward
    #: estimate beats the silent ~0 that a missing usage object otherwise bills.
    estimated: bool = False


#: An async callback invoked once per LLM call with an :class:`LLMUsageEvent`.
OnLLMComplete = Callable[[LLMUsageEvent], Awaitable[None]]


def _has_usage(meta: Mapping[str, Any] | None) -> bool:
    m = meta or {}
    return m.get("prompt_tokens") is not None or m.get("completion_tokens") is not None


def _sum_content_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def _estimated_usage_event(
    model: str, role: str, input_chars: int, output_chars: int
) -> LLMUsageEvent:
    """Char/4 usage estimate for a call whose backend reported no usage object.

    Cache fields stay 0 — never estimate a cache read/write.
    """
    return LLMUsageEvent(
        model=model,
        tokens_in=max(1, input_chars // 4),
        tokens_out=max(1, output_chars // 4),
        role=role,
        estimated=True,
    )


def _log_usage_fallback(already_logged: bool, event: LLMUsageEvent) -> bool:
    """First occurrence per adapter at INFO (operator signal), then DEBUG."""
    msg = "llm_usage_fallback model=%s role=%s est_in=%d est_out=%d"
    args = (event.model, event.role, event.tokens_in, event.tokens_out)
    if already_logged:
        logger.debug(msg, *args)
    else:
        logger.info(msg, *args)
    return True


def _usage_event(
    meta: Mapping[str, Any] | None, model: str, role: str = ""
) -> LLMUsageEvent:
    m = meta or {}
    event = LLMUsageEvent(
        model=model,
        tokens_in=int(m.get("prompt_tokens", 0) or 0),
        tokens_out=int(m.get("completion_tokens", 0) or 0),
        cached=int(m.get("cache_read_tokens", 0) or 0),
        cache_write=int(m.get("cache_write_tokens", 0) or 0),
        role=role,
    )
    # Cache-hit telemetry (measure the prompt cache, never hand-place it): a
    # low read ratio on a warm call means a prompt prefix went unstable — the
    # byte-identical-prefix invariant broke somewhere upstream.
    if event.tokens_in > 0 and (event.cached or event.cache_write):
        logger.debug(
            "[llm-usage] role=%s cache_read_ratio=%.2f (in=%d cached=%d write=%d)",
            role,
            event.cached / event.tokens_in,
            event.tokens_in,
            event.cached,
            event.cache_write,
        )
    return event


class ProviderDirector:
    """``CompletesLLM`` adapter: one plain-text completion per verdict."""

    def __init__(
        self, provider: LLMProvider, on_llm_complete: OnLLMComplete | None = None
    ) -> None:
        self._p = provider
        #: Optional async billing hook; set by DialogMachine.register_llm_callback.
        self.on_llm_complete = on_llm_complete
        self._logged_fallback = False

    @property
    def model_id(self) -> str:
        return getattr(self._p, "model", "")

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        """Return the provider completion's text for ``messages``."""
        msgs = list(messages)
        result = await self._p.complete(msgs, **kw)
        if self.on_llm_complete is not None:
            if not _has_usage(result.metadata) and result.text:
                event = _estimated_usage_event(
                    self.model_id,
                    "director",
                    _sum_content_chars(msgs),
                    len(result.text),
                )
                self._logged_fallback = _log_usage_fallback(
                    self._logged_fallback, event
                )
            else:
                event = _usage_event(result.metadata, self.model_id, role="director")
            await self.on_llm_complete(event)
        return result.text


class ProviderTalker:
    """``StreamsLLM`` adapter: yield non-empty text tokens from the provider."""

    def __init__(
        self, provider: LLMProvider, on_llm_complete: OnLLMComplete | None = None
    ) -> None:
        self._p = provider
        #: Optional async billing hook; set by DialogMachine.register_llm_callback.
        self.on_llm_complete = on_llm_complete
        self._logged_fallback = False

    @property
    def model_id(self) -> str:
        return getattr(self._p, "model", "")

    async def stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[str]:
        """Yield each streamed chunk's text, skipping empty/None chunks.

        Captures the trailing usage-only chunk (``chunk.usage``) and reports it
        to ``on_llm_complete`` once the stream is exhausted, so the talker's
        token usage (incl. cache read/write) reaches billing. When the backend
        streamed no usage object at all, falls back to a char/4 estimate rather
        than billing zero.
        """
        msgs = list(messages)
        usage: dict[str, Any] = {}
        output_chars = 0
        async for chunk in self._p.stream(msgs, **kw):
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if chunk.text:
                output_chars += len(chunk.text)
                yield chunk.text
        if self.on_llm_complete is not None:
            if not usage and output_chars:
                event = _estimated_usage_event(
                    self.model_id, "talker", _sum_content_chars(msgs), output_chars
                )
                self._logged_fallback = _log_usage_fallback(
                    self._logged_fallback, event
                )
            else:
                event = _usage_event(usage, self.model_id, role="talker")
            await self.on_llm_complete(event)


def provider_adapters(
    provider: LLMProvider,
    on_llm_complete: OnLLMComplete | None = None,
) -> tuple[ProviderDirector, ProviderTalker]:
    """Build the (Director, Talker) adapter pair for ``provider``."""
    return (
        ProviderDirector(provider, on_llm_complete),
        ProviderTalker(provider, on_llm_complete),
    )


__all__ = [
    "LLMUsageEvent",
    "OnLLMComplete",
    "ProviderDirector",
    "ProviderTalker",
    "provider_adapters",
]
