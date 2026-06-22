"""Reusable Director/Talker adapters over an :class:`LLMProvider`.

A Playbook's Director needs a plain-text completion per verdict
(:class:`~superdialog.playbook.director.CompletesLLM`) and its Talker needs
a token stream (:class:`~superdialog.playbook.talker.StreamsLLM`). Both shapes
are trivially backed by the project's :class:`LLMProvider`; these adapters
do that wiring once so CLIs, examples, and hosts share it instead of
re-declaring inline classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from ..llm.provider import LLMProvider


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


#: An async callback invoked once per LLM call with an :class:`LLMUsageEvent`.
OnLLMComplete = Callable[[LLMUsageEvent], Awaitable[None]]


def _usage_event(meta: Mapping[str, Any] | None, model: str) -> LLMUsageEvent:
    m = meta or {}
    return LLMUsageEvent(
        model=model,
        tokens_in=int(m.get("prompt_tokens", 0) or 0),
        tokens_out=int(m.get("completion_tokens", 0) or 0),
        cached=int(m.get("cache_read_tokens", 0) or 0),
        cache_write=int(m.get("cache_write_tokens", 0) or 0),
    )


class ProviderDirector:
    """``CompletesLLM`` adapter: one plain-text completion per verdict."""

    def __init__(
        self, provider: LLMProvider, on_llm_complete: OnLLMComplete | None = None
    ) -> None:
        self._p = provider
        #: Optional async billing hook; set by DialogMachine.register_llm_callback.
        self.on_llm_complete = on_llm_complete

    @property
    def model_id(self) -> str:
        return getattr(self._p, "model", "")

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        """Return the provider completion's text for ``messages``."""
        result = await self._p.complete(list(messages), **kw)
        if self.on_llm_complete is not None:
            await self.on_llm_complete(_usage_event(result.metadata, self.model_id))
        return result.text


class ProviderTalker:
    """``StreamsLLM`` adapter: yield non-empty text tokens from the provider."""

    def __init__(
        self, provider: LLMProvider, on_llm_complete: OnLLMComplete | None = None
    ) -> None:
        self._p = provider
        #: Optional async billing hook; set by DialogMachine.register_llm_callback.
        self.on_llm_complete = on_llm_complete

    @property
    def model_id(self) -> str:
        return getattr(self._p, "model", "")

    async def stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[str]:
        """Yield each streamed chunk's text, skipping empty/None chunks.

        Captures the trailing usage-only chunk (``chunk.usage``) and reports it
        to ``on_llm_complete`` once the stream is exhausted, so the talker's
        token usage (incl. cache read/write) reaches billing.
        """
        usage: dict[str, Any] = {}
        async for chunk in self._p.stream(list(messages), **kw):
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if chunk.text:
                yield chunk.text
        if self.on_llm_complete is not None:
            await self.on_llm_complete(_usage_event(usage, self.model_id))


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
