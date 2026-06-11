"""Reusable Director/Talker adapters over an :class:`LLMProvider`.

A Playbook's Director needs a plain-text completion per verdict
(:class:`~superdialog.playbook.director.CompletesLLM`) and its Talker needs
a token stream (:class:`~superdialog.playbook.talker.StreamsLLM`). Both shapes
are trivially backed by the project's :class:`LLMProvider`; these adapters
do that wiring once so CLIs, examples, and hosts share it instead of
re-declaring inline classes.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..llm.provider import LLMProvider


class ProviderDirector:
    """``CompletesLLM`` adapter: one plain-text completion per verdict."""

    def __init__(self, provider: LLMProvider) -> None:
        self._p = provider

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        """Return the provider completion's text for ``messages``."""
        return (await self._p.complete(list(messages), **kw)).text


class ProviderTalker:
    """``StreamsLLM`` adapter: yield non-empty text tokens from the provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self._p = provider

    async def stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[str]:
        """Yield each streamed chunk's text, skipping empty/None chunks."""
        async for chunk in self._p.stream(list(messages), **kw):
            if chunk.text:
                yield chunk.text


def provider_adapters(
    provider: LLMProvider,
) -> tuple[ProviderDirector, ProviderTalker]:
    """Build the (Director, Talker) adapter pair for ``provider``."""
    return ProviderDirector(provider), ProviderTalker(provider)


__all__ = ["ProviderDirector", "ProviderTalker", "provider_adapters"]
