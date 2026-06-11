"""Tests for the reusable provider adapters (no network).

A fake LLMProvider drives ``ProviderDirector`` / ``ProviderTalker`` so the
adapter wiring is verified without any litellm call.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from superdialog.llm.provider import CompletionResult, StreamChunk
from superdialog.playbook import (
    ProviderDirector,
    ProviderTalker,
    provider_adapters,
)


class FakeProvider:
    """Minimal LLMProvider: returns a fixed completion and stream chunks."""

    def __init__(self, text: str, pieces: list[str | None]) -> None:
        self._text = text
        self._pieces = pieces

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        return CompletionResult(text=self._text, tool_calls=[], metadata={})

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        for piece in self._pieces:
            yield StreamChunk(text=piece, tool_call_delta=None, done=False)


async def test_provider_director_returns_completion_text() -> None:
    director = ProviderDirector(FakeProvider("verdict json", []))
    out = await director.complete([{"role": "user", "content": "hi"}])
    assert out == "verdict json"


async def test_provider_talker_yields_text_pieces_skipping_empty() -> None:
    talker = ProviderTalker(FakeProvider("", ["Hel", "", "lo", None, "!"]))
    pieces = [p async for p in talker.stream([{"role": "user", "content": "hi"}])]
    assert pieces == ["Hel", "lo", "!"]


async def test_provider_adapters_returns_director_talker_pair() -> None:
    provider = FakeProvider("ok", ["a"])
    director, talker = provider_adapters(provider)
    assert isinstance(director, ProviderDirector)
    assert isinstance(talker, ProviderTalker)
    assert await director.complete([]) == "ok"
    assert [p async for p in talker.stream([])] == ["a"]
