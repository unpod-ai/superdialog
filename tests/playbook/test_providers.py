"""Tests for the reusable provider adapters (no network).

A fake LLMProvider drives ``ProviderDirector`` / ``ProviderTalker`` so the
adapter wiring is verified without any litellm call.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from superdialog.llm.openai_provider import OpenAIProvider
from superdialog.llm.provider import CompletionResult, StreamChunk, apply_json_mode
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


# --- json_mode normalization (capability playbook-verdict-reliability) ---


def test_apply_json_mode_sets_response_format() -> None:
    assert apply_json_mode({"json_mode": True}) == {
        "response_format": {"type": "json_object"}
    }


def test_apply_json_mode_is_noop_without_flag() -> None:
    assert apply_json_mode({"temperature": 0}) == {"temperature": 0}


def test_apply_json_mode_preserves_explicit_response_format() -> None:
    out = apply_json_mode(
        {"json_mode": True, "response_format": {"type": "json_schema"}}
    )
    assert out == {"response_format": {"type": "json_schema"}}


async def test_openai_provider_forwards_json_mode_to_sdk() -> None:
    # json_mode is normalized to the backend's response_format and never leaks
    # to the SDK as an unknown kwarg.
    captured: dict[str, Any] = {}

    class _Msg:
        content = "{}"
        tool_calls = None

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]
        usage = None

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    p = OpenAIProvider("openai/gpt-4o-mini")
    p._client = _FakeClient()
    await p.complete([{"role": "user", "content": "hi"}], json_mode=True)
    assert captured["response_format"] == {"type": "json_object"}
    assert "json_mode" not in captured
