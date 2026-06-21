"""Provider-backend tests: resolver routing + cross-backend tool-call parity.

Routing tests are offline (construction only). Parity tests make real calls and
skip when ``OPENAI_API_KEY`` is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import anyio
import pytest
from dotenv import load_dotenv

from superdialog.llm import (
    AnyLlmProvider,
    LitellmProvider,
    OpenAIProvider,
    resolve_llm,
)

# Load the repo .env so live tests see provider keys when present.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_MODEL = "openai/gpt-4.1-mini"

_PICK_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "pick_color",
            "description": "Record the color the user names.",
            "parameters": {
                "type": "object",
                "properties": {"color": {"type": "string"}},
                "required": ["color"],
            },
        },
    }
]
_PICK_MESSAGES = [{"role": "user", "content": "Use pick_color with color blue."}]


# ----------------------------------------------------------------------------
# Offline: backend routing (task 1.6 / default-backend)
# ----------------------------------------------------------------------------


def test_default_backend_is_anyllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERDIALOG_LLM_BACKEND", raising=False)
    assert isinstance(resolve_llm("openai/gpt-4.1-mini").inner, AnyLlmProvider)
    assert isinstance(resolve_llm("anthropic/claude-haiku-4-5").inner, AnyLlmProvider)


def test_scheme_selects_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERDIALOG_LLM_BACKEND", raising=False)
    assert isinstance(resolve_llm("litellm/openai/gpt-4.1-mini").inner, LitellmProvider)
    assert isinstance(resolve_llm("oai/gpt-4.1-mini").inner, OpenAIProvider)
    assert isinstance(
        resolve_llm("anyllm/anthropic/claude-haiku-4-5").inner, AnyLlmProvider
    )


def test_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERDIALOG_LLM_BACKEND", "litellm")
    assert isinstance(resolve_llm("openai/gpt-4.1-mini").inner, LitellmProvider)
    monkeypatch.setenv("SUPERDIALOG_LLM_BACKEND", "openai")
    assert isinstance(resolve_llm("openai/gpt-4.1-mini").inner, OpenAIProvider)


def test_litellm_specific_uris_stay_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    # vllm@host and custom/ are LiteLLM features regardless of default backend.
    monkeypatch.delenv("SUPERDIALOG_LLM_BACKEND", raising=False)
    assert isinstance(
        resolve_llm("vllm/my-model@http://localhost:8000").inner, LitellmProvider
    )


def test_anyllm_uri_splits_provider_and_model() -> None:
    p = AnyLlmProvider("anthropic/claude-haiku-4-5")
    assert (p._provider, p._model) == ("anthropic", "claude-haiku-4-5")
    bare = AnyLlmProvider("gpt-4.1-mini")
    assert (bare._provider, bare._model) == (None, "gpt-4.1-mini")


def test_anyllm_provider_reuses_client_across_calls() -> None:
    """Regression: the AnyLLM client (and its keep-alive connection pool) is
    built once and reused across turns, instead of rebuilt on every call."""
    pytest.importorskip("any_llm")
    from unittest.mock import AsyncMock, MagicMock, patch

    msg = MagicMock(content="hi", tool_calls=None)
    resp = MagicMock(choices=[MagicMock(message=msg)], usage=None)
    fake_client = MagicMock()
    fake_client.acompletion = AsyncMock(return_value=resp)

    with patch("any_llm.AnyLLM.create", return_value=fake_client) as create:
        p = AnyLlmProvider("openai/gpt-4.1-mini")

        async def _two_turns() -> None:
            await p.complete([{"role": "user", "content": "a"}])
            await p.complete([{"role": "user", "content": "b"}])

        anyio.run(_two_turns)

    assert create.call_count == 1  # client built once...
    assert fake_client.acompletion.await_count == 2  # ...reused for both turns


# ----------------------------------------------------------------------------
# Live: cross-backend parity (task 1.5)
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason="OPENAI_API_KEY not set")
def test_tool_call_shape_parity_across_backends() -> None:
    """Same prompt + tools through all three backends yields the same tool call."""
    providers = {
        "openai": OpenAIProvider(_MODEL),
        "litellm": LitellmProvider(_MODEL),
        "anyllm": AnyLlmProvider(_MODEL),
    }
    forced = {"type": "function", "function": {"name": "pick_color"}}
    names: dict[str, str] = {}
    for backend, provider in providers.items():
        result = anyio.run(
            lambda p=provider: p.complete(
                _PICK_MESSAGES, tools=_PICK_TOOL, tool_choice=forced
            )
        )
        assert result.tool_calls, f"{backend}: expected a tool call"
        names[backend] = result.tool_calls[0]["function"]["name"]
    # Every backend selected the same tool, in the same normalized shape.
    assert set(names.values()) == {"pick_color"}, names


@pytest.mark.skipif(not _HAS_OPENAI, reason="OPENAI_API_KEY not set")
def test_text_completion_parity_across_backends() -> None:
    """Plain completion returns non-empty text on every backend."""
    msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
    for provider in (
        OpenAIProvider(_MODEL),
        LitellmProvider(_MODEL),
        AnyLlmProvider(_MODEL),
    ):
        result = anyio.run(lambda p=provider: p.complete(msgs))
        assert result.text.strip(), f"{type(provider).__name__}: empty text"
