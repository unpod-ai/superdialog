import pytest

import superdialog.llm.openai_provider as op
from superdialog.llm.anyllm_provider import AnyLlmProvider
from superdialog.llm.litellm_provider import LitellmProvider
from superdialog.llm.openai_provider import OpenAIProvider
from superdialog.llm.registry import register_llm_provider
from superdialog.llm.resolver import resolve_llm

# ``resolve_llm`` now wraps the backend in a ResilientProvider; inspect
# ``.inner`` for the selected backend. Attribute access (``.model`` etc.)
# delegates to the inner backend, so those assertions are unchanged.


def test_openai_uri() -> None:
    # Default backend is now any-llm (official-SDK delegation); the model URI is
    # preserved. Use the `litellm/` scheme to force the LiteLLM backend.
    p = resolve_llm("openai/gpt-5.1")
    assert isinstance(p.inner, AnyLlmProvider)
    assert p.model == "openai/gpt-5.1"


def test_litellm_scheme_forces_litellm() -> None:
    p = resolve_llm("litellm/openai/gpt-5.1")
    assert isinstance(p.inner, LitellmProvider)
    assert p.model == "openai/gpt-5.1"


def test_anthropic_uri() -> None:
    p = resolve_llm("anthropic/claude-opus-4-7")
    assert isinstance(p.inner, AnyLlmProvider)
    assert p.model == "anthropic/claude-opus-4-7"


def test_vllm_with_host() -> None:
    p = resolve_llm("vllm/llama-3@http://my-vllm:8000")
    assert isinstance(p.inner, LitellmProvider)
    assert p.model == "hosted_vllm/llama-3"
    assert p.default_opts.get("api_base") == "http://my-vllm:8000"


def test_ollama_with_host() -> None:
    p = resolve_llm("ollama/llama3@http://localhost:11434")
    assert isinstance(p.inner, LitellmProvider)
    assert p.model == "ollama/llama3"
    assert p.default_opts.get("api_base") == "http://localhost:11434"


def test_custom_provider_requires_registration() -> None:
    with pytest.raises(ValueError, match="Unknown custom provider"):
        resolve_llm("custom/unknown/model")


def test_custom_provider_after_registration() -> None:
    register_llm_provider("kerali", "https://llm.kerali.io/v1", "key-123")
    p = resolve_llm("custom/kerali/llama-3-70b")
    assert isinstance(p.inner, LitellmProvider)
    assert p.model == "openai/llama-3-70b"
    assert p.default_opts.get("api_base") == "https://llm.kerali.io/v1"
    assert p.default_opts.get("api_key") == "key-123"


def test_livekit_scheme_routes_to_gateway() -> None:
    # ``livekit/<model>`` forces the OpenAI SDK backend against the LiveKit
    # inference gateway. Construction is lazy (no client until first call), so
    # this needs no LiveKit creds or network. The scheme is stripped to a bare
    # gateway model id.
    p = resolve_llm("livekit/google/gemma-4-31b-it")
    assert isinstance(p.inner, OpenAIProvider)
    assert p.inner._livekit is True
    assert p.model == "google/gemma-4-31b-it"


def test_livekit_client_refreshes_before_token_expiry(monkeypatch) -> None:
    # The gateway JWT has a bounded TTL; a call longer than the token lifetime
    # must rebuild the client with a fresh token instead of 401ing mid-stream.
    clock = {"t": 1000.0}
    builds = {"n": 0}

    def fake_make_livekit_client(ttl: float = op._LK_TOKEN_TTL_S):
        builds["n"] += 1
        return f"client-{builds['n']}", clock["t"] + ttl

    monkeypatch.setattr(op, "make_livekit_client", fake_make_livekit_client)
    monkeypatch.setattr(op.time, "monotonic", lambda: clock["t"])

    prov = OpenAIProvider(model="google/gemma-4-31b-it", livekit=True)
    assert prov._ensure_client() == "client-1"  # first build
    assert prov._ensure_client() == "client-1"  # cached, token still fresh
    # Advance to within the refresh margin of expiry -> rebuild.
    clock["t"] = 1000.0 + op._LK_TOKEN_TTL_S - op._LK_REFRESH_MARGIN_S + 1
    assert prov._ensure_client() == "client-2"
    assert builds["n"] == 2


def test_livekit_client_requires_credentials(monkeypatch) -> None:
    for var in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="LIVEKIT_API_KEY"):
        op.make_livekit_client()
