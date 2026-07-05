import pytest

import superdialog.llm.livekit_gateway as lkg
import superdialog.llm.openai_provider as op
from superdialog.llm.anyllm_provider import AnyLlmProvider
from superdialog.llm.litellm_provider import (
    LitellmProvider,
    _resolve_dynamic_credentials,
)
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


def test_custom_lk_inference_auto_registers_from_env(monkeypatch) -> None:
    # `custom/lk-inference/<model>` self-registers the gateway from env (no
    # explicit hook) and routes through LiteLLM with a refreshing token source
    # as api_key. URL + registration need no livekit-agents import.
    monkeypatch.setenv("LIVEKIT_API_KEY", "lk_key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "lk_secret")
    monkeypatch.delenv("LIVEKIT_INFERENCE_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    p = resolve_llm("custom/lk-inference/google/gemma-4-31b-it")
    assert isinstance(p.inner, LitellmProvider)
    assert p.model == "openai/google/gemma-4-31b-it"
    assert p.default_opts.get("api_base") == "https://agent-gateway.livekit.cloud/v1"
    assert isinstance(p.default_opts.get("api_key"), lkg.LiveKitTokenSource)


def test_livekit_gateway_url_env(monkeypatch) -> None:
    monkeypatch.delenv("LIVEKIT_INFERENCE_URL", raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://x.livekit.cloud")
    assert lkg.livekit_gateway_url() == "https://agent-gateway.livekit.cloud/v1"
    monkeypatch.setenv("LIVEKIT_URL", "wss://x.staging.livekit.cloud")
    assert lkg.livekit_gateway_url() == "https://agent-gateway.staging.livekit.cloud/v1"
    monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "https://my-gw/v1")
    assert lkg.livekit_gateway_url() == "https://my-gw/v1"


def test_livekit_token_source_caches_and_refreshes(monkeypatch) -> None:
    clock = {"t": 500.0}
    mints = {"n": 0}

    def fake_mint(ttl: float) -> str:
        mints["n"] += 1
        return f"tok-{mints['n']}"

    monkeypatch.setattr(lkg, "mint_livekit_token", fake_mint)
    monkeypatch.setattr(lkg.time, "monotonic", lambda: clock["t"])
    src = lkg.LiveKitTokenSource(ttl=1000.0, margin=100.0)
    assert src() == "tok-1"  # first mint
    assert src() == "tok-1"  # cached, still fresh
    clock["t"] = 500.0 + 1000.0 - 100.0 + 1  # within refresh margin of expiry
    assert src() == "tok-2"  # re-minted
    assert mints["n"] == 2


def test_litellm_resolves_callable_api_key() -> None:
    # A callable api_key is invoked per request; a string passes through.
    dynamic = {"api_key": lambda: "fresh-token", "api_base": "https://gw/v1"}
    _resolve_dynamic_credentials(dynamic)
    assert dynamic["api_key"] == "fresh-token"
    static = {"api_key": "static-key"}
    _resolve_dynamic_credentials(static)
    assert static["api_key"] == "static-key"


def test_livekit_path_preserves_provider_prefix_in_model() -> None:
    # The gateway routes by <provider>/<model>, so the livekit path must keep
    # the prefix; the direct OpenAI API wants a bare id, so it strips it.
    msgs = [{"role": "user", "content": "x"}]
    lk = OpenAIProvider(model="openai/gpt-4o-mini", livekit=True)
    assert lk._build_kwargs(msgs, None)["model"] == "openai/gpt-4o-mini"
    direct = OpenAIProvider(model="openai/gpt-4o-mini")
    assert direct._build_kwargs(msgs, None)["model"] == "gpt-4o-mini"
    gemma = OpenAIProvider(model="google/gemma-4-31b-it", livekit=True)
    assert gemma._build_kwargs(msgs, None)["model"] == "google/gemma-4-31b-it"


def test_openai_stream_captures_usage(monkeypatch) -> None:
    # stream() must request + surface usage on the final chunk, else streamed
    # turns (e.g. the playbook talker on the livekit/ path) report zero tokens.
    import anyio

    class _Delta:
        def __init__(self, content=None):
            self.content = content
            self.tool_calls = None

    class _Choice:
        def __init__(self, content=None, finish=None):
            self.delta = _Delta(content)
            self.finish_reason = finish

    class _Usage:
        prompt_tokens = 123
        completion_tokens = 7

    class _Chunk:
        def __init__(self, choices=None, usage=None):
            self.choices = choices or []
            self.usage = usage

    captured: dict = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)

        async def _gen():
            yield _Chunk(choices=[_Choice("hel")])
            yield _Chunk(choices=[_Choice("lo", finish="stop")])
            yield _Chunk(choices=[], usage=_Usage())  # trailing usage-only chunk

        return _gen()

    class _Client:
        class chat:
            class completions:
                create = staticmethod(_fake_create)

    prov = OpenAIProvider(model="google/gemma-4-31b-it", livekit=True)
    monkeypatch.setattr(prov, "_ensure_client", lambda: _Client())

    async def _run():
        return [c async for c in prov.stream([{"role": "user", "content": "hi"}])]

    chunks = anyio.run(_run)
    assert captured.get("stream_options") == {"include_usage": True}
    assert chunks[-1].done is True
    assert chunks[-1].usage["prompt_tokens"] == 123
    assert chunks[-1].usage["completion_tokens"] == 7
