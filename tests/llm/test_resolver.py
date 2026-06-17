import pytest

from superdialog.llm.anyllm_provider import AnyLlmProvider
from superdialog.llm.litellm_provider import LitellmProvider
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
