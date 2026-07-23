"""Tests for the shared per-URI backend cache in the resolver.

The cache must share SDK-pool backends (any-llm / OpenAI) across resolves so
pooled sessions reuse one warm client pool, while LiteLLM-backed results stay
uncached (litellm has its own global client cache; ``custom/`` credentials
live in a mutable registry). The ambient ``SUPERDIALOG_LLM_BACKEND`` is part
of the key so an env flip never serves a stale backend.
"""

from __future__ import annotations

import pytest

from superdialog.llm.litellm_provider import LitellmProvider
from superdialog.llm.openai_provider import OpenAIProvider
from superdialog.llm.registry import register_llm_provider
from superdialog.llm.resolver import _backend_cache, resolve_backend, resolve_llm


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch: pytest.MonkeyPatch):
    """Isolate cache state per test (module-level dict survives otherwise)."""
    monkeypatch.delenv("SUPERDIALOG_LLM_BACKEND", raising=False)
    _backend_cache.clear()
    yield
    _backend_cache.clear()


def test_same_uri_shares_one_backend() -> None:
    a = resolve_backend("oai/gpt-4.1-mini")
    b = resolve_backend("oai/gpt-4.1-mini")
    assert a is b


def test_resolve_llm_wraps_the_shared_backend() -> None:
    # Fresh cheap ResilientProvider per resolve, same backend underneath —
    # session 2 rides the client pool session 1 (or the boot warm) opened.
    p1 = resolve_llm("oai/gpt-4.1-mini")
    p2 = resolve_llm("oai/gpt-4.1-mini")
    assert p1 is not p2
    assert p1.inner is p2.inner
    assert p1.inner is resolve_backend("oai/gpt-4.1-mini")


def test_distinct_uris_cache_separately() -> None:
    a = resolve_backend("oai/gpt-4.1-mini")
    b = resolve_backend("oai/gpt-4.1")
    assert a is not b


def test_litellm_backed_uris_are_not_cached() -> None:
    a = resolve_backend("litellm/openai/gpt-4.1-mini")
    b = resolve_backend("litellm/openai/gpt-4.1-mini")
    assert isinstance(a, LitellmProvider)
    assert a is not b
    assert not _backend_cache


def test_custom_registry_update_is_honored() -> None:
    # custom/ credentials come from a mutable registry — a cached provider
    # would pin the old base_url/key forever.
    register_llm_provider("cachetest", "https://one.example/v1", "k1")
    p1 = resolve_llm("custom/cachetest/m")
    assert p1.default_opts.get("api_base") == "https://one.example/v1"
    register_llm_provider("cachetest", "https://two.example/v1", "k2")
    p2 = resolve_llm("custom/cachetest/m")
    assert p2.default_opts.get("api_base") == "https://two.example/v1"


def test_env_flip_never_serves_stale_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression guard for the test_env_overrides_default semantics: the same
    # bare URI under a different ambient backend selector is a different key.
    monkeypatch.setenv("SUPERDIALOG_LLM_BACKEND", "openai")
    assert isinstance(resolve_llm("openai/gpt-4.1-mini").inner, OpenAIProvider)
    monkeypatch.setenv("SUPERDIALOG_LLM_BACKEND", "litellm")
    assert isinstance(resolve_llm("openai/gpt-4.1-mini").inner, LitellmProvider)
    # Back to openai -> the first backend is served again, still cached.
    monkeypatch.setenv("SUPERDIALOG_LLM_BACKEND", "openai")
    assert isinstance(resolve_llm("openai/gpt-4.1-mini").inner, OpenAIProvider)


def test_livekit_scheme_is_cached() -> None:
    # The gateway provider refreshes its JWT internally, so sharing is safe.
    a = resolve_backend("livekit/google/gemma-4-31b-it")
    b = resolve_backend("livekit/google/gemma-4-31b-it")
    assert a is b
    assert isinstance(a, OpenAIProvider)
