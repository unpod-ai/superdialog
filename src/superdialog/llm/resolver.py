"""Resolve a model URI into an LLMProvider instance.

Backend selection (which SDK actually makes the call) is layered on top of the
model URI:

* Explicit backend scheme prefix wins: ``anyllm/…``, ``litellm/…``, ``oai/…``.
* Else the ``SUPERDIALOG_LLM_BACKEND`` env var (``anyllm`` | ``litellm`` |
  ``openai``).
* Else the default backend (``anyllm``), which falls back to LiteLLM when the
  optional ``any-llm-sdk`` package is not installed.

LiteLLM-specific URI forms (``custom/…``, ``…@host``) always resolve through
LiteLLM regardless of the selected backend, since they rely on litellm features.

The resolved backend is wrapped in a :class:`ResilientProvider` so timeout,
retry, and optional hedge apply once for every engine path (D4); inspect
``resolve_llm(uri).inner`` for the underlying backend.
"""

from __future__ import annotations

import os

from .litellm_provider import LitellmProvider
from .provider import LLMProvider
from .registry import get_custom
from .resilience import ResilienceConfig, ResilientProvider

_DEFAULT_BACKEND = "anyllm"
_BACKEND_SCHEMES = {"anyllm/": "anyllm", "litellm/": "litellm", "oai/": "openai"}


def _litellm_resolve(uri: str) -> LLMProvider:
    """Resolve a URI through LiteLLM (handles custom/ and @host forms)."""
    if uri.startswith("custom/"):
        parts = uri.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"Custom URI requires model: {uri}")
        _, name, model = parts
        cfg = get_custom(name)
        if not cfg:
            raise ValueError(f"Unknown custom provider: {name}")
        return LitellmProvider(
            model=f"openai/{model}", api_base=cfg.base_url, api_key=cfg.api_key
        )
    if "@" in uri:
        provider_model, host = uri.split("@", 1)
        scheme, model = provider_model.split("/", 1)
        litellm_scheme = {"vllm": "hosted_vllm", "ollama": "ollama"}.get(scheme, scheme)
        return LitellmProvider(model=f"{litellm_scheme}/{model}", api_base=host)
    return LitellmProvider(model=uri)


def _anyllm_resolve(uri: str) -> LLMProvider:
    """Resolve through any-llm; fall back to LiteLLM if the package is missing."""
    try:
        import any_llm  # noqa: F401  -- availability probe

        from .anyllm_provider import AnyLlmProvider

        return AnyLlmProvider(model=uri)
    except Exception:
        # any-llm-sdk not installed (or failed to import) -> graceful fallback.
        return _litellm_resolve(uri)


def resolve_backend(uri: str) -> LLMProvider:
    """Parse a model URI + backend selector and return the raw backend provider.

    No resilience wrapping (used directly for hedge legs). Most callers want
    :func:`resolve_llm`, which wraps the result.

    Examples:
        openai/gpt-4.1-mini                 -> default backend (any-llm)
        anyllm/anthropic/claude-haiku-4-5   -> AnyLlmProvider
        litellm/anthropic/claude-haiku-4-5  -> LitellmProvider
        oai/gpt-4.1-mini                    -> OpenAIProvider (naked openai SDK)
        custom/<name>/<model>               -> LiteLLM (registered base_url+key)
        vllm/<model>@<host>                  -> LiteLLM (hosted_vllm via api_base)
    """
    backend: str | None = None
    for scheme, name in _BACKEND_SCHEMES.items():
        if uri.startswith(scheme):
            backend = name
            uri = uri[len(scheme) :]
            break
    if backend is None:
        backend = os.environ.get("SUPERDIALOG_LLM_BACKEND", _DEFAULT_BACKEND).lower()

    # LiteLLM-specific URI forms always use LiteLLM.
    if uri.startswith("custom/") or "@" in uri:
        return _litellm_resolve(uri)

    if backend == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=uri)
    if backend == "litellm":
        return _litellm_resolve(uri)
    return _anyllm_resolve(uri)


def resolve_llm(uri: str) -> LLMProvider:
    """Resolve ``uri`` to a backend and wrap it with per-request resilience.

    The returned provider applies a configurable timeout, bounded retry, and an
    optional cross-provider hedge (see :class:`ResilienceConfig`). The raw
    backend is available as ``.inner``.
    """
    inner = resolve_backend(uri)
    cfg = ResilienceConfig.from_env()
    hedge: LLMProvider | None = None
    if cfg.hedge_enabled and cfg.hedge_model:
        try:
            hedge = resolve_backend(cfg.hedge_model)
        except Exception:
            hedge = None
    return ResilientProvider(inner, cfg, hedge)
