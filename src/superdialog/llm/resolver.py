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
from dataclasses import replace

from .fallback import FallbackConfig, FallbackProvider
from .litellm_provider import LitellmProvider
from .livekit_gateway import LIVEKIT_PROVIDER_NAME, register_livekit_inference
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
        if cfg is None and name == LIVEKIT_PROVIDER_NAME:
            # Self-register the LiveKit gateway from env so
            # ``custom/lk-inference/<model>`` works with no explicit registration
            # hook (parity with the ``livekit/`` scheme). Its api_key is a
            # refreshing token source, so long calls never reuse a stale JWT.
            register_livekit_inference()
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


def _build_backend(uri: str) -> LLMProvider:
    """Construct a fresh backend for ``uri`` (uncached; see resolve_backend)."""
    # LiveKit inference gateway: force the OpenAI SDK backend against the
    # gateway (minted JWT + gateway base_url), independent of the ambient
    # LLM_BACKEND so a stray LIVEKIT_API_KEY can't reroute unrelated models.
    if uri.startswith("livekit/"):
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=uri[len("livekit/") :], livekit=True)

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


# Per-process shared backends. Backends are stateless besides their SDK client
# pool (AsyncOpenAI is documented concurrency-safe; the livekit variant
# refreshes its JWT internally), so sessions 2..n reuse the warm connection
# pool session 1 opened. Concurrency notes:
#   * asyncio single-thread: a concurrent same-key resolve is a benign
#     last-write-wins double-build (both instances valid, neither closed) —
#     no lock needed.
#   * Cached clients are NEVER closed; nothing in-tree closes providers today
#     and cache entries live for the process lifetime.
#   * The key includes the ambient SUPERDIALOG_LLM_BACKEND so an env flip
#     (tests, dev) can never serve a backend built under a different selector.
#   * SDK clients are built lazily on first call and their connections bind to
#     the running loop — the cache assumes one long-lived loop per process
#     (true for the pool, servers, and CLI runs).
_backend_cache: dict[tuple[str, str], LLMProvider] = {}


def resolve_backend(uri: str) -> LLMProvider:
    """Parse a model URI + backend selector and return the backend provider.

    No resilience wrapping (used directly for hedge legs). Most callers want
    :func:`resolve_llm`, which wraps the result.

    SDK-pool backends (any-llm / OpenAI) are cached per URI and shared across
    sessions so repeat resolves reuse one warm client pool. LiteLLM-backed
    results (``litellm/``, ``custom/``, ``@host``, any-llm import fallback)
    are built fresh each time: litellm keeps its own global client cache, and
    ``custom/`` credentials live in a mutable registry a cache would go stale
    against.

    Examples:
        openai/gpt-4.1-mini                 -> default backend (any-llm)
        anyllm/anthropic/claude-haiku-4-5   -> AnyLlmProvider
        litellm/anthropic/claude-haiku-4-5  -> LitellmProvider
        oai/gpt-4.1-mini                    -> OpenAIProvider (naked openai SDK)
        livekit/google/gemma-4-31b-it       -> OpenAIProvider on the LiveKit gateway
        custom/<name>/<model>               -> LiteLLM (registered base_url+key)
        vllm/<model>@<host>                  -> LiteLLM (hosted_vllm via api_base)
    """
    key = (os.environ.get("SUPERDIALOG_LLM_BACKEND", "").lower(), uri)
    cached = _backend_cache.get(key)
    if cached is not None:
        return cached
    provider = _build_backend(uri)
    if not isinstance(provider, LitellmProvider):
        _backend_cache[key] = provider
    return provider


def resolve_llm(uri: str) -> LLMProvider:
    """Resolve ``uri`` to a backend and wrap it with per-request resilience.

    The returned provider applies a configurable timeout, bounded retry, and an
    optional cross-provider hedge (see :class:`ResilienceConfig`). The raw
    backend is available as ``.inner``.

    When ``SUPERDIALOG_LLM_FALLBACK_MODELS`` is set (comma-separated URIs) the
    result is a :class:`FallbackProvider` chain: leg 0 is today's exact wrap,
    fallback legs are cheap wraps (no retry, no hedge). Unset ⇒ byte-identical
    to the plain resilient wrap.
    """
    inner = resolve_backend(uri)
    cfg = ResilienceConfig.from_env()
    hedge: LLMProvider | None = None
    if cfg.hedge_enabled and cfg.hedge_model:
        try:
            hedge = resolve_backend(cfg.hedge_model)
        except Exception:
            hedge = None
    primary = ResilientProvider(inner, cfg, hedge)

    fb = FallbackConfig.from_env()
    if not fb.models:
        return primary
    leg_cfg = replace(cfg, max_retries=0, hedge_enabled=False)
    legs: list[tuple[str, LLMProvider]] = [(uri, primary)]
    for m in fb.models:
        if m == uri:
            continue  # a fallback equal to the primary re-dials the same outage
        try:
            legs.append((m, ResilientProvider(resolve_backend(m), leg_cfg)))
        except Exception:
            # A bad fallback URI must never break primary resolution (same
            # containment the hedge leg gets above).
            continue
    if len(legs) == 1:
        return primary
    return FallbackProvider(legs, cooldown_s=fb.cooldown_s)


async def check_model_available(
    uri: str, *, api_key: str | None = None
) -> bool | None:
    """Whether ``uri`` (``provider/model``) is real and currently servable.

    Checked against the provider's OWN live endpoint (litellm's
    ``get_valid_models(check_provider_endpoint=True)``), not a fixed list —
    self-contained in superdialog, no host-specific allowlist required.

    Returns ``None`` (not ``False``) when availability can't be determined at
    all — no resolvable API key for the provider — so a caller can tell "not
    available" apart from "couldn't check" rather than treating both as a
    hard rejection.
    """
    import asyncio

    provider, _, model = uri.partition("/")
    if not provider or not model:
        return False
    key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
    if not key:
        return None

    def _check() -> bool:
        import litellm

        valid = litellm.get_valid_models(
            check_provider_endpoint=True,
            custom_llm_provider=provider,
            api_key=key,
        )
        return any(model == v or v.endswith(f"/{model}") for v in valid)

    try:
        return await asyncio.to_thread(_check)
    except Exception:
        # A transient network/provider-API failure is "couldn't check", not
        # "model doesn't exist" — never let this path silently reject a real
        # model over a flaky endpoint.
        return None
