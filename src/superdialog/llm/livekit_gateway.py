"""Shared helpers for the LiveKit inference gateway (OpenAI-compatible).

The gateway authenticates with a short-lived JWT minted from
``LIVEKIT_API_KEY``/``LIVEKIT_API_SECRET`` (~10 min TTL). Long calls must
refresh it, so :class:`LiveKitTokenSource` re-mints before expiry. Both engine
paths reuse this: the openai-SDK provider (``livekit/<model>``) via
:func:`mint_livekit_token`, and the LiteLLM provider (``custom/lk-inference/
<model>``) via a token-source ``api_key`` that :class:`LitellmProvider` invokes
per request.

The gateway URL is resolved without importing ``livekit-agents`` (mirrors
supervoice's own ``_livekit_inference_url``) so URI resolution/registration stay
dependency-free; only actual token minting needs the SDK.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

# Registry name for the gateway as a custom LiteLLM provider. Matches the parent
# repo's lite_v2 ``llm_routing.INFERENCE_PROVIDER_NAME`` so both emit the same URI.
LIVEKIT_PROVIDER_NAME = "lk-inference"

# JWT lifetime + refresh guard for the token source (litellm path).
TOKEN_TTL_S = 1800.0
REFRESH_MARGIN_S = 120.0

_DEFAULT_GATEWAY_URL = "https://agent-gateway.livekit.cloud/v1"
_STAGING_GATEWAY_URL = "https://agent-gateway.staging.livekit.cloud/v1"


def livekit_credentials() -> tuple[str, str]:
    """Return ``(api_key, api_secret)`` from env, or raise a clear error."""
    key = os.environ.get("LIVEKIT_API_KEY") or os.environ.get(
        "LIVEKIT_INFERENCE_API_KEY"
    )
    secret = os.environ.get("LIVEKIT_API_SECRET") or os.environ.get(
        "LIVEKIT_INFERENCE_API_SECRET"
    )
    if not (key and secret):
        raise ValueError(
            "livekit inference requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
        )
    return key, secret


def livekit_gateway_url() -> str:
    """Resolve the gateway base URL (honors ``LIVEKIT_INFERENCE_URL``).

    Reimplemented locally (no ``livekit-agents`` import) so URI resolution works
    in envs without the SDK — mirrors ``get_default_inference_url`` upstream.
    """
    url = os.environ.get("LIVEKIT_INFERENCE_URL")
    if url:
        return url
    if ".staging.livekit.cloud" in os.environ.get("LIVEKIT_URL", ""):
        return _STAGING_GATEWAY_URL
    return _DEFAULT_GATEWAY_URL


def mint_livekit_token(ttl: float = TOKEN_TTL_S) -> str:
    """Mint a gateway access token (JWT). Needs the ``livekit-agents`` SDK."""
    key, secret = livekit_credentials()
    try:
        from livekit.agents.inference.llm import create_access_token
    except ImportError as exc:  # eval envs may lack the agents SDK
        raise ValueError(
            "livekit inference needs the livekit-agents package "
            "(install superdialog's 'livekit' extra: uv sync --extra livekit)"
        ) from exc
    return create_access_token(key, secret, ttl=ttl)


class LiveKitTokenSource:
    """Callable returning a cached gateway JWT, re-minted before it lapses.

    Passed as a *callable* ``api_key`` so :class:`LitellmProvider` re-reads a
    fresh token every request instead of capturing one at construction (which
    would 401 a call longer than the token TTL).
    """

    def __init__(
        self, ttl: float = TOKEN_TTL_S, margin: float = REFRESH_MARGIN_S
    ) -> None:
        self._ttl = ttl
        self._margin = margin
        self._token: str | None = None
        self._expiry = 0.0

    def __call__(self) -> str:
        now = time.monotonic()
        if self._token is None or now >= self._expiry - self._margin:
            self._token = mint_livekit_token(self._ttl)
            self._expiry = now + self._ttl
        return self._token


def register_livekit_inference(name: str = LIVEKIT_PROVIDER_NAME) -> None:
    """Register the gateway as a custom LiteLLM provider with a refreshing token."""
    from .registry import register_llm_provider

    register_llm_provider(name, livekit_gateway_url(), LiveKitTokenSource())


__all__: list[str] = [
    "LIVEKIT_PROVIDER_NAME",
    "LiveKitTokenSource",
    "livekit_credentials",
    "livekit_gateway_url",
    "mint_livekit_token",
    "register_livekit_inference",
]


# Callable-key type used by the registry + litellm provider.
DynamicKey = Callable[[], str]
