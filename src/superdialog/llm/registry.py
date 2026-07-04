"""Custom-provider registry for resolver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# ``api_key`` may be a plain string, or a zero-arg callable returning a fresh
# token per call (e.g. the LiveKit gateway's short-lived JWT — see
# ``livekit_gateway.LiveKitTokenSource``). LitellmProvider resolves the callable
# per request so long calls never reuse an expired token.
ApiKey = str | Callable[[], str]


@dataclass
class CustomProviderConfig:
    base_url: str
    api_key: ApiKey
    api_style: str = "openai"


_REGISTRY: dict[str, CustomProviderConfig] = {}


def register_llm_provider(
    name: str, base_url: str, api_key: ApiKey, api_style: str = "openai"
) -> None:
    _REGISTRY[name] = CustomProviderConfig(
        base_url=base_url, api_key=api_key, api_style=api_style
    )


def get_custom(name: str) -> CustomProviderConfig | None:
    return _REGISTRY.get(name)
