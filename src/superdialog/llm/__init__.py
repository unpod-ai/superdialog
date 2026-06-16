"""LLM provider layer for superdialog."""

from .anyllm_provider import AnyLlmProvider
from .litellm_provider import LitellmProvider
from .openai_provider import OpenAIProvider
from .provider import CompletionResult, LLMProvider, StreamChunk
from .registry import CustomProviderConfig, get_custom, register_llm_provider
from .resilience import LLMResilienceError, ResilienceConfig, ResilientProvider
from .resolver import resolve_backend, resolve_llm

__all__ = [
    "AnyLlmProvider",
    "CompletionResult",
    "CustomProviderConfig",
    "LLMProvider",
    "LLMResilienceError",
    "LitellmProvider",
    "OpenAIProvider",
    "ResilienceConfig",
    "ResilientProvider",
    "StreamChunk",
    "get_custom",
    "register_llm_provider",
    "resolve_backend",
    "resolve_llm",
]
