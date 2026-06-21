"""LLM provider layer for superdialog."""

from .anyllm_provider import AnyLlmProvider
from .litellm_provider import LitellmProvider
from .openai_provider import OpenAIProvider
from .prompt_cache import CACHE_PREFIX_KEY, PromptCacheConfig, mark_cache_prefix
from .provider import CompletionResult, LLMProvider, StreamChunk
from .registry import CustomProviderConfig, get_custom, register_llm_provider
from .resilience import LLMResilienceError, ResilienceConfig, ResilientProvider
from .resolver import resolve_backend, resolve_llm

__all__ = [
    "CACHE_PREFIX_KEY",
    "AnyLlmProvider",
    "CompletionResult",
    "CustomProviderConfig",
    "LLMProvider",
    "LLMResilienceError",
    "LitellmProvider",
    "OpenAIProvider",
    "PromptCacheConfig",
    "ResilienceConfig",
    "ResilientProvider",
    "StreamChunk",
    "get_custom",
    "mark_cache_prefix",
    "register_llm_provider",
    "resolve_backend",
    "resolve_llm",
]
