# src/superdialog/playbook/eval/cache.py
"""In-memory LLM response cache for eval runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol


class SpeaksUser(Protocol):
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class EvalCache:
    """In-memory cache keyed by (model_id, sha256(messages))."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    def hash_messages(self, messages: list[dict]) -> str:
        serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, model_id: str, key: str) -> str | None:
        return self._store.get(model_id, {}).get(key)

    def put(self, model_id: str, key: str, value: str) -> None:
        if model_id not in self._store:
            self._store[model_id] = {}
        self._store[model_id][key] = value

    def clear(self, model_id: str | None = None) -> None:
        if model_id is None:
            self._store.clear()
        else:
            self._store.pop(model_id, None)


def cached_speaker(llm: SpeaksUser, cache: EvalCache, model_id: str) -> SpeaksUser:
    """Wrap a SpeaksUser with transparent EvalCache caching."""

    class _Cached:
        async def complete(
            self, messages: list[dict[str, str]], **kwargs: Any
        ) -> str:
            key = cache.hash_messages(messages)
            hit = cache.get(model_id, key)
            if hit is not None:
                return hit
            result = await llm.complete(messages, **kwargs)
            cache.put(model_id, key, result)
            return result

    return _Cached()


__all__ = ["EvalCache", "cached_speaker"]