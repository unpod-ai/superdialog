# tests/playbook/eval/test_cache.py
"""EvalCache: in-memory LLM response cache + cached_speaker wrapper."""

from typing import Any

from superdialog.playbook.eval.cache import EvalCache, cached_speaker


async def test_hash_messages_deterministic() -> None:
    cache = EvalCache()
    msgs = [{"role": "user", "content": "hello"}]
    assert cache.hash_messages(msgs) == cache.hash_messages(msgs)


async def test_different_messages_different_hashes() -> None:
    cache = EvalCache()
    h1 = cache.hash_messages([{"role": "user", "content": "hello"}])
    h2 = cache.hash_messages([{"role": "user", "content": "world"}])
    assert h1 != h2


async def test_cache_miss_returns_none() -> None:
    cache = EvalCache()
    assert cache.get("model/id", "nonexistent") is None


async def test_cache_hit_returns_value() -> None:
    cache = EvalCache()
    msgs = [{"role": "user", "content": "hi"}]
    key = cache.hash_messages(msgs)
    cache.put("my-model", key, "the answer")
    assert cache.get("my-model", key) == "the answer"


async def test_different_model_ids_isolated() -> None:
    cache = EvalCache()
    msgs = [{"role": "user", "content": "same"}]
    key = cache.hash_messages(msgs)
    cache.put("model-a", key, "answer-a")
    assert cache.get("model-b", key) is None
    assert cache.get("model-a", key) == "answer-a"


async def test_cached_speaker_calls_llm_once_for_same_messages() -> None:
    cache = EvalCache()
    call_count = 0

    class CountingLLM:
        async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            return "cached answer"

    wrapped = cached_speaker(CountingLLM(), cache, "gpt-4")
    msgs = [{"role": "user", "content": "what is 2+2?"}]
    r1 = await wrapped.complete(msgs)
    r2 = await wrapped.complete(msgs)
    assert r1 == r2 == "cached answer"
    assert call_count == 1


async def test_cached_speaker_calls_llm_for_different_messages() -> None:
    cache = EvalCache()
    calls: list[str] = []

    class LoggingLLM:
        async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
            calls.append(messages[-1]["content"])
            return f"answer to {messages[-1]['content']}"

    wrapped = cached_speaker(LoggingLLM(), cache, "gpt-4")
    await wrapped.complete([{"role": "user", "content": "q1"}])
    await wrapped.complete([{"role": "user", "content": "q2"}])
    assert calls == ["q1", "q2"]


async def test_cached_speaker_model_id_scopes_cache() -> None:
    cache = EvalCache()

    class TaggedLLM:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
            return self.tag

    msgs = [{"role": "user", "content": "same question"}]
    a = cached_speaker(TaggedLLM("model-a"), cache, "model-a")
    b = cached_speaker(TaggedLLM("model-b"), cache, "model-b")
    assert await a.complete(msgs) == "model-a"
    assert await b.complete(msgs) == "model-b"