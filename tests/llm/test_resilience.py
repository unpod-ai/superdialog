"""Resilience tests (task 3.4 / capability ``llm-call-resilience``).

Offline and deterministic: fake providers with controllable stalls/failures
exercise the timeout, retry, and hedge paths of :class:`ResilientProvider`
without any network. Timing assertions use tiny sleeps with generous bounds so
they stay robust on slow CI while still proving the tail is *bounded*.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import anyio
import pytest

from superdialog.llm.provider import CompletionResult, StreamChunk
from superdialog.llm.resilience import (
    LLMResilienceError,
    ResilienceConfig,
    ResilientProvider,
    _is_retryable,
)

_MSGS = [{"role": "user", "content": "hi"}]


def _run(coro_factory: Any) -> Any:
    return anyio.run(coro_factory)


async def _collect(provider: ResilientProvider) -> list[str]:
    out: list[str] = []
    async for chunk in provider.stream(_MSGS):
        if chunk.text:
            out.append(chunk.text)
    return out


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _SlowProvider:
    """``complete`` sleeps ``delay`` then returns; counts attempts."""

    def __init__(self, delay: float, text: str = "ok") -> None:
        self.delay = delay
        self.text = text
        self.calls = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> CompletionResult:
        self.calls += 1
        await anyio.sleep(self.delay)
        return CompletionResult(text=self.text, tool_calls=[], metadata={})

    async def stream(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        await anyio.sleep(self.delay)
        yield StreamChunk(text=self.text, tool_call_delta=None, done=True)


class _FlakyComplete:
    """Stalls (long sleep) for the first ``fail_times`` calls, then is fast."""

    def __init__(self, fail_times: int, slow: float = 5.0) -> None:
        self.fail_times = fail_times
        self.slow = slow
        self.calls = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> CompletionResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            await anyio.sleep(self.slow)
        return CompletionResult(text="ok", tool_calls=[], metadata={})

    async def stream(self, *a: Any, **k: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="ok", tool_call_delta=None, done=True)


class _BadComplete:
    """Always raises a non-transient error."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> CompletionResult:
        self.calls += 1
        raise ValueError("bad request")

    async def stream(self, *a: Any, **k: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", tool_call_delta=None, done=True)


class _FlakyStream:
    """Stalls before the first token for the first ``fail_times`` calls."""

    def __init__(self, fail_times: int, slow: float = 5.0) -> None:
        self.fail_times = fail_times
        self.slow = slow
        self.calls = 0

    async def complete(self, *a: Any, **k: Any) -> CompletionResult:
        return CompletionResult(text="ok", tool_calls=[], metadata={})

    async def stream(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.calls <= self.fail_times:
            await anyio.sleep(self.slow)
        yield StreamChunk(text="hi", tool_call_delta=None, done=True)


class _StallMidStream:
    """Yields one token, then stalls forever before the next."""

    def __init__(self, slow: float = 5.0) -> None:
        self.slow = slow
        self.calls = 0

    async def complete(self, *a: Any, **k: Any) -> CompletionResult:
        return CompletionResult(text="ok", tool_calls=[], metadata={})

    async def stream(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        yield StreamChunk(text="a", tool_call_delta=None, done=False)
        await anyio.sleep(self.slow)
        yield StreamChunk(text="b", tool_call_delta=None, done=True)


_FAST = ResilienceConfig(timeout_s=0.05, max_retries=2, backoff_base_s=0.001)


# ---------------------------------------------------------------------------
# _is_retryable predicate
# ---------------------------------------------------------------------------


def test_is_retryable_classification() -> None:
    assert _is_retryable(TimeoutError())
    assert _is_retryable(RuntimeError("Service Unavailable"))
    assert _is_retryable(RuntimeError("rate limit exceeded"))

    class _RateLimitError(Exception):
        pass

    assert _is_retryable(_RateLimitError())

    class _Resp(Exception):
        status_code = 503

    assert _is_retryable(_Resp())
    # Non-transient conditions fail fast.
    assert not _is_retryable(ValueError("invalid api key"))
    assert not _is_retryable(KeyError("missing field"))


# ---------------------------------------------------------------------------
# Timeout (task 3.1) + bounded tail
# ---------------------------------------------------------------------------


def test_induced_stall_is_capped() -> None:
    """A stalled call is abandoned near the timeout, not after the full stall."""
    inner = _SlowProvider(delay=10.0)
    rp = ResilientProvider(inner, ResilienceConfig(timeout_s=0.05, max_retries=0))
    t0 = time.perf_counter()
    with pytest.raises(LLMResilienceError):
        _run(lambda: rp.complete(_MSGS))
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"stall not capped: {elapsed:.2f}s"
    assert inner.calls == 1


def test_timeout_disabled_passes_through() -> None:
    inner = _SlowProvider(delay=0.0, text="done")
    rp = ResilientProvider(inner, ResilienceConfig(timeout_s=None, max_retries=0))
    result = _run(lambda: rp.complete(_MSGS))
    assert result.text == "done"


# ---------------------------------------------------------------------------
# Retry (task 3.2)
# ---------------------------------------------------------------------------


def test_transient_timeout_recovers_via_retry() -> None:
    inner = _FlakyComplete(fail_times=1)
    rp = ResilientProvider(inner, _FAST)
    result = _run(lambda: rp.complete(_MSGS))
    assert result.text == "ok"
    assert inner.calls == 2  # first stalled, retry succeeded


def test_retries_exhausted_surface_controlled_error() -> None:
    inner = _SlowProvider(delay=10.0)
    rp = ResilientProvider(inner, _FAST)
    t0 = time.perf_counter()
    with pytest.raises(LLMResilienceError):
        _run(lambda: rp.complete(_MSGS))
    elapsed = time.perf_counter() - t0
    # 3 attempts * ~timeout + tiny backoffs — still well bounded.
    assert elapsed < 2.0, f"exhaustion not bounded: {elapsed:.2f}s"
    assert inner.calls == 3


def test_non_retryable_error_fails_fast() -> None:
    inner = _BadComplete()
    rp = ResilientProvider(inner, _FAST)
    with pytest.raises(ValueError, match="bad request"):
        _run(lambda: rp.complete(_MSGS))
    assert inner.calls == 1  # no retries on a non-transient error


# ---------------------------------------------------------------------------
# Hedge (task 3.3)
# ---------------------------------------------------------------------------


def test_slow_primary_is_hedged() -> None:
    primary = _SlowProvider(delay=5.0, text="primary")
    hedge = _SlowProvider(delay=0.0, text="hedge")
    cfg = ResilienceConfig(timeout_s=5.0, hedge_enabled=True, hedge_delay_s=0.02)
    rp = ResilientProvider(primary, cfg, hedge=hedge)
    t0 = time.perf_counter()
    result = _run(lambda: rp.complete(_MSGS))
    elapsed = time.perf_counter() - t0
    assert result.text == "hedge"
    assert elapsed < 1.0, f"hedge not within budget: {elapsed:.2f}s"


def test_hedge_off_by_default_uses_only_primary() -> None:
    primary = _SlowProvider(delay=0.0, text="primary")
    hedge = _SlowProvider(delay=0.0, text="hedge")
    # Default config has hedging disabled even if a hedge provider is supplied.
    rp = ResilientProvider(primary, ResilienceConfig(), hedge=hedge)
    result = _run(lambda: rp.complete(_MSGS))
    assert result.text == "primary"
    assert hedge.calls == 0


# ---------------------------------------------------------------------------
# Stream resilience
# ---------------------------------------------------------------------------


def test_stream_retries_before_first_token() -> None:
    inner = _FlakyStream(fail_times=1)
    rp = ResilientProvider(inner, _FAST)
    chunks = _run(lambda: _collect(rp))
    assert chunks == ["hi"]
    assert inner.calls == 2


def test_stream_stall_after_first_token_is_surfaced() -> None:
    inner = _StallMidStream(slow=10.0)
    rp = ResilientProvider(inner, _FAST)

    async def _drain() -> list[str]:
        out: list[str] = []
        async for chunk in rp.stream(_MSGS):
            if chunk.text:
                out.append(chunk.text)
        return out

    with pytest.raises(TimeoutError):
        _run(_drain)
    assert inner.calls == 1  # not silently restarted after a partial stream


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = ResilienceConfig.from_env({})
    assert cfg.timeout_s == 60.0
    assert cfg.max_retries == 2
    assert cfg.hedge_enabled is False


def test_config_from_env_overrides() -> None:
    cfg = ResilienceConfig.from_env(
        {
            "SUPERDIALOG_LLM_TIMEOUT_S": "3.5",
            "SUPERDIALOG_LLM_MAX_RETRIES": "1",
            "SUPERDIALOG_LLM_HEDGE": "true",
            "SUPERDIALOG_LLM_HEDGE_MODEL": "anthropic/claude-haiku-4-5",
            "SUPERDIALOG_LLM_HEDGE_DELAY_S": "0.4",
        }
    )
    assert cfg.timeout_s == 3.5
    assert cfg.max_retries == 1
    assert cfg.hedge_enabled is True
    assert cfg.hedge_model == "anthropic/claude-haiku-4-5"
    assert cfg.hedge_delay_s == 0.4


def test_config_timeout_disable_sentinels() -> None:
    for raw in ("0", "none", "off"):
        cfg = ResilienceConfig.from_env({"SUPERDIALOG_LLM_TIMEOUT_S": raw})
        assert cfg.timeout_s is None, raw
