"""Tests for the FallbackProvider chain (offline, fake legs).

Covers: chain order, cooldown skip/expiry/clear, all-down pass, mid-stream
surface, pre-first-token failover (including auth errors — chain legs are
different providers/keys, so non-retryable errors DO fail over), model
attribution after failover, compat surface, warmup fan-out, and the
``resolve_llm`` env wiring.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import anyio
import pytest

from superdialog.llm.fallback import (
    FallbackConfig,
    FallbackProvider,
    _down_until,
)
from superdialog.llm.provider import CompletionResult, StreamChunk
from superdialog.llm.resilience import LLMResilienceError

_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    """Module-level cooldown map survives tests otherwise; env stays unset."""
    monkeypatch.delenv("SUPERDIALOG_LLM_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("SUPERDIALOG_LLM_FALLBACK_COOLDOWN_S", raising=False)
    _down_until.clear()
    yield
    _down_until.clear()


class _Leg:
    """Programmable fake leg: fails N times, then answers; streams chunks."""

    def __init__(
        self,
        name: str,
        *,
        fail_times: int = 0,
        error: Exception | None = None,
        chunks: tuple[str, ...] = ("ok",),
        fail_mid_stream: bool = False,
    ) -> None:
        self.model = name
        self.inner = f"inner-{name}"
        self._fail_times = fail_times
        self._error = error or RuntimeError(f"{name} down")
        self._chunks = chunks
        self._fail_mid_stream = fail_mid_stream
        self.complete_calls = 0
        self.stream_calls = 0
        self.warmed = 0

    async def complete(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> CompletionResult:
        self.complete_calls += 1
        if self.complete_calls <= self._fail_times:
            raise self._error
        return CompletionResult(text=self.model, tool_calls=[], metadata={})

    async def stream(
        self, messages: list[dict[str, Any]], tools: Any = None, **opts: Any
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        if self.stream_calls <= self._fail_times:
            raise self._error
        if not self._chunks:
            return  # a truly empty stream: no text, no done chunk
        for i, text in enumerate(self._chunks):
            yield StreamChunk(text=text, tool_call_delta=None, done=False)
            if self._fail_mid_stream and i == 0:
                raise self._error
        yield StreamChunk(text=None, tool_call_delta=None, done=True)

    async def warmup(self) -> None:
        self.warmed += 1


def _chain(*legs: _Leg, cooldown_s: float = 30.0) -> FallbackProvider:
    return FallbackProvider([(leg.model, leg) for leg in legs], cooldown_s)


async def _collect(provider: FallbackProvider) -> list[str]:
    return [c.text async for c in provider.stream(_MSGS) if c.text]


# ── complete: chain order + cooldown ─────────────────────


def test_failover_answers_from_next_leg_and_marks_down() -> None:
    primary = _Leg("primary", fail_times=99)
    backup = _Leg("backup")
    chain = _chain(primary, backup)
    result = anyio.run(chain.complete, _MSGS)
    assert result.text == "backup"
    assert "primary" in _down_until  # cooldown started
    assert "backup" not in _down_until


def test_cooldown_skips_dead_leg_next_turn() -> None:
    primary = _Leg("primary", fail_times=99)
    backup = _Leg("backup")
    chain = _chain(primary, backup)
    anyio.run(chain.complete, _MSGS)
    anyio.run(chain.complete, _MSGS)
    assert primary.complete_calls == 1  # discovered dead once, then skipped
    assert backup.complete_calls == 2


def test_cooldown_expiry_retries_primary_and_clears() -> None:
    primary = _Leg("primary", fail_times=1)  # dead once, then healthy
    backup = _Leg("backup")
    chain = _chain(primary, backup, cooldown_s=0.02)
    anyio.run(chain.complete, _MSGS)  # failover, primary on cooldown
    time.sleep(0.05)  # cooldown expires
    result = anyio.run(chain.complete, _MSGS)
    assert result.text == "primary"
    assert "primary" not in _down_until  # success cleared the cooldown


def test_all_down_still_tries_every_leg() -> None:
    primary = _Leg("primary")
    backup = _Leg("backup")
    chain = _chain(primary, backup)
    _down_until["primary"] = time.monotonic() + 999
    _down_until["backup"] = time.monotonic() + 999
    result = anyio.run(chain.complete, _MSGS)
    assert result.text == "primary"
    assert "primary" not in _down_until


def test_all_legs_failing_raises_resilience_error() -> None:
    chain = _chain(_Leg("a", fail_times=99), _Leg("b", fail_times=99))
    with pytest.raises(LLMResilienceError):
        anyio.run(chain.complete, _MSGS)


def test_auth_error_fails_over() -> None:
    # Non-retryable for same-leg retry, but chain legs are DIFFERENT
    # providers/keys — a revoked primary key is the fallback scenario.
    primary = _Leg("primary", fail_times=99, error=ValueError("invalid api key"))
    backup = _Leg("backup")
    result = anyio.run(_chain(primary, backup).complete, _MSGS)
    assert result.text == "backup"


# ── stream: pre-first-token vs mid-stream ────────────────


def test_stream_fails_over_before_first_token() -> None:
    primary = _Leg("primary", fail_times=99)
    backup = _Leg("backup", chunks=("b1", "b2"))
    chain = _chain(primary, backup)
    assert anyio.run(_collect, chain) == ["b1", "b2"]
    assert "primary" in _down_until


def test_stream_midstream_error_surfaces_never_restarts() -> None:
    primary = _Leg("primary", fail_mid_stream=True, chunks=("p1", "p2"))
    backup = _Leg("backup")
    chain = _chain(primary, backup)
    with pytest.raises(RuntimeError, match="primary down"):
        anyio.run(_collect, chain)
    assert backup.stream_calls == 0  # spoken turn is never restarted
    assert "primary" not in _down_until  # mid-stream is not a leg-down signal


def test_empty_stream_falls_to_next_leg_without_cooldown() -> None:
    primary = _Leg("primary", chunks=())
    backup = _Leg("backup", chunks=("b",))
    chain = _chain(primary, backup)
    assert anyio.run(_collect, chain) == ["b"]
    assert "primary" not in _down_until  # empty ≠ proven down


def test_stream_success_clears_cooldown_at_first_token() -> None:
    primary = _Leg("primary", chunks=("p",))
    backup = _Leg("backup")
    _down_until["primary"] = 0.0  # expired entry left from an earlier outage
    chain = _chain(primary, backup)
    anyio.run(_collect, chain)
    assert "primary" not in _down_until


# ── compat surface ───────────────────────────────────────


def test_model_attributes_the_leg_that_answered() -> None:
    primary = _Leg("primary", fail_times=99)
    backup = _Leg("backup")
    chain = _chain(primary, backup)
    assert chain.model == "primary"  # before any turn: leg 0
    anyio.run(chain.complete, _MSGS)
    assert chain.model == "backup"  # after failover: the answering leg


def test_inner_and_getattr_delegate_to_leg0() -> None:
    primary = _Leg("primary")
    chain = _chain(primary, _Leg("backup"))
    assert chain.inner == "inner-primary"
    assert chain.complete_calls == 0  # arbitrary attr -> leg 0


def test_warmup_warms_every_leg() -> None:
    primary, backup = _Leg("primary"), _Leg("backup")
    chain = _chain(primary, backup)
    anyio.run(chain.warmup)
    assert (primary.warmed, backup.warmed) == (1, 1)


# ── config + resolve_llm wiring ──────────────────────────


def test_fallback_config_from_env() -> None:
    cfg = FallbackConfig.from_env(
        {
            "SUPERDIALOG_LLM_FALLBACK_MODELS": " oai/gpt-4.1 , oai/gpt-4o ",
            "SUPERDIALOG_LLM_FALLBACK_COOLDOWN_S": "5",
        }
    )
    assert cfg.models == ("oai/gpt-4.1", "oai/gpt-4o")
    assert cfg.cooldown_s == 5.0
    assert FallbackConfig.from_env({}).models == ()


def test_resolve_llm_env_unset_is_plain_resilient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superdialog.llm.resilience import ResilientProvider
    from superdialog.llm.resolver import _backend_cache, resolve_llm

    monkeypatch.delenv("SUPERDIALOG_LLM_BACKEND", raising=False)
    _backend_cache.clear()
    p = resolve_llm("oai/gpt-4.1-mini")
    assert isinstance(p, ResilientProvider)
    assert not isinstance(p, FallbackProvider)


def test_resolve_llm_builds_chain_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from superdialog.llm.resilience import ResilientProvider
    from superdialog.llm.resolver import _backend_cache, resolve_backend, resolve_llm

    monkeypatch.delenv("SUPERDIALOG_LLM_BACKEND", raising=False)
    monkeypatch.setenv(
        "SUPERDIALOG_LLM_FALLBACK_MODELS",
        "oai/gpt-4.1-mini,oai/gpt-4.1",  # first entry == primary -> deduped
    )
    _backend_cache.clear()
    p = resolve_llm("oai/gpt-4.1-mini")
    assert isinstance(p, FallbackProvider)
    uris = [uri for uri, _leg in p._legs]
    assert uris == ["oai/gpt-4.1-mini", "oai/gpt-4.1"]
    leg0, leg1 = (leg for _uri, leg in p._legs)
    assert isinstance(leg0, ResilientProvider)
    assert leg0.cfg.max_retries >= 0  # today's exact wrap
    assert leg1.cfg.max_retries == 0 and leg1.cfg.hedge_enabled is False
    # .inner compat: the chain's inner is the primary's cached backend.
    assert p.inner is resolve_backend("oai/gpt-4.1-mini")
