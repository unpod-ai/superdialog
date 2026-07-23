"""Session-transparent LLM fallback chain.

``FallbackProvider`` tries an ordered list of ``ResilientProvider``-wrapped
legs. A leg that fails is put on a cooldown shared process-wide (module-level
``_down_until`` keyed by URI — providers are per-session, so instance state
would rediscover a dead primary once per session instead of once per cooldown
per process). Failover fires on ANY exception before the first token: unlike
per-request retry, chain legs are different providers/keys, so a revoked
primary key is precisely the fallback scenario. Mid-stream failures surface
unchanged (same contract as ``ResilientProvider.stream``) — a partially-spoken
turn is never silently restarted.

Recovery is probe-free: a dead leg is skipped for ``cooldown_s``, then the
next real turn retries it. When every leg is on cooldown, all legs are tried
anyway (an outage must not make the chain refuse to even attempt).
"""

# ponytail: recovery = try-on-next-turn; add a cheap probe only if the
# cooldown-expiry turn's latency shows in p99.

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

import anyio

from .provider import CompletionResult, LLMProvider, StreamChunk
from .resilience import LLMResilienceError, _aclose

_log = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_S = 30.0

# uri -> monotonic time until which the leg is skipped. Plain dict write under
# asyncio (single thread) — no lock needed.
_down_until: dict[str, float] = {}


@dataclass(frozen=True)
class FallbackConfig:
    """Env knobs for the fallback chain (off unless models are configured)."""

    models: tuple[str, ...] = ()
    cooldown_s: float = _DEFAULT_COOLDOWN_S

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "FallbackConfig":
        e = env if env is not None else os.environ
        raw = e.get("SUPERDIALOG_LLM_FALLBACK_MODELS") or ""
        models = tuple(m.strip() for m in raw.split(",") if m.strip())
        v = e.get("SUPERDIALOG_LLM_FALLBACK_COOLDOWN_S")
        cooldown = float(v) if v not in (None, "") else _DEFAULT_COOLDOWN_S
        return cls(models=models, cooldown_s=cooldown)


class FallbackProvider:
    """Ordered chain of ``(uri, provider)`` legs with per-URI cooldown.

    Leg 0 is the primary (today's exact retry+hedge wrap); legs 1..n are
    cheap wraps (no retry, no hedge) so worst-case attempts stay bounded at
    ``(max_retries + 1) + n_fallbacks``.
    """

    def __init__(
        self,
        legs: list[tuple[str, LLMProvider]],
        cooldown_s: float = _DEFAULT_COOLDOWN_S,
    ) -> None:
        if not legs:
            raise ValueError("FallbackProvider needs at least one leg")
        self._legs = legs
        self._cooldown_s = cooldown_s
        self._last_model: str | None = None

    # -- compat surface (the chain must look like the wrap it replaced) ------

    @property
    def inner(self) -> Any:
        """The primary's raw backend, matching ``resolve_llm(uri).inner``."""
        return self._legs[0][1].inner  # type: ignore[attr-defined]

    @property
    def model(self) -> str:
        """The model that actually answered last (billing attribution after a
        failover), falling back to the primary's model before any turn."""
        return self._last_model or getattr(self._legs[0][1], "model", "") or ""

    def __getattr__(self, name: str) -> Any:
        legs = self.__dict__.get("_legs")
        if not legs:
            raise AttributeError(name)
        return getattr(legs[0][1], name)

    # -- chain mechanics -----------------------------------------------------

    def _live_legs(self) -> list[tuple[str, LLMProvider]]:
        now = time.monotonic()
        live = [(u, p) for u, p in self._legs if _down_until.get(u, 0.0) <= now]
        return live or list(self._legs)  # all down -> try everything anyway

    def _mark_down(self, uri: str, exc: BaseException) -> None:
        _down_until[uri] = time.monotonic() + self._cooldown_s
        _log.warning(
            "llm_fallback leg_down uri=%s cooldown_s=%s error=%s: %s",
            uri,
            self._cooldown_s,
            type(exc).__name__,
            exc,
        )

    def _mark_up(self, uri: str, provider: LLMProvider) -> None:
        _down_until.pop(uri, None)
        self._last_model = getattr(provider, "model", None)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        errors: list[BaseException] = []
        for uri, leg in self._live_legs():
            try:
                result = await leg.complete(messages, tools=tools, **opts)
            except Exception as exc:
                errors.append(exc)
                self._mark_down(uri, exc)
                continue
            self._mark_up(uri, leg)
            return result
        raise LLMResilienceError(
            f"all {len(self._legs)} fallback leg(s) failed"
        ) from (errors[-1] if errors else None)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        errors: list[BaseException] = []
        for uri, leg in self._live_legs():
            agen = leg.stream(messages, tools=tools, **opts)
            emitted = False
            try:
                async for chunk in agen:
                    if not emitted:
                        emitted = True
                        # The leg answered: clear its cooldown + attribute the
                        # turn to it before the host consumes the first token.
                        self._mark_up(uri, leg)
                    yield chunk
                if emitted:
                    return
                # An empty stream is a per-leg outcome, not an answer — treat
                # like a pre-first-token failure and let the next leg try.
                errors.append(LLMResilienceError(f"empty stream from {uri}"))
            except Exception as exc:
                if emitted:
                    raise  # mid-stream: surface, never restart a spoken turn
                errors.append(exc)
                self._mark_down(uri, exc)
            finally:
                await _aclose(agen)
        raise LLMResilienceError(
            f"all {len(self._legs)} fallback leg(s) failed before first token"
        ) from (errors[-1] if errors else None)

    async def warmup(self) -> None:
        """Warm every leg concurrently (each leg guards its own hedge/errors),
        so failover never pays a cold connect at the worst possible moment."""

        async def _warm(provider: LLMProvider) -> None:
            fn = getattr(provider, "warmup", None)
            if fn is None:
                return
            try:
                await fn()
            except Exception:  # noqa: BLE001 — warmup must never raise
                pass

        async with anyio.create_task_group() as tg:
            for _uri, leg in self._legs:
                tg.start_soon(_warm, leg)


__all__ = ["FallbackConfig", "FallbackProvider"]
