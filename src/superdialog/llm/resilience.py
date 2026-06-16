"""Per-request resilience for any :class:`LLMProvider` (D4).

Timeout, bounded retry-with-backoff, and an optional cross-provider hedge are
applied *once* by wrapping any backend in :class:`ResilientProvider`. Because
``resolve_llm`` returns the wrapped provider, every engine path (flow
``toolcall``/``llm`` adapters, playbook Director/Talker, flow edge-evaluation)
inherits the policy without per-call-site code.

The wrapper is transparent on the happy path: a single attempt that returns
before the timeout behaves exactly like calling the bare backend. Retries fire
only on timeouts and transient failures; hedging is off unless configured.

Configuration is read from the environment (``ResilienceConfig.from_env``):

* ``SUPERDIALOG_LLM_TIMEOUT_S``   — per-request timeout seconds (``0``/``none`` disables).
* ``SUPERDIALOG_LLM_MAX_RETRIES`` — extra attempts after the first.
* ``SUPERDIALOG_LLM_BACKOFF_BASE_S`` / ``SUPERDIALOG_LLM_BACKOFF_MAX_S``.
* ``SUPERDIALOG_LLM_HEDGE`` (bool) + ``SUPERDIALOG_LLM_HEDGE_MODEL`` (URI)
  + ``SUPERDIALOG_LLM_HEDGE_DELAY_S``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

import anyio

from .provider import CompletionResult, LLMProvider, StreamChunk


class LLMResilienceError(RuntimeError):
    """Raised when all retries/hedge legs are exhausted without a result."""


_TRANSIENT_TYPE_MARKERS = (
    "timeout",
    "connection",
    "ratelimit",
    "serviceunavailable",
    "apiconnection",
    "internalservererror",
    "overloaded",
    "apitimeout",
)
_TRANSIENT_MSG_MARKERS = (
    "timeout",
    "timed out",
    "temporarily",
    "overloaded",
    "rate limit",
    "too many requests",
    "service unavailable",
    "connection reset",
    "connection error",
    "bad gateway",
    "gateway timeout",
)
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    """True for timeouts and common transient provider failures.

    Backend-agnostic: matches our own timeout, transient HTTP status codes, and
    transient markers in the exception type name or message. Non-transient
    errors (auth, bad request) return False so they fail fast.
    """
    if isinstance(exc, TimeoutError):
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()
    if any(marker in name for marker in _TRANSIENT_TYPE_MARKERS):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MSG_MARKERS)


@dataclass(frozen=True)
class ResilienceConfig:
    """Knobs for :class:`ResilientProvider`. Defaults are a safety net (a
    generous timeout that bounds infinite hangs) — latency-sensitive voice
    deployments tune the timeout down and/or enable hedging."""

    timeout_s: float | None = 60.0
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    hedge_enabled: bool = False
    hedge_delay_s: float = 2.0
    hedge_model: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ResilienceConfig":
        """Build a config from environment variables, falling back to defaults."""
        e = env if env is not None else os.environ

        def _float(key: str, default: float) -> float:
            v = e.get(key)
            return float(v) if v not in (None, "") else default

        def _int(key: str, default: int) -> int:
            v = e.get(key)
            return int(v) if v not in (None, "") else default

        def _bool(key: str, default: bool) -> bool:
            v = e.get(key)
            if v in (None, ""):
                return default
            return v.strip().lower() in ("1", "true", "yes", "on")

        def _timeout(key: str, default: float | None) -> float | None:
            v = e.get(key)
            if v in (None, ""):
                return default
            v = v.strip().lower()
            return None if v in ("0", "none", "off") else float(v)

        return cls(
            timeout_s=_timeout("SUPERDIALOG_LLM_TIMEOUT_S", cls.timeout_s),
            max_retries=_int("SUPERDIALOG_LLM_MAX_RETRIES", cls.max_retries),
            backoff_base_s=_float("SUPERDIALOG_LLM_BACKOFF_BASE_S", cls.backoff_base_s),
            backoff_max_s=_float("SUPERDIALOG_LLM_BACKOFF_MAX_S", cls.backoff_max_s),
            hedge_enabled=_bool("SUPERDIALOG_LLM_HEDGE", cls.hedge_enabled),
            hedge_delay_s=_float("SUPERDIALOG_LLM_HEDGE_DELAY_S", cls.hedge_delay_s),
            hedge_model=e.get("SUPERDIALOG_LLM_HEDGE_MODEL") or cls.hedge_model,
        )


async def _aclose(agen: Any) -> None:
    """Best-effort close of an async generator on retry/exit."""
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        pass


class ResilientProvider:
    """Wrap an :class:`LLMProvider` with timeout + retry + optional hedge."""

    def __init__(
        self,
        inner: LLMProvider,
        config: ResilienceConfig | None = None,
        hedge: LLMProvider | None = None,
    ) -> None:
        self.inner = inner
        self.cfg = config or ResilienceConfig()
        self._hedge = hedge

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attributes (e.g. ``model``) to the wrapped backend so
        # the wrapper is transparent for logging/introspection.
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def _backoff_delay(self, attempt: int) -> float:
        return min(self.cfg.backoff_base_s * (2**attempt), self.cfg.backoff_max_s)

    async def _with_timeout(self, factory: Callable[[], Awaitable[Any]]) -> Any:
        """Await ``factory()`` under the configured per-request timeout."""
        if self.cfg.timeout_s is None:
            return await factory()
        with anyio.fail_after(self.cfg.timeout_s):
            return await factory()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        """One completion with timeout + retry (and hedge when configured)."""
        if self.cfg.hedge_enabled and self._hedge is not None:
            return await self._complete_hedged(messages, tools, opts)
        return await self._complete_with_retry(self.inner, messages, tools, opts)

    async def _complete_with_retry(
        self,
        provider: LLMProvider,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        opts: dict[str, Any],
    ) -> CompletionResult:
        last: BaseException | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return await self._with_timeout(
                    lambda: provider.complete(messages, tools=tools, **opts)
                )
            except Exception as exc:
                last = exc
                if not _is_retryable(exc):
                    raise  # fail fast, transparent — not a transient condition
                if attempt == self.cfg.max_retries:
                    break
                await anyio.sleep(self._backoff_delay(attempt))
        raise LLMResilienceError(
            f"LLM complete failed after {self.cfg.max_retries + 1} attempt(s)"
        ) from last

    async def _complete_hedged(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        opts: dict[str, Any],
    ) -> CompletionResult:
        """Race the primary against a delayed alternate; first success wins."""
        hedge = self._hedge
        assert hedge is not None
        legs: list[tuple[float, Callable[[], Awaitable[CompletionResult]]]] = [
            (
                0.0,
                lambda: self._with_timeout(
                    lambda: self.inner.complete(messages, tools=tools, **opts)
                ),
            ),
            (
                self.cfg.hedge_delay_s,
                lambda: self._with_timeout(
                    lambda: hedge.complete(messages, tools=tools, **opts)
                ),
            ),
        ]
        return await self._race(legs)

    async def _race(
        self, legs: list[tuple[float, Callable[[], Awaitable[Any]]]]
    ) -> Any:
        """Run legs concurrently (each after its delay); return the first
        success and cancel the rest. Raise if every leg fails."""
        outcome: dict[str, Any] = {}
        errors: list[BaseException] = []

        async def _run(delay: float, factory: Callable[[], Awaitable[Any]]) -> None:
            if delay:
                await anyio.sleep(delay)
            if "value" in outcome:
                return
            try:
                value = await factory()
            except Exception as exc:  # cancellation (BaseException) propagates
                errors.append(exc)
                return
            if "value" not in outcome:
                outcome["value"] = value
                tg.cancel_scope.cancel()

        async with anyio.create_task_group() as tg:
            for delay, factory in legs:
                tg.start_soon(_run, delay, factory)

        if "value" in outcome:
            return outcome["value"]
        raise LLMResilienceError("all hedge legs failed") from (
            errors[-1] if errors else None
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks, bounding the wait for each chunk by the timeout.

        Retries only while nothing has been emitted yet (re-issuing the request
        is safe before the first token). Once a chunk is yielded, a stall is
        surfaced rather than silently restarting a partially-spoken turn.
        """
        attempt = 0
        while True:
            agen = self.inner.stream(messages, tools=tools, **opts)
            emitted = False
            try:
                while True:
                    try:
                        chunk = await self._anext_with_timeout(agen)
                    except StopAsyncIteration:
                        return
                    except Exception as exc:
                        if emitted or not _is_retryable(exc):
                            raise
                        if attempt >= self.cfg.max_retries:
                            raise LLMResilienceError(
                                "LLM stream failed before first token after retries"
                            ) from exc
                        break  # nothing emitted yet -> retry the whole stream
                    emitted = True
                    yield chunk
            finally:
                await _aclose(agen)
            attempt += 1
            await anyio.sleep(self._backoff_delay(attempt - 1))

    async def _anext_with_timeout(self, agen: Any) -> StreamChunk:
        if self.cfg.timeout_s is None:
            return await agen.__anext__()
        with anyio.fail_after(self.cfg.timeout_s):
            return await agen.__anext__()


__all__ = ["LLMResilienceError", "ResilienceConfig", "ResilientProvider"]
