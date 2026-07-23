"""warmup() is optional, fire-and-forget, and warms both hedge legs.

Offline: fake providers/clients; no network. The 5s fail_after bound is a
structural guard read in source — not timed here (a real stall test would sleep
5s); these prove the never-raise contract and the both-legs behaviour instead.
"""

from __future__ import annotations

import anyio

from superdialog.llm.litellm_provider import LitellmProvider
from superdialog.llm.openai_provider import OpenAIProvider
from superdialog.llm.resilience import ResilienceConfig, ResilientProvider


class _Warmable:
    model = "fake/model"

    def __init__(self, boom: bool = False) -> None:
        self.warmed = 0
        self._boom = boom

    async def warmup(self) -> None:
        self.warmed += 1
        if self._boom:
            raise RuntimeError("warmup exploded")


class _NoWarmup:
    model = "fake/nowarm"


def test_resilient_provider_warms_inner_and_hedge():
    inner, hedge = _Warmable(), _Warmable()
    rp = ResilientProvider(inner, ResilienceConfig(), hedge)
    anyio.run(rp.warmup)
    assert inner.warmed == 1 and hedge.warmed == 1


def test_resilient_provider_warmup_tolerates_a_failing_leg():
    inner, hedge = _Warmable(boom=True), _Warmable()
    rp = ResilientProvider(inner, ResilienceConfig(), hedge)
    anyio.run(rp.warmup)  # must not raise
    assert inner.warmed == 1 and hedge.warmed == 1  # other leg still warmed


def test_resilient_provider_warmup_noop_without_backend_warmup():
    rp = ResilientProvider(_NoWarmup(), ResilienceConfig(), None)
    anyio.run(rp.warmup)  # no hedge, backend has no warmup -> clean no-op


def test_litellm_warmup_is_noop():
    anyio.run(LitellmProvider(model="openai/gpt-4.1-mini").warmup)


def test_openai_warmup_swallows_client_error(monkeypatch):
    p = OpenAIProvider(model="gpt-4.1-mini")

    class _Boom:
        class models:
            @staticmethod
            async def list():
                raise RuntimeError("gateway 404")

    monkeypatch.setattr(p, "_ensure_client", lambda: _Boom())
    anyio.run(p.warmup)  # must not raise


def test_openai_warmup_calls_models_list(monkeypatch):
    called: list[bool] = []

    class _OK:
        class models:
            @staticmethod
            async def list():
                called.append(True)
                return object()

    p = OpenAIProvider(model="gpt-4.1-mini")
    monkeypatch.setattr(p, "_ensure_client", lambda: _OK())
    anyio.run(p.warmup)
    assert called == [True]
