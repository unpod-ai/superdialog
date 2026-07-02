"""Determinism: two identical runs agree on all non-latency results."""

import superdialog.eval.endpoints.in_process as ip
from superdialog.eval.metrics.registry import build_suite
from superdialog.eval.runner import run_ab

from tests.eval.test_e2e_golden import (
    _ConstUser,
    _FakeMachine,
    _dataset,
    _factories,
    _judge,
)


def _snapshot(result):
    return {
        m.mode: {
            "composite": round(m.composite_mean, 6),
            "guardrail_rate": m.guardrail_violation_rate,
            "aggs": {
                k: (v.mean, v.n, v.errored, v.skipped)
                for k, v in m.aggregates.items()
                if k != "efficiency"  # efficiency carries wall-clock latency
            },
        }
        for m in result.modes
    }


async def _run():
    metrics = ["task_success", "slot_accuracy", "guardrail", "efficiency"]
    return await run_ab(
        _dataset(),
        modes=["vanilla", "playbook"],
        endpoint_factories=_factories(),
        suite=build_suite(metrics, _judge(), "fake"),
        user_llm=_ConstUser("hi"),
        metric_names=metrics,
        repeats=1,
    )


async def test_two_runs_agree_on_non_latency(monkeypatch):
    monkeypatch.setattr(ip, "DialogMachine", _FakeMachine)
    r1 = await _run()
    r2 = await _run()
    assert _snapshot(r1) == _snapshot(r2)
