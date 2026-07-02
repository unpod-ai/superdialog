"""Task 4.5: EfficiencyMetric percentiles (pure code, never errors)."""

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.metrics.custom import EfficiencyMetric, _percentile


async def test_efficiency_percentiles_and_turns():
    s = EvalSample(
        kind="conversation",
        user_input=[],
        metadata={"latencies_ms": [100, 200, 300, 400, 500], "turns": 3},
    )
    r = await EfficiencyMetric().score(s)
    assert r.value == 3.0 and not r.errored
    assert r.extra["turns"] == 3
    assert r.extra["latency_p50"] == 300.0
    assert r.extra["latency_p95"] == 480.0


async def test_efficiency_empty_latencies_are_zero():
    s = EvalSample(kind="conversation", user_input=[], metadata={})
    r = await EfficiencyMetric().score(s)
    assert r.value == 0.0 and not r.errored
    assert r.extra["latency_p50"] == 0.0
    assert r.extra["latency_p95"] == 0.0


def test_percentile_helper_empty_list():
    assert _percentile([], 95.0) == 0.0
