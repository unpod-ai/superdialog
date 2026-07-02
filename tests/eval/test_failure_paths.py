"""Aggregation is honest: errored/skipped are counted, never folded into the mean."""

from __future__ import annotations

from superdialog.eval.metrics.base import MetricResult
from superdialog.eval.results import CaseResult
from superdialog.eval.runner import _aggregate_mode
from superdialog.eval.scoring import DEFAULT_WEIGHTS


def _case(metrics: dict[str, list[MetricResult]]) -> CaseResult:
    return CaseResult(
        case_id="c",
        mode="m",
        metric_results=metrics,
        guardrail_failed=False,
        turns=0,
    )


async def test_aggregate_excludes_errored_and_skipped_from_mean() -> None:
    """Mean uses only scored values; errored/skipped are counted separately."""
    case_results = [
        _case(
            {
                "quality": [
                    MetricResult(name="quality", value=1.0),
                    MetricResult(name="quality", value=None, errored=True),
                ]
            }
        ),
        _case(
            {
                "quality": [
                    MetricResult(name="quality", value=3.0),
                    MetricResult(name="quality", value=None, skipped=True),
                ],
                "dead": [MetricResult(name="dead", value=None, errored=True)],
            }
        ),
    ]

    mode = _aggregate_mode("m", case_results, DEFAULT_WEIGHTS)

    quality = mode.aggregates["quality"]
    # mean of only [1.0, 3.0] == 2.0 — NOT (1+3)/4 with errored/skipped as 0.
    assert quality.mean == 2.0
    assert quality.n == 2
    assert quality.errored == 1
    assert quality.skipped == 1

    # A metric where every result errored -> mean is None (never a fake 0.0).
    dead = mode.aggregates["dead"]
    assert dead.mean is None
    assert dead.n == 0
    assert dead.errored == 1
    assert dead.skipped == 0
