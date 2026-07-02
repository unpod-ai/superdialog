"""case_composite: guardrail hard-gate, efficiency normalization, honest empties."""

from __future__ import annotations

from superdialog.eval.metrics.base import MetricResult
from superdialog.eval.results import CaseResult
from superdialog.eval.scoring import DEFAULT_WEIGHTS, case_composite


def _case(
    metrics: dict[str, list[MetricResult]], *, guardrail_failed: bool
) -> CaseResult:
    return CaseResult(
        case_id="c",
        mode="m",
        metric_results=metrics,
        guardrail_failed=guardrail_failed,
        turns=0,
    )


def test_guardrail_failure_hard_gates_to_zero() -> None:
    """A perfect task_success is wiped to 0.0 when a guardrail was violated."""
    case = _case(
        {"task_success": [MetricResult(name="task_success", value=1.0)]},
        guardrail_failed=True,
    )
    assert case_composite(case, DEFAULT_WEIGHTS) == 0.0


def test_clean_case_scores_positive() -> None:
    """No guardrail failure: task_success 1.0 + 8-turn efficiency -> (0, 1]."""
    case = _case(
        {
            "task_success": [MetricResult(name="task_success", value=1.0)],
            "efficiency": [MetricResult(name="efficiency", value=8.0)],
        },
        guardrail_failed=False,
    )
    composite = case_composite(case, DEFAULT_WEIGHTS)
    assert 0.0 < composite <= 1.0


def test_only_errored_metrics_scores_zero() -> None:
    """All metrics errored -> total_w is 0 -> composite 0.0 (not a fake mean)."""
    case = _case(
        {
            "task_success": [
                MetricResult(name="task_success", value=None, errored=True),
            ],
        },
        guardrail_failed=False,
    )
    assert case_composite(case, DEFAULT_WEIGHTS) == 0.0
