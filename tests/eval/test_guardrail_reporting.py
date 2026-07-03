"""Guardrail honesty: 0% on zero scored probes must read 'not tested'."""

from superdialog.eval.metrics.base import MetricResult
from superdialog.eval.report import render_markdown
from superdialog.eval.results import CaseResult, RunResult
from superdialog.eval.runner import _aggregate_mode
from superdialog.eval.scoring import DEFAULT_WEIGHTS


def _case(metric_results: dict, guardrail_failed: bool = False) -> CaseResult:
    return CaseResult(
        case_id="c1",
        mode="playbook",
        metric_results=metric_results,
        guardrail_failed=guardrail_failed,
        turns=2,
    )


def test_rate_is_none_when_no_guardrail_scored():
    cr = _case({"task_success": [MetricResult(name="task_success", value=1.0)]})
    mode = _aggregate_mode("playbook", [cr], DEFAULT_WEIGHTS)
    assert mode.guardrail_violation_rate is None


def test_rate_is_zero_when_probes_scored_and_clean():
    cr = _case({"guardrail": [MetricResult(name="guardrail", value=1.0, passed=True)]})
    mode = _aggregate_mode("playbook", [cr], DEFAULT_WEIGHTS)
    assert mode.guardrail_violation_rate == 0.0


def test_rate_counts_violations_when_scored():
    cr = _case(
        {"guardrail": [MetricResult(name="guardrail", value=0.0, passed=False)]},
        guardrail_failed=True,
    )
    mode = _aggregate_mode("playbook", [cr], DEFAULT_WEIGHTS)
    assert mode.guardrail_violation_rate == 1.0


def test_report_renders_not_tested_for_none_rate():
    cr = _case({"task_success": [MetricResult(name="task_success", value=1.0)]})
    mode = _aggregate_mode("playbook", [cr], DEFAULT_WEIGHTS)
    md = render_markdown(RunResult(dataset="d", metrics=["task_success"], modes=[mode]))
    assert "not tested" in md
    assert "0.0%" not in md
