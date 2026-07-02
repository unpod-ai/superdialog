"""Report renderer: JSON round-trip + Markdown headline/delta/guardrail."""

from __future__ import annotations

import json

from superdialog.eval.metrics.base import MetricResult
from superdialog.eval.report import write_report
from superdialog.eval.results import (
    CaseResult,
    MetricAggregate,
    ModeResult,
    RunResult,
)


def _mode(name: str, mean: float, guardrail_failed: bool) -> ModeResult:
    """Build a one-metric, one-case ModeResult for ``task_success``."""
    return ModeResult(
        mode=name,
        aggregates={
            "task_success": MetricAggregate(
                metric="task_success", mean=mean, std=0.0, n=1, errored=0, skipped=0
            )
        },
        composite_mean=mean,
        guardrail_violation_rate=1.0 if guardrail_failed else 0.0,
        case_results=[
            CaseResult(
                case_id="c1",
                mode=name,
                metric_results={
                    "task_success": [
                        MetricResult(name="task_success", value=mean, passed=True)
                    ]
                },
                guardrail_failed=guardrail_failed,
                turns=3,
            )
        ],
    )


def test_write_report_json_and_markdown(tmp_path):
    """write_report emits both files; JSON round-trips; MD carries the headline."""
    run = RunResult(
        dataset="demo",
        metrics=["task_success"],
        modes=[
            _mode("vanilla", 0.5, guardrail_failed=True),
            _mode("playbook", 0.9, guardrail_failed=False),
        ],
    )

    json_path, md_path = write_report(run, str(tmp_path))

    assert json_path.endswith("report.json")
    assert md_path.endswith("report.md")

    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["modes"][0]["mode"] == "vanilla"

    md = open(md_path, encoding="utf-8").read()
    assert "task_success" in md
    assert "Δ (playbook−vanilla)" in md
    assert "+0.400" in md
    assert "## Composite & guardrails" in md
    assert "GUARDRAIL FAIL" in md
