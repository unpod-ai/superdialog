"""New scenario-eval metrics: evidence-gated success, recovery, consistency,
constraint adherence, and PII-leak checks.

Every metric returns superdialog.eval.metrics.base.MetricResult -- the same
shape the A/B harness's custom metrics use (superdialog/eval/metrics/custom.py)
-- so reports/aggregation code needs no adapter. These metrics score a
session's parsed event list (see events.py) and/or transcript directly,
NOT an EvalSample -- they run once per (case, mode), not per-sample.
"""

from __future__ import annotations

from typing import Any

from superdialog.eval.metrics.base import MetricResult

from .events import any_real_success


def evidence_gated_task_success(
    completed: bool, events: list[dict[str, Any]], *, tool_ids: set[str] | None = None
) -> MetricResult:
    """completed AND a real tool call succeeded -- never judge-only.

    A playbook reaching a terminal checkpoint (completed=True) is NOT proof a
    booking happened -- see the golfai S1 incident, where the agent said
    "payment link sent" with zero real hold/confirm calls. This is the hard
    gate from the design spec's "Success must be evidence-gated" section.
    """
    has_evidence = any_real_success(events, tool_ids=tool_ids)
    ok = completed and has_evidence
    if ok:
        reason = "completed with real tool-call evidence"
    elif not completed:
        reason = "session never reached a terminal checkpoint"
    else:
        reason = "completed, but no real tool-call evidence (possible fabrication)"
    return MetricResult(name="task_success", value=1.0 if ok else 0.0, passed=ok, reason=reason)


def pass_at_1(
    completed: bool, events: list[dict[str, Any]], *, tool_ids: set[str] | None = None
) -> MetricResult:
    """Evidence-gated success reached with ZERO checkpoint revisits.

    A revisit is the SAME to_checkpoint appearing twice in the advance-event
    sequence -- the state machine backtracked (a retry), so the task did not
    succeed on the first pass through the flow.
    """
    base = evidence_gated_task_success(completed, events, tool_ids=tool_ids)
    if base.value != 1.0:
        return MetricResult(name="pass_at_1", value=0.0, passed=False, reason=base.reason)
    visited: list[str] = [
        e["to_checkpoint"] for e in events if e.get("type") == "advance"
    ]
    if len(visited) != len(set(visited)):
        return MetricResult(
            name="pass_at_1",
            value=0.0,
            passed=False,
            reason="succeeded, but a checkpoint was revisited (retry/backtrack)",
        )
    return MetricResult(
        name="pass_at_1", value=1.0, passed=True, reason="succeeded on the first pass"
    )
