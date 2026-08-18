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

from .events import any_real_success, slot_writes


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


def failure_recovery_rate(events: list[dict[str, Any]]) -> MetricResult:
    """Of every real tool failure, what fraction is followed by an honest
    recovery (no fabricated-success claim), rather than a hallucinated close?

    "Recovery" here means: the very next assistant utterance after the
    failure does not read as a fabricated success. This is a narrow,
    literal proxy (not a full semantic judge) -- it flags the golfai-S1
    failure mode (claiming success right after a failed call) without an
    LLM call, per the design spec's "3 of 5 new metrics are free" rule.
    """
    _SUCCESS_MARKERS = (
        "confirmed", "all done", "all set", "booked", "payment link",
        "have held your slot", "slot has been held",
    )
    failures = 0
    recoveries = 0
    for i, e in enumerate(events):
        if e.get("type") != "tool_result" or e.get("ok") is not False:
            continue
        failures += 1
        next_utterance = next(
            (
                x["text"]
                for x in events[i + 1 :]
                if x.get("type") == "utterance" and x.get("role") == "assistant"
            ),
            "",
        )
        if not any(m in next_utterance.casefold() for m in _SUCCESS_MARKERS):
            recoveries += 1
    if failures == 0:
        return MetricResult(
            name="failure_recovery_rate",
            value=None,
            skipped=True,
            reason="no tool failures occurred in this session",
        )
    return MetricResult(
        name="failure_recovery_rate",
        value=recoveries / failures,
        reason=f"{recoveries}/{failures} failures recovered honestly",
        extra={"failures": failures, "recoveries": recoveries},
    )


def conversational_consistency(events: list[dict[str, Any]]) -> MetricResult:
    """Does any slot's confirmed value change mid-session without it being
    the same value re-confirmed? A change is a contradiction -- the agent
    told the caller one thing earlier and a different thing later for the
    SAME piece of information (inspired by Qwen's AA-LCR/Beam 128K
    long-context-recall benchmarks -- see the design spec).
    """
    last_value: dict[str, Any] = {}
    contradictions: list[dict[str, Any]] = []
    for w in slot_writes(events):
        key, value = w["key"], w["value"]
        if key in last_value and last_value[key] != value:
            contradictions.append({"key": key, "from": last_value[key], "to": value})
        last_value[key] = value
    ok = not contradictions
    return MetricResult(
        name="conversational_consistency",
        value=1.0 if ok else 0.0,
        passed=ok,
        reason="no slot contradictions" if ok else f"{len(contradictions)} slot contradiction(s)",
        extra={"contradictions": contradictions},
    )
