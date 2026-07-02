"""Composite scoring with a guardrail hard-gate."""

from __future__ import annotations

from superdialog.eval.results import CaseResult

# Extends playbook/eval/scorer.py's 0.4/0.3/0.2/0.1 objective with a guardrail gate.
DEFAULT_WEIGHTS: dict[str, float] = {
    "task_success": 0.4,
    "slot_accuracy": 0.3,
    "faithfulness": 0.2,
    "efficiency": 0.1,
}
_EXPECTED_TURNS = 8.0  # ponytail: fixed normalizer; swap for min-max if it matters


def _efficiency_norm(turns: float) -> float:
    """Fewer turns -> closer to 1.0. 8 turns -> ~0.5."""
    return 1.0 / (1.0 + turns / _EXPECTED_TURNS)


def _metric_mean(case: CaseResult, metric: str) -> float | None:
    vals = [
        r.value
        for r in case.metric_results.get(metric, [])
        if r.value is not None and not r.errored and not r.skipped
    ]
    return sum(vals) / len(vals) if vals else None


def case_composite(case: CaseResult, weights: dict[str, float]) -> float:
    """Weighted score over available metrics; 0.0 if any guardrail was violated."""
    if case.guardrail_failed:
        return 0.0
    acc = 0.0
    total_w = 0.0
    for metric, w in weights.items():
        v = _metric_mean(case, metric)
        if v is None:
            continue
        if metric == "efficiency":
            v = _efficiency_norm(v)
        acc += w * v
        total_w += w
    return acc / total_w if total_w else 0.0
