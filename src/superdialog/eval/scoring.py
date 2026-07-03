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


# Framework score: quality is a GATE, not a tradeable weight — a case scores 0
# unless every target below is met (and no guardrail fired); passing cases are
# then ranked purely by cost: lower latency and fewer input tokens win.
QUALITY_TARGETS: dict[str, float] = {
    "task_success": 1.0,
    "slot_accuracy": 1.0,
}
# ponytail: fixed normalizers; recalibrate from fleet p50s if rankings look off.
_LAT_NORM_MS = 2000.0
_TOK_NORM = 1500.0


def _metric_extra_mean(case: CaseResult, metric: str, key: str) -> float | None:
    vals = [
        float(r.extra[key])
        for r in case.metric_results.get(metric, [])
        if not r.errored
        and not r.skipped
        and isinstance((r.extra or {}).get(key), (int, float))
    ]
    return sum(vals) / len(vals) if vals else None


def case_framework(case: CaseResult) -> float:
    """Quality-gated cost score in (0, 1]; 0.0 when the quality gate fails.

    Gate: guardrail clean AND every QUALITY_TARGETS metric scored at target.
    Cost: 1 / ((1 + p95_latency/2s) * (1 + tokens_per_turn/1500)) — halves at
    each normalizer, so an instant zero-token run scores 1.0.
    """
    if case.guardrail_failed:
        return 0.0
    for metric, target in QUALITY_TARGETS.items():
        v = _metric_mean(case, metric)
        if v is None or v < target:
            return 0.0
    lat_p95 = _metric_extra_mean(case, "efficiency", "latency_p95") or 0.0
    tokens = _metric_extra_mean(case, "token_cost", "tokens_mean") or 0.0
    return 1.0 / ((1.0 + lat_p95 / _LAT_NORM_MS) * (1.0 + tokens / _TOK_NORM))
