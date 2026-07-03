"""Result models for A/B eval runs (dataclasses; JSON via dataclasses.asdict)."""

from __future__ import annotations

from dataclasses import dataclass, field

from superdialog.eval.metrics.base import MetricResult


@dataclass
class CaseResult:
    """Scored outcome of one (case, mode) run."""

    case_id: str
    mode: str
    metric_results: dict[str, list[MetricResult]]  # metric name -> per-sample results
    guardrail_failed: bool
    turns: int
    degraded: bool = False  # not yet surfaced: endpoint.turn() returns only text
    # ponytail: wire when the endpoint exposes turn metadata


@dataclass
class MetricAggregate:
    """Aggregate of one metric across a mode's cases/repeats."""

    metric: str
    mean: float | None  # None when nothing scored (all errored/skipped)
    std: float
    n: int  # scored values counted into mean
    errored: int
    skipped: int
    # Mean per numeric MetricResult.extra key (e.g. latency_p95, tokens_mean).
    extras: dict[str, float] = field(default_factory=dict)


@dataclass
class ModeResult:
    """One mode's aggregated scores plus its per-case detail."""

    mode: str
    aggregates: dict[str, MetricAggregate]
    composite_mean: float
    guardrail_violation_rate: float
    case_results: list[CaseResult] = field(default_factory=list)
    # Quality-gated framework score: 0 unless quality targets met, then
    # rewards low latency + low tokens/turn (scoring.case_framework).
    framework_mean: float = 0.0


@dataclass
class RunResult:
    """Top-level A/B outcome: one ModeResult per mode compared."""

    dataset: str
    metrics: list[str]
    modes: list[ModeResult]
