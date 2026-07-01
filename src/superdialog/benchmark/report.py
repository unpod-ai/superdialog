"""Aggregate per-sample scores into a report the panel can render.

A ``ModeReport`` is one column of the benchmark panel — e.g. "With SuperDialog"
or "Raw LLM". It holds every sample's scores plus the aggregate means for the
4 deterministic metrics and the 7 RAGAS metrics (per judge).

Scorer-first: ``build_report`` runs the deterministic metrics always (free,
offline) and the RAGAS metrics only when ``run_ragas=True`` (costs judge-LLM
calls). So the pipeline can be validated on golden data at zero LLM cost, then
the same call flipped to spend on RAGAS when ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .deterministic import DeterministicScores, score_deterministic
from .loader import BenchmarkSample
from .ragas_scorer import DEFAULT_JUDGES, METRIC_KEYS, RagasScores, score_ragas

DET_KEYS = ("completion", "data_capture", "smoothness", "repairs")


@dataclass
class SampleResult:
    id: str
    scenario_type: str
    difficulty: str
    deterministic: DeterministicScores
    ragas: dict[str, RagasScores] = field(default_factory=dict)  # {judge: scores}


@dataclass
class ModeReport:
    label: str                       # panel column header, e.g. "With SuperDialog"
    per_sample: list[SampleResult]
    det_mean: dict[str, float]       # {metric: mean}
    ragas_mean: dict[str, dict[str, float]]  # {judge: {metric: mean}}
    # Total system-under-test LLM cost (USD) for this mode's run, priced via
    # litellm. 0.0 in scorer-first (golden data — no SUT run yet); the runner
    # sets it once it actually runs the model. Judge/eval cost is NOT included.
    cost_usd: float = 0.0

    @property
    def n(self) -> int:
        return len(self.per_sample)


def _mean(values: list[float]) -> float | None:
    # None (not 0.0) when nothing scored -> deferred metrics render as "—", not "0%"
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def build_report(
    label: str,
    samples: list[BenchmarkSample],
    *,
    run_ragas: bool = False,
    judges: tuple[str, ...] = DEFAULT_JUDGES,
) -> ModeReport:
    """Score every sample and aggregate into a ModeReport.

    ``run_ragas=False`` (default) scores only the deterministic metrics — no LLM
    cost. Set ``run_ragas=True`` to also spend on the RAGAS judges.
    """
    results: list[SampleResult] = []
    for s in samples:
        det = score_deterministic(s)
        ragas: dict[str, RagasScores] = {}
        if run_ragas:
            ragas = score_ragas(s, judges)
        results.append(
            SampleResult(
                id=s.id,
                scenario_type=s.scenario_type,
                difficulty=s.difficulty,
                deterministic=det,
                ragas=ragas,
            )
        )

    det_mean = {
        k: _mean([getattr(r.deterministic, k) for r in results]) for k in DET_KEYS
    }

    ragas_mean: dict[str, dict[str, float]] = {}
    if run_ragas:
        for judge in judges:
            ragas_mean[judge] = {
                mk: _mean(
                    [
                        getattr(r.ragas[judge], mk)
                        for r in results
                        if judge in r.ragas
                    ]
                )
                for mk in METRIC_KEYS
            }

    return ModeReport(
        label=label,
        per_sample=results,
        det_mean=det_mean,
        ragas_mean=ragas_mean,
    )


__all__ = ["SampleResult", "ModeReport", "build_report", "DET_KEYS"]
