"""Framework-agnostic metric seam. RAGAS/deepeval/custom all implement Metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from superdialog.eval.dataset.models import EvalSample


@dataclass
class MetricResult:
    name: str
    value: float | None  # None => errored/unscored (never silently 0)
    passed: bool | None = None  # for gate metrics (guardrail)
    reason: str = ""
    errored: bool = False
    skipped: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    name: str

    async def score(self, sample: EvalSample) -> MetricResult: ...

    # Optional attr, NOT part of the runtime-checkable contract: a metric may
    # expose ``applies_to: tuple[str, ...]`` to restrict which sample kinds it
    # scores. MetricSuite reads it via getattr with a permissive default, so a
    # metric without it (like a simple const metric) still counts as a Metric.


class MetricSuite:
    """A mixed list of metrics (RAGAS + custom), scored per sample."""

    def __init__(self, metrics: list[Any]) -> None:
        self._metrics = metrics

    async def score(self, sample: EvalSample) -> list[MetricResult]:
        results: list[MetricResult] = []
        for m in self._metrics:
            applies = getattr(m, "applies_to", ("conversation", "probe"))
            if sample.kind not in applies:
                continue
            try:
                results.append(await m.score(sample))
            except Exception as exc:  # judge crash => errored, NOT a 0 score
                results.append(
                    MetricResult(
                        name=m.name,
                        value=None,
                        errored=True,
                        reason=f"metric error: {exc}",
                    )
                )
        return results
