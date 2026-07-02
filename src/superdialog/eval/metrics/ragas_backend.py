"""Adapter: any RAGAS metric -> our Metric protocol. RAGAS is one provider."""

from __future__ import annotations

from typing import Any

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.dataset.ragas_io import to_ragas_sample
from superdialog.eval.metrics.base import MetricResult


class RagasMetric:
    """Wrap a configured RAGAS metric instance behind the Metric protocol."""

    def __init__(
        self,
        ragas_metric: Any,
        *,
        applies_to: tuple[str, ...] = ("conversation", "probe"),
        name: str | None = None,
    ) -> None:
        self._m = ragas_metric
        self.applies_to = applies_to
        self.name = name or getattr(ragas_metric, "name", "ragas")

    async def score(self, sample: EvalSample) -> MetricResult:
        rs = to_ragas_sample(sample)
        try:
            if sample.kind == "conversation" and hasattr(self._m, "multi_turn_ascore"):
                val = await self._m.multi_turn_ascore(rs)
            else:
                val = await self._m.single_turn_ascore(rs)
            return MetricResult(name=self.name, value=float(val))
        except Exception as exc:
            return MetricResult(
                name=self.name,
                value=None,
                errored=True,
                reason=f"ragas error: {exc}",
            )


def build_ragas_judge(judge_model: str) -> Any:
    """Wrap superdialog's resolver as a RAGAS LLM (LangchainLLMWrapper)."""
    from langchain_core.language_models import BaseChatModel  # noqa: F401
    from ragas.llms import LangchainLLMWrapper

    from superdialog.eval.metrics._langchain_shim import ResolverChatModel

    return LangchainLLMWrapper(ResolverChatModel(model_uri=judge_model))
