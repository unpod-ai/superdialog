"""EvalSample <-> RAGAS interchange. RAGAS is imported lazily and optionally."""

from __future__ import annotations

from typing import Any

from superdialog.eval.dataset.models import EvalSample


def sample_to_ragas_dict(sample: EvalSample) -> dict[str, Any]:
    """Framework-neutral dict matching RAGAS Single/MultiTurnSample fields."""
    d: dict[str, Any] = {"user_input": sample.user_input}
    if sample.response:
        d["response"] = sample.response
    if sample.reference is not None:
        d["reference"] = sample.reference
    if sample.retrieved_contexts is not None:
        d["retrieved_contexts"] = sample.retrieved_contexts
    return d


def to_ragas_sample(sample: EvalSample) -> Any:
    """Build a real RAGAS sample (requires the `ragas` extra)."""
    from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
    from ragas.messages import AIMessage, HumanMessage

    if sample.kind == "conversation":
        msgs = [
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in sample.user_input  # type: ignore[union-attr]
        ]
        return MultiTurnSample(user_input=msgs, reference=sample.reference)
    return SingleTurnSample(**sample_to_ragas_dict(sample))


def to_evaluation_dataset(samples: list[EvalSample]) -> Any:
    """Wrap samples in a ragas EvaluationDataset."""
    from ragas.dataset_schema import EvaluationDataset

    return EvaluationDataset(samples=[to_ragas_sample(s) for s in samples])
