"""Assemble a mixed MetricSuite from metric names (config-driven)."""

from __future__ import annotations

from typing import Any

from superdialog.eval.metrics.base import MetricSuite
from superdialog.eval.metrics.custom import (
    EfficiencyMetric,
    GuardrailJudge,
    SlotAccuracyJudge,
    TaskSuccessJudge,
    TokenCostMetric,
)
from superdialog.playbook.eval.runner import SpeaksUser

_CUSTOM = {
    "task_success": lambda judge, model: TaskSuccessJudge(judge),
    "slot_accuracy": lambda judge, model: SlotAccuracyJudge(judge),
    "guardrail": lambda judge, model: GuardrailJudge(judge),
    "efficiency": lambda judge, model: EfficiencyMetric(),
    "token_cost": lambda judge, model: TokenCostMetric(),
}

# Pure-code metrics: no judge tokens, no extra latency. Always included so a
# run can never silently lose latency/token evidence (goals #4/#5).
_FREE = ("efficiency", "token_cost")

# ragas name -> (ragas.metrics class name, applies_to)
_RAGAS = {
    "faithfulness": ("Faithfulness", ("probe",)),
    "answer_correctness": ("AnswerCorrectness", ("probe",)),
    "topic_adherence": ("TopicAdherenceScore", ("conversation",)),
    "goal_accuracy": ("AgentGoalAccuracyWithReference", ("conversation",)),
}


def build_suite(
    names: list[str], judge_llm: SpeaksUser, judge_model: str
) -> MetricSuite:
    """Map metric names to a mixed suite. Custom names always work; ragas names
    need a working `ragas` install (raise a clear error otherwise)."""
    metrics: list[Any] = []
    for name in names:
        if name in _CUSTOM:
            metrics.append(_CUSTOM[name](judge_llm, judge_model))
        elif name in _RAGAS:
            metrics.append(_build_ragas(name, judge_model))
        else:
            valid = sorted([*_CUSTOM, *_RAGAS])
            raise KeyError(f"unknown metric {name!r}. valid: {valid}")
    for free in _FREE:
        if free not in names:
            metrics.append(_CUSTOM[free](judge_llm, judge_model))
    return MetricSuite(metrics)


def _build_ragas(name: str, judge_model: str) -> Any:
    try:
        import ragas.metrics as rm

        from superdialog.eval.metrics.ragas_backend import (
            RagasMetric,
            build_ragas_judge,
        )
    except Exception as exc:  # ragas missing or broken import
        raise RuntimeError(
            f"metric {name!r} needs the 'ragas' extra and a working ragas "
            f"import (uv sync --extra ragas); got: {exc}"
        ) from exc
    cls_name, applies_to = _RAGAS[name]
    obj = getattr(rm, cls_name)(llm=build_ragas_judge(judge_model))
    return RagasMetric(obj, applies_to=applies_to, name=name)
