"""Score one EvalCase under both playbook and vanilla modes on all 9 metrics."""

from __future__ import annotations

from pydantic import BaseModel

from superdialog.eval.dataset.models import EvalCase
from superdialog.eval.metrics.base import MetricResult

from .events import parse_events
from .models import SessionMetrics
from .runner import SpeaksUser
from .scenario_metrics import (
    constraint_adherence,
    conversational_consistency,
    evidence_gated_task_success,
    failure_recovery_rate,
    pass_at_1,
    pii_violation_rate,
)
from .vanilla import VanillaSessionMetrics


class ScenarioResult(BaseModel):
    """All scored metrics for one (case, mode)."""

    case_id: str
    mode: str
    metrics: dict[str, MetricResult]

    model_config = {"arbitrary_types_allowed": True}


async def _playbook_side(
    case: EvalCase, sm: SessionMetrics, never_say: list[str], judge_llm: SpeaksUser
) -> ScenarioResult:
    events = parse_events(sm.event_log_jsonl)
    utterances = [
        e["text"] for e in events if e.get("type") == "utterance" and e.get("role") == "assistant"
    ]
    metrics: dict[str, MetricResult] = {
        "task_success": evidence_gated_task_success(sm.completed, events),
        "pass_at_1": pass_at_1(sm.completed, events),
        "failure_recovery_rate": failure_recovery_rate(events),
        "conversational_consistency": conversational_consistency(events),
        "slot_accuracy": MetricResult(
            name="slot_accuracy", value=sm.slot_accuracy,
            reason=f"diffs: {sm.slot_diffs}" if sm.slot_diffs else "exact match",
        ),
        "constraint_adherence": await constraint_adherence(utterances, never_say, judge_llm),
        "pii_violation_rate": await pii_violation_rate(utterances, judge_llm),
    }
    return ScenarioResult(case_id=case.id, mode="playbook", metrics=metrics)


async def _vanilla_side(
    case: EvalCase, vm: VanillaSessionMetrics, never_say: list[str], judge_llm: SpeaksUser
) -> ScenarioResult:
    # Vanilla has no tools and no event log -- it can NEVER produce real tool-
    # call evidence, so task_success/pass_at_1/failure_recovery_rate are all
    # evaluated against an empty event list, which fails the evidence gate by
    # construction (see scenario_metrics.evidence_gated_task_success). This
    # is correct, not a bug: it's the concrete demonstration of the
    # playbook's grounding advantage the design spec calls for.
    utterances = [t.text for t in vm.transcript if t.role == "assistant"]
    metrics: dict[str, MetricResult] = {
        "task_success": evidence_gated_task_success(True, []),
        "pass_at_1": pass_at_1(True, []),
        "failure_recovery_rate": failure_recovery_rate([]),
        "conversational_consistency": MetricResult(
            name="conversational_consistency", value=None, skipped=True,
            reason="vanilla has no structured slots to track",
        ),
        "slot_accuracy": MetricResult(
            name="slot_accuracy", value=None, skipped=True,
            reason="vanilla has no structured slots to grade",
        ),
        "constraint_adherence": await constraint_adherence(utterances, never_say, judge_llm),
        "pii_violation_rate": await pii_violation_rate(utterances, judge_llm),
    }
    return ScenarioResult(case_id=case.id, mode="vanilla", metrics=metrics)


async def score_case_both_modes(
    *,
    case: EvalCase,
    playbook_metrics: SessionMetrics,
    vanilla_metrics: VanillaSessionMetrics,
    never_say: list[str],
    judge_llm: SpeaksUser,
) -> dict[str, ScenarioResult]:
    """Score one case's playbook run and vanilla run on all 9 metrics.

    task_success/slot_accuracy are the pieces where each side is built
    differently (playbook: real event-log evidence; vanilla: structurally
    incapable, scored 0/skipped by construction) -- guardrail and efficiency
    are intentionally NOT duplicated here; they stay on the existing
    superdialog.eval A/B path (superdialog/eval/metrics/custom.py), which
    already covers them for both modes via the standard bench command.
    """
    return {
        "playbook": await _playbook_side(case, playbook_metrics, never_say, judge_llm),
        "vanilla": await _vanilla_side(case, vanilla_metrics, never_say, judge_llm),
    }
