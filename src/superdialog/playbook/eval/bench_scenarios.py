"""Score one EvalCase under both playbook and vanilla modes on all 9 metrics."""

from __future__ import annotations

from pydantic import BaseModel

from superdialog.eval.dataset.models import EvalCase
from superdialog.eval.metrics.base import MetricResult

from .events import assistant_utterances_by_checkpoint, parse_events
from .models import SessionMetrics
from .runner import SpeaksUser
from .scenario_metrics import (
    constraint_adherence,
    constraint_adherence_scoped,
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
    case: EvalCase,
    sm: SessionMetrics,
    never_say: list[str],
    judge_llm: SpeaksUser,
    *,
    never_say_by_checkpoint: dict[str, list[str]] | None = None,
    initial_checkpoint: str | None = None,
    interrupt_checkpoints: set[str] | None = None,
    designed_edges: set[tuple[str, str]] | None = None,
) -> ScenarioResult:
    events = parse_events(sm.event_log_jsonl)
    utterances = [
        e["text"] for e in events if e.get("type") == "utterance" and e.get("role") == "assistant"
    ]
    # Checkpoint-scoped when the caller provides the per-checkpoint map
    # (test_scenarios.py does); falls back to the flat global check
    # otherwise, so existing callers (test_bench_scenarios.py) are unaffected.
    if never_say_by_checkpoint is not None:
        pairs = assistant_utterances_by_checkpoint(events, initial_checkpoint)
        constraint_result = await constraint_adherence_scoped(
            pairs, never_say_by_checkpoint, judge_llm
        )
    else:
        constraint_result = await constraint_adherence(utterances, never_say, judge_llm)
    metrics: dict[str, MetricResult] = {
        "task_success": evidence_gated_task_success(sm.completed, events),
        "pass_at_1": pass_at_1(
            sm.completed, events,
            interrupt_checkpoints=interrupt_checkpoints, designed_edges=designed_edges,
        ),
        "failure_recovery_rate": failure_recovery_rate(events),
        "conversational_consistency": conversational_consistency(events),
        "slot_accuracy": MetricResult(
            name="slot_accuracy", value=sm.slot_accuracy,
            reason=f"diffs: {sm.slot_diffs}" if sm.slot_diffs else "exact match",
        ),
        "constraint_adherence": constraint_result,
        "pii_violation_rate": await pii_violation_rate(utterances, judge_llm),
    }
    return ScenarioResult(case_id=case.id, mode="playbook", metrics=metrics)


async def _vanilla_side(
    case: EvalCase, vm: VanillaSessionMetrics, never_say: list[str], judge_llm: SpeaksUser
) -> ScenarioResult:
    # Vanilla can now make REAL tool calls too (native LLM function-calling,
    # see vanilla.py) -- task_success/pass_at_1/failure_recovery_rate read
    # vm.events the same way the playbook side reads its EventLog. A vanilla
    # run given no tools (or one that never calls any) still gets an empty
    # events list, so it fails the evidence gate exactly as before this
    # change -- this isn't a free pass, it's the same evidence bar applied
    # fairly to both modes.
    utterances = [t.text for t in vm.transcript if t.role == "assistant"]
    metrics: dict[str, MetricResult] = {
        "task_success": evidence_gated_task_success(True, vm.events),
        "pass_at_1": pass_at_1(True, vm.events),
        "failure_recovery_rate": failure_recovery_rate(vm.events),
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
    never_say_by_checkpoint: dict[str, list[str]] | None = None,
    initial_checkpoint: str | None = None,
    interrupt_checkpoints: set[str] | None = None,
    designed_edges: set[tuple[str, str]] | None = None,
) -> dict[str, ScenarioResult]:
    """Score one case's playbook run and vanilla run on all 9 metrics.

    task_success/slot_accuracy are the pieces where each side is built
    differently (playbook: real event-log evidence; vanilla: structurally
    incapable, scored 0/skipped by construction) -- guardrail and efficiency
    are intentionally NOT duplicated here; they stay on the existing
    superdialog.eval A/B path (superdialog/eval/metrics/custom.py), which
    already covers them for both modes via the standard bench command.

    ``never_say_by_checkpoint`` (optional): when given, the playbook side's
    constraint_adherence is scoped per-checkpoint instead of checked against
    a flattened union of every checkpoint's rules (see
    ``scenario_metrics.constraint_adherence_scoped``). Vanilla has no
    checkpoints, so its constraint check always uses the flat ``never_say``
    list regardless.
    """
    return {
        "playbook": await _playbook_side(
            case, playbook_metrics, never_say, judge_llm,
            never_say_by_checkpoint=never_say_by_checkpoint,
            initial_checkpoint=initial_checkpoint,
            interrupt_checkpoints=interrupt_checkpoints,
            designed_edges=designed_edges,
        ),
        "vanilla": await _vanilla_side(case, vanilla_metrics, never_say, judge_llm),
    }
