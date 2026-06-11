"""Reflective prose optimizer: scoring, reflection, paired-round loop."""

from __future__ import annotations

from statistics import mean

from pydantic import BaseModel

from .eval_bridge import EvalReport

W_COMPLETION = 0.4
W_SLOT = 0.3
W_SMOOTHNESS = 0.2
W_REPAIR = 0.1


class ObjectiveBreakdown(BaseModel):
    """Scalar objective plus its per-dimension breakdown."""

    objective: float
    completion_rate: float
    slot_accuracy: float
    mean_turns_per_checkpoint: float
    repair_rate: float


def _smoothness(mean_turns_per_checkpoint: float) -> float:
    """Map mean turns/checkpoint to [0, 1]; 1 turn -> 1.0, more -> less."""
    return 1.0 / (1.0 + max(0.0, mean_turns_per_checkpoint - 1.0))


def score_report(report: EvalReport) -> ObjectiveBreakdown:
    """Score an eval report. Pure: no LLM, no I/O.

    Smoothness is averaged over completed sessions only, so fail-fast
    incomplete sessions cannot game the mean (they pay via completion).
    """
    if not report.sessions:
        return ObjectiveBreakdown(
            objective=0.0,
            completion_rate=0.0,
            slot_accuracy=0.0,
            mean_turns_per_checkpoint=0.0,
            repair_rate=0.0,
        )
    per_completed = [
        mean(s.turns_per_checkpoint.values())
        for s in report.sessions
        if s.completed and s.turns_per_checkpoint
    ]
    mean_tpc = mean(per_completed) if per_completed else 0.0
    total_turns = sum(s.turns for s in report.sessions)
    total_repairs = sum(s.repair_count for s in report.sessions)
    repair_rate = total_repairs / total_turns if total_turns else 0.0
    smooth = _smoothness(mean_tpc) if per_completed else 0.0
    objective = (
        W_COMPLETION * report.completion_rate
        + W_SLOT * report.mean_slot_accuracy
        + W_SMOOTHNESS * smooth
        + W_REPAIR * (1.0 - min(1.0, repair_rate))
    )
    return ObjectiveBreakdown(
        objective=objective,
        completion_rate=report.completion_rate,
        slot_accuracy=report.mean_slot_accuracy,
        mean_turns_per_checkpoint=mean_tpc,
        repair_rate=repair_rate,
    )
