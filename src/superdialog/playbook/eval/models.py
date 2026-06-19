# src/superdialog/playbook/eval/models.py
"""Data models for playbook evaluation: session metrics, audit, corpus, multi-model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PersonaSpec(BaseModel):
    """A simulated caller: who they are, what they want, what they convey."""

    name: str
    traits: str
    goal: str
    max_turns: int = 12
    opening: str = "Hello"
    ground_truth_slots: dict[str, Any] = Field(default_factory=dict)


class SessionMetrics(BaseModel):
    """Measurements folded from one persona-driven session."""

    persona: str
    completed: bool
    outcome: str | None
    turns: int
    turns_per_checkpoint: dict[str, int]
    slot_accuracy: float
    slot_diffs: dict[str, tuple[Any, Any]]
    repair_count: int
    degraded_count: int
    event_log_jsonl: str


class EvalReport(BaseModel):
    """Aggregate of persona sessions."""

    sessions: list[SessionMetrics]

    @property
    def completion_rate(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(s.completed for s in self.sessions) / len(self.sessions)

    @property
    def mean_slot_accuracy(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(s.slot_accuracy for s in self.sessions) / len(self.sessions)


class EdgeScenario(BaseModel):
    """One test utterance at a specific checkpoint."""

    checkpoint_id: str
    utterance: str
    expected_advance: str | None  # target checkpoint id if advancing, None if blocking


class CorpusSpec(BaseModel):
    """Auto-generated test suite for a playbook."""

    playbook_file: str = ""
    persona_tests: list[PersonaSpec] = Field(default_factory=list)
    edge_scenarios: list[EdgeScenario] = Field(default_factory=list)
    generated_by: str = "corpus_generator"


class AuditReport(BaseModel):
    """Post-session analysis: path validity, slot completeness, response quality."""

    session_id: str = ""
    checkpoint_path: list[str] = Field(default_factory=list)
    path_valid: bool = True
    path_violations: list[str] = Field(default_factory=list)
    slot_coverage: dict[str, bool] = Field(default_factory=dict)
    slot_completeness: float = 0.0
    response_quality: float = 0.0
    overall_score: float = 0.0
    critical_issues: list[str] = Field(default_factory=list)


class ModelScore(BaseModel):
    """Eval results for one model ID."""

    model_id: str
    completion_rate: float = 0.0
    mean_slot_accuracy: float = 0.0
    mean_turns_per_checkpoint: float = 0.0
    repair_rate: float = 0.0
    objective: float = 0.0
    sessions: list[SessionMetrics] = Field(default_factory=list)


class MultiModelReport(BaseModel):
    """Side-by-side eval results across multiple model IDs."""

    playbook_file: str = ""
    generated_at: str = ""
    models: list[ModelScore] = Field(default_factory=list)


__all__ = [
    "AuditReport",
    "CorpusSpec",
    "EdgeScenario",
    "EvalReport",
    "ModelScore",
    "MultiModelReport",
    "PersonaSpec",
    "SessionMetrics",
]