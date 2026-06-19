from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Eval models (Tools 1 + 2) ─────────────────────────────────────────────


class EdgeTest(BaseModel):
    node_id: str
    edge_id: str
    condition: str = ""
    utterances: list[str] = Field(default_factory=list)
    negative_utterances: list[str] = Field(default_factory=list)


class EdgeResult(BaseModel):
    passed: bool
    actual_edge: str | None = None
    expected_edge: str = ""
    error: str | None = None


class NegativeEdgeResult(BaseModel):
    passed: bool
    actual_edge: str | None = None
    error: str | None = None


class PathStep(BaseModel):
    utterance: str
    expected_edge: str
    expected_node: str


class PathTest(BaseModel):
    name: str
    steps: list[PathStep] = Field(default_factory=list)


class PathStepResult(BaseModel):
    utterance: str
    expected_edge: str
    actual_edge: str | None = None
    expected_node: str
    actual_node: str | None = None
    passed: bool = False


class PathResult(BaseModel):
    name: str = ""
    completed: bool = False
    steps: list[PathStepResult] = Field(default_factory=list)


class PersonaConfig(BaseModel):
    name: str
    traits: str = ""
    goal: str = ""
    expected_final_node: str | None = None
    max_turns: int = 10


class PersonaResult(BaseModel):
    persona_name: str
    model_id: str = ""
    final_node: str = ""
    expected_final_node: str | None = None
    reached_final: bool = False
    turns_taken: int = 0
    transitions: list[dict[str, Any]] = Field(default_factory=list)
    conversation: list[dict[str, str]] = Field(default_factory=list)


class TestCorpus(BaseModel):
    flow_file: str = ""
    edge_tests: list[EdgeTest] = Field(default_factory=list)
    path_tests: list[PathTest] = Field(default_factory=list)
    persona_tests: list[PersonaConfig] = Field(default_factory=list)
    generated_by: str = "corpus_generator"
    reviewed: bool = False


class ModelScore(BaseModel):
    model_id: str = ""
    edge_accuracy: float = 0.0
    path_accuracy: float = 0.0
    persona_completion: float = 0.0
    edge_results: list[EdgeResult] = Field(default_factory=list)
    negative_results: list[NegativeEdgeResult] = Field(default_factory=list)
    path_results: list[PathResult] = Field(default_factory=list)
    persona_results: list[PersonaResult] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    negative_failures: list[str] = Field(default_factory=list)

    @property
    def edges_total(self) -> int:
        return len(self.edge_results)

    @property
    def edges_passed(self) -> int:
        return sum(1 for r in self.edge_results if r.passed)

    @property
    def negatives_total(self) -> int:
        return len(self.negative_results)

    @property
    def negatives_passed(self) -> int:
        return sum(1 for r in self.negative_results if r.passed)

    @property
    def paths_total(self) -> int:
        return len(self.path_results)

    @property
    def paths_passed(self) -> int:
        return sum(1 for r in self.path_results if r.completed)


class EvalReport(BaseModel):
    flow_file: str = ""
    generated_at: str = ""
    models: list[ModelScore] = Field(default_factory=list)

    def summary(self) -> str:
        """One human-readable line per model: edge/negative/path/persona tallies."""
        lines = [f"Eval report: {self.flow_file}"]
        for m in self.models:
            lines.append(
                f"  {m.model_id}: "
                f"edges={m.edges_passed}/{m.edges_total} "
                f"neg={m.negatives_passed}/{m.negatives_total} "
                f"paths={m.paths_passed}/{m.paths_total} "
                f"personas={int(m.persona_completion * 100)}%"
            )
        return "\n".join(lines)


# ── Audit models (Tool 3 — SessionAuditor) ───────────────────────────────


class PathViolation(BaseModel):
    step: int
    edge_id: str | None
    from_node: str | None
    to_node: str | None
    reason: str


class EdgeVerdict(BaseModel):
    step: int
    edge_id: str | None
    from_node: str | None
    correct: bool
    confidence: Literal["high", "medium", "low"] = "high"
    preferred_edge: str | None = None
    reason: str = ""


class ResponseVerdict(BaseModel):
    step: int
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    routing_leak: bool = False
    bot_message: str = ""


class AuditReport(BaseModel):
    session_id: str = ""
    flow_file: str = ""
    final_node: str = ""
    reached_final: bool = False

    path_valid: bool = True
    path_violations: list[PathViolation] = Field(default_factory=list)

    edge_verdicts: list[EdgeVerdict] = Field(default_factory=list)
    edge_accuracy: float = 0.0

    response_verdicts: list[ResponseVerdict] = Field(default_factory=list)
    response_quality: float = 0.0
    routing_leaks: list[str] = Field(default_factory=list)

    slot_coverage: dict[str, bool] = Field(default_factory=dict)
    slot_completeness: float = 0.0

    overall_score: float = 0.0
    critical_issues: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_markdown(self) -> str:
        lines = [
            "SESSION AUDIT REPORT",
            "════════════════════",
            f"Session: {self.session_id}   Flow: {self.flow_file}",
            f"Final node: {self.final_node}   {'✓ reached final' if self.reached_final else '✗ did not reach final'}",
            "",
            f"LAYER 1 — PATH VALIDITY       {'✓ PASS' if self.path_valid else '✗ FAIL'} ({len(self.path_violations)} violations)",
        ]
        for v in self.path_violations:
            lines.append(f"  ✗ Step {v.step}: {v.reason}")

        correct = sum(1 for e in self.edge_verdicts if e.correct)
        total = len(self.edge_verdicts)
        lines.append(
            f"LAYER 2 — EDGE ACCURACY       {'✓ PASS' if self.edge_accuracy >= 0.9 else '✗ WARN'} "
            f"({correct}/{total} = {int(self.edge_accuracy * 100)}%)"
        )
        for v in self.edge_verdicts:
            if not v.correct:
                lines.append(
                    f"  ✗ Step {v.step}: took '{v.edge_id}' — preferred '{v.preferred_edge}' ({v.confidence})"
                )

        lines.append(
            f"LAYER 3 — RESPONSE QUALITY    {'✓ PASS' if self.response_quality >= 4.0 else '✗ WARN'} "
            f"(avg {self.response_quality:.1f}/5)"
        )
        for leak in self.routing_leaks:
            lines.append(f"  ✗ routing leak: {leak[:80]}")

        captured = sum(1 for v in self.slot_coverage.values() if v)
        total_slots = len(self.slot_coverage)
        lines.append(
            f"LAYER 4 — SLOT COMPLETENESS   {'✓ PASS' if self.slot_completeness >= 0.9 else '✗ WARN'} "
            f"({captured}/{total_slots} captured)"
        )
        for slot, ok in self.slot_coverage.items():
            if not ok:
                lines.append(f"  ✗ missing: {slot}")

        lines += ["", f"OVERALL SCORE: {int(self.overall_score * 100)}%"]
        return "\n".join(lines)