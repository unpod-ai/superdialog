"""Load benchmark dataset JSONL files into typed samples.

Datasets live in ``superdialog/examples/datasets/`` — one JSON object per line.
Each line carries a golden transcript plus the RAGAS ground-truth fields. See
``examples/datasets/universal_dataset.jsonl`` for the canonical shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkSample:
    """One scenario: golden transcript + ground truth for scoring."""

    id: str
    playbook: str | None
    scenario_type: str
    difficulty: str
    persona: dict
    # golden transcript, roles are "user"/"agent"
    conversation: list[dict]
    ground_truth: dict
    # RAGAS fields: user_input ([{role: human|ai}]), reference, retrieved_contexts,
    # reference_topics
    ragas_sample: dict
    without_framework_context: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def ground_truth_slots(self) -> dict:
        return self.persona.get("ground_truth_slots", {}) or {}

    @property
    def expected_outcome(self) -> str:
        return self.ground_truth.get("expected_outcome", "")

    @property
    def expected_checkpoint_path(self) -> list[str]:
        return self.ground_truth.get("expected_checkpoint_path", []) or []

    @property
    def reference(self) -> str:
        return self.ragas_sample.get("reference", "")

    @property
    def reference_topics(self) -> list[str]:
        return self.ragas_sample.get("reference_topics", []) or []


def load_dataset(path: str | Path) -> list[BenchmarkSample]:
    """Parse a ``.jsonl`` dataset into ``BenchmarkSample`` objects.

    Raises ``ValueError`` with the line number on the first malformed row so a
    bad dataset fails fast instead of silently scoring garbage.
    """
    p = Path(path)
    samples: list[BenchmarkSample] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p.name}:{lineno} invalid JSON: {e}") from e
        try:
            samples.append(
                BenchmarkSample(
                    id=obj["id"],
                    playbook=obj.get("playbook"),
                    scenario_type=obj.get("scenario_type", ""),
                    difficulty=obj.get("difficulty", ""),
                    persona=obj.get("persona", {}),
                    conversation=obj.get("conversation", []),
                    ground_truth=obj.get("ground_truth", {}),
                    ragas_sample=obj.get("ragas_sample", {}),
                    without_framework_context=obj.get("without_framework_context", ""),
                    metadata=obj.get("metadata", {}),
                )
            )
        except KeyError as e:
            raise ValueError(f"{p.name}:{lineno} missing field {e}") from e
    return samples


DATASETS_DIR = Path(__file__).resolve().parents[3] / "examples" / "datasets"


def load_named(name: str) -> list[BenchmarkSample]:
    """Load a dataset by short name, e.g. ``load_named("universal")``."""
    return load_dataset(DATASETS_DIR / f"{name}_dataset.jsonl")


__all__ = ["BenchmarkSample", "load_dataset", "load_named", "DATASETS_DIR"]
