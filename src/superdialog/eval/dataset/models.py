"""Neutral, RAGAS-shaped dataset schema for the A/B harness."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from superdialog.playbook.eval.models import PersonaSpec

ProbeKind = Literal["factual", "guardrail", "slot_challenge"]


class Probe(BaseModel):
    """A scripted single-turn test injected into (or run standalone) a case."""

    kind: ProbeKind
    utterance: str
    expect: str  # the fact, "refuse", or the expected captured value
    inject_after: str | int | None = None  # checkpoint id / turn idx / None
    reference_contexts: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One evaluation unit: a persona-driven journey plus scripted probes."""

    id: str
    playbook: str
    persona: PersonaSpec
    probes: list[Probe] = Field(default_factory=list)
    expected_outcome: str | None = None
    reference: str | None = None


class EvalSample(BaseModel):
    """What metrics consume — RAGAS-shaped. One per conversation or probe."""

    kind: Literal["conversation", "probe"]
    # str for a probe; list[{role,content}] for a conversation.
    user_input: str | list[dict[str, str]]
    response: str = ""
    reference: str | None = None
    retrieved_contexts: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A loadable/saveable list of EvalCases (the `<playbook>.evalcases.yaml`)."""

    playbook: str = ""
    cases: list[EvalCase] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "EvalDataset":
        import yaml

        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def save(self, path: str) -> None:
        import yaml

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                self.model_dump(mode="json", exclude_defaults=True),
                fh,
                sort_keys=False,
                allow_unicode=True,
            )
