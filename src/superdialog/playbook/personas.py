"""Persona suites for the optimizer: load, save, generate, derive."""

from __future__ import annotations

import json
import os

import yaml

from .director import CompletesLLM
from .eval_bridge import PersonaSpec
from .models import Playbook

_AXES = (
    "cooperative and forthcoming",
    "terse and impatient",
    "tangent-prone (wanders off-topic, must be steered back)",
    "error-making (gives one wrong slot value, then corrects it when asked)",
)

_GEN_SYSTEM = """\
You create test personas for evaluating a conversational agent. Given the
playbook summary and its slot schema, return ONLY a JSON array of exactly
{count} personas, one per diversity axis:
{axes}

Each persona: {{"name": str, "traits": str, "goal": str,
"ground_truth_slots": {{...}}}}. ground_truth_slots MUST contain a concrete,
plausible value for EVERY required slot listed. No commentary, no fences.
"""


def persona_cache_path(playbook_path: str) -> str:
    """The conventional persona-suite cache path beside the playbook."""
    root, _ = os.path.splitext(playbook_path)
    return f"{root}.personas.yaml"


def load_personas(path: str) -> list[PersonaSpec]:
    """Load a YAML/JSON list of PersonaSpec dicts; ValueError when malformed."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    data = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of personas")
    return [PersonaSpec.model_validate(item) for item in data]


def save_personas(personas: list[PersonaSpec], path: str) -> None:
    """Write a persona suite as reviewable YAML."""
    dumped = yaml.safe_dump(
        [p.model_dump() for p in personas], sort_keys=False, allow_unicode=True
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumped)


def derive_default_persona(playbook: Playbook) -> PersonaSpec:
    """Single fallback persona derived from the initial checkpoint's goal."""
    cp = playbook.checkpoint(playbook.initial_checkpoint_id)
    goal = cp.goal or "complete the conversation"
    return PersonaSpec(
        name="default",
        traits="cooperative, concise",
        goal=f"Work with the agent so it can: {goal}",
    )


def _required_slots(playbook: Playbook) -> dict[str, str]:
    """Map of required slot key -> type across all checkpoints."""
    out: dict[str, str] = {}
    for journey in playbook.journeys.values():
        for cp in journey.checkpoints:
            for key, spec in cp.slots.items():
                if spec.required:
                    out[key] = spec.type
    return out


def _summary(playbook: Playbook) -> str:
    lines: list[str] = []
    for jname, journey in playbook.journeys.items():
        for cp in journey.checkpoints:
            lines.append(f"- {jname}.{cp.id}: goal={cp.goal!r}")
    return "\n".join(lines)


async def generate_personas(
    playbook: Playbook,
    llm: CompletesLLM,
    *,
    count: int = 4,
    max_attempts: int = 3,
) -> list[PersonaSpec]:
    """Generate a diverse persona suite; ValueError after max_attempts."""
    required = _required_slots(playbook)
    system = _GEN_SYSTEM.format(count=count, axes="\n".join(f"- {a}" for a in _AXES))
    user = (
        f"CHECKPOINTS:\n{_summary(playbook)}\n\n"
        f"REQUIRED SLOTS (every persona needs concrete values for all):\n"
        + "\n".join(f"- {k} ({t})" for k, t in required.items())
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = "no attempts made"
    for _ in range(max_attempts):
        raw = await llm.complete(messages)
        try:
            data = json.loads(raw.strip().strip("`"))
            if not isinstance(data, list):
                raise ValueError("expected a JSON array of personas")
            personas = [PersonaSpec.model_validate(item) for item in data]
            missing = [
                p.name
                for p in personas
                if not set(required) <= set(p.ground_truth_slots)
            ]
            if missing:
                raise ValueError(f"personas missing required slots: {missing}")
        except ValueError as exc:
            last_error = str(exc)
            continue
        return personas
    raise ValueError(f"persona generation failed: {last_error}")
