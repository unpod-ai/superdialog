# src/superdialog/playbook/eval/corpus.py
"""CorpusGenerator: auto-generate EdgeScenarios and PersonaSpecs from a playbook."""

from __future__ import annotations

import json

from .models import CorpusSpec, EdgeScenario, PersonaSpec
from .personas import generate_personas
from ..director import CompletesLLM
from ..models import Playbook

_EDGE_SYSTEM = """\
You generate test utterances for a conversational agent checkpoint.
Given the checkpoint details, return ONLY a JSON array:
[{{"utterance": str, "expected_advance": str | null}}]

Rules:
- {advance_count} items must satisfy the advance condition (expected_advance = the target checkpoint id).
- {block_count} items must NOT satisfy it (expected_advance = null).
- No commentary, no fences.
"""


def _first_advance_target(cp_id: str, playbook: Playbook) -> str | None:
    """Return the first declared advance target for a checkpoint, or None."""
    jname, cpname = (cp_id.split(".", 1) + [""])[:2]
    journey = playbook.journeys.get(jname)
    if journey is None:
        return None
    for cp in journey.checkpoints:
        if cp.id == cpname and cp.advance_when:
            target = cp.advance_when[0].to
            return target if "." in target else f"{jname}.{target}"
    return None


class CorpusGenerator:
    """Generate a CorpusSpec for ``playbook`` using ``llm`` to write test utterances."""

    def __init__(
        self,
        playbook: Playbook,
        llm: CompletesLLM,
        utterances_per_checkpoint: int = 2,
        negatives_per_checkpoint: int = 1,
    ) -> None:
        self._playbook = playbook
        self._llm = llm
        self._advance_count = utterances_per_checkpoint
        self._block_count = negatives_per_checkpoint

    async def generate(self, playbook_file: str = "") -> CorpusSpec:
        edge_scenarios = await self._generate_edge_scenarios()
        try:
            personas = await generate_personas(
                self._playbook, self._llm, count=4, max_attempts=3
            )
        except ValueError:
            personas = []

        return CorpusSpec(
            playbook_file=playbook_file,
            persona_tests=personas,
            edge_scenarios=edge_scenarios,
        )

    async def _generate_edge_scenarios(self) -> list[EdgeScenario]:
        scenarios: list[EdgeScenario] = []
        for jname, journey in self._playbook.journeys.items():
            for cp in journey.checkpoints:
                if not cp.advance_when:
                    continue
                cp_id = f"{jname}.{cp.id}"
                target = _first_advance_target(cp_id, self._playbook)
                raw_scenarios = await self._generate_for_checkpoint(cp_id, cp, target)
                scenarios.extend(raw_scenarios)
        return scenarios

    async def _generate_for_checkpoint(
        self, cp_id: str, cp: object, target: str | None
    ) -> list[EdgeScenario]:
        system = _EDGE_SYSTEM.format(
            advance_count=self._advance_count, block_count=self._block_count
        )
        slot_lines = "\n".join(
            f"  - {k}: {v.type} (required={v.required})"
            for k, v in getattr(cp, "slots", {}).items()
        )
        advance_lines = "\n".join(
            f"  - when: {r.when!r}, to: {r.to!r}"
            for r in getattr(cp, "advance_when", [])
        )
        user = (
            f"CHECKPOINT: {cp_id}\n"
            f"GOAL: {getattr(cp, 'goal', '')!r}\n"
            f"SLOTS:\n{slot_lines or '  (none)'}\n"
            f"ADVANCE RULES:\n{advance_lines or '  (none)'}\n"
            f"ADVANCE TARGET (for expected_advance): {target!r}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = await self._llm.complete(messages)
        try:
            data = json.loads(raw.strip().strip("`"))
            if not isinstance(data, list):
                return []
            return [
                EdgeScenario(
                    checkpoint_id=cp_id,
                    utterance=item.get("utterance", ""),
                    expected_advance=item.get("expected_advance"),
                )
                for item in data
                if isinstance(item, dict) and item.get("utterance")
            ]
        except (ValueError, KeyError):
            return []


__all__ = ["CorpusGenerator"]