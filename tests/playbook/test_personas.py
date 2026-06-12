"""Tests for persona suite load/save/generate/derive."""

import json

import pytest

from superdialog.playbook.eval_bridge import PersonaSpec
from superdialog.playbook.models import Playbook
from superdialog.playbook.personas import (
    derive_default_persona,
    generate_personas,
    load_personas,
    persona_cache_path,
    save_personas,
)
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_optimize import CannedEditsLLM


def _personas_json(count: int = 4, drop_slot: bool = False) -> str:
    slots = {"city": "Pune", "date": "2026-06-12"}
    if drop_slot:
        slots = {"city": "Pune"}
    return json.dumps(
        [
            {
                "name": f"p{i}",
                "traits": "direct",
                "goal": "book a slot",
                "ground_truth_slots": slots,
            }
            for i in range(count)
        ]
    )


def test_save_load_round_trip(tmp_path) -> None:
    personas = [
        PersonaSpec(name="a", traits="t", goal="g", ground_truth_slots={"city": "Pune"})
    ]
    path = tmp_path / "x.personas.yaml"
    save_personas(personas, str(path))
    loaded = load_personas(str(path))
    assert loaded == personas


def test_load_rejects_non_list(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: not-a-list\n")
    with pytest.raises(ValueError):
        load_personas(str(path))


def test_cache_path_sits_beside_the_playbook(tmp_path) -> None:
    p = tmp_path / "booking.yaml"
    assert persona_cache_path(str(p)) == str(tmp_path / "booking.personas.yaml")


def test_derive_default_uses_initial_checkpoint_goal() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    persona = derive_default_persona(pb)
    assert "Have city and date" in persona.goal
    assert persona.ground_truth_slots == {}


async def test_generate_returns_validated_suite() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    llm = CannedEditsLLM([_personas_json()])
    personas = await generate_personas(pb, llm)
    assert len(personas) == 4
    assert all(p.ground_truth_slots.keys() >= {"city", "date"} for p in personas)
    # the prompt enumerated the slot schema and the diversity axes
    prompt = " ".join(m["content"] for m in llm.calls[0])
    assert "city" in prompt and "date" in prompt
    assert "tangent" in prompt


async def test_generate_retries_then_raises_on_missing_required_slots() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    llm = CannedEditsLLM([_personas_json(drop_slot=True)])
    with pytest.raises(ValueError):
        await generate_personas(pb, llm, max_attempts=2)
    assert len(llm.calls) == 2
