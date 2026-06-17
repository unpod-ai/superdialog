# tests/playbook/eval/test_corpus.py
"""CorpusGenerator: auto-generate EdgeScenarios and PersonaSpecs from a playbook."""

import json
from typing import Any

from superdialog.playbook.eval.corpus import CorpusGenerator
from superdialog.playbook.eval.models import CorpusSpec
from superdialog.playbook.models import Playbook
from tests.playbook.test_models import MINIMAL_YAML


class EdgeLLM:
    """Returns scripted JSON for edge scenarios and personas."""

    def __init__(self, advancing: list[str], blocking: list[str]) -> None:
        self._advancing = advancing
        self._blocking = blocking

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        if "personas" in messages[0]["content"].lower() or "persona" in messages[1]["content"].lower():
            # persona generation call
            personas = [
                {
                    "name": "default",
                    "traits": "cooperative",
                    "goal": "book a tee time",
                    "ground_truth_slots": {"city": "Pune", "date": "2026-06-12"},
                }
            ]
            return json.dumps(personas)
        # edge scenario call
        scenarios = (
            [{"utterance": u, "expected_advance": "booking.confirm"} for u in self._advancing]
            + [{"utterance": u, "expected_advance": None} for u in self._blocking]
        )
        return json.dumps(scenarios)


async def test_corpus_has_edge_scenarios() -> None:
    playbook = Playbook.from_yaml(MINIMAL_YAML)
    llm = EdgeLLM(
        advancing=["I want to book in Pune on June 15"],
        blocking=["I have no idea"],
    )
    gen = CorpusGenerator(playbook=playbook, llm=llm, utterances_per_checkpoint=1, negatives_per_checkpoint=1)
    corpus = await gen.generate(playbook_file="booking.yaml")

    assert isinstance(corpus, CorpusSpec)
    assert corpus.playbook_file == "booking.yaml"
    assert len(corpus.edge_scenarios) > 0


async def test_corpus_edge_scenario_fields() -> None:
    playbook = Playbook.from_yaml(MINIMAL_YAML)
    llm = EdgeLLM(
        advancing=["Book in Pune tomorrow"],
        blocking=["Maybe later"],
    )
    gen = CorpusGenerator(playbook=playbook, llm=llm, utterances_per_checkpoint=1, negatives_per_checkpoint=1)
    corpus = await gen.generate()

    advancing = [s for s in corpus.edge_scenarios if s.expected_advance is not None]
    blocking = [s for s in corpus.edge_scenarios if s.expected_advance is None]
    assert len(advancing) > 0
    assert len(blocking) > 0
    for scenario in corpus.edge_scenarios:
        assert scenario.checkpoint_id
        assert scenario.utterance


async def test_corpus_contains_personas() -> None:
    playbook = Playbook.from_yaml(MINIMAL_YAML)
    llm = EdgeLLM(advancing=["yes"], blocking=["no"])
    gen = CorpusGenerator(playbook=playbook, llm=llm)
    corpus = await gen.generate()

    assert len(corpus.persona_tests) > 0
    for p in corpus.persona_tests:
        assert p.name
        assert p.goal


async def test_corpus_generated_by_tag() -> None:
    playbook = Playbook.from_yaml(MINIMAL_YAML)
    llm = EdgeLLM(advancing=["yes"], blocking=["no"])
    gen = CorpusGenerator(playbook=playbook, llm=llm)
    corpus = await gen.generate()
    assert corpus.generated_by == "corpus_generator"