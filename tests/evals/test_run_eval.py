"""Eval: run FlowEvaluator against a flow using a generated corpus.

Run:
    pytest tests/evals/test_run_eval.py -s -v \
        --flow /path/to/flow.json \
        --corpus /path/to/flow_corpus.json

If --corpus not provided, generates corpus on the fly first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from superdialog.machine.eval.corpus_generator import CorpusGenerator
from superdialog.machine.eval.evaluator import FlowEvaluator
from superdialog.machine.eval.models import TestCorpus


# NOTE: --corpus is registered in tests/evals/conftest.py (pytest ignores
# pytest_addoption in a test module, which is why this option never took effect).


@pytest.fixture
def corpus_path(request: pytest.FixtureRequest):
    return request.config.getoption("--corpus")


@pytest.mark.anyio
async def test_run_eval(flow, flow_path, corpus_path, llm_fn, eval_model) -> None:
    if corpus_path and Path(corpus_path).exists():
        corpus = TestCorpus.model_validate_json(Path(corpus_path).read_text())
        print(f"\nLoaded corpus: {len(corpus.edge_tests)} edge tests")
    else:
        print("\nNo corpus provided — generating from flow...")
        generator = CorpusGenerator(flow=flow, llm_fn=llm_fn, utterances_per_edge=2, negative_per_edge=1)
        corpus = await generator.generate_corpus(flow_file=flow_path)
        print(f"Generated {len(corpus.edge_tests)} edge tests")

    evaluator = FlowEvaluator(
        flow=flow,
        llm_factory=lambda model_id: llm_fn,
    )
    report = await evaluator.eval_corpus(corpus, model_ids=[eval_model])

    for score in report.models:
        print(f"\n{'='*50}")
        print(f"Model: {score.model_id}")
        print(f"Edge accuracy:      {score.edge_accuracy:.1%}")
        print(f"Persona completion: {score.persona_completion:.1%}")
        if score.failures:
            print(f"\nFailures ({len(score.failures)}):")
            for f in score.failures:
                print(f"  - {f}")

    assert len(report.models) > 0
    assert report.models[0].edge_accuracy >= 0.0