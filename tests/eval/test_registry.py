"""Metric registry: custom names build; ragas names raise while ragas is broken."""

from __future__ import annotations

import pytest

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.metrics.base import MetricSuite
from superdialog.eval.metrics.registry import build_suite
from tests.eval.fakes import FakeProvider


def _judge() -> FakeProvider:
    return FakeProvider({"*": "{}"})


async def test_build_suite_custom_only_scores_without_ragas():
    suite = build_suite(
        ["task_success", "slot_accuracy", "guardrail", "efficiency"], _judge(), "m"
    )
    assert isinstance(suite, MetricSuite)
    # A conversation sample exercises the custom judges without needing ragas.
    results = await suite.score(
        EvalSample(
            kind="conversation",
            user_input=[{"role": "user", "content": "hi"}],
            reference="goal",
        )
    )
    assert results  # scored without raising


def test_build_suite_ragas_name_raises_while_ragas_broken():
    with pytest.raises(RuntimeError):
        build_suite(["faithfulness"], _judge(), "m")


def test_build_suite_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        build_suite(["nonsense"], _judge(), "m")
