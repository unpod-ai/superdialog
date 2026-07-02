"""Metric registry: custom names build always; ragas names build when installed."""

from __future__ import annotations

import builtins

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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_build_suite_ragas_name_builds_when_available():
    # With the `ragas` extra installed, ragas metric names resolve to RagasMetric
    # instances (constructed offline — no API call at build time).
    pytest.importorskip("ragas")
    suite = build_suite(["faithfulness"], _judge(), "openai/gpt-4.1-mini")
    assert isinstance(suite, MetricSuite)
    assert any(getattr(m, "name", "") == "faithfulness" for m in suite._metrics)


def test_build_suite_ragas_name_raises_when_ragas_unavailable(monkeypatch):
    # If `import ragas.metrics` fails (extra not installed / broken), the registry
    # surfaces a clear RuntimeError instead of a bare ImportError.
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("ragas.metrics"):
            raise ImportError("simulated: ragas unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError):
        build_suite(["faithfulness"], _judge(), "m")


def test_build_suite_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        build_suite(["nonsense"], _judge(), "m")
