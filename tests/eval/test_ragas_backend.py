"""Unit tests for the RagasMetric adapter (no working ragas needed)."""

from __future__ import annotations

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.metrics.ragas_backend import RagasMetric


class _StubRagasMetric:
    name = "faithfulness"

    async def single_turn_ascore(self, sample, callbacks=None):
        return 0.75


class _BoomRagasMetric:
    name = "faithfulness"

    async def single_turn_ascore(self, sample, callbacks=None):
        raise RuntimeError("judge exploded")


async def test_ragas_metric_adapter_wraps_single_turn(monkeypatch):
    import superdialog.eval.metrics.ragas_backend as rb

    monkeypatch.setattr(rb, "to_ragas_sample", lambda s: s)  # bypass ragas import
    m = RagasMetric(_StubRagasMetric(), applies_to=("probe",))
    r = await m.score(
        EvalSample(
            kind="probe",
            user_input="q",
            response="a",
            reference="a",
            retrieved_contexts=["a"],
        )
    )
    assert r.name == "faithfulness" and r.value == 0.75


async def test_ragas_metric_adapter_errors_are_captured(monkeypatch):
    import superdialog.eval.metrics.ragas_backend as rb

    monkeypatch.setattr(rb, "to_ragas_sample", lambda s: s)
    m = RagasMetric(_BoomRagasMetric(), applies_to=("probe",))
    r = await m.score(
        EvalSample(kind="probe", user_input="q", response="a", reference="a")
    )
    assert r.errored is True and r.value is None
