"""Tests for the EvalSample <-> RAGAS interchange (lazy RAGAS import)."""

import pytest

from superdialog.eval.dataset import ragas_io
from superdialog.eval.dataset.models import EvalSample


def test_sample_to_ragas_dict_probe():
    s = EvalSample(
        kind="probe",
        user_input="hours?",
        response="9-5",
        reference="9 to 5",
        retrieved_contexts=["Open 9-5"],
    )
    d = ragas_io.sample_to_ragas_dict(s)
    assert d["user_input"] == "hours?"
    assert d["response"] == "9-5"
    assert d["retrieved_contexts"] == ["Open 9-5"]


def test_sample_to_ragas_dict_conversation_shapes_messages():
    s = EvalSample(
        kind="conversation",
        user_input=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        reference="greeted",
    )
    d = ragas_io.sample_to_ragas_dict(s)
    assert isinstance(d["user_input"], list)
    assert d["reference"] == "greeted"


@pytest.mark.integration
def test_to_evaluation_dataset_builds_ragas_object():
    pytest.importorskip("ragas")
    s = EvalSample(kind="probe", user_input="q", response="a", reference="a")
    ds = ragas_io.to_evaluation_dataset([s])
    assert ds is not None  # ragas.EvaluationDataset
