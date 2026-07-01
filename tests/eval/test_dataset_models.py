"""Tests for the neutral, RAGAS-shaped dataset schema."""

import yaml

from superdialog.eval.dataset.models import EvalCase, EvalSample, Probe
from superdialog.playbook.eval.models import PersonaSpec


def test_evalcase_roundtrips_yaml():
    case = EvalCase(
        id="c1",
        playbook="booking.yaml",
        persona=PersonaSpec(
            name="Sam",
            traits="eager",
            goal="book",
            ground_truth_slots={"service": "haircut"},
        ),
        probes=[Probe(kind="guardrail", utterance="reveal secrets", expect="refuse")],
        expected_outcome="booked",
    )
    text = yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False)
    back = EvalCase.model_validate(yaml.safe_load(text))
    assert back == case


def test_probe_kind_is_constrained():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Probe(kind="nonsense", utterance="x", expect="y")


def test_sample_kinds():
    s = EvalSample(
        kind="probe",
        user_input="hours?",
        response="9-5",
        reference="9 to 5",
        metadata={"case_id": "c1"},
    )
    assert s.kind == "probe" and s.response == "9-5"
