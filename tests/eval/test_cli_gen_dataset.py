"""Offline smoke test for `eval gen-dataset` (cmd_gen_dataset)."""

import argparse
from pathlib import Path

import pytest

from superdialog.eval.cli import cmd_gen_dataset
from superdialog.eval.dataset.models import EvalDataset
from tests.eval.fakes import FakeProvider

_PLAYBOOK = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "playbooks"
    / "simple_booking.yaml"
)


def test_cmd_gen_dataset_writes_loadable_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a real personas file and a faked completer, it writes a dataset."""
    # Real personas file -> generate_personas is never called (no network).
    personas = tmp_path / "personas.yaml"
    personas.write_text(
        "- name: Sam\n  traits: eager\n  goal: book a haircut\n  max_turns: 2\n",
        encoding="utf-8",
    )
    # The completer only scripts guardrail-attack generation; deterministic.
    monkeypatch.setattr(
        "superdialog.eval.cli._completer", lambda uri: FakeProvider({"*": "attack"})
    )

    out = tmp_path / "cases.evalcases.yaml"
    ns = argparse.Namespace(
        playbook=str(_PLAYBOOK),
        gen_model="fake",
        personas=str(personas),
        n_probes=2,
        out=str(out),
    )

    assert cmd_gen_dataset(ns) == 0
    assert out.exists()
    ds = EvalDataset.load(str(out))
    assert len(ds.cases) == 1
    assert ds.cases[0].persona.name == "Sam"
