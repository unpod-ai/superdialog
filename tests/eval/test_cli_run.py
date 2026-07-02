"""Offline smoke test for `eval run` (cmd_run) — no network."""

import argparse
from pathlib import Path

import pytest

from superdialog.eval.cli import cmd_run
from tests.eval.fakes import FakeProvider

_JUDGE = '{"completed": true, "graded": 1.0, "reason": "ok"}'

_DATASET = """\
playbook: pb
cases:
  - id: c1
    playbook: pb
    persona:
      name: p
      traits: terse
      goal: get help
      max_turns: 1
    reference: the assistant should help the caller
"""


def test_cmd_run_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-mode run over one case writes report.json + report.md."""
    dataset = tmp_path / "d.yaml"
    dataset.write_text(_DATASET, encoding="utf-8")
    playbook = tmp_path / "p.txt"
    playbook.write_text("You are a helpful assistant.", encoding="utf-8")
    out = tmp_path / "out"

    # Judge / user-sim completer is deterministic; graded 1.0 for everything.
    monkeypatch.setattr(
        "superdialog.eval.cli._completer", lambda uri: FakeProvider({"*": _JUDGE})
    )
    # The vanilla endpoint's provider comes from resolve_llm as imported here.
    monkeypatch.setattr(
        "superdialog.eval.cli.resolve_llm", lambda uri: FakeProvider({"*": "ok"})
    )

    ns = argparse.Namespace(
        playbook=str(playbook),
        dataset=str(dataset),
        modes="vanilla",
        agent_model="fake",
        director_model=None,
        talker_model=None,
        judge_model="fake",
        user_model=None,
        metrics="task_success,efficiency",
        repeats=1,
        out=str(out),
    )

    assert cmd_run(ns) == 0
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
