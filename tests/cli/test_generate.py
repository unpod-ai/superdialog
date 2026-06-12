"""CLI tests for the top-level `generate` subcommand (playbook by default)."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

from superdialog.cli.main import main

_cli = importlib.import_module("superdialog.cli.main")

_YAML = 'goal: "x"\nplaybook:\n  - {id: a, say: "hi", done_when: "done"}\n'


def test_generate_writes_playbook_and_prints_path(tmp_path: Path, capsys) -> None:
    out = tmp_path / "agent.yaml"
    with patch.object(_cli, "_run_generate_playbook", return_value=_YAML) as m:
        rc = main(["generate", "a demo booking agent", "--output", str(out)])
    assert rc == 0
    m.assert_called_once()
    assert out.read_text() == _YAML
    assert str(out) in capsys.readouterr().out


def test_generate_default_output_is_playbook_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(_cli, "_run_generate_playbook", return_value=_YAML):
        rc = main(["generate", "an agent"])
    assert rc == 0
    assert (tmp_path / "playbook.yaml").read_text() == _YAML


def test_generate_without_prompt_errors_cleanly(capsys) -> None:
    with patch.object(_cli, "_run_generate_playbook") as m:
        rc = main(["generate"])
    assert rc == 1
    m.assert_not_called()
    assert "description" in capsys.readouterr().err.lower()


def test_chat_missing_files_hint_points_to_generate(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)  # neither playbook.yaml nor flow.json exists
    rc = main(["chat"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "superdialog generate" in err


def test_chat_prefers_playbook_yaml_over_flow_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "playbook.yaml").write_text(
        'persona: "p"\n'
        "journeys:\n  main:\n    checkpoints:\n"
        '      - {id: only, guidance: "hi", terminal: true}\n'
    )
    (tmp_path / "flow.json").write_text("{}")
    with patch.object(_cli, "_run_playbook_repl") as mock_play:
        rc = main(["chat"])
    assert rc == 0
    mock_play.assert_called_once()
    assert "playbook.yaml" in mock_play.call_args[0][0]
