"""CLI tests for the optimize subcommand (heavy lifting patched out)."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

from superdialog.cli.main import main
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_simple import SIMPLE

_cli = importlib.import_module("superdialog.cli.main")


def _write(tmp_path: Path, text: str = MINIMAL_YAML) -> Path:
    p = tmp_path / "play.yaml"
    p.write_text(text)
    return p


def test_optimize_writes_out_and_prints_trace(tmp_path, capsys) -> None:
    src = _write(tmp_path)
    out = tmp_path / "improved.yaml"
    improved = MINIMAL_YAML.replace("Collect naturally.", "Collect warmly.")
    lines = ["round 1: incumbent 0.40 vs candidate 0.70 - accepted (1 edit)"]
    with patch.object(_cli, "_run_optimize", return_value=(improved, lines)) as m:
        rc = main(
            ["optimize", "--playbook", str(src), "--rounds", "1", "--out", str(out)]
        )
    assert rc == 0
    m.assert_called_once()
    assert out.read_text() == improved
    printed = capsys.readouterr().out
    assert "round 1" in printed and "accepted" in printed
    assert str(out) in printed


def test_optimize_default_out_is_improved_basename(tmp_path) -> None:
    src = _write(tmp_path)
    with patch.object(_cli, "_run_optimize", return_value=("y: 1\n", [])):
        rc = main(["optimize", "--playbook", str(src)])
    assert rc == 0
    assert (tmp_path / "improved.play.yaml").read_text() == "y: 1\n"


def test_optimize_missing_playbook_returns_1(capsys) -> None:
    rc = main(["optimize", "--playbook", "/nope.yaml"])
    assert rc == 1
    assert "/nope.yaml" in capsys.readouterr().err


def test_optimize_invalid_playbook_exits_clean(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("journeys: not-a-dict\n")
    with patch.object(_cli, "_run_optimize") as m:
        rc = main(["optimize", "--playbook", str(bad)])
    assert rc == 1
    m.assert_not_called()
    assert "Invalid playbook" in capsys.readouterr().err


def test_optimize_accepts_simple_format(tmp_path) -> None:
    src = tmp_path / "simple.yaml"
    src.write_text(SIMPLE)
    with patch.object(_cli, "_run_optimize", return_value=("y: 1\n", [])) as m:
        rc = main(["optimize", "--playbook", str(src)])
    assert rc == 0
    m.assert_called_once()


def test_package_exports() -> None:
    import superdialog.playbook as pb

    for name in (
        "optimize",
        "OptimizeReport",
        "ObjectiveBreakdown",
        "RoundTrace",
        "FullDoc",
        "SimpleDoc",
        "MutationError",
        "Edit",
        "make_editable",
        "load_personas",
        "generate_personas",
    ):
        assert hasattr(pb, name), name
        assert name in pb.__all__, name
