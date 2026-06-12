"""CLI tests for ``superdialog.cli.main``.

The ``chat`` subcommand drives a live LLM so we don't exercise it here;
``flow lint`` and ``flow draw`` are pure flow-file operations and can be
verified end-to-end against the bundled fixtures.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superdialog import Flow
from superdialog.cli.main import _build_parser, _lint_flow, main

_cli_main_module = importlib.import_module("superdialog.cli.main")

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "flow"


def test_flow_lint_clean_fixture(capsys: pytest.CaptureFixture) -> None:
    rc = main(["flow", "lint", str(FIXTURES / "kyc.json")])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "OK"


def test_flow_lint_reports_broken_edge(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    flow = {
        "id": "broken",
        "system_prompt": "test",
        "initial_node": "a",
        "nodes": [
            {
                "id": "a",
                "name": "A",
                "edges": [
                    {
                        "id": "to_nowhere",
                        "condition": "always",
                        "target_node_id": "missing",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(flow))

    rc = main(["flow", "lint", str(path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "unknown target 'missing'" in out


def test_flow_draw_emits_mermaid(capsys: pytest.CaptureFixture) -> None:
    rc = main(["flow", "draw", str(FIXTURES / "kyc.json")])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.strip().splitlines()
    assert lines[0] == "graph TD"
    assert any("-->" in line for line in lines[1:])


def test_parser_requires_subcommand(capsys: pytest.CaptureFixture) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_flow_draw_handles_appointment_fixture(
    capsys: pytest.CaptureFixture,
) -> None:
    rc = main(["flow", "draw", str(FIXTURES / "appointment.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "graph TD" in out


def _make_mock_flow(num_nodes=2):
    mock = MagicMock()
    mock.nodes = [MagicMock(edges=[MagicMock()]) for _ in range(num_nodes)]
    return mock


def test_flow_generate_saves_to_output_file(tmp_path, capsys):
    mock_flow = _make_mock_flow()
    out = tmp_path / "flow.json"

    with (
        patch.object(
            _cli_main_module, "create_dialog_flow", new_callable=MagicMock
        ) as mock_create,
        patch("asyncio.run", return_value=mock_flow),
    ):
        rc = main(["flow", "generate", "A booking agent", "--output", str(out)])

    assert rc == 0
    mock_create.assert_called_once()
    mock_flow.save.assert_called_once_with(str(out))
    assert "Saved" in capsys.readouterr().out


def test_flow_generate_from_file(tmp_path, capsys):
    desc = tmp_path / "desc.txt"
    desc.write_text("A golf tee-time booking agent")
    mock_flow = _make_mock_flow()
    out = tmp_path / "flow.json"

    with (
        patch.object(
            _cli_main_module, "create_dialog_flow", new_callable=MagicMock
        ) as mock_create,
        patch("asyncio.run", return_value=mock_flow),
    ):
        rc = main(["flow", "generate", "--from", str(desc), "--output", str(out)])

    assert rc == 0
    mock_create.assert_called_once()
    mock_flow.save.assert_called_once_with(str(out))


def test_flow_generate_from_file_not_found(capsys):
    rc = main(["flow", "generate", "--from", "/nonexistent/desc.txt"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_flow_generate_default_output_is_flow_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    mock_flow = _make_mock_flow()

    with (
        patch.object(_cli_main_module, "create_dialog_flow", new_callable=MagicMock),
        patch("asyncio.run", return_value=mock_flow),
    ):
        rc = main(["flow", "generate", "A simple agent"])

    assert rc == 0
    mock_flow.save.assert_called_once_with("flow.json")


def test_chat_flow_json_defaults_to_playbook_engine(tmp_path, monkeypatch):
    """Default mode runs flow JSON on the Playbook engine (auto-compiled)."""
    flow_data = {
        "id": "t",
        "system_prompt": "s",
        "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    (tmp_path / "flow.json").write_text(json.dumps(flow_data))
    monkeypatch.chdir(tmp_path)

    with (
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_legacy,
    ):
        rc = main(["chat"])

    assert rc == 0
    mock_play.assert_called_once()
    mock_legacy.assert_not_called()


def test_chat_explicit_flow_path_mode_flow(tmp_path):
    """--mode flow opts into the legacy DialogMachine engine."""
    flow_data = {
        "id": "t",
        "system_prompt": "s",
        "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    flow_file = tmp_path / "custom.json"
    flow_file.write_text(json.dumps(flow_data))

    with patch.object(_cli_main_module, "_run_chat_repl") as mock_repl:
        rc = main(["chat", "--flow", str(flow_file), "--mode", "flow"])

    assert rc == 0
    mock_repl.assert_called_once()
    call_args = mock_repl.call_args
    from superdialog import Flow

    assert isinstance(call_args[0][0], Flow)
    assert isinstance(call_args[0][1], str)


def test_chat_missing_flow_returns_1_with_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main(["chat"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "generate" in err.lower()


def _make_flow_file(tmp_path: Path, flow_data: dict) -> str:
    """Helper: write flow_data to a temp JSON file and return the path."""
    import json

    flow_file = tmp_path / "test_flow.json"
    flow_file.write_text(json.dumps(flow_data))
    return str(flow_file)


def test_lint_warns_when_criteria_key_not_in_any_edge_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Warn when a criteria key is never referenced in any edge input_schema."""
    flow_data = {
        "id": "test",
        "system_prompt": "test",
        "initial_node": "collect",
        "nodes": [
            {
                "id": "collect",
                "name": "Collect",
                "instruction": "Ask for email.",
                "completion_criteria": [
                    {"key": "email", "description": "User email", "required": True}
                ],
                "edges": [
                    {
                        "id": "phone_given",
                        "condition": "User gave phone number",
                        "target_node_id": "done",
                        "input_schema": {
                            "type": "object",
                            "properties": {"phone": {"type": "string"}},
                        },
                    }
                ],
            },
            {"id": "done", "name": "Done", "edges": []},
        ],
    }
    flow = Flow.load(_make_flow_file(tmp_path, flow_data))
    issues = _lint_flow(flow)
    assert any("email" in i and "criteria" in i.lower() for i in issues)


def test_lint_no_warn_when_criteria_key_in_edge_schema(
    tmp_path: Path,
) -> None:
    """No warning when required criteria key matches an edge input_schema property."""
    flow_data = {
        "id": "test",
        "system_prompt": "test",
        "initial_node": "collect",
        "nodes": [
            {
                "id": "collect",
                "name": "Collect",
                "instruction": "Ask for email.",
                "completion_criteria": [
                    {"key": "email", "description": "User email", "required": True}
                ],
                "edges": [
                    {
                        "id": "email_given",
                        "condition": "User gave email",
                        "target_node_id": "done",
                        "input_schema": {
                            "type": "object",
                            "properties": {"email": {"type": "string"}},
                        },
                    }
                ],
            },
            {"id": "done", "name": "Done", "edges": []},
        ],
    }
    flow = Flow.load(_make_flow_file(tmp_path, flow_data))
    issues = _lint_flow(flow)
    criteria_issues = [i for i in issues if "criteria" in i.lower() and "email" in i]
    assert criteria_issues == []


def test_lint_no_warn_for_optional_criteria_not_in_edge_schema(
    tmp_path: Path,
) -> None:
    """No warning for optional criteria keys not found in edge schemas."""
    flow_data = {
        "id": "test",
        "system_prompt": "test",
        "initial_node": "collect",
        "nodes": [
            {
                "id": "collect",
                "name": "Collect",
                "instruction": "Ask for email.",
                "completion_criteria": [
                    {"key": "email", "description": "User email", "required": False}
                ],
                "edges": [
                    {
                        "id": "done_edge",
                        "condition": "User is done",
                        "target_node_id": "done",
                        "input_schema": {
                            "type": "object",
                            "properties": {"other": {"type": "string"}},
                        },
                    }
                ],
            },
            {"id": "done", "name": "Done", "edges": []},
        ],
    }
    flow = Flow.load(_make_flow_file(tmp_path, flow_data))
    issues = _lint_flow(flow)
    criteria_issues = [i for i in issues if "criteria" in i.lower()]
    assert criteria_issues == []


def test_generate_runs_lint_and_prints_issues(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """flow generate should run lint and print any issues found."""
    from unittest.mock import MagicMock, patch

    broken_node = MagicMock()
    broken_node.id = "node_a"
    broken_node.name = "Node A"
    broken_node.completion_criteria = []
    broken_edge = MagicMock()
    broken_edge.id = "edge_1"
    broken_edge.target_node_id = "missing_node"
    broken_edge.input_schema = None
    broken_node.edges = [broken_edge]

    broken_flow = MagicMock()
    broken_flow.nodes = [broken_node]
    broken_flow.global_edges = []

    output_file = str(tmp_path / "flow.json")
    broken_flow.save = MagicMock()

    with (
        patch("superdialog.cli.main.create_dialog_flow", return_value=broken_flow),
        patch("superdialog.cli.main.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = broken_flow
        main(["flow", "generate", "test prompt", "--output", output_file])

    out = capsys.readouterr().out
    assert "Lint warnings" in out
    assert "warning: " in out
    assert "missing_node" in out
    assert "superdialog flow lint" in out


def test_generate_prints_lint_ok_when_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """flow generate should print 'Lint: OK' when no issues found."""
    from unittest.mock import MagicMock, patch

    clean_node = MagicMock()
    clean_node.id = "node_a"
    clean_node.name = "Node A"
    clean_node.completion_criteria = []
    clean_edge = MagicMock()
    clean_edge.id = "edge_1"
    clean_edge.target_node_id = "done"
    clean_edge.input_schema = None
    clean_node.edges = [clean_edge]

    done_node = MagicMock()
    done_node.id = "done"
    done_node.name = "Done"
    done_node.completion_criteria = []
    done_node.edges = []

    clean_flow = MagicMock()
    clean_flow.nodes = [clean_node, done_node]
    clean_flow.global_edges = []

    output_file = str(tmp_path / "flow.json")
    clean_flow.save = MagicMock()

    with (
        patch("superdialog.cli.main.create_dialog_flow", return_value=clean_flow),
        patch("superdialog.cli.main.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run.return_value = clean_flow
        main(["flow", "generate", "test prompt", "--output", output_file])

    out = capsys.readouterr().out
    assert "Lint: OK" in out


# -- playbook detection / dispatch -------------------------------------------

_MINIMAL_PLAYBOOK = """\
persona: "You are a tiny demo agent."
journeys:
  demo:
    checkpoints:
      - id: done
        terminal: true
        outcome: finished
"""


def _write_playbook(tmp_path: Path) -> Path:
    path = tmp_path / "play.yaml"
    path.write_text(_MINIMAL_PLAYBOOK)
    return path


def test_chat_detects_playbook(tmp_path: Path) -> None:
    """A --flow path whose content has top-level 'journeys' runs the playbook."""
    path = _write_playbook(tmp_path)

    with (
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--flow", str(path)])

    assert rc == 0
    mock_flow.assert_not_called()
    mock_play.assert_called_once()
    assert mock_play.call_args[0][0] == str(path)


def test_chat_explicit_playbook_flag(tmp_path: Path) -> None:
    """--playbook PATH runs the playbook REPL with that path."""
    path = _write_playbook(tmp_path)

    with (
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--playbook", str(path)])

    assert rc == 0
    mock_flow.assert_not_called()
    mock_play.assert_called_once()
    assert mock_play.call_args[0][0] == str(path)


def test_chat_mode_flow_still_runs_dialogmachine(tmp_path: Path) -> None:
    """--mode flow runs flow JSON on the legacy DialogMachine REPL."""
    flow_data = {
        "id": "t",
        "system_prompt": "s",
        "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    flow_file = tmp_path / "flow.json"
    flow_file.write_text(json.dumps(flow_data))

    with (
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--flow", str(flow_file), "--mode", "flow"])

    assert rc == 0
    mock_play.assert_not_called()
    mock_flow.assert_called_once()
    assert isinstance(mock_flow.call_args[0][0], Flow)


def test_chat_missing_playbook_file(
    capsys: pytest.CaptureFixture,
) -> None:
    """--playbook with a missing path exits 1 with a helpful message."""
    rc = main(["chat", "--playbook", "/nope.yaml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "/nope.yaml" in err


def test_chat_malformed_playbook_exits_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A file with 'journeys' but an invalid schema exits 1, no traceback."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("journeys: not-a-dict\n")  # detected as playbook, fails to load
    with patch.object(_cli_main_module, "_run_playbook_repl") as mock_play:
        rc = main(["chat", "--flow", str(bad)])
    assert rc == 1
    mock_play.assert_not_called()
    assert "Invalid playbook" in capsys.readouterr().err


def test_chat_malformed_flow_exits_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Malformed input exits 1 with a clean message, not a traceback."""
    bad = tmp_path / "f.json"
    bad.write_text("{not json")
    with patch.object(_cli_main_module, "_run_playbook_repl") as mock_play:
        rc = main(["chat", "--flow", str(bad)])
    assert rc == 1
    mock_play.assert_not_called()
    assert "Invalid playbook" in capsys.readouterr().err


def test_chat_malformed_flow_exits_clean_in_flow_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    bad = tmp_path / "f.json"
    bad.write_text("{not json")
    with patch.object(_cli_main_module, "_run_chat_repl") as mock_flow:
        rc = main(["chat", "--flow", str(bad), "--mode", "flow"])
    assert rc == 1
    mock_flow.assert_not_called()
    assert "Could not load flow" in capsys.readouterr().err


# -- simple playbook detection / dispatch ------------------------------------

_SIMPLE_PLAYBOOK = """\
name: "Tiny"
persona:
  identity: "You are a tiny demo agent."
playbook:
  - id: only
    purpose: "Say hi and stop."
    say: "Hi there!"
    done_when: "greeted"
"""


def _write_simple(tmp_path: Path) -> Path:
    path = tmp_path / "simple.yaml"
    path.write_text(_SIMPLE_PLAYBOOK)
    return path


def test_chat_detects_simple_playbook(tmp_path: Path) -> None:
    path = _write_simple(tmp_path)
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--flow", str(path)])
    assert rc == 0
    mock_flow.assert_not_called()
    mock_play.assert_not_called()
    mock_simple.assert_called_once()
    assert mock_simple.call_args[0][0] == str(path)


def test_chat_explicit_simple_flag(tmp_path: Path) -> None:
    path = _write_simple(tmp_path)
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--simple", str(path)])
    assert rc == 0
    mock_flow.assert_not_called()
    mock_simple.assert_called_once()
    assert mock_simple.call_args[0][0] == str(path)


def test_chat_journeys_wins_over_playbook_list(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(_MINIMAL_PLAYBOOK)
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
    ):
        rc = main(["chat", "--flow", str(path)])
    assert rc == 0
    mock_simple.assert_not_called()
    mock_play.assert_called_once()


def test_chat_missing_simple_file(capsys: pytest.CaptureFixture) -> None:
    rc = main(["chat", "--simple", "/nope-simple.yaml"])
    assert rc == 1
    assert "/nope-simple.yaml" in capsys.readouterr().err


def test_chat_malformed_simple_exits_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("playbook:\n  - not_a_step_mapping\n")
    with patch.object(_cli_main_module, "_run_simple_repl") as mock_simple:
        rc = main(["chat", "--flow", str(bad)])
    assert rc == 1
    mock_simple.assert_not_called()
    assert "Invalid simple playbook" in capsys.readouterr().err


def test_chat_flow_json_not_swallowed_by_simple_detection(tmp_path: Path) -> None:
    flow_data = {
        "id": "t",
        "system_prompt": "s",
        "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    flow_file = tmp_path / "flow.json"
    flow_file.write_text(json.dumps(flow_data))
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
    ):
        rc = main(["chat", "--flow", str(flow_file)])
    assert rc == 0
    mock_simple.assert_not_called()
    mock_play.assert_called_once()  # default: compiled onto the Playbook engine


def test_chat_playbook_flag_accepts_simple_format(tmp_path: Path) -> None:
    """The unified loader lets --playbook take a simple-format file."""
    from tests.playbook.test_simple import SIMPLE

    p = tmp_path / "simple.yaml"
    p.write_text(SIMPLE)
    with patch.object(_cli_main_module, "_run_playbook_repl") as mock_play:
        rc = main(["chat", "--playbook", str(p)])
    assert rc == 0
    mock_play.assert_called_once()
