"""Routing tests for the nested `eval` subcommand group."""

from superdialog.cli.main import _build_parser, _cmd_eval
from superdialog.eval.cli import cmd_gen_dataset, cmd_run


def test_eval_flow_routes_to_legacy_cmd_eval() -> None:
    """`eval flow --flow x.json` keeps routing to the legacy _cmd_eval."""
    args = _build_parser().parse_args(["eval", "flow", "--flow", "x.json"])
    assert args.fn is _cmd_eval
    assert args.flow == "x.json"


def test_eval_gen_dataset_routes_to_cmd_gen_dataset() -> None:
    """`eval gen-dataset --playbook p.yaml` routes to cmd_gen_dataset."""
    args = _build_parser().parse_args(["eval", "gen-dataset", "--playbook", "p.yaml"])
    assert args.fn is cmd_gen_dataset


def test_eval_run_routes_to_cmd_run() -> None:
    """`eval run` routes to cmd_run with required args parsed."""
    args = _build_parser().parse_args(
        ["eval", "run", "--playbook", "p.yaml", "--dataset", "d.yaml", "--out", "o"]
    )
    assert args.fn is cmd_run
