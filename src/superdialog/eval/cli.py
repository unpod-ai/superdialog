"""CLI handlers for the eval A/B harness (wired into superdialog.cli.main)."""

from __future__ import annotations

import argparse
import os

import anyio

from superdialog.llm.resolver import resolve_llm
from superdialog.playbook.providers import provider_adapters


def _completer(uri: str):
    """Str-returning `.complete` completer (Director adapter) for a model URI.

    resolve_llm() yields a provider whose complete() returns a CompletionResult;
    the judges / personas / user-simulator need plain text, which the Director
    adapter provides.
    """
    director, _talker = provider_adapters(resolve_llm(uri))
    return director


def cmd_gen_dataset(args: argparse.Namespace) -> int:
    """`superdialog eval gen-dataset` — build <playbook>.evalcases.yaml offline."""
    from superdialog.eval.dataset.generate import build_dataset
    from superdialog.playbook.eval.personas import generate_personas, load_personas
    from superdialog.playbook.models import Playbook

    async def _run() -> None:
        pb = Playbook.load(args.playbook)
        gen = _completer(args.gen_model)
        personas = (
            load_personas(args.personas)
            if args.personas
            else await generate_personas(pb, gen)
        )
        ds = await build_dataset(pb, personas, gen, n_probes=args.n_probes)
        out = args.out or f"{os.path.splitext(args.playbook)[0]}.evalcases.yaml"
        ds.save(out)
        print(f"wrote {out} ({len(ds.cases)} cases)")

    anyio.run(_run)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """`superdialog eval run` — A/B evaluate a dataset and write a report."""
    from superdialog.eval.dataset.models import EvalCase, EvalDataset
    from superdialog.eval.endpoints.in_process import (
        InProcessPlaybook,
        InProcessVanilla,
    )
    from superdialog.eval.metrics.registry import build_suite
    from superdialog.eval.report import write_report
    from superdialog.eval.runner import run_ab

    async def _run() -> None:
        dataset = EvalDataset.load(args.dataset)
        with open(args.playbook, encoding="utf-8") as fh:
            playbook_text = fh.read()
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

        def _vanilla(case: EvalCase) -> InProcessVanilla:
            return InProcessVanilla(playbook_text, resolve_llm(args.agent_model))

        def _playbook(case: EvalCase) -> InProcessPlaybook:
            return InProcessPlaybook(
                args.playbook,
                agent_model=args.agent_model,
                director_model=args.director_model,
                talker_model=args.talker_model,
            )

        factories = {"vanilla": _vanilla, "playbook": _playbook}
        suite = build_suite(metrics, _completer(args.judge_model), args.judge_model)
        user_llm = _completer(args.user_model or args.agent_model)

        result = await run_ab(
            dataset,
            modes=modes,
            endpoint_factories={m: factories[m] for m in modes},
            suite=suite,
            user_llm=user_llm,
            metric_names=metrics,
            repeats=args.repeats,
        )
        json_path, md_path = write_report(result, args.out)
        print(f"wrote {json_path} and {md_path}")

    anyio.run(_run)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """`superdialog eval serve` — run the OpenAI-compatible server."""
    import uvicorn

    from superdialog.eval.server.openai_server import build_app

    app = build_app(
        args.playbook,
        agent_model=args.agent_model,
        director_model=getattr(args, "director_model", None),
        talker_model=getattr(args, "talker_model", None),
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    return 0
