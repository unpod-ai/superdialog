"""CLI handlers for the eval A/B harness (wired into superdialog.cli.main)."""

from __future__ import annotations

import argparse
import os

import anyio
from dotenv import load_dotenv

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
    load_dotenv()
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
    load_dotenv()
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


def cmd_bench(args: argparse.Namespace) -> int:
    """`superdialog eval bench` — one shot: gen the dataset (if missing) then A/B
    evaluate every ``--models`` entry into ``<out>/<model>/report.{json,md}``."""
    load_dotenv()
    from superdialog.eval.dataset.generate import build_dataset
    from superdialog.eval.dataset.models import EvalCase, EvalDataset
    from superdialog.eval.endpoints.in_process import (
        InProcessPlaybook,
        InProcessVanilla,
    )
    from superdialog.eval.metrics.registry import build_suite
    from superdialog.eval.report import render_combined, write_combined, write_report
    from superdialog.eval.runner import run_ab
    from superdialog.playbook.eval.personas import generate_personas, load_personas
    from superdialog.playbook.models import Playbook

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    dataset_path = (
        args.dataset or f"{os.path.splitext(args.playbook)[0]}.evalcases.yaml"
    )

    async def _run() -> None:
        # 1) Dataset: reuse an existing one unless --regen (or it's missing).
        if args.regen or not os.path.exists(dataset_path):
            pb = Playbook.load(args.playbook)
            gen = _completer(args.gen_model)
            personas = (
                load_personas(args.personas)
                if args.personas
                else await generate_personas(pb, gen)
            )
            ds = await build_dataset(pb, personas, gen, n_probes=args.n_probes)
            ds.save(dataset_path)
            print(f"[gen] wrote {dataset_path} ({len(ds.cases)} cases)")
        else:
            print(f"[gen] reusing {dataset_path} (pass --regen to rebuild)")

        # 2) A/B every model into its own report dir.
        dataset = EvalDataset.load(dataset_path)
        # Override each persona's turn budget so B's longer capture flow can
        # finish (default is short; the playbook asks one slot per turn).
        if args.max_turns:
            dataset = dataset.model_copy(
                update={
                    "cases": [
                        c.model_copy(
                            update={
                                "persona": c.persona.model_copy(
                                    update={"max_turns": args.max_turns}
                                )
                            }
                        )
                        for c in dataset.cases
                    ]
                }
            )
        with open(args.playbook, encoding="utf-8") as fh:
            playbook_text = fh.read()

        for model in models:

            def _vanilla(case: EvalCase, _m: str = model) -> InProcessVanilla:
                return InProcessVanilla(playbook_text, resolve_llm(_m))

            def _playbook(case: EvalCase, _m: str = model) -> InProcessPlaybook:
                # director/talker default to the model unless overridden — lets a
                # strong Director drive B while a cheap Talker speaks.
                return InProcessPlaybook(
                    args.playbook,
                    agent_model=_m,
                    director_model=args.director_model,
                    talker_model=args.talker_model,
                )

            factories = {"vanilla": _vanilla, "playbook": _playbook}
            suite = build_suite(metrics, _completer(args.judge_model), args.judge_model)
            out_dir = os.path.join(args.out, model.replace("/", "_"))
            print(f"[run] {model} -> {out_dir}")
            result = await run_ab(
                dataset,
                modes=modes,
                endpoint_factories={m: factories[m] for m in modes},
                suite=suite,
                user_llm=_completer(args.user_model or model),
                metric_names=metrics,
                repeats=args.repeats,
            )
            write_report(result, out_dir)  # per-model report.json (detail)
            entries.append((model, result))

        # Single combined report: one big table (every metric × every model/mode),
        # written once at the out root and printed to the terminal.
        combined_path = write_combined(entries, args.out)
        md = render_combined(entries)
        print("\n" + md + "\n")
        print(f"[report] combined single-file report -> {combined_path}")

    entries: list = []
    anyio.run(_run)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """`superdialog eval serve` — run the OpenAI-compatible server."""
    load_dotenv()
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
