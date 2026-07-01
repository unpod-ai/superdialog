#!/usr/bin/env python3
"""One-command benchmark: Raw LLM vs With-SuperDialog, side by side.

Glue only — wires the benchmark module's pieces into a single run:

    load dataset -> run raw mode -> run with-SD mode -> print 2-column panel(Δ)

Both modes answer the SAME fixed dataset questions with the SAME system-under-
test model, so the panel's Δ column is the framework's measured lift.

Usage
-----
    # deterministic only (FREE — no LLM calls):
    python examples/run_benchmark.py kairali --no-ragas --dry

    # full run, one model (spends: SUT calls x scenarios x 2 modes + judges):
    python examples/run_benchmark.py kairali --model gpt-4o-mini

    # both models (a panel per model):
    python examples/run_benchmark.py kairali --both

Env: OPENAI_API_KEY and/or ANTHROPIC_API_KEY must be set for a live run.
Install: `uv sync --extra benchmark` inside superdialog/ first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from superdialog.benchmark.loader import load_named
from superdialog.benchmark.panel import render_big_table, render_panel
from superdialog.benchmark.ragas_scorer import DEFAULT_JUDGES
from superdialog.benchmark.report import build_report
from superdialog.benchmark.runner import run_raw_mode, run_sd_mode

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PROMPT = "/home/ankit/Downloads/FLow_Testing/Kairali.txt"

# friendly aliases -> litellm model strings
MODEL_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "claude-haiku": "anthropic/claude-haiku-4-5-20251001",
}

# the SUT set run by --both (3 models)
ALL_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "claude-haiku"]


def _resolve_playbook(samples, override: str | None) -> Path:
    if override:
        return Path(override)
    rel = next((s.playbook for s in samples if s.playbook), None)
    if not rel:
        raise SystemExit(
            "dataset has no playbook field — pass --playbook <path> explicitly"
        )
    return REPO_ROOT / rel


def _run_for_model(args, samples, model_str: str):
    """Run one model, return (rendered terminal panel, list[ModeReport])."""
    run_ragas = not args.no_ragas
    playbook = _resolve_playbook(samples, args.playbook)

    if args.dry:
        # no LLM — score the golden transcripts so the panel shape is visible
        golden = build_report("Golden (no run)", samples, run_ragas=False)
        reports = [golden]
        return (
            render_panel(reports, dataset=args.dataset, playbook=str(playbook.name)),
            reports,
        )

    reports = []
    if not args.sd_only:
        reports.append(
            run_raw_mode(
                samples,
                model_str,
                args.raw_prompt,
                label=f"Raw LLM ({model_str})",
                run_ragas=run_ragas,
            )
        )
    if not args.raw_only:
        reports.append(
            run_sd_mode(
                samples,
                model_str,
                playbook,
                label=f"With SuperDialog ({model_str})",
                run_ragas=run_ragas,
            )
        )
    # raw first, SD second -> Δ = SD - raw (framework lift). One mode -> one column.
    return (
        render_panel(reports, dataset=args.dataset, playbook=str(playbook.name)),
        reports,
    )


# Metrics deferred this run (embedding-based — need an embeddings model wired).
SKIPPED_METRICS = ("answer_correctness", "answer_relevancy")


def _build_report_doc(args, samples, reports_by_model: dict) -> str:
    """Assemble a shareable markdown report with aligned percent tables."""
    mode = "With SuperDialog (framework)" if args.sd_only else (
        "Raw LLM only" if args.raw_only else "Raw vs With-SuperDialog"
    )
    playbook = _resolve_playbook(samples, args.playbook).name
    # flatten every model's mode-reports into one ordered column list
    flat = [r for reports in reports_by_model.values() for r in reports]
    judge = None if args.no_ragas else DEFAULT_JUDGES[0]
    lines = [
        "# SuperDialog Benchmark Report",
        "",
        f"- **dataset:** {args.dataset} ({len(samples)} scenarios)",
        f"- **playbook:** {playbook}",
        f"- **mode:** {mode}",
        f"- **models:** {', '.join(reports_by_model.keys())}",
        f"- **metrics:** 4 deterministic + 7 RAGAS "
        f"({'RAGAS off' if args.no_ragas else 'RAGAS on'})",
        f"- **judge:** {judge or 'n/a (RAGAS off)'} (single fixed judge)",
        "- **scores:** whole-integer percent",
        "- **cost:** system-under-test tokens only (judge/eval LLM cost excluded)",
        "",
        "## Results",
        "",
        render_big_table(flat, dataset=args.dataset, playbook=playbook, judge=judge),
        "",
        "## Notes",
        "",
        "- `—` = metric not scored this run.",
        f"- Deferred metrics (need an embeddings model): "
        f"{', '.join(SKIPPED_METRICS)}. The other RAGAS metrics + all 4 "
        "deterministic metrics scored normally.",
        "- Cost counts only the benchmarked model's tokens, not the RAGAS "
        "judge LLM calls.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Raw LLM vs With-SuperDialog benchmark")
    ap.add_argument("dataset", nargs="?", default="kairali", help="dataset short name")
    ap.add_argument("--model", default="gpt-4o-mini", help="alias or litellm model id")
    ap.add_argument("--both", action="store_true", help="run gpt-4o-mini AND claude-haiku")
    ap.add_argument("--no-ragas", action="store_true", help="deterministic metrics only")
    ap.add_argument("--sd-only", action="store_true", help="run only with-SuperDialog")
    ap.add_argument("--raw-only", action="store_true", help="run only raw LLM")
    ap.add_argument("--dry", action="store_true", help="no LLM — score golden data only")
    ap.add_argument("--raw-prompt", default=DEFAULT_RAW_PROMPT, help="raw-LLM system prompt file")
    ap.add_argument("--playbook", default=None, help="playbook YAML (else from dataset)")
    ap.add_argument(
        "--out",
        nargs="?",
        const="__auto__",
        default=None,
        help="write a markdown report (default path if flag given with no value)",
    )
    args = ap.parse_args()

    samples = load_named(args.dataset)
    print(f"# dataset={args.dataset}  scenarios={len(samples)}  "
          f"ragas={'off' if args.no_ragas else 'on'}  dry={args.dry}\n")

    models = ALL_MODELS if args.both else [args.model]
    reports_by_model: dict = {}
    for m in models:
        model_str = MODEL_ALIASES.get(m, m)
        print(f"\n########## MODEL: {model_str} ##########\n")
        panel, reports = _run_for_model(args, samples, model_str)
        reports_by_model[model_str] = reports
        print(panel)

    if args.out is not None:
        out_path = (
            Path.cwd() / f"benchmark_{args.dataset}.md"
            if args.out == "__auto__"
            else Path(args.out)
        )
        doc = _build_report_doc(args, samples, reports_by_model)
        out_path.write_text(doc, encoding="utf-8")
        print(f"\n# report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
