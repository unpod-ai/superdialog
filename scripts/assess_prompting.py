"""North-star assessment: simple-format vs full-format playbook prompting.

Runs the same persona suite against both artifacts and reports the
Conversation Objective per arm. Protocol and north-star definition:
docs/plans/2026-06-12-prompting-assessment.md.

Run from the repo root:

    uv run python scripts/assess_prompting.py [--model openai/gpt-4o-mini]
        [--n 1] [--personas examples/playbooks/realestate_site_visit.personas.yaml]
        [--only-persona eager_buyer]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from superdialog.llm.resolver import resolve_llm
from superdialog.playbook import (
    Playbook,
    PlaybookAgent,
    httpx_http,
    load_simple,
    provider_adapters,
    run_eval,
)
from superdialog.playbook.optimize import ObjectiveBreakdown, score_report
from superdialog.playbook.personas import load_personas

ROOT = Path(__file__).parent.parent
SIMPLE_PATH = ROOT / "examples/playbooks/realestate_site_visit.simple.yaml"
FULL_PATH = ROOT / "examples/playbooks/realestate_site_visit.yaml"
ENRICHED_PATH = ROOT / "examples/playbooks/realestate_site_visit.enriched.yaml"
PERSONAS_PATH = ROOT / "examples/playbooks/realestate_site_visit.personas.yaml"
OUT_DIR = ROOT / "docs/plans/assessments"


def _structure(pb: Playbook) -> dict:
    """Checkpoint ids + slot keys — the structural identity of an artifact."""
    return {
        f"{jname}.{cp.id}": sorted(cp.slots)
        for jname, journey in pb.journeys.items()
        for cp in journey.checkpoints
    }


def _check_comparable(simple_pb: Playbook, full_pb: Playbook) -> None:
    """Print structural drift; identical structure makes the A/A claim honest."""
    s, f = _structure(simple_pb), _structure(full_pb)
    if s == f:
        print("structure: IDENTICAL (checkpoints + slot keys) — A/A null test")
        return
    print("structure: DIFFERS — this run is an A/B, not a null test")
    for key in sorted(set(s) | set(f)):
        if s.get(key) != f.get(key):
            print(f"  {key}: simple={s.get(key)} full={f.get(key)}")


def _fmt(b: ObjectiveBreakdown) -> str:
    return (
        f"CO={b.objective:.3f}  completion={b.completion_rate:.2f}  "
        f"slots={b.slot_accuracy:.2f}  turns/cp={b.mean_turns_per_checkpoint:.1f}  "
        f"repairs={b.repair_rate:.2f}"
    )


async def _assess(
    model: str, n: int, personas_path: str, only: str | None, enriched: bool
) -> dict:
    personas = load_personas(personas_path)
    if only:
        personas = [p for p in personas if p.name == only]
        if not personas:
            raise SystemExit(f"no persona named {only!r} in {personas_path}")

    arms = {
        "simple": load_simple(str(SIMPLE_PATH)),
        "full": Playbook.load(str(FULL_PATH)),
    }
    _check_comparable(arms["simple"], arms["full"])
    if enriched:
        arms["enriched"] = Playbook.load(str(ENRICHED_PATH))
        print(
            "enriched arm: structure intentionally differs "
            "(interrupts, fallback checkpoint, silence policy) — Phase 2 A/B"
        )

    director, talker = provider_adapters(resolve_llm(model))
    results: dict = {}
    for arm_name, playbook in arms.items():
        print(f"\n=== arm: {arm_name} — {len(personas)} persona(s) × n={n} ===")
        t0 = time.monotonic()
        report = await run_eval(
            lambda: PlaybookAgent(
                playbook=playbook,
                talker_llm=talker,
                director_llm=director,
                http=httpx_http,
            ),
            personas,
            director,  # the provider adapter satisfies SpeaksUser
            n,
        )
        elapsed = time.monotonic() - t0
        breakdown = score_report(report)
        print(f"{arm_name}: {_fmt(breakdown)}  [{elapsed:.0f}s]")
        for s in report.sessions:
            print(
                f"  {s.persona}: completed={s.completed} outcome={s.outcome} "
                f"turns={s.turns} slot_acc={s.slot_accuracy:.2f} "
                f"repairs={s.repair_count} diffs={dict(list(s.slot_diffs.items())[:4])}"
            )
        results[arm_name] = {
            "breakdown": breakdown.model_dump(),
            "elapsed_s": round(elapsed, 1),
            "sessions": [
                s.model_dump(exclude={"event_log_jsonl"}) for s in report.sessions
            ],
            "event_logs": {
                f"{s.persona}#{i}": s.event_log_jsonl
                for i, s in enumerate(report.sessions)
            },
        }

    objectives = {a: r["breakdown"]["objective"] for a, r in results.items()}
    print()
    names = list(objectives)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            print(f"ΔCO ({a} − {b}) = {objectives[a] - objectives[b]:+.3f}")
    if not enriched:
        print(
            "Interpretation: with identical structure this gap estimates the "
            "eval noise floor (see plan §2)."
        )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "n": n,
        "personas": [p.name for p in personas],
        "objectives": {a: round(v, 4) for a, v in objectives.items()},
        "arms": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--personas", default=str(PERSONAS_PATH))
    parser.add_argument("--only-persona", default=None)
    parser.add_argument(
        "--enriched",
        action="store_true",
        help="Add the Phase-2 structure-enriched arm (see plan §2)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    record = asyncio.run(
        _assess(args.model, args.n, args.personas, args.only_persona, args.enriched)
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{record['timestamp'].replace(':', '-')}.json"
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nrecord: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
