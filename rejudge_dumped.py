#!/usr/bin/env python3
"""Re-judge already-dumped transcripts with a (possibly different) judge
model -- the conversation and probes are NEVER re-driven, only the scoring.

Reads the JSON files test_scenarios.py's _dump_case() writes (one per case,
under <playbook>.transcripts/ by default) and recomputes every metric:
code-only ones (task_success, pass_at_1, failure_recovery_rate,
conversational_consistency, slot_accuracy) for free, judge-only ones
(constraint_adherence, pii_violation_rate, guardrail, faithfulness,
topic_adherence, goal_accuracy) against --judge-model.

Caveat: slot_accuracy here is unscoped (checks ALL of
persona.ground_truth_slots, not just the playbook's required: true subset)
-- the required-slot set itself wasn't persisted in the dump.

Usage:
    python rejudge_dumped.py --dump-dir /path/to/playbook.transcripts \
        --judge-model anthropic/claude-opus-5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from superdialog.eval.dataset.models import EvalCase, Probe  # noqa: E402
from superdialog.eval.endpoints.base import Transcript  # noqa: E402
from superdialog.eval.metrics.base import MetricResult  # noqa: E402
from superdialog.eval.metrics.registry import build_suite  # noqa: E402
from superdialog.eval.runner import samples_from_run  # noqa: E402
from superdialog.llm.resolver import resolve_llm  # noqa: E402
from superdialog.playbook.eval.events import (  # noqa: E402
    assistant_utterances_by_checkpoint,
    slot_writes,
)
from superdialog.playbook.eval.models import PersonaSpec  # noqa: E402
from superdialog.playbook.eval.personas import required_slots  # noqa: E402
from superdialog.playbook.models import Playbook  # noqa: E402
from superdialog.playbook.eval.scenario_metrics import (  # noqa: E402
    constraint_adherence,
    constraint_adherence_scoped,
    conversational_consistency,
    evidence_gated_task_success,
    failure_recovery_rate,
    pass_at_1,
    pii_violation_rate,
)
from superdialog.playbook.providers import provider_adapters  # noqa: E402

_PROBE_METRICS = ["guardrail", "faithfulness", "topic_adherence", "goal_accuracy"]


def _judge(model_uri: str):
    director, _talker = provider_adapters(resolve_llm(model_uri))
    return director


def _slot_accuracy(expected: dict[str, Any], final_slots: dict[str, Any]):
    if not expected:
        return 1.0, {}
    diffs: dict[str, tuple[Any, Any]] = {}
    correct = 0
    for key, want in expected.items():
        got = final_slots.get(key)
        w = str(want).strip().lower()
        g = str(got).strip().lower() if got is not None else ""
        if got is not None and (w == g or w in g or g in w):
            correct += 1
        else:
            diffs[key] = (want, got)
    return correct / len(expected), diffs


async def _score_probes_from_dump(
    case: EvalCase, mode: str, transcript: Transcript,
    probe_pairs: list[tuple[Probe, str]], judge, judge_model: str,
) -> dict[str, MetricResult]:
    samples = samples_from_run(case, mode, transcript, probe_pairs)
    suite = build_suite(_PROBE_METRICS, judge, judge_model)
    sample_results = await asyncio.gather(*[suite.score(s) for s in samples])
    by_metric: dict[str, list[MetricResult]] = {}
    for results in sample_results:
        for res in results:
            if res.name not in _PROBE_METRICS:
                continue
            by_metric.setdefault(res.name, []).append(res)
    out: dict[str, MetricResult] = {}
    for name, results in by_metric.items():
        scored = [r.value for r in results if not r.skipped and not r.errored and r.value is not None]
        if scored:
            out[name] = MetricResult(
                name=name, value=sum(scored) / len(scored),
                reason=f"mean over {len(scored)}/{len(results)} sample(s)",
            )
        else:
            errored = any(r.errored for r in results)
            out[name] = MetricResult(
                name=name, value=None, skipped=not errored, errored=errored,
                reason="no scorable sample" if not errored else "all samples errored",
            )
    return out


async def _rejudge_one(
    path: Path, judge_model: str, required: set[str] | None,
) -> dict[str, dict[str, MetricResult]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    judge = _judge(judge_model)
    persona = PersonaSpec.model_validate(d["persona"])
    case = EvalCase(
        id=d["case_id"], playbook="dumped", persona=persona, probes=[],
        reference=persona.goal,
    )

    never_say: list[str] = d["never_say"]
    never_say_by_checkpoint: dict[str, list[str]] = d["never_say_by_checkpoint"]

    # ---- playbook side ----
    pb_events = d["playbook_events"]
    pb_utterances = [
        e["text"] for e in pb_events if e.get("type") == "utterance" and e.get("role") == "assistant"
    ]
    final_slots = {w["key"]: w["value"] for w in slot_writes(pb_events)}
    expected_slots = persona.ground_truth_slots
    if required is not None:
        expected_slots = {k: v for k, v in expected_slots.items() if k in required}
    slot_acc, slot_diffs = _slot_accuracy(expected_slots, final_slots)
    pb_pairs = assistant_utterances_by_checkpoint(pb_events)
    pb_conv = Transcript()
    for e in pb_events:
        if e.get("type") == "utterance":
            pb_conv.add(e["role"], e["text"])
    pb_probe_pairs = [
        (Probe.model_validate(item["probe"]), item["reply"]) for item in d["probes_playbook"]
    ]

    pb_constraint, pb_pii, pb_probes = await asyncio.gather(
        constraint_adherence_scoped(pb_pairs, never_say_by_checkpoint, judge),
        pii_violation_rate(pb_utterances, judge),
        _score_probes_from_dump(case, "playbook", pb_conv, pb_probe_pairs, judge, judge_model),
    )
    playbook_metrics: dict[str, MetricResult] = {
        "task_success": evidence_gated_task_success(True, pb_events),
        "pass_at_1": pass_at_1(True, pb_events),
        "failure_recovery_rate": failure_recovery_rate(pb_events),
        "conversational_consistency": conversational_consistency(pb_events),
        "slot_accuracy": MetricResult(
            name="slot_accuracy", value=slot_acc,
            reason=f"diffs: {slot_diffs}" if slot_diffs else "exact match",
        ),
        "constraint_adherence": pb_constraint,
        "pii_violation_rate": pb_pii,
        **pb_probes,
    }

    # ---- vanilla side ----
    va_transcript_raw = d["vanilla_transcript"]
    va_events = d["vanilla_events"]
    va_utterances = [t["text"] for t in va_transcript_raw if t["role"] == "assistant"]
    va_conv = Transcript()
    for t in va_transcript_raw:
        va_conv.add(t["role"], t["text"])
    va_probe_pairs = [
        (Probe.model_validate(item["probe"]), item["reply"]) for item in d["probes_vanilla"]
    ]

    va_constraint, va_pii, va_probes = await asyncio.gather(
        constraint_adherence(va_utterances, never_say, judge),
        pii_violation_rate(va_utterances, judge),
        _score_probes_from_dump(case, "vanilla", va_conv, va_probe_pairs, judge, judge_model),
    )
    vanilla_metrics: dict[str, MetricResult] = {
        "task_success": evidence_gated_task_success(True, va_events),
        "pass_at_1": pass_at_1(True, va_events),
        "failure_recovery_rate": failure_recovery_rate(va_events),
        "conversational_consistency": MetricResult(
            name="conversational_consistency", value=None, skipped=True,
            reason="vanilla has no structured slots to track",
        ),
        "slot_accuracy": MetricResult(
            name="slot_accuracy", value=None, skipped=True,
            reason="vanilla has no structured slots to grade",
        ),
        "constraint_adherence": va_constraint,
        "pii_violation_rate": va_pii,
        **va_probes,
    }

    return {"playbook": playbook_metrics, "vanilla": vanilla_metrics}


_AGGREGATE_METRICS = (
    "task_success", "pass_at_1", "failure_recovery_rate",
    "conversational_consistency", "slot_accuracy",
    "constraint_adherence", "pii_violation_rate", *_PROBE_METRICS,
)


def _print_case(case_id: str, results: dict[str, dict[str, MetricResult]]) -> None:
    print(f"\n{'=' * 74}\n SCENARIO — {case_id}\n{'=' * 74}")
    for mode, metrics in results.items():
        print(f"  [{mode}]")
        for name, m in metrics.items():
            status = "skip" if m.skipped else ("err" if m.errored else f"{m.value}")
            print(f"    {name:28s} {status}  {m.reason}")


def _print_aggregate(all_results: list[dict[str, dict[str, MetricResult]]]) -> None:
    print(f"\n{'=' * 74}\n AGGREGATE — {len(all_results)} case(s), playbook vs vanilla (RE-JUDGED)\n{'=' * 74}")
    for mode in ("playbook", "vanilla"):
        print(f"  [{mode}]")
        for name in _AGGREGATE_METRICS:
            values = [
                r[mode][name].value for r in all_results
                if name in r[mode] and r[mode][name].value is not None
            ]
            n_total = len(all_results)
            if not values:
                print(f"    {name:28s} n/a   (0/{n_total} scored)")
                continue
            avg = sum(values) / len(values)
            print(f"    {name:28s} {avg * 100:5.1f}%  ({len(values)}/{n_total} scored)")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Re-judge dumped eval transcripts")
    ap.add_argument("--dump-dir", required=True, help="directory of <case_id>.json dumps")
    ap.add_argument("--judge-model", default="anthropic/claude-opus-5")
    ap.add_argument(
        "--playbook", default=None,
        help="playbook YAML (same one the dump came from) -- when given, "
        "slot_accuracy scopes to required:true slots only, same as the "
        "live harness, instead of the full generated ground_truth_slots dict",
    )
    args = ap.parse_args()

    files = sorted(Path(args.dump_dir).glob("*.json"))
    if not files:
        print(f"no dumped cases found in {args.dump_dir}")
        return 1

    required = None
    if args.playbook:
        pb = Playbook.load(args.playbook)
        required = set(required_slots(pb))
        print(f"[rejudge] slot_accuracy scoped to {len(required)} required slot(s)")

    all_results = []
    for f in files:
        print(f"[rejudge] {f.stem} ...", flush=True)
        results = await _rejudge_one(f, args.judge_model, required)
        _print_case(f.stem, results)
        all_results.append(results)

    _print_aggregate(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
