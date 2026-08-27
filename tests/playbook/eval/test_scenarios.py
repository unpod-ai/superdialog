# tests/playbook/eval/test_scenarios.py
"""Auto-generating scenario eval: any playbook, no hand-written scenarios.

Usage:
    export SCENARIO_EVAL_PLAYBOOK=/path/to/playbook.yaml
    .venv/bin/pytest -m integration tests/playbook/eval/test_scenarios.py -s

Generates (or reuses, if <playbook>.evalcases.yaml already exists)
personas + probes from the playbook itself via the same machinery
`superdialog eval gen-dataset` uses -- no scenario is ever hand-written.

Scores 7 of 9 metrics per case (task_success, pass_at_1,
failure_recovery_rate, conversational_consistency, slot_accuracy,
constraint_adherence, pii_violation_rate). guardrail and efficiency run
separately via `superdialog eval bench` (different session driver) --
see bench_scenarios.py for why they aren't merged into this report.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path

import anyio
import pytest

from superdialog.agents.llm_agent import LLMAgent
from superdialog.eval.dataset.generate import build_dataset
from superdialog.eval.dataset.models import EvalDataset
from superdialog.eval.endpoints.base import Transcript
from superdialog.eval.endpoints.in_process import InProcessPlaybook, InProcessVanilla
from superdialog.eval.metrics.base import MetricResult
from superdialog.eval.metrics.registry import build_suite
from superdialog.eval.runner import samples_from_run
from superdialog.llm.resolver import resolve_llm
from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.events import SlotWriteEvent
from superdialog.playbook.eval.bench_scenarios import score_case_both_modes
from superdialog.playbook.eval.events import parse_events
from superdialog.playbook.eval.personas import generate_personas, required_slots
from superdialog.playbook.eval.runner import run_session
from superdialog.playbook.eval.vanilla import (
    run_vanilla_session,
    tool_call_guidance,
    tool_schemas,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.providers import provider_adapters
from superdialog.playbook.toolexec import httpx_http

# guardrail/faithfulness/topic_adherence/goal_accuracy: judge-only metrics
# (no evidence to gate on), scored via the existing eval-bench machinery
# (probes + MetricSuite) reused here so they land in the SAME report instead
# of requiring a separate `superdialog eval bench` invocation. faithfulness
# needs `uv sync --extra ragas`.
_PROBE_METRICS = ["guardrail", "faithfulness", "topic_adherence", "goal_accuracy"]

pytestmark = pytest.mark.integration

PB_PATH = Path(
    os.environ.get(
        "SCENARIO_EVAL_PLAYBOOK",
        os.path.expanduser("~/Downloads/flow_golf_ai_updated_full.v2.yaml"),
    )
)

# Raw per-case data (transcripts + probe replies + the never_say lists used)
# gets dumped here so a run can be RE-JUDGED later with a different judge
# model without re-driving the actual conversation/probes -- the conversation
# generation is the expensive/slow part; the judging is cheap and swappable.
# Previously nothing was persisted at all, so a judge-model comparison always
# meant a full re-run. Set SCENARIO_EVAL_DUMP_DIR="" to disable.
_DUMP_DIR_RAW = os.environ.get(
    "SCENARIO_EVAL_DUMP_DIR", f"{os.path.splitext(str(PB_PATH))[0]}.transcripts"
)
_DUMP_DIR = Path(_DUMP_DIR_RAW) if _DUMP_DIR_RAW else None
# Persona/dataset generator + LLM judge: strongest available, per the design
# spec's LLM-roles table (one-time/bounded-volume cost, quality gates every
# downstream score).
_GEN_MODEL = os.environ.get("SCENARIO_EVAL_GEN_MODEL", "anthropic/claude-opus-5")
_JUDGE_MODEL = os.environ.get("SCENARIO_EVAL_JUDGE_MODEL", "anthropic/claude-opus-5")
# Simulated caller: highest call volume in the pipeline, per the design spec
# -- fast/cheap, proven in this exact role by golfai's own test.
_CALLER_MODEL = os.environ.get("SCENARIO_EVAL_CALLER_MODEL", "openai/gpt-4.1-mini")
_N_PROBES = int(os.environ.get("SCENARIO_EVAL_N_PROBES", "5"))
# PersonaSpec.max_turns defaults to 12 (models.py) -- too few for a deep
# playbook (westgate has 32 checkpoints; sessions were hitting 12/12 turns
# every time, still mid-flow, never reaching the terminal checkpoint that
# fires the tool call task_success/pass_at_1 gate on). Unset by default so
# existing runs/datasets are unaffected; set to raise the budget.
_MAX_TURNS = os.environ.get("SCENARIO_EVAL_MAX_TURNS")
# Seeds slots.phone (confirmed, by=compiler) before the playbook-mode session
# starts -- mirrors production (the handler seeds phone the same way before
# the flow runs) and agent_playbook_traced.py's TRACE_PHONE. Playbooks whose
# greeting on_enter looks up the caller by phone (golfai) need a REAL,
# registered number or every persona 404s on players-search regardless of
# what the generated dataset's ground_truth phone_number says -- the mock
# API only has one registered player. Empty by default (unset) so playbooks
# that don't need it (e.g. westgate) are unaffected. Vanilla mode has no
# slot/tool framework at all, so this only applies to the playbook side.
_PHONE = os.environ.get("SCENARIO_EVAL_PHONE", "")


def _skip_if_missing() -> None:
    if not PB_PATH.exists():
        pytest.skip(f"playbook not found: {PB_PATH}")


def _dataset_path() -> Path:
    stem = os.path.splitext(str(PB_PATH))[0]
    return Path(f"{stem}.evalcases.yaml")


def _completer(model_uri: str):
    director, _talker = provider_adapters(resolve_llm(model_uri))
    return director


async def _load_or_generate_dataset() -> EvalDataset:
    """Reuse superdialog eval gen-dataset's own logic -- no new generator."""
    ds_path = _dataset_path()
    if ds_path.exists():
        return EvalDataset.load(str(ds_path))
    pb = Playbook.load(str(PB_PATH))
    gen = _completer(_GEN_MODEL)
    personas = await generate_personas(pb, gen)
    dataset = await build_dataset(pb, personas, gen, n_probes=_N_PROBES)
    dataset.save(str(ds_path))
    return dataset


async def _playbook_agent() -> PlaybookAgent:
    """Director/Talker come from the playbook's own llm: block -- never
    overridden, per the design spec's LLM-roles table. Each resolved
    provider is wrapped through provider_adapters() before use --
    PlaybookAgent needs the StreamsLLM/CompletesLLM adapters, not the raw
    provider resolve_llm_providers() returns (passing the raw provider
    crashes the Talker's stream filter pipeline with a StreamChunk/str
    type mismatch on every turn)."""
    pb = Playbook.load(str(PB_PATH))
    talker_provider, director_provider = await pb.resolve_llm_providers()
    _, talker_adapter = provider_adapters(talker_provider)
    director_adapter, _ = provider_adapters(director_provider)
    return PlaybookAgent(
        playbook=pb, talker_llm=talker_adapter, director_llm=director_adapter, http=httpx_http,
    )


async def _vanilla_agent() -> LLMAgent:
    """Mirrors the playbook's OWN declared Talker model -- A/B validity
    requires isolating structure vs. no-structure, not model vs. model.

    Also gets the playbook's own tools as native function-calling schemas
    (see vanilla.tool_schemas) -- vanilla is "no Director/checkpoints/slots",
    not "no tools"; without this it could never produce real tool-call
    evidence regardless of what it says, which isn't a fair comparison.
    """
    pb = Playbook.load(str(PB_PATH))
    model_uri = pb.llm_uri() or _CALLER_MODEL
    text = PB_PATH.read_text(encoding="utf-8") + tool_call_guidance(pb.tools)
    opts = {"tools": tool_schemas(pb.tools)} if pb.tools else {}
    return LLMAgent(resolve_llm(model_uri), system_prompt=text, **opts)


def _never_say(pb: Playbook) -> list[str]:
    """Collect every never_say string declared across the playbook's
    checkpoints, for constraint_adherence."""
    out: list[str] = []
    for journey in pb.journeys.values():
        for cp in journey.checkpoints:
            out.extend(getattr(cp, "never_say", None) or [])
    return out


def _never_say_by_checkpoint(pb: Playbook) -> dict[str, list[str]]:
    """Per-checkpoint never_say map, for constraint_adherence_scoped.

    Keyed by the SAME ``to_checkpoint`` id an AdvanceEvent records --
    journey-qualified (``"main.xxx"``), matching ``Playbook.initial_checkpoint_id``
    / ``checkpoint_ids()``. A bare-id key here previously never matched a
    qualified ``to_checkpoint``, so ``constraint_adherence_scoped`` silently
    fell back to "no visited checkpoint declares never_say constraints" on
    every playbook run -- it was never actually judging anything.
    """
    out: dict[str, list[str]] = {}
    for jname, journey in pb.journeys.items():
        for cp in journey.checkpoints:
            rules = getattr(cp, "never_say", None) or []
            if rules:
                out[f"{jname}.{cp.id}"] = list(rules)
    return out


def _checkpoint_graph(pb: Playbook) -> tuple[set[str], set[tuple[str, str]]]:
    """(interrupt_checkpoints, designed_edges) for pass_at_1's revisit check.

    Both are keyed journey-qualified (``"main.xxx"``), same as ``to_checkpoint``
    on an AdvanceEvent and each ``AdvanceRule.to``/``InterruptSpec.to`` as
    authored in the YAML.

    interrupt_checkpoints = the ``to`` targets of top-level
    ``Playbook.interrupts`` entries with ``resume: true`` -- the playbook's
    OWN "answer this off-flow, then restore whatever was in progress"
    mechanism (``InterruptSpec.resume``). This is the actual ground-truth
    signal, NOT a proxy: an earlier version of this function guessed
    "checkpoint has no advance_when" as a stand-in and got 2/4 westgate
    cases wrong (e.g. main.answer_direct_questions is a resume:true
    interrupt target but also has its own advance_when for when it's
    entered as a normal flow step -- the empty-advance_when guess missed it
    entirely). Checkpoints with resume:false (e.g. global_goodbye ->
    deliver_closing) are real one-way transitions, not revisit-exempt.

    designed_edges = every checkpoint's own declared advance_when targets
    (e.g. announce_starting_price -> present_pricing, a two-step
    announce/react hub authored as a single checkpoint, not a retry loop).
    """
    interrupts = {itr.to for itr in pb.interrupts if itr.resume}
    edges: set[tuple[str, str]] = set()
    for jname, journey in pb.journeys.items():
        for cp in journey.checkpoints:
            qid = f"{jname}.{cp.id}"
            for rule in (cp.advance_when or []):
                edges.add((qid, rule.to))
    return interrupts, edges


def _run(coro):
    return anyio.run(lambda: coro)


# Accumulates every scored case this session, for the aggregate summary
# pytest_sessionfinish prints once at the end -- per-scenario prints alone
# don't answer "what's the overall playbook vs vanilla score."
_ALL_RESULTS: list[dict] = []

_AGGREGATE_METRICS = (
    "task_success", "pass_at_1", "failure_recovery_rate",
    "conversational_consistency", "slot_accuracy",
    "constraint_adherence", "pii_violation_rate",
    *_PROBE_METRICS,
)


def _print_models_once() -> None:
    pb = Playbook.load(str(PB_PATH))
    director = pb.director_llm_uri() or pb.llm_uri()
    talker = pb.llm_uri()
    print(f"\n{'=' * 74}\n MODELS — {PB_PATH.name}\n{'=' * 74}")
    print(f"  playbook director : {director}")
    print(f"  playbook talker    : {talker}")
    print(f"  vanilla (= talker) : {talker or _CALLER_MODEL}")
    print(f"  caller simulator   : {_CALLER_MODEL}")
    print(f"  judge / gen model  : {_JUDGE_MODEL} / {_GEN_MODEL}")


def pytest_generate_tests(metafunc):
    """Parametrize test_scenario by the auto-generated dataset's cases -- the
    ONLY mechanism used to drive per-case tests in this file."""
    if "case" not in metafunc.fixturenames:
        return
    _skip_if_missing()
    _print_models_once()
    ds = _run(_load_or_generate_dataset())
    if _MAX_TURNS:
        n = int(_MAX_TURNS)
        ds = ds.model_copy(update={
            "cases": [
                c.model_copy(update={"persona": c.persona.model_copy(update={"max_turns": n})})
                for c in ds.cases
            ]
        })
        print(f"  max_turns override : {n} (was persona-default 12)")
    metafunc.parametrize("case", ds.cases, ids=[c.id for c in ds.cases])


def _transcript_from_utterances(entries: list[tuple[str, str]]) -> Transcript:
    """Build an eval.endpoints.base.Transcript from (role, text) pairs we
    already have from our OWN persona session -- avoids re-driving the
    conversation a second time just to get a "conversation" sample."""
    t = Transcript()
    for role, text in entries:
        t.add(role, text)
    return t


def _probe_endpoint(pb: Playbook, mode: str, director: str, talker: str):
    """A fresh ConversationEndpoint -- each probe gets its own instance so
    they can run concurrently (run_probes resets one shared instance
    between probes, which forces them sequential; we don't reuse it)."""
    if mode == "playbook":
        return InProcessPlaybook(
            str(PB_PATH), agent_model=talker, director_model=director, talker_model=talker,
        )
    return InProcessVanilla(PB_PATH.read_text(encoding="utf-8"), talker)


async def _run_one_probe(probe, pb: Playbook, mode: str, director: str, talker: str):
    endpoint = _probe_endpoint(pb, mode, director, talker)
    await endpoint.start()
    reply = await endpoint.turn(probe.utterance)
    return probe, reply


async def _score_probe_metrics(
    case,
    pb: Playbook,
    mode: str,
    transcript: Transcript,
    judge: object,
) -> tuple[dict[str, MetricResult], list[tuple[object, str]]]:
    """guardrail/faithfulness/topic_adherence/goal_accuracy -- judge-only
    metrics with no evidence to gate on, so they reuse the existing
    eval-bench sample/suite machinery, but run every probe and every
    sample-score concurrently (the shared run_probes()/sequential-loop
    versions were a real bottleneck: ~20+ sequential judge round-trips per
    mode against a rate-limited model). The "conversation" sample reuses OUR
    already-run persona session via ``transcript``, so it's never re-driven.

    Returns (aggregated metrics, raw probe_results) -- the caller persists
    the raw pairs so a re-judge later doesn't need to re-run the probes.
    """
    director = pb.director_llm_uri() or pb.llm_uri() or _CALLER_MODEL
    talker = pb.llm_uri() or _CALLER_MODEL
    probe_results = list(await asyncio.gather(
        *[_run_one_probe(p, pb, mode, director, talker) for p in case.probes]
    ))
    samples = samples_from_run(case, mode, transcript, probe_results)
    suite = build_suite(_PROBE_METRICS, judge, _JUDGE_MODEL)
    sample_results = await asyncio.gather(*[suite.score(sample) for sample in samples])
    by_metric: dict[str, list[MetricResult]] = {}
    for results in sample_results:
        for res in results:
            # build_suite always appends efficiency/token_cost (the "free"
            # metrics) regardless of what was asked for -- drop them here,
            # they aren't part of what this round wired in.
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
    return out, probe_results


def _dump_case(
    case,
    never_say: list[str],
    never_say_by_checkpoint: dict[str, list[str]],
    pb_events: list[dict],
    vanilla_metrics,
    pb_probe_results: list[tuple],
    va_probe_results: list[tuple],
    results: dict,
) -> None:
    """Persist everything a re-judge needs -- transcripts + probe replies +
    the never_say lists actually used -- so a later run can score this SAME
    conversation with a different judge model without re-driving it.

    Also persists the ALREADY-COMPUTED metrics (``results``) -- this is the
    cross-process-safe source of truth conftest.py's aggregate reads from.
    Under pytest-xdist each persona runs in its own worker process, so the
    in-memory ``_ALL_RESULTS`` list only ever holds that one worker's
    subset; the dump directory is a real shared filesystem all workers
    write into, so reading it back is correct regardless of worker count.
    """
    if _DUMP_DIR is None:
        return
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case.id,
        "persona": case.persona.model_dump(),
        "never_say": never_say,
        "never_say_by_checkpoint": never_say_by_checkpoint,
        "playbook_events": pb_events,
        "vanilla_transcript": [t.model_dump() for t in vanilla_metrics.transcript],
        "vanilla_events": vanilla_metrics.events,
        "probes_playbook": [
            {"probe": p.model_dump(), "reply": reply} for p, reply in pb_probe_results
        ],
        "probes_vanilla": [
            {"probe": p.model_dump(), "reply": reply} for p, reply in va_probe_results
        ],
        "metrics": {
            mode: {name: dataclasses.asdict(m) for name, m in result.metrics.items()}
            for mode, result in results.items()
        },
    }
    out_path = _DUMP_DIR / f"{case.id}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def test_scenario(case) -> None:
    """One case, both modes, 7 of 9 metrics -- printed and asserted.

    guardrail/efficiency are NOT computed here -- they run on a separate
    session driver via `superdialog eval bench` (see bench_scenarios.py's
    module docstring for why). Run that command too for full 9-metric
    coverage on the same playbook.
    """
    pb = Playbook.load(str(PB_PATH))
    never_say = _never_say(pb)
    never_say_by_checkpoint = _never_say_by_checkpoint(pb)
    interrupt_checkpoints, designed_edges = _checkpoint_graph(pb)
    judge = _completer(_JUDGE_MODEL)
    caller = _completer(_CALLER_MODEL)

    agent = await _playbook_agent()
    if _PHONE:
        # Append BEFORE run_session()'s runtime.start(): golfai's greeting
        # on_enter (action-players-search) fires synchronously inside
        # start() itself, keyed on slots.phone -- seeding after start()
        # would be too late and the lookup would run with an empty phone.
        agent.runtime.log.append(
            SlotWriteEvent(key="phone", value=_PHONE, status="confirmed", by="compiler")
        )
    vanilla = await _vanilla_agent()
    required = set(required_slots(pb))
    playbook_metrics, vanilla_metrics = await asyncio.gather(
        run_session(agent, case.persona, caller, required_slots=required),
        run_vanilla_session(vanilla, case.persona, caller, tools=pb.tools),
    )

    results = await score_case_both_modes(
        case=case,
        playbook_metrics=playbook_metrics,
        vanilla_metrics=vanilla_metrics,
        never_say=never_say,
        judge_llm=judge,
        never_say_by_checkpoint=never_say_by_checkpoint,
        initial_checkpoint=pb.initial_checkpoint_id,
        interrupt_checkpoints=interrupt_checkpoints,
        designed_edges=designed_edges,
    )

    pb_events = parse_events(playbook_metrics.event_log_jsonl)
    pb_transcript = _transcript_from_utterances(
        [(e["role"], e["text"]) for e in pb_events if e.get("type") == "utterance"]
    )
    va_transcript = _transcript_from_utterances(
        [(t.role, t.text) for t in vanilla_metrics.transcript]
    )
    (probe_playbook, pb_probe_results), (probe_vanilla, va_probe_results) = await asyncio.gather(
        _score_probe_metrics(case, pb, "playbook", pb_transcript, judge),
        _score_probe_metrics(case, pb, "vanilla", va_transcript, judge),
    )
    results["playbook"].metrics.update(probe_playbook)
    results["vanilla"].metrics.update(probe_vanilla)

    _ALL_RESULTS.append(results)
    _dump_case(
        case, never_say, never_say_by_checkpoint,
        pb_events, vanilla_metrics, pb_probe_results, va_probe_results,
        results,
    )

    print(f"\n{'=' * 74}\n SCENARIO — {case.id} ({case.persona.name})\n{'=' * 74}")
    for mode, result in results.items():
        print(f"  [{mode}]")
        for name, m in result.metrics.items():
            status = "skip" if m.skipped else ("err" if m.errored else f"{m.value}")
            print(f"    {name:28s} {status}  {m.reason}")

    # Hard assertions -- a playbook run must never fabricate success, and
    # must never leak PII/violate a declared constraint, regardless of the
    # persona's goal being achieved.
    pb_result = results["playbook"]
    assert pb_result.metrics["pii_violation_rate"].value == 0.0, (
        f"PII LEAK: {pb_result.metrics['pii_violation_rate'].reason}"
    )
    assert pb_result.metrics["constraint_adherence"].value == 1.0, (
        f"CONSTRAINT VIOLATED: {pb_result.metrics['constraint_adherence'].reason}"
    )
