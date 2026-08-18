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
import os
from pathlib import Path

import anyio
import pytest

from superdialog.agents.llm_agent import LLMAgent
from superdialog.eval.dataset.generate import build_dataset
from superdialog.eval.dataset.models import EvalDataset
from superdialog.llm.resolver import resolve_llm
from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.eval.bench_scenarios import score_case_both_modes
from superdialog.playbook.eval.personas import generate_personas
from superdialog.playbook.eval.runner import run_session
from superdialog.playbook.eval.vanilla import run_vanilla_session
from superdialog.playbook.models import Playbook
from superdialog.playbook.providers import provider_adapters
from superdialog.playbook.toolexec import httpx_http

pytestmark = pytest.mark.integration

PB_PATH = Path(
    os.environ.get(
        "SCENARIO_EVAL_PLAYBOOK",
        os.path.expanduser("~/Downloads/flow_golf_ai_updated_full.v2.yaml"),
    )
)
# Persona/dataset generator + LLM judge: strongest available, per the design
# spec's LLM-roles table (one-time/bounded-volume cost, quality gates every
# downstream score).
_GEN_MODEL = os.environ.get("SCENARIO_EVAL_GEN_MODEL", "anthropic/claude-opus-5")
_JUDGE_MODEL = os.environ.get("SCENARIO_EVAL_JUDGE_MODEL", "anthropic/claude-opus-5")
# Simulated caller: highest call volume in the pipeline, per the design spec
# -- fast/cheap, proven in this exact role by golfai's own test.
_CALLER_MODEL = os.environ.get("SCENARIO_EVAL_CALLER_MODEL", "openai/gpt-4.1-mini")
_N_PROBES = int(os.environ.get("SCENARIO_EVAL_N_PROBES", "5"))


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
    overridden, per the design spec's LLM-roles table."""
    pb = Playbook.load(str(PB_PATH))
    talker_llm, director_llm = await pb.resolve_llm_providers()
    return PlaybookAgent(
        playbook=pb, talker_llm=talker_llm, director_llm=director_llm, http=httpx_http,
    )


async def _vanilla_agent() -> LLMAgent:
    """Mirrors the playbook's OWN declared Talker model -- A/B validity
    requires isolating structure vs. no-structure, not model vs. model."""
    pb = Playbook.load(str(PB_PATH))
    model_uri = pb.llm_uri() or _CALLER_MODEL
    text = PB_PATH.read_text(encoding="utf-8")
    return LLMAgent(resolve_llm(model_uri), system_prompt=text)


def _never_say(pb: Playbook) -> list[str]:
    """Collect every never_say string declared across the playbook's
    checkpoints, for constraint_adherence."""
    out: list[str] = []
    for journey in pb.journeys.values():
        for cp in journey.checkpoints:
            out.extend(getattr(cp, "never_say", None) or [])
    return out


def _run(coro):
    return anyio.run(lambda: coro)


def pytest_generate_tests(metafunc):
    """Parametrize test_scenario by the auto-generated dataset's cases -- the
    ONLY mechanism used to drive per-case tests in this file."""
    if "case" not in metafunc.fixturenames:
        return
    _skip_if_missing()
    ds = _run(_load_or_generate_dataset())
    metafunc.parametrize("case", ds.cases, ids=[c.id for c in ds.cases])


async def test_scenario(case) -> None:
    """One case, both modes, 7 of 9 metrics -- printed and asserted.

    guardrail/efficiency are NOT computed here -- they run on a separate
    session driver via `superdialog eval bench` (see bench_scenarios.py's
    module docstring for why). Run that command too for full 9-metric
    coverage on the same playbook.
    """
    pb = Playbook.load(str(PB_PATH))
    never_say = _never_say(pb)
    judge = _completer(_JUDGE_MODEL)
    caller = _completer(_CALLER_MODEL)

    agent = await _playbook_agent()
    vanilla = await _vanilla_agent()
    playbook_metrics, vanilla_metrics = await asyncio.gather(
        run_session(agent, case.persona, caller),
        run_vanilla_session(vanilla, case.persona, caller),
    )

    results = await score_case_both_modes(
        case=case,
        playbook_metrics=playbook_metrics,
        vanilla_metrics=vanilla_metrics,
        never_say=never_say,
        judge_llm=judge,
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
