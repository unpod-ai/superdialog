"""Run a model against the dataset questions, then score its behaviour.

Design (replay / fixed-question benchmark):
    The dataset holds the USER turns (every question a caller can ask) plus the
    golden AGENT responses (the ground truth). The runner replays the *user*
    turns at the system-under-test (SUT) model and collects the SUT's *fresh*
    agent responses — the model answers the same fixed questions every run, so
    models are compared apples-to-apples. The SUT sees its own prior answers as
    history (free-running: honest about drift, still reproducible because the
    user turns are fixed).

Two modes:
    with-SD : questions fed to a PlaybookAgent (director + talker) built from the
              playbook YAML — full framework routing / slot state machine.
    raw     : questions fed to a plain litellm chat with a system prompt (e.g.
              the Kairali.txt production prompt) — no framework.

Each fresh transcript is packed back into a BenchmarkSample that KEEPS the
original ground truth (reference / reference_topics / expected_outcome /
ground_truth_slots), so ``build_report`` scores the SUT's answers against the
dataset. Cost = SUT tokens only (judges excluded), priced via litellm.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from .cost import cost_from_response, cost_from_tokens
from .loader import BenchmarkSample
from .ragas_scorer import DEFAULT_JUDGES
from .report import ModeReport, build_report


def _user_turns(sample: BenchmarkSample) -> list[str]:
    """The fixed user questions, in order, from the golden conversation."""
    return [
        t.get("content", "")
        for t in sample.conversation
        if t.get("role") in ("user", "human")
    ]


def _pack_fresh(sample: BenchmarkSample, pairs: list[tuple[str, str]]) -> BenchmarkSample:
    """Rebuild a sample with the SUT-generated transcript, ground truth intact.

    ``pairs`` is [(user_text, agent_text), ...] in turn order.
    """
    conversation = []
    user_input = []
    for user_text, agent_text in pairs:
        conversation.append({"role": "user", "content": user_text})
        user_input.append({"role": "human", "content": user_text})
        conversation.append({"role": "agent", "content": agent_text})
        user_input.append({"role": "ai", "content": agent_text})
    fresh_ragas = {**sample.ragas_sample, "user_input": user_input}
    return replace(sample, conversation=conversation, ragas_sample=fresh_ragas)


# ---------------------------------------------------------------- raw LLM mode


async def run_raw(
    sample: BenchmarkSample, model: str, system_prompt: str
) -> tuple[BenchmarkSample, float]:
    """Replay the user turns at a plain litellm chat. Returns (fresh sample, cost)."""
    import litellm

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    pairs: list[tuple[str, str]] = []
    cost = 0.0
    for user_text in _user_turns(sample):
        messages.append({"role": "user", "content": user_text})
        resp = await litellm.acompletion(model=model, messages=messages, temperature=0.3)
        agent_text = resp.choices[0].message.content or ""
        cost += cost_from_response(resp)  # SUT cost only
        messages.append({"role": "assistant", "content": agent_text})
        pairs.append((user_text, agent_text))
    return _pack_fresh(sample, pairs), cost


# --------------------------------------------------------- with-SuperDialog mode


async def run_with_sd(
    sample: BenchmarkSample, model: str, playbook_path: str | Path
) -> tuple[BenchmarkSample, float]:
    """Replay the user turns at a PlaybookAgent built from the YAML.

    Returns (fresh sample, cost). Cost is summed from the SUT usage callback.
    """
    from superdialog.llm.litellm_provider import LitellmProvider
    from superdialog.playbook import PlaybookAgent, httpx_http
    from superdialog.playbook.models import Playbook
    from superdialog.playbook.providers import provider_adapters

    playbook = Playbook.load(str(playbook_path))

    cost = 0.0

    async def _on_usage(ev) -> None:  # LLMUsageEvent(model, tokens_in, tokens_out, ...)
        nonlocal cost
        cost += cost_from_tokens(ev.model, ev.tokens_in, ev.tokens_out)

    provider = LitellmProvider(model)
    director, talker = provider_adapters(provider, on_llm_complete=_on_usage)
    agent = PlaybookAgent(
        playbook=playbook,
        talker_llm=talker,
        director_llm=director,
        http=httpx_http,
    )
    await agent.runtime.start()

    pairs: list[tuple[str, str]] = []
    for user_text in _user_turns(sample):
        if agent.runtime.state.ended:
            break
        result = await agent.turn(user_text, stream=False)
        agent_text = getattr(result, "text", "") or ""
        pairs.append((user_text, agent_text))
    return _pack_fresh(sample, pairs), cost


# ------------------------------------------------------------------- mode runners


def _run(coro):
    return asyncio.run(coro)


def run_raw_mode(
    samples: list[BenchmarkSample],
    model: str,
    system_prompt_path: str | Path,
    *,
    label: str | None = None,
    run_ragas: bool = False,
    judges: tuple[str, ...] = DEFAULT_JUDGES,
) -> ModeReport:
    """Run every sample raw, score, and aggregate into a ModeReport."""
    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")

    async def _all():
        return await asyncio.gather(
            *(run_raw(s, model, system_prompt) for s in samples)
        )

    fresh_and_cost = _run(_all())
    fresh = [fc[0] for fc in fresh_and_cost]
    total_cost = sum(fc[1] for fc in fresh_and_cost)
    report = build_report(
        label or f"Raw LLM ({model})", fresh, run_ragas=run_ragas, judges=judges
    )
    report.cost_usd = round(total_cost, 6)
    return report


def run_sd_mode(
    samples: list[BenchmarkSample],
    model: str,
    playbook_path: str | Path,
    *,
    label: str | None = None,
    run_ragas: bool = False,
    judges: tuple[str, ...] = DEFAULT_JUDGES,
) -> ModeReport:
    """Run every sample through the PlaybookAgent, score, aggregate."""

    async def _all():
        return await asyncio.gather(
            *(run_with_sd(s, model, playbook_path) for s in samples)
        )

    fresh_and_cost = _run(_all())
    fresh = [fc[0] for fc in fresh_and_cost]
    total_cost = sum(fc[1] for fc in fresh_and_cost)
    report = build_report(
        label or f"With SuperDialog ({model})", fresh, run_ragas=run_ragas, judges=judges
    )
    report.cost_usd = round(total_cost, 6)
    return report


__all__ = ["run_raw", "run_with_sd", "run_raw_mode", "run_sd_mode"]
