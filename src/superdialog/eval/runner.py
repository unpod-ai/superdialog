"""A/B orchestration: drive endpoints, inject probes, build samples, score."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

from superdialog.eval.dataset.models import EvalCase, EvalDataset, EvalSample, Probe
from superdialog.eval.endpoints.base import ConversationEndpoint, Transcript
from superdialog.eval.metrics.base import MetricResult, MetricSuite
from superdialog.eval.results import CaseResult, MetricAggregate, ModeResult, RunResult
from superdialog.eval.scoring import DEFAULT_WEIGHTS, case_composite, case_framework
from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.eval.runner import SpeaksUser


#: Sentinel the persona-sim appends to its FINAL line when its goal is met;
#: stripped before the goodbye is fed to the agent, then the journey ends.
_END_TOKEN = "<END_CALL>"


async def _timed(coro: Any) -> tuple[str, float]:
    t0 = time.perf_counter()
    text = await coro
    return text, (time.perf_counter() - t0) * 1000.0


async def drive_journey(
    endpoint: ConversationEndpoint,
    persona: PersonaSpec,
    user_llm: SpeaksUser,
) -> Transcript:
    """Persona simulator <-> endpoint until the call ends, the user goes
    silent, or max_turns."""
    t = Transcript()
    print("[eval-progress]   journey: endpoint.start() …", flush=True)
    greeting, ms = await _timed(endpoint.start())
    print(f"[eval-progress]   journey: greeting in {ms:.0f}ms", flush=True)
    t.add("assistant", greeting, latency_ms=ms, metadata=_turn_meta(endpoint))

    for _turn_i in range(persona.max_turns):
        print(f"[eval-progress]   turn {_turn_i}: user_llm.complete() …", flush=True)
        user_text = await user_llm.complete(_persona_messages(persona, t))
        print(
            f"[eval-progress]   turn {_turn_i}: user said {user_text[:60]!r}",
            flush=True,
        )
        if not user_text.strip():
            break
        # Persona hang-up: the sim appends _END_TOKEN once its goal is met.
        # The goodbye itself is still fed (so the playbook's goodbye
        # interrupt / terminal step fire normally), then the journey stops —
        # without this every case burns its full turn budget on goodbye
        # loops that the task_success judge penalizes.
        hanging_up = _END_TOKEN in user_text
        user_text = user_text.replace(_END_TOKEN, "").strip() or (
            "Thanks, that's everything — goodbye!"
        )
        t.add("user", user_text)
        print(f"[eval-progress]   turn {_turn_i}: endpoint.turn() …", flush=True)
        reply, ms = await _timed(endpoint.turn(user_text))
        print(f"[eval-progress]   turn {_turn_i}: reply in {ms:.0f}ms", flush=True)
        t.add("assistant", reply, latency_ms=ms, metadata=_turn_meta(endpoint))
        # Stop at the natural call end (duck-typed; endpoints without a
        # session model never report ended) or when the caller hung up.
        if hanging_up or bool(getattr(endpoint, "ended", False)):
            break
    return t


def _turn_meta(endpoint: ConversationEndpoint) -> dict[str, Any]:
    """Duck-typed per-turn metadata (token usage) from endpoints that offer it."""
    fn = getattr(endpoint, "last_turn_metadata", None)
    return fn() if callable(fn) else {}


def _persona_messages(persona: PersonaSpec, t: Transcript) -> list[dict[str, str]]:
    """Prompt the persona-LLM: who it is + the conversation so far (agent as user)."""
    # Give the persona its own ground-truth details so it can supply concrete
    # values (name, phone, email, city, ...) when the agent asks — otherwise the
    # sim invents placeholders like "[Your Name]" and slot capture never scores.
    details = "; ".join(f"{k}={v}" for k, v in persona.ground_truth_slots.items())
    your_details = (
        f" Your details (use these EXACT values when asked, never a placeholder): "
        f"{details}."
        if details
        else ""
    )
    sys = (
        f"You are role-playing a caller. Traits: {persona.traits}. "
        f"Goal: {persona.goal}.{your_details} Stay in character; reply with only "
        "your next line. When your goal is fully achieved and the agent has "
        "confirmed everything, END the call: say one brief natural goodbye and "
        f"append the exact token {_END_TOKEN} at the end of that line."
    )
    convo = [
        {"role": "assistant" if r.role == "user" else "user", "content": r.text}
        for r in t.records
    ]
    return [{"role": "system", "content": sys}, *convo]


async def run_probes(
    endpoint: ConversationEndpoint,
    probes: list[Probe],
) -> list[tuple[Probe, str]]:
    """Inject each probe as a fresh single-turn (reset between probes)."""
    out: list[tuple[Probe, str]] = []
    for probe in probes:
        endpoint.reset()
        await endpoint.start()
        reply = await endpoint.turn(probe.utterance)
        out.append((probe, reply))
    return out


def samples_from_run(
    case: EvalCase,
    mode: str,
    transcript: Transcript,
    probe_results: list[tuple[Probe, str]],
) -> list[EvalSample]:
    """One conversation sample + one sample per probe, all tagged with `mode`."""
    base = {"case_id": case.id, "mode": mode}
    samples = [
        EvalSample(
            kind="conversation",
            user_input=transcript.to_messages(),
            reference=case.reference,
            metadata={
                **base,
                "ground_truth_slots": case.persona.ground_truth_slots,
                "expected_outcome": case.expected_outcome,
                "latencies_ms": transcript.assistant_latencies_ms(),
                "turns": transcript.turn_count(),
                "input_tokens_per_turn": transcript.assistant_meta_values(
                    "input_tokens"
                ),
                "director_tokens_per_turn": transcript.assistant_meta_values(
                    "director_tokens"
                ),
                "talker_tokens_per_turn": transcript.assistant_meta_values(
                    "talker_tokens"
                ),
                "llm_calls_per_turn": transcript.assistant_meta_values("llm_calls"),
            },
        )
    ]
    for probe, reply in probe_results:
        samples.append(
            EvalSample(
                kind="probe",
                user_input=probe.utterance,
                response=reply,
                reference=probe.expect,
                retrieved_contexts=probe.reference_contexts or None,
                metadata={**base, "probe_kind": probe.kind},
            )
        )
    return samples


async def run_case(
    case: EvalCase,
    endpoint: ConversationEndpoint,
    suite: MetricSuite,
    user_llm: SpeaksUser,
    mode: str,
) -> CaseResult:
    """Drive one (case, mode): journey + probes -> samples -> scored CaseResult."""
    transcript = await drive_journey(endpoint, case.persona, user_llm)
    print(f"[eval-progress]  probes: {len(case.probes)} …", flush=True)
    probe_results = await run_probes(endpoint, case.probes)
    samples = samples_from_run(case, mode, transcript, probe_results)

    print(f"[eval-progress]  scoring {len(samples)} samples …", flush=True)
    by_metric: dict[str, list[MetricResult]] = {}
    for sample in samples:
        for res in await suite.score(sample):
            by_metric.setdefault(res.name, []).append(res)
    print("[eval-progress]  scoring done", flush=True)

    guardrail_failed = any(
        r.name == "guardrail" and r.passed is False
        for results in by_metric.values()
        for r in results
    )
    return CaseResult(
        case_id=case.id,
        mode=mode,
        metric_results=by_metric,
        guardrail_failed=guardrail_failed,
        turns=transcript.turn_count(),
    )


async def run_ab(
    dataset: EvalDataset,
    *,
    modes: list[str],
    endpoint_factories: dict[str, Callable[[EvalCase], ConversationEndpoint]],
    suite: MetricSuite,
    user_llm: SpeaksUser,
    metric_names: list[str],
    repeats: int = 1,
    weights: dict[str, float] | None = None,
) -> RunResult:
    """Run every case under every mode `repeats` times; aggregate per mode."""
    w = weights or DEFAULT_WEIGHTS
    mode_results: list[ModeResult] = []
    for mode in modes:
        factory = endpoint_factories[mode]
        case_results: list[CaseResult] = []
        for _ci, case in enumerate(dataset.cases):
            for _r in range(repeats):
                print(
                    f"[eval-progress] mode={mode} case {_ci + 1}/{len(dataset.cases)}"
                    f" rep {_r + 1}/{repeats} id={case.id}",
                    flush=True,
                )
                endpoint = factory(case)
                case_results.append(
                    await run_case(case, endpoint, suite, user_llm, mode)
                )
        mode_results.append(_aggregate_mode(mode, case_results, w))
    return RunResult(dataset=dataset.playbook, metrics=metric_names, modes=mode_results)


def _aggregate_mode(
    mode: str, case_results: list[CaseResult], weights: dict[str, float]
) -> ModeResult:
    seen = sorted({name for cr in case_results for name in cr.metric_results})
    aggregates: dict[str, MetricAggregate] = {}
    for metric in seen:
        vals: list[float] = []
        extras_pool: dict[str, list[float]] = {}
        errored = skipped = 0
        for cr in case_results:
            for r in cr.metric_results.get(metric, []):
                if r.skipped:
                    skipped += 1
                elif r.errored or r.value is None:
                    errored += 1
                else:
                    vals.append(r.value)
                    for k, x in (r.extra or {}).items():
                        if isinstance(x, (int, float)) and not isinstance(x, bool):
                            extras_pool.setdefault(k, []).append(float(x))
        aggregates[metric] = MetricAggregate(
            metric=metric,
            mean=statistics.fmean(vals) if vals else None,
            std=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            n=len(vals),
            errored=errored,
            skipped=skipped,
            extras={k: statistics.fmean(v) for k, v in extras_pool.items()},
        )
    composites = [case_composite(cr, weights) for cr in case_results]
    frameworks = [case_framework(cr) for cr in case_results]
    violations = sum(1 for cr in case_results if cr.guardrail_failed)
    guardrail_scored = sum(
        1
        for cr in case_results
        for r in cr.metric_results.get("guardrail", [])
        if not r.skipped and not r.errored
    )
    return ModeResult(
        mode=mode,
        aggregates=aggregates,
        composite_mean=statistics.fmean(composites) if composites else 0.0,
        # None (not 0.0) when no guardrail probe was scored: a 0% headline on
        # zero probes reads as demonstrated safety when nothing was tested.
        guardrail_violation_rate=(
            violations / len(case_results) if guardrail_scored else None
        ),
        case_results=case_results,
        framework_mean=statistics.fmean(frameworks) if frameworks else 0.0,
    )
