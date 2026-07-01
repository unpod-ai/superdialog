"""A/B orchestration: drive endpoints, inject probes, build samples, score."""

from __future__ import annotations

import time
from typing import Any

from superdialog.eval.dataset.models import EvalCase, EvalSample, Probe
from superdialog.eval.endpoints.base import ConversationEndpoint, Transcript
from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.eval.runner import SpeaksUser


async def _timed(coro: Any) -> tuple[str, float]:
    t0 = time.perf_counter()
    text = await coro
    return text, (time.perf_counter() - t0) * 1000.0


async def drive_journey(
    endpoint: ConversationEndpoint,
    persona: PersonaSpec,
    user_llm: SpeaksUser,
) -> Transcript:
    """Persona simulator <-> endpoint until max_turns or the user goes silent."""
    t = Transcript()
    greeting, ms = await _timed(endpoint.start())
    t.add("assistant", greeting, latency_ms=ms)

    for _ in range(persona.max_turns):
        user_text = await user_llm.complete(_persona_messages(persona, t))
        if not user_text.strip():
            break
        t.add("user", user_text)
        reply, ms = await _timed(endpoint.turn(user_text))
        t.add("assistant", reply, latency_ms=ms)
    return t


def _persona_messages(persona: PersonaSpec, t: Transcript) -> list[dict[str, str]]:
    """Prompt the persona-LLM: who it is + the conversation so far (agent as user)."""
    sys = (
        f"You are role-playing a caller. Traits: {persona.traits}. "
        f"Goal: {persona.goal}. Stay in character; reply with only your next line."
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
