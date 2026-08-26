# src/superdialog/playbook/eval/runner.py
"""Persona-driven session runner: the core measurement substrate."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .models import EvalReport, PersonaSpec, SessionMetrics
from ..agent import PlaybookAgent
from ..events import (
    AdvanceEvent,
    DegradedEvent,
    EventLog,
    SteeringNoteEvent,
    UtteranceEvent,
)
from ..state import ConversationState

_PERSONA_SYSTEM = (
    "You are role-playing a caller. Traits: {traits}. Your goal: {goal}."
    "{your_details} Reply with ONLY the caller's next utterance, 1-2 sentences."
)
_TRANSCRIPT_WINDOW = 10


class SpeaksUser(Protocol):
    """Persona-LLM seam: anything that completes chat messages to text."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


async def run_session(
    agent: PlaybookAgent,
    persona: PersonaSpec,
    user_llm: SpeaksUser,
    *,
    required_slots: set[str] | None = None,
) -> SessionMetrics:
    """Drive one persona session against ``agent`` and measure it.

    ``required_slots`` (optional): when given, slot_accuracy is scored only
    against these keys instead of the full ``persona.ground_truth_slots``
    dict. The generated ground truth includes many playbook-optional,
    harvest-only backstory fields (``required: false``, "capture ONLY if
    volunteered") a well-behaved Director correctly never asks for -- scoring
    against the full dict punishes correct behavior. Pass
    ``superdialog.playbook.eval.personas.required_slots(playbook)`` to scope
    it. Omit for the old (unscoped) behavior.
    """
    await agent.runtime.start()
    user_text = persona.opening
    turns = 0
    while turns < persona.max_turns and not agent.runtime.state.ended:
        await agent.turn(user_text)
        turns += 1
        if agent.runtime.state.ended or turns >= persona.max_turns:
            break
        messages = _persona_messages(persona, agent.runtime.state)
        user_text = (await user_llm.complete(messages)).strip()
        if not user_text:
            break
    return _measure(agent, persona, turns, required_slots)


async def run_eval(
    playbook_factory: Callable[[], PlaybookAgent],
    personas: list[PersonaSpec],
    user_llm: SpeaksUser,
    n: int = 1,
) -> EvalReport:
    """Run each persona ``n`` times against fresh agents; aggregate metrics."""
    import asyncio

    results = await asyncio.gather(*[
        run_session(playbook_factory(), persona, user_llm)
        for persona in personas
        for _ in range(n)
    ])
    return EvalReport(sessions=list(results))


def _persona_messages(
    persona: PersonaSpec, state: ConversationState
) -> list[dict[str, str]]:
    # Give the persona its own ground-truth details so it answers with the
    # SAME values the eval scores it against, instead of inventing a random
    # name/language/etc every time it's asked (see slot_accuracy: a caller
    # sim with no ground truth in its own prompt has nothing consistent to
    # be "truthful" about).
    details = "; ".join(f"{k}={v}" for k, v in persona.ground_truth_slots.items())
    your_details = (
        f" Your details (use these EXACT values when asked, never invent "
        f"different ones): {details}."
        if details
        else ""
    )
    system = _PERSONA_SYSTEM.format(
        traits=persona.traits, goal=persona.goal, your_details=your_details
    )
    messages = [{"role": "system", "content": system}]
    for entry in state.transcript[-_TRANSCRIPT_WINDOW:]:
        if entry.role == "system":
            continue
        role = "assistant" if entry.role == "user" else "user"
        messages.append({"role": role, "content": entry.text})
    # The agent's last turn can legitimately produce no new spoken line (a
    # checkpoint choosing to say nothing) -- the transcript's last entry is
    # then still the caller's own prior line, which flips to role=assistant
    # here. Some providers (Anthropic) reject a message list that doesn't
    # end in "user"; a synthetic silence turn keeps this provider-agnostic.
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "(silence)"})
    return messages


def _measure(
    agent: PlaybookAgent,
    persona: PersonaSpec,
    turns: int,
    required_slots: set[str] | None = None,
) -> SessionMetrics:
    log = agent.runtime.log
    state = agent.runtime.state
    expected = persona.ground_truth_slots
    if required_slots is not None:
        expected = {k: v for k, v in expected.items() if k in required_slots}
    accuracy, diffs = _slot_accuracy(expected, state)
    return SessionMetrics(
        persona=persona.name,
        completed=state.ended,
        outcome=state.outcome,
        turns=turns,
        turns_per_checkpoint=_turns_per_checkpoint(log),
        slot_accuracy=accuracy,
        slot_diffs=diffs,
        repair_count=sum(
            1
            for e in log.events
            if isinstance(e, SteeringNoteEvent) and e.kind == "repair"
        ),
        degraded_count=sum(1 for e in log.events if isinstance(e, DegradedEvent)),
        event_log_jsonl=log.to_jsonl(),
    )


def _turns_per_checkpoint(log: EventLog) -> dict[str, int]:
    counts: dict[str, int] = {}
    current: str | None = None
    for e in log.events:
        if isinstance(e, AdvanceEvent):
            current = e.to_checkpoint
        elif isinstance(e, UtteranceEvent) and e.role == "user" and current:
            counts[current] = counts.get(current, 0) + 1
    return counts


def _slot_accuracy(
    expected: dict[str, Any], state: ConversationState
) -> tuple[float, dict[str, tuple[Any, Any]]]:
    if not expected:
        return 1.0, {}
    diffs: dict[str, tuple[Any, Any]] = {}
    correct = 0
    for key, want in expected.items():
        got = state.slot_value(key)
        if got is not None and slot_matches(str(want), str(got)):
            correct += 1
        else:
            diffs[key] = (want, got)
    return correct / len(expected), diffs


def slot_matches(want: str, got: str) -> bool:
    """Exact match, or one side contained in the other, case-insensitive.

    Ground truth is often a verbose persona sentence ("Yes, this is a good
    time -- I have about ten minutes") while the director captures a
    compressed value ("yes"); a strict equality check scores that as wrong
    even though it's correct. Containment credits real paraphrases while
    still failing genuinely different facts (different city/area names
    share no substring). ponytail: heuristic, not semantic -- revisit with
    embedding similarity if this starts over/under-crediting in practice.
    """
    w, g = want.strip().lower(), got.strip().lower()
    return bool(w) and bool(g) and (w == g or w in g or g in w)


__all__ = [
    "EvalReport", "PersonaSpec", "SessionMetrics", "SpeaksUser",
    "run_eval", "run_session", "slot_matches",
]