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
    "You are role-playing a caller. Traits: {traits}. Your goal: {goal}. "
    "Reply with ONLY the caller's next utterance, 1-2 sentences."
)
_TRANSCRIPT_WINDOW = 10


class SpeaksUser(Protocol):
    """Persona-LLM seam: anything that completes chat messages to text."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


async def run_session(
    agent: PlaybookAgent, persona: PersonaSpec, user_llm: SpeaksUser
) -> SessionMetrics:
    """Drive one persona session against ``agent`` and measure it."""
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
    return _measure(agent, persona, turns)


async def run_eval(
    playbook_factory: Callable[[], PlaybookAgent],
    personas: list[PersonaSpec],
    user_llm: SpeaksUser,
    n: int = 1,
) -> EvalReport:
    """Run each persona ``n`` times against fresh agents; aggregate metrics."""
    sessions = [
        await run_session(playbook_factory(), persona, user_llm)
        for persona in personas
        for _ in range(n)
    ]
    return EvalReport(sessions=sessions)


def _persona_messages(
    persona: PersonaSpec, state: ConversationState
) -> list[dict[str, str]]:
    system = _PERSONA_SYSTEM.format(traits=persona.traits, goal=persona.goal)
    messages = [{"role": "system", "content": system}]
    for entry in state.transcript[-_TRANSCRIPT_WINDOW:]:
        if entry.role == "system":
            continue
        role = "assistant" if entry.role == "user" else "user"
        messages.append({"role": role, "content": entry.text})
    return messages


def _measure(agent: PlaybookAgent, persona: PersonaSpec, turns: int) -> SessionMetrics:
    log = agent.runtime.log
    state = agent.runtime.state
    accuracy, diffs = _slot_accuracy(persona.ground_truth_slots, state)
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
        if got is not None and str(want).lower() == str(got).lower():
            correct += 1
        else:
            diffs[key] = (want, got)
    return correct / len(expected), diffs


__all__ = ["EvalReport", "PersonaSpec", "SessionMetrics", "SpeaksUser", "run_eval", "run_session"]