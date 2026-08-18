"""Vanilla-mode session driver: the run_session shape, for a flat-prompt LLMAgent.

run_session (runner.py) is typed to PlaybookAgent and reads .runtime.start()/
.runtime.state.ended — APIs a vanilla LLMAgent doesn't have. This mirrors its
turn-taking loop against LLMAgent.turn() directly. Vanilla has no tools, so
there is no event log here (see events.py) — its metrics read .transcript.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from .models import PersonaSpec
from .runner import SpeaksUser


class TranscriptEntry(BaseModel):
    role: str  # "user" | "assistant"
    text: str


class VanillaSessionMetrics(BaseModel):
    """The vanilla-side counterpart to SessionMetrics — no event log, no
    slots, no checkpoints: a flat LLMAgent has none of those concepts."""

    persona: str
    turns: int
    transcript: list[TranscriptEntry] = Field(default_factory=list)


class TurnsText(Protocol):
    """Duck-typed: anything with an async turn(text) -> object-with-.text."""

    async def turn(self, text: str) -> Any: ...


async def run_vanilla_session(
    agent: TurnsText, persona: PersonaSpec, user_llm: SpeaksUser
) -> VanillaSessionMetrics:
    """Drive one persona session against a flat vanilla agent; measure it."""
    transcript: list[TranscriptEntry] = []
    user_text = persona.opening
    turns = 0
    while turns < persona.max_turns:
        transcript.append(TranscriptEntry(role="user", text=user_text))
        result = await agent.turn(user_text)
        reply = result.text if hasattr(result, "text") else str(result)
        transcript.append(TranscriptEntry(role="assistant", text=reply))
        turns += 1
        if turns >= persona.max_turns:
            break
        messages = _persona_messages(persona, transcript)
        user_text = (await user_llm.complete(messages)).strip()
        if not user_text:
            break
    return VanillaSessionMetrics(persona=persona.name, turns=turns, transcript=transcript)


def _persona_messages(
    persona: PersonaSpec, transcript: list[TranscriptEntry]
) -> list[dict[str, str]]:
    system = (
        f"You are role-playing a caller. Traits: {persona.traits}. "
        f"Your goal: {persona.goal}. Reply with ONLY the caller's next "
        "utterance, 1-2 sentences."
    )
    messages = [{"role": "system", "content": system}]
    for entry in transcript[-10:]:
        role = "assistant" if entry.role == "user" else "user"
        messages.append({"role": role, "content": entry.text})
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "(silence)"})
    return messages
