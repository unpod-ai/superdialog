# src/superdialog/playbook/eval/from_traversal.py
"""Derive evals from a saved traversal JSON.

Two workflows:

  Regression (exact replay):
      traversal = load_traversal("traversal_xxx.json")
      user     = traversal_to_scripted_user(traversal)   # replays exact utterances
      persona  = traversal_to_persona(traversal)          # ground_truth from final_slots
      metrics  = await run_session(agent, persona, user)

  Persona-driven (LLM user):
      persona  = traversal_to_persona(traversal)          # traits + goal + ground_truth
      metrics  = await run_session(agent, persona, llm_user)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .models import PersonaSpec


class ScriptedUser:
    """Replays a fixed list of utterances — deterministic regression baseline.

    When the list is exhausted, returns "" which causes run_session to stop.
    """

    def __init__(self, messages: list[str]) -> None:
        self._messages = list(messages)
        self._idx = 0

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        if self._idx >= len(self._messages):
            return ""
        msg = self._messages[self._idx]
        self._idx += 1
        return msg


def load_traversal(path: str | Path) -> dict[str, Any]:
    """Load a traversal JSON file."""
    return json.loads(Path(path).read_text())


def traversal_to_scripted_user(traversal: dict[str, Any]) -> ScriptedUser:
    """Extract user utterances in traversal order for exact replay."""
    messages = [
        step["user_message"]
        for step in traversal.get("traversal", [])
        if step.get("user_message")
    ]
    return ScriptedUser(messages)


def traversal_to_persona(traversal: dict[str, Any]) -> PersonaSpec:
    """Derive a PersonaSpec from a real traversal's slots and outcome.

    The returned persona has:
    - ``traits``            — readable summary of captured slot values
    - ``goal``              — "achieve outcome: <outcome>"
    - ``ground_truth_slots``— the traversal's final_slots (for slot_accuracy)
    - ``opening``           — first user utterance in the traversal
    - ``max_turns``         — traversal length + 5 buffer
    """
    slots: dict[str, Any] = {
        k: v["value"] for k, v in traversal.get("final_slots", {}).items()
    }

    trait_keys = [
        "name", "investment_or_self_use", "staying", "job_location",
        "configuration", "selected_language",
    ]
    trait_parts = [
        f"{k}={slots[k]!r}" for k in trait_keys if k in slots
    ]
    traits = ", ".join(trait_parts) if trait_parts else "cooperative caller"

    steps = traversal.get("traversal", [])
    opening = "Hello"
    for step in steps:
        if step.get("user_message"):
            opening = step["user_message"]
            break

    outcome = traversal.get("outcome") or "complete"

    return PersonaSpec(
        name=traversal.get("session_id", "traversal_replay"),
        traits=traits,
        goal=f"achieve outcome: {outcome}",
        max_turns=len(steps) + 5,
        opening=opening,
        ground_truth_slots=slots,
    )


class TimingLLM:
    """Wraps a CompletesLLM and records wall-clock latency per call.

    Use instead of the raw LLM when you want latency stats:

        timed = TimingLLM(director_llm)
        agent = PlaybookAgent(..., director_llm=timed, ...)
        await run_session(agent, persona, user)
        print(timed.mean_ms, timed.p95_ms)
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.latencies_ms: list[float] = []

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        t0 = time.perf_counter()
        result = await self._inner.complete(messages, **kw)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return result

    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        t0 = time.perf_counter()
        try:
            async for chunk in self._inner.stream(messages, **kw):
                yield chunk
        finally:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000)

    @property
    def mean_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)]

    @property
    def total_calls(self) -> int:
        return len(self.latencies_ms)


__all__ = [
    "ScriptedUser",
    "TimingLLM",
    "load_traversal",
    "traversal_to_persona",
    "traversal_to_scripted_user",
]
