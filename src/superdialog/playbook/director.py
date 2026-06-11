"""Director: async supervisor — extract, judge, steer (design doc §2)."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .events import AdvanceEvent, Event, SlotWriteEvent, SteeringNoteEvent
from .expr import ExprError, evaluate
from .models import Checkpoint, Playbook
from .state import ConversationState, SlotValue


class CompletesLLM(Protocol):
    """Minimal structured-completion surface the Director depends on."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class DirectorDecision(BaseModel):
    """Outcome of one Director evaluation: events to append, or degraded."""

    events: list[Event] = Field(default_factory=list)
    degraded: bool = False  # LLM failed; Talker continues solo


def _verdict_prompt(
    pb: Playbook, cp: Checkpoint, state: ConversationState
) -> list[dict[str, str]]:
    rules = [r for r in cp.advance_when if r.judge == "llm"]
    rule_lines = "\n".join(f"- to={r.to!r}: {r.when}" for r in rules) or "(none)"
    interrupt_lines = (
        "\n".join(f"- id={i.id!r}: {i.when}" for i in pb.interrupts if i.judge == "llm")
        or "(none)"
    )
    slot_lines = (
        "\n".join(
            f"- {k} ({s.type}{', required' if s.required else ''}): {s.description}"
            for k, s in cp.slots.items()
        )
        or "(none)"
    )
    known = {k: v.value for k, v in state.slots.items()}
    transcript = "\n".join(f"{m.role}: {m.text}" for m in state.transcript[-12:])
    system = (
        "You supervise a live conversation. Read the transcript and respond with "
        'STRICT JSON only: {"slots": {<key>: <value> for any newly evident slot '
        'values}, "advance": <target id from the rules below, or null>, '
        '"note": <one-or-two-sentence direction for the speaking agent, or null>, '
        '"interrupt": <interrupt id if one clearly applies, else omit>}.\n\n'
        f"Current step: {cp.id} — goal: {cp.goal}\n"
        f"Slots to extract:\n{slot_lines}\n"
        f"Already known: {json.dumps(known, default=str)}\n"
        f"Advance rules:\n{rule_lines}\n"
        f"Interrupts:\n{interrupt_lines}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": transcript},
    ]


class _PipelineNs:
    """expr namespace: pipeline.ok / pipeline.failed over the 'pipeline' result key.

    Holds only two booleans — no state reference — as sandbox defense in depth.
    """

    def __init__(self, state: ConversationState) -> None:
        result = state.tool_results.get("pipeline")
        self.ok = bool(result and result.ok)
        self.failed = bool(result and not result.ok)


class Director:
    """ONE structured LLM call per user utterance: extract, judge, steer."""

    def __init__(self, playbook: Playbook, llm: CompletesLLM) -> None:
        self._pb = playbook
        self._llm = llm

    def _requires_met(
        self, requires: list[str], cp: Checkpoint, state: ConversationState
    ) -> bool:
        if cp.gate == "hard":
            return state.confirmed(requires)
        return state.filled(requires)

    def _expr_advance(
        self, cp: Checkpoint, state: ConversationState, cp_ref: str
    ) -> list[Event]:
        for rule in cp.advance_when:
            if rule.judge != "expr":
                continue
            try:
                fired = bool(
                    evaluate(rule.when, state, extra={"pipeline": _PipelineNs(state)})
                )
            except ExprError:
                fired = False
            if fired and self._requires_met(rule.requires, cp, state):
                events: list[Event] = [
                    SlotWriteEvent(key=k, value=v, status="confirmed", by="director")
                    for k, v in rule.set.items()
                ]
                events.append(
                    AdvanceEvent(
                        from_checkpoint=cp_ref,
                        to_checkpoint=rule.to,
                        rule=rule.rule_id,
                        by="expr",
                    )
                )
                return events
        return []

    async def evaluate(
        self, state: ConversationState, expr_only: bool = False
    ) -> DirectorDecision:
        """Evaluate the current state: expr rules first, then one LLM verdict."""
        if state.checkpoint_id is None or state.ended:
            return DirectorDecision()
        cp_ref = state.checkpoint_id
        cp = self._pb.checkpoint(cp_ref)

        expr_events = self._expr_advance(cp, state, cp_ref)
        if expr_events:
            return DirectorDecision(events=expr_events)
        if expr_only:
            return DirectorDecision()

        try:
            raw = await self._llm.complete(_verdict_prompt(self._pb, cp, state))
            verdict = json.loads(_strip_fences(raw))
        except Exception:
            return DirectorDecision(degraded=True)
        if not isinstance(verdict, dict):
            return DirectorDecision(degraded=True)

        events: list[Event] = []
        for key, value in (verdict.get("slots") or {}).items():
            if key in cp.slots or self._pb.slot_spec(key):
                events.append(
                    SlotWriteEvent(
                        key=key, value=value, status="confirmed", by="director"
                    )
                )
        # apply slot writes to a copy so requires sees them
        peek = state.model_copy(deep=True)
        for e in events:
            if isinstance(e, SlotWriteEvent):
                peek.slots[e.key] = SlotValue(
                    value=e.value,
                    status="confirmed",
                    by="director",
                    version=peek.version,
                )

        interrupt_id = verdict.get("interrupt")
        if interrupt_id:
            spec = next((i for i in self._pb.interrupts if i.id == interrupt_id), None)
            if spec is not None:
                events.append(
                    AdvanceEvent(
                        from_checkpoint=cp_ref,
                        to_checkpoint=spec.to,
                        rule=f"interrupt:{spec.id}",
                    )
                )
                return DirectorDecision(events=events)

        target = verdict.get("advance")
        if target:
            rule = next(
                (r for r in cp.advance_when if r.judge == "llm" and r.to == target),
                None,
            )
            if rule is not None:
                if self._requires_met(rule.requires, cp, peek):
                    for k, v in rule.set.items():
                        events.append(
                            SlotWriteEvent(
                                key=k, value=v, status="confirmed", by="director"
                            )
                        )
                    events.append(
                        AdvanceEvent(
                            from_checkpoint=cp_ref,
                            to_checkpoint=rule.to,
                            rule=rule.rule_id,
                        )
                    )
                else:
                    missing = [k for k in rule.requires if k not in peek.slots]
                    events.append(
                        SteeringNoteEvent(
                            text=(
                                "Cannot move on yet — still need: "
                                f"{', '.join(missing)}. Ask for these naturally."
                            ),
                            kind="steer",
                        )
                    )
        note = verdict.get("note")
        if note and not any(isinstance(e, SteeringNoteEvent) for e in events):
            events.append(SteeringNoteEvent(text=str(note), kind="steer"))
        return DirectorDecision(events=events)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()
