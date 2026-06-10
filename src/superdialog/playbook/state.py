"""ConversationState: a pure fold over the event log (design doc §3)."""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field

from .events import (
    AdvanceEvent,
    EnvWriteEvent,
    EventLog,
    ExternalEvent,
    ScratchpadEvent,
    SessionEndEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
    SummaryEvent,
    ToolCallEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from .models import Playbook, SlotSpec


class SlotValue(BaseModel):
    """A slot's current value plus its provenance and confirmation status."""

    value: Any
    status: Literal["provisional", "confirmed"]
    by: str
    version: int


class ToolResult(BaseModel):
    """Stored outcome of a tool call, keyed by its `store_as` name."""

    tool: str
    ok: bool
    status: int | None = None
    data: Any = None
    error: str | None = None
    version: int


class TranscriptEntry(BaseModel):
    """One utterance in the conversation transcript."""

    role: str
    text: str
    version: int


class ConversationState(BaseModel):
    """Derived snapshot of a conversation, computed by folding the event log."""

    version: int = 0
    checkpoint_id: str | None = None
    checkpoint_entered_version: int = 0  # version of the AdvanceEvent that entered it
    # Append-only audit trail of exited checkpoints (revisits duplicate).
    completed: list[str] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    tool_results: dict[str, ToolResult] = Field(default_factory=dict)
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    scratchpad: list[str] = Field(default_factory=list)
    steering_note: str | None = None
    steering_kind: str = "steer"
    summary: str = ""
    silence_count: int = 0  # silences since checkpoint entry (reset on advance)
    ended: bool = False
    outcome: str | None = None
    user_turns_in_checkpoint: int = 0

    @classmethod
    def fold(
        cls, log: EventLog, playbook: Playbook | None = None
    ) -> "ConversationState":
        """Fold the event log into a fresh state snapshot.

        Returns a snapshot whose payloads are deep-copied from events, so
        mutating the result never aliases the log. Slot invalidation is
        NON-TRANSITIVE (authors list the transitive closure in `invalidates`)
        and is skipped when a write re-asserts the same value.
        """
        # Precompute slot specs once; first declaration wins (matches
        # Playbook.slot_spec semantics).
        specs: dict[str, SlotSpec] = {}
        if playbook:
            for journey in playbook.journeys.values():
                for cp in journey.checkpoints:
                    for key, slot_spec in cp.slots.items():
                        specs.setdefault(key, slot_spec)
        s = cls()
        for e in log.replay():
            s.version = e.version
            if isinstance(e, UtteranceEvent):
                s.transcript.append(
                    TranscriptEntry(role=e.role, text=e.text, version=e.version)
                )
                if e.role == "user":
                    s.user_turns_in_checkpoint += 1
            elif isinstance(e, SlotWriteEvent):
                spec = specs.get(e.key)
                if spec and spec.authoritative and e.by == "talker":
                    continue  # authoritative slots are Director/tool-only
                existing = s.slots.get(e.key)
                if (
                    existing
                    and existing.status == "confirmed"
                    and e.status == "provisional"
                ):
                    continue  # never downgrade
                unchanged = existing is not None and existing.value == e.value
                s.slots[e.key] = SlotValue(
                    value=copy.deepcopy(e.value),
                    status=e.status,
                    by=e.by,
                    version=e.version,
                )
                if spec and not unchanged:
                    for dep in spec.invalidates:
                        if dep != e.key:
                            s.slots.pop(dep, None)
                            s.tool_results.pop(dep, None)
            elif isinstance(e, AdvanceEvent):
                if e.from_checkpoint:
                    s.completed.append(e.from_checkpoint)
                s.checkpoint_id = e.to_checkpoint
                s.checkpoint_entered_version = e.version
                s.user_turns_in_checkpoint = 0
                s.silence_count = 0
            elif isinstance(e, SteeringNoteEvent):
                s.steering_note, s.steering_kind = e.text, e.kind
            elif isinstance(e, ToolCallEvent):
                s.tool_call_counts[e.tool] = s.tool_call_counts.get(e.tool, 0) + 1
            elif isinstance(e, ToolResultEvent):
                if e.store_as:
                    s.tool_results[e.store_as] = ToolResult(
                        tool=e.tool,
                        ok=e.ok,
                        status=e.status,
                        data=copy.deepcopy(e.data),
                        error=e.error,
                        version=e.version,
                    )
            elif isinstance(e, EnvWriteEvent):
                s.env[e.key] = e.value
            elif isinstance(e, ScratchpadEvent):
                s.scratchpad.append(e.text)
            elif isinstance(e, SummaryEvent):
                s.summary = e.text
            elif isinstance(e, ExternalEvent):
                if e.kind == "silence":
                    s.silence_count += 1
            elif isinstance(e, SessionEndEvent):
                s.ended, s.outcome = True, e.outcome
        return s

    # convenience used by expr judge and renderer
    def slot_value(self, key: str) -> Any:
        """Return the slot's value, or None when the slot is unset."""
        sv = self.slots.get(key)
        return sv.value if sv else None

    def confirmed(self, keys: list[str]) -> bool:
        """True when every key is filled with a confirmed value."""
        return all(
            k in self.slots and self.slots[k].status == "confirmed" for k in keys
        )

    def filled(self, keys: list[str]) -> bool:
        """True when every key has a value (provisional or confirmed)."""
        return all(k in self.slots for k in keys)
