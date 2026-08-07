"""ConversationState: a pure fold over the event log (design doc §3)."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .events import (
    AdvanceEvent,
    EnvWriteEvent,
    EventLog,
    ExternalEvent,
    RevertEvent,
    ScratchpadEvent,
    SessionEndEvent,
    SessionStartEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
    SummaryEvent,
    ToolCallEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from .models import Playbook, SlotSpec


#: A resume-stack entry older than this many subsequent advances is stale:
#: "resuming" to it would teleport the caller minutes backward. Reaped by the
#: fold (pure — no event needed; the non-resume is visible in traversals).
_RESUME_STACK_MAX_AGE = 6


def _superseded_versions(log: EventLog) -> set[int]:
    """Versions whose STATE effects were rewound by an active ``RevertEvent``.

    Walked newest-first so a later revert that covers an earlier revert's own
    version deactivates it — the earlier revert's range then re-applies, which
    is exactly "state as of the later revert's target".
    """
    dead: set[int] = set()
    for e in reversed(log.events):
        if isinstance(e, RevertEvent) and e.version not in dead:
            dead.update(range(e.superseded_from, e.superseded_to + 1))
    return dead


def _ekey(entity: str, key: str) -> str:
    """Composite storage key. 'caller' keeps the BARE key so single-entity
    playbooks (multi_entity off, entity defaults 'caller') store exactly as
    today — backward compatible. Other entities namespace: 'partner:dob'."""
    return key if entity == "caller" else f"{entity}:{key}"


class SlotValue(BaseModel):
    """A slot's current value plus its provenance and confirmation status."""

    value: Any
    status: Literal["provisional", "confirmed"]
    by: str
    version: int
    entity: str = "caller"  # whose slot; "caller" stays backward compatible


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
    now: datetime | None = None  # per-call anchor from SessionStartEvent
    checkpoint_id: str | None = None
    checkpoint_entered_version: int = 0  # version of the AdvanceEvent that entered it
    # Append-only audit trail of exited checkpoints (revisits duplicate).
    completed: list[str] = Field(default_factory=list)
    # Checkpoints to return to after a resume=True interrupt detour finishes
    # (LIFO). Pushed with the step we left when such an interrupt fires; popped
    # by the synthesized "resume" advance. Derived purely from the log.
    resume_stack: list[str] = Field(default_factory=list)
    # Advance ordinal at push, parallel to resume_stack (same length, same
    # order); entries whose ordinal falls >_RESUME_STACK_MAX_AGE advances
    # behind are expired by the fold.
    resume_stack_seq: list[int] = Field(default_factory=list)
    # True when the CURRENT checkpoint was entered by a resume=True interrupt —
    # the runtime uses this to know it should return once the detour is done.
    entered_via_resume: bool = False
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
    language: str | None = None  # sticky reply language; adheres to the bridge

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
        interrupt_resume: dict[str, bool] = {}
        if playbook:
            for journey in playbook.journeys.values():
                for cp in journey.checkpoints:
                    for key, slot_spec in cp.slots.items():
                        specs.setdefault(key, slot_spec)
            for itr in playbook.interrupts:
                interrupt_resume[itr.id] = itr.resume
        s = cls()
        dead = _superseded_versions(log)
        # Advance ordinal for resume-stack aging. Superseded (rewound)
        # advances `continue` above before reaching the AdvanceEvent branch,
        # so they never bump it — dead detours don't age live ones.
        adv_seq = 0
        for e in log.replay():
            s.version = e.version
            if e.version in dead and not isinstance(e, SessionStartEvent):
                # Rewound: the event's STATE effect is superseded. Speech is
                # irreversible — utterances stay in the transcript (and keep
                # the sticky language: the caller's language doesn't un-happen)
                # but stop counting toward checkpoint progress. SessionStart is
                # never superseded: the call's time anchor isn't conversational
                # state.
                if isinstance(e, UtteranceEvent):
                    s.transcript.append(
                        TranscriptEntry(role=e.role, text=e.text, version=e.version)
                    )
                    if e.role == "user" and e.language:
                        s.language = e.language
                continue
            if isinstance(e, UtteranceEvent):
                s.transcript.append(
                    TranscriptEntry(role=e.role, text=e.text, version=e.version)
                )
                if e.role == "user":
                    s.user_turns_in_checkpoint += 1
                    if e.language:
                        # Adhere to the bridge's per-turn detection; sticky
                        # against a missing signal so an undetected turn never
                        # flips us.
                        # ponytail: trusts the bridge's script thresholds; add
                        # N-turn hysteresis only if [turn-trace] shows
                        # oscillation.
                        s.language = e.language
            elif isinstance(e, SlotWriteEvent):
                spec = specs.get(e.key)
                if spec and spec.authoritative and e.by == "talker":
                    continue  # authoritative slots are Director/tool-only
                skey = _ekey(e.entity, e.key)
                existing = s.slots.get(skey)
                if (
                    existing
                    and existing.status == "confirmed"
                    and e.status == "provisional"
                ):
                    continue  # never downgrade
                unchanged = existing is not None and existing.value == e.value
                s.slots[skey] = SlotValue(
                    value=copy.deepcopy(e.value),
                    status=e.status,
                    by=e.by,
                    version=e.version,
                    entity=e.entity,
                )
                if spec and not unchanged:
                    for dep in spec.invalidates:
                        if dep != e.key:
                            s.slots.pop(_ekey(e.entity, dep), None)
                            s.tool_results.pop(dep, None)
            elif isinstance(e, AdvanceEvent):
                if e.from_checkpoint:
                    s.completed.append(e.from_checkpoint)
                # Resume bookkeeping: a resume=True interrupt records the step we
                # are leaving so the detour can return to it; the synthesized
                # "resume" advance pops that entry. entered_via_resume tracks
                # whether the checkpoint we just entered is such a detour target.
                adv_seq += 1
                entered_via_resume = False
                if e.rule.startswith("interrupt:") and e.from_checkpoint:
                    _iid = e.rule.split(":", 1)[1]
                    if interrupt_resume.get(_iid):
                        s.resume_stack = s.resume_stack + [e.from_checkpoint]
                        s.resume_stack_seq = s.resume_stack_seq + [adv_seq]
                        entered_via_resume = True
                elif e.rule == "resume" and s.resume_stack:
                    s.resume_stack = s.resume_stack[:-1]
                    s.resume_stack_seq = s.resume_stack_seq[:-1]
                s.entered_via_resume = entered_via_resume
                # Expire stale entries oldest-first: an entry stranded for
                # more than _RESUME_STACK_MAX_AGE advances would teleport the
                # caller backward if ever "resumed".
                while (
                    s.resume_stack
                    and adv_seq - s.resume_stack_seq[0] > _RESUME_STACK_MAX_AGE
                ):
                    s.resume_stack = s.resume_stack[1:]
                    s.resume_stack_seq = s.resume_stack_seq[1:]
                s.checkpoint_id = e.to_checkpoint
                s.checkpoint_entered_version = e.version
                s.user_turns_in_checkpoint = 0
                s.silence_count = 0
                # A steer note is advice for the current step; it must not
                # bleed into the next checkpoint.
                s.steering_note = None
                s.steering_kind = "steer"
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
            elif isinstance(e, SessionStartEvent):
                try:
                    s.now = datetime.fromisoformat(e.started_at)
                except (ValueError, TypeError):
                    s.now = None
            elif isinstance(e, SessionEndEvent):
                s.ended, s.outcome = True, e.outcome
        return s

    # convenience used by expr judge and renderer
    def slot_value(self, key: str, entity: str = "caller") -> Any:
        """Return the slot's value, or None when the slot is unset."""
        sv = self.slots.get(_ekey(entity, key))
        return sv.value if sv else None

    def confirmed(self, keys: list[str], entity: str = "caller") -> bool:
        """True when every key is filled with a confirmed value."""
        return all(
            _ekey(entity, k) in self.slots
            and self.slots[_ekey(entity, k)].status == "confirmed"
            for k in keys
        )

    def filled(self, keys: list[str], entity: str = "caller") -> bool:
        """True when every key has a value (provisional or confirmed)."""
        return all(_ekey(entity, k) in self.slots for k in keys)
