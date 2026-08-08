"""Append-only event log: the single source of truth for a conversation."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Iterator, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 0  # stamped by EventLog.append; 0 == unstamped


class UtteranceEvent(_Base):
    type: Literal["utterance"] = "utterance"
    role: Literal["user", "assistant", "system"]
    text: str
    spoke_from_version: int | None = None  # Talker: state version it rendered
    language: str | None = None  # bridge-detected language of this turn (user turns)


class SlotWriteEvent(_Base):
    type: Literal["slot_write"] = "slot_write"
    key: str
    value: Any
    status: Literal["provisional", "confirmed"]
    by: Literal["talker", "director", "tool", "compiler"]
    entity: str = "caller"  # whose slot; "caller" stays backward compatible


class AdvanceEvent(_Base):
    type: Literal["advance"] = "advance"
    from_checkpoint: str | None
    to_checkpoint: str
    # rule id, "init", "auto", "pipeline", "on_failure",
    # "interrupt:<id>", "policy:<name>", "supervisor:<reason>"
    rule: str
    by: Literal["director", "expr", "policy", "supervisor"] = "director"
    # v2 accountability: True = slot evidence / expr backed this advance;
    # False = prose-only (steered); None = unclassified (interrupts, policy,
    # resume, pre-v2 logs).
    corroborated: bool | None = None


class SteeringNoteEvent(_Base):
    type: Literal["steering_note"] = "steering_note"
    text: str
    kind: Literal["steer", "repair"] = "steer"


class ToolCallEvent(_Base):
    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(_Base):
    type: Literal["tool_result"] = "tool_result"
    tool: str
    store_as: str | None = None
    ok: bool
    status: int | None = None  # HTTP status when applicable
    data: Any = None
    error: str | None = None


class EnvWriteEvent(_Base):
    type: Literal["env_write"] = "env_write"
    key: str
    value: str


class SessionStartEvent(_Base):
    type: Literal["session_start"] = "session_start"
    started_at: str = ""  # ISO-8601, tz-aware; the per-call date/time anchor
    timezone: str = "UTC"


class ScratchpadEvent(_Base):
    type: Literal["scratchpad"] = "scratchpad"
    text: str


class SummaryEvent(_Base):
    type: Literal["summary"] = "summary"
    text: str


class ExternalEvent(_Base):
    type: Literal["external"] = "external"
    kind: Literal["silence", "webhook", "timer"]
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DegradedEvent(_Base):
    """Director failure marker — degraded mode is auditable, never silent."""

    type: Literal["degraded"] = "degraded"
    component: Literal["director", "talker", "supervisor", "runtime"] = "director"
    detail: str = ""


class RevertEvent(_Base):
    """Supersede a range of earlier state effects — rewind state, never speech.

    The fold skips state-bearing events whose version falls in
    ``[superseded_from, superseded_to]``; utterances always stay in the
    transcript (the caller heard them) and ``SessionStartEvent`` is never
    superseded (the call's time anchor is not conversational state). The log
    stays append-only: a revert is itself an event, so the audit trail and
    traversal exports remain complete. A later revert may supersede an earlier
    ``RevertEvent``, which re-activates the events that one had superseded.
    """

    type: Literal["revert"] = "revert"
    superseded_from: int
    superseded_to: int
    reason: str = ""
    by: Literal["director", "supervisor", "runtime"] = "runtime"


class SpeechCorrectionEvent(_Base):
    """Correct an assistant utterance to what the caller actually heard.

    Append-only barge-in truncation: the original UtteranceEvent stays
    in the log (what was GENERATED); the fold's transcript shows this
    text (what was DELIVERED). Its presence is the completed=False
    marker for the corrected utterance.
    """

    type: Literal["speech_correction"] = "speech_correction"
    utterance_version: int
    heard_text: str


class SessionEndEvent(_Base):
    type: Literal["session_end"] = "session_end"
    outcome: str | None = None


Event = Annotated[
    Union[
        UtteranceEvent,
        SlotWriteEvent,
        AdvanceEvent,
        SteeringNoteEvent,
        ToolCallEvent,
        ToolResultEvent,
        EnvWriteEvent,
        SessionStartEvent,
        ScratchpadEvent,
        SummaryEvent,
        ExternalEvent,
        DegradedEvent,
        SessionEndEvent,
        RevertEvent,
        SpeechCorrectionEvent,
    ],
    Field(discriminator="type"),
]
_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


class EventLog:
    """Append-only, monotonically versioned event sequence."""

    def __init__(self, events: list[Event] | None = None) -> None:
        self.events: list[Event] = list(events or [])
        versions = [e.version for e in self.events]
        if versions != list(range(1, len(self.events) + 1)):
            raise ValueError(
                f"event versions must be contiguous starting at 1, got {versions}"
            )
        # Live observers (Shepherd-style non-perturbing observation): every
        # appended event is pushed to each subscriber queue. Observers read
        # only; the log never blocks on them (put_nowait on unbounded queues).
        self._subscribers: list["asyncio.Queue[Event]"] = []

    @property
    def version(self) -> int:
        return self.events[-1].version if self.events else 0

    def append(self, event: Event) -> Event:
        if event.version != 0:
            raise ValueError(f"event already stamped with version {event.version}")
        stamped = event.model_copy(update={"version": self.version + 1})
        self.events.append(stamped)
        for q in self._subscribers:
            q.put_nowait(stamped)
        return stamped

    def subscribe(self) -> "asyncio.Queue[Event]":
        """Register a live observer; every future append lands on the queue.

        Observation is non-perturbing: the log's behavior is identical with or
        without subscribers. A wholesale ``load_log`` swap orphans subscribers
        of the old log — re-subscribe after restoring a session.
        """
        q: "asyncio.Queue[Event]" = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[Event]") -> None:
        """Remove a subscriber queue; unknown queues are ignored."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def replay(self) -> Iterator[Event]:
        return iter(self.events)

    def to_jsonl(self) -> str:
        return "\n".join(e.model_dump_json() for e in self.events)

    @classmethod
    def from_jsonl(cls, text: str) -> "EventLog":
        events = [
            _event_adapter.validate_json(line)
            for line in text.splitlines()
            if line.strip()
        ]
        return cls(events=events)
