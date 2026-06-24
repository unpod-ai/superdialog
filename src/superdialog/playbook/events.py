"""Append-only event log: the single source of truth for a conversation."""

from __future__ import annotations

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


class SlotWriteEvent(_Base):
    type: Literal["slot_write"] = "slot_write"
    key: str
    value: Any
    status: Literal["provisional", "confirmed"]
    by: Literal["talker", "director", "tool", "compiler"]


class AdvanceEvent(_Base):
    type: Literal["advance"] = "advance"
    from_checkpoint: str | None
    to_checkpoint: str
    # rule id, "init", "auto", "pipeline", "on_failure",
    # "interrupt:<id>", "policy:<name>"
    rule: str
    by: Literal["director", "expr", "policy"] = "director"


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
    component: Literal["director", "talker"] = "director"
    detail: str = ""


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

    @property
    def version(self) -> int:
        return self.events[-1].version if self.events else 0

    def append(self, event: Event) -> Event:
        if event.version != 0:
            raise ValueError(f"event already stamped with version {event.version}")
        stamped = event.model_copy(update={"version": self.version + 1})
        self.events.append(stamped)
        return stamped

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
