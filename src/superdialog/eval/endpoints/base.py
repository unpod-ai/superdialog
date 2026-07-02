"""Transport seam: the ConversationEndpoint protocol + a Transcript container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TurnRecord:
    """One line of the observable conversation."""

    role: str  # "assistant" | "user"
    text: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transcript:
    """Ordered record of a conversation — the ONLY substrate metrics may read."""

    records: list[TurnRecord] = field(default_factory=list)

    def add(
        self,
        role: str,
        text: str,
        *,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one turn to the transcript."""
        self.records.append(TurnRecord(role, text, latency_ms, metadata or {}))

    def to_messages(self) -> list[dict[str, str]]:
        """RAGAS/LLM-judge friendly [{role, content}] view."""
        return [{"role": r.role, "content": r.text} for r in self.records]

    def assistant_latencies_ms(self) -> list[float]:
        """Latencies (ms) of every assistant turn that recorded one."""
        return [
            r.latency_ms
            for r in self.records
            if r.role == "assistant" and r.latency_ms is not None
        ]

    def turn_count(self) -> int:
        """Number of user turns (a proxy for conversation length)."""
        return sum(1 for r in self.records if r.role == "user")


@runtime_checkable
class ConversationEndpoint(Protocol):
    """A black-box conversational agent the harness drives, transport-agnostic."""

    async def start(self) -> str:
        """Produce the opening assistant utterance (greeting)."""
        ...

    async def turn(self, text: str) -> str:
        """Feed one user utterance; return the assistant's reply text."""
        ...

    def reset(self) -> None:
        """Clear conversation state for a fresh case."""
        ...
