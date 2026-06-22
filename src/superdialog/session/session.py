"""Session — per-conversation lifecycle shell + SessionHandle helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..chat_context import ChatContext, ChatMessage
from ..flow_state import FlowState
from ..stream import StreamChunk

if TYPE_CHECKING:
    from ..agent import Agent, TurnResult


@dataclass
class SessionInit:
    """Per-session construction context for a context-aware agent factory.

    Carries the ``session_id`` and arbitrary ``metadata`` (e.g. ``playbook_id``)
    supplied at :meth:`SessionWorker.acquire` time, so one worker process can
    build a *different* Agent per session.
    """

    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Per-conversation persistent state.

    A Session carries only durable data (id, chat_ctx, flow_state). It does
    not hold a reference to an Agent — that lifetime is owned by the
    SessionWorker. The Worker yields a :class:`SessionHandle` that pairs
    the Session with its currently-loaded Agent for the duration of an
    ``acquire(...)`` block.
    """

    id: str
    chat_ctx: ChatContext = field(default_factory=ChatContext)
    flow_state: FlowState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def assist(self, text: str) -> None:
        """Append a system message to ``chat_ctx``; takes effect on the next turn."""
        if not text:
            return
        self.chat_ctx.items.append(ChatMessage(role="system", content=text))


class SessionHandle:
    """The object yielded by ``SessionWorker.acquire``.

    Pairs a Session with its currently-loaded Agent so callers can drive a
    turn without touching either directly. Delegates ``turn`` to the agent
    and ``assist`` to the session.
    """

    def __init__(self, session: Session, agent: "Agent") -> None:
        self._session = session
        self._agent = agent

    @property
    def session(self) -> Session:
        return self._session

    @property
    def agent(self) -> "Agent":
        return self._agent

    async def turn(
        self, text: str, *, stream: bool = False
    ) -> "TurnResult | AsyncIterator[StreamChunk]":
        return await self._agent.turn(text, stream=stream)

    def assist(self, text: str) -> None:
        """Push a system instruction to both the live Agent and the Session.

        Mutating both keeps the Session's snapshot honest if the Agent has
        already loaded chat_ctx and is now appending live messages.
        """
        self._session.assist(text)
        self._agent.assist(text)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "id": self._session.id,
            "chat_ctx": self._session.chat_ctx,
            "flow_state": self._session.flow_state,
        }


__all__ = ["Session", "SessionHandle", "SessionInit"]
