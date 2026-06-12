"""PlaybookAgent — the Playbook engine behind the public Agent protocol.

Implements :class:`superdialog.agent.Agent` so SessionWorker and every host
adapter run a Playbook unchanged. Each turn runs the Director
(``runtime.on_user_text``) concurrently with the Talker stream: the Talker
speaks from the current state while the Director settles in the background;
hard gates barrier on ``director_done``, which resolves to the quiescent
state once ``on_user_text`` returns.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, cast

import anyio
from anyio.abc import TaskStatus

from ..agent import TurnResult
from ..chat_context import ChatContext, ChatMessage, Role
from ..stream import StreamChunk, Turn
from .director import CompletesLLM
from .events import EventLog, UtteranceEvent
from .models import Playbook
from .runtime import PlaybookRuntime
from .state import ConversationState
from .talker import SpeechChunk, StreamsLLM, Talker
from .toolexec import HttpFn, PythonToolFn


def _abandoned(eg: BaseExceptionGroup) -> bool:
    """True when every leaf is a GeneratorExit/cancellation (barge-in)."""
    rest = eg.split((GeneratorExit, anyio.get_cancelled_exc_class()))[1]
    return rest is None


class PlaybookAgent:
    """Playbook engine as a drop-in :class:`superdialog.agent.Agent`.

    ``runtime`` is public: hosts may call ``agent.runtime.start()`` to seed
    the session eagerly, feed external events, or inspect state. A turn on a
    never-started runtime starts it automatically.
    """

    def __init__(
        self,
        playbook: Playbook,
        talker_llm: StreamsLLM,
        director_llm: CompletesLLM,
        http: HttpFn,
        python_tools: dict[str, PythonToolFn] | None = None,
        token_budget: int = 4000,
        barrier_timeout: float = 0.4,
        hold_timeout: float = 5.0,
    ) -> None:
        self.runtime = PlaybookRuntime(
            playbook,
            director_llm=director_llm,
            http=http,
            python_tools=python_tools,
        )
        self._talker = Talker(
            playbook,
            talker_llm,
            token_budget=token_budget,
            barrier_timeout=barrier_timeout,
            hold_timeout=hold_timeout,
        )

    # ---- Agent Protocol -----------------------------------------------------

    async def turn(
        self,
        text: str,
        *,
        stream: bool = False,
    ) -> TurnResult | AsyncIterator[StreamChunk]:
        """Process one user turn; stream chunks live when ``stream=True``."""
        if stream:
            return self._stream_turn(text)
        final = Turn(text="")
        async for chunk in self._stream_turn(text):
            if chunk.done and chunk.turn is not None:
                final = chunk.turn
        return TurnResult(text=final.text, metadata=final.metadata)

    def assist(self, text: str) -> None:
        """Push a system-level note into the log; takes effect next turn."""
        if not text:
            return
        self.runtime.log.append(UtteranceEvent(role="system", text=text))

    @property
    def chat_ctx(self) -> ChatContext:
        """Brain-agnostic view of the transcript (roles map 1:1)."""
        return ChatContext(
            items=[
                ChatMessage(role=cast(Role, e.role), content=e.text)
                for e in self.runtime.state.transcript
            ]
        )

    def load_chat_ctx(self, ctx: ChatContext) -> None:
        """Seed a fresh event log from the context's utterances."""
        log = EventLog()
        for m in ctx.items:
            if m.role == "tool":
                continue  # tool messages have no utterance shape in the log
            log.append(UtteranceEvent(role=m.role, text=m.content))
        self.load_event_log(log)

    # ---- full-fidelity persistence -------------------------------------------

    @property
    def event_log(self) -> EventLog:
        """The runtime's append-only event log (single source of truth)."""
        return self.runtime.log

    def load_event_log(self, log: EventLog) -> None:
        """Replace the runtime's event log wholesale (lossless restore)."""
        self.runtime.load_log(log)

    # ---- internals ------------------------------------------------------------

    async def _stream_turn(self, text: str) -> AsyncIterator[StreamChunk]:
        """Talker chunks live, then pass-through, then the done chunk.

        Barge-in semantics: aborting this generator (host ``aclose()`` or
        cancellation mid-stream) interrupts SPEECH, not the state machine.
        The Talker stream stops, but the Director (``on_user_text``) runs
        to completion in a shielded scope — the user interrupted speech,
        so its decision still lands and the log stays quiescent. Cleanup
        is shielded too: partial talker speech is always logged exactly
        once and ``check_repairs`` always runs, even mid-abort, and the
        abort never escapes ``aclose()``.
        """
        pass_through = await self._ensure_started()
        quiescent = anyio.Event()

        async def run_director(
            *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED
        ) -> None:
            # Shielded: a barge-in cancels the Talker, never the Director.
            # started() fires only once the shield is engaged, so tg.start
            # guarantees the decision lands even on an immediate abort.
            with anyio.CancelScope(shield=True):
                task_status.started()
                # The utterance is already in the log (appended before the
                # snapshot below); record=False avoids a double-append.
                pass_through.extend(await self.runtime.on_user_text(text, record=False))
                quiescent.set()

        async def director_done() -> ConversationState:
            # Event-guarded: idempotent and cancellation-safe, as the
            # Talker's barrier contract requires. "Done" == quiescent,
            # because on_user_text only returns at quiescence.
            await quiescent.wait()
            return self.runtime.state

        # Append the current user utterance BEFORE snapshotting so the Talker
        # renders a transcript that ends at THIS turn (not the previous one).
        # run_director runs on_user_text with record=False to avoid a
        # double-append.
        self.runtime.log.append(UtteranceEvent(role="user", text=text))
        # Snapshot AFTER the append (includes the current turn) but BEFORE the
        # Director mutates state further: the Talker speaks from this snapshot;
        # hard gates barrier via director_done.
        speak_state = self.runtime.state
        entry_cp = speak_state.checkpoint_id
        talker_chunks: list[SpeechChunk] = []
        speech = self._talker.speak(speak_state, director_done=director_done)
        aborted = False
        try:
            async with anyio.create_task_group() as tg:
                await tg.start(run_director)
                async for chunk in speech:
                    talker_chunks.append(chunk)
                    if chunk.text:
                        yield StreamChunk(text=chunk.text)
            # task-group exit == Director done: pass-through is complete here
        except BaseExceptionGroup as eg:
            # A GeneratorExit thrown at the yield (or a cancellation) can
            # surface from the task group wrapped in a group; a pure-abort
            # group just means the host abandoned the stream.
            if not _abandoned(eg):
                raise
            aborted = True
        finally:
            # Runs on GeneratorExit too; shield so the async cleanup
            # survives the abort. The task group has already waited for the
            # shielded Director, so the log is quiescent here.
            with anyio.CancelScope(shield=True):
                await speech.aclose()  # close the Talker's LLM stream now
                talker_text = "".join(c.text for c in talker_chunks).strip()
                # If the Director advanced the checkpoint this turn AND there
                # is pass-through, the Talker spoke from the PRE-advance state:
                # its reply is stale (e.g. re-asking for a slot just filled).
                # Suppress it — keep the LOG honest and the final Turn.text
                # clean. The new checkpoint's verbatim pass-through is the
                # authoritative reply. (Live-streamed Talker chunks were
                # already yielded; that is an accepted streaming limitation.)
                advanced = self.runtime.state.checkpoint_id != entry_cp
                suppress = advanced and bool(pass_through)
                if talker_text and not suppress:
                    # The runtime never sees Talker speech; log it here —
                    # exactly once, partial or complete.
                    self.runtime.log.append(
                        UtteranceEvent(
                            role="assistant",
                            text=talker_text,
                            spoke_from_version=talker_chunks[-1].spoke_from_version,
                        )
                    )
                await self.runtime.check_repairs()
        if aborted:
            return  # exit without yielding so the host's aclose() is clean

        for line in pass_through:
            yield StreamChunk(text=line)
        if suppress:
            full = " ".join(pass_through).strip()
        else:
            full = talker_text
            if pass_through:
                full = (talker_text + " " + " ".join(pass_through)).strip()
        yield StreamChunk(done=True, turn=Turn(text=full, metadata=self._metadata()))

    async def _ensure_started(self) -> list[str]:
        """Start a never-started runtime; return its pass-through speech."""
        state = self.runtime.state
        if state.checkpoint_id is None and not state.ended:
            return await self.runtime.start()
        return []

    def _metadata(self) -> dict[str, Any]:
        state = self.runtime.state
        meta: dict[str, Any] = {
            "checkpoint": state.checkpoint_id,
            "version": state.version,
            "ended": state.ended,
        }
        if state.ended:
            meta["outcome"] = state.outcome
        return meta


__all__ = ["PlaybookAgent"]
