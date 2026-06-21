"""PlaybookAgent — the Playbook engine behind the public Agent protocol.

Implements :class:`superdialog.agent.Agent` so SessionWorker and every host
adapter run a Playbook unchanged. Each turn runs the Director
(``runtime.on_user_text``) concurrently with the Talker stream: the Talker
speaks from the current state while the Director settles in the background;
hard gates barrier on ``director_done``, which resolves to the quiescent
state once ``on_user_text`` returns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, cast

import anyio

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

logger = logging.getLogger(__name__)


class _LLMTimer:
    """Wraps director/talker LLM, records latency per call and per user turn.

    Call begin_turn() / end_turn() around each user turn so per-turn totals
    can be reported alongside the overall mean/p95.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.latencies_ms: list[float] = []          # every call, flat
        self._turn_buckets: list[list[float]] = []   # per turn, list of call durations
        self._current_bucket: list[float] | None = None

    def begin_turn(self) -> None:
        self._current_bucket = []
        self._turn_buckets.append(self._current_bucket)

    def end_turn(self) -> None:
        self._current_bucket = None

    def _record(self, elapsed_ms: float) -> None:
        self.latencies_ms.append(elapsed_ms)
        if self._current_bucket is not None:
            self._current_bucket.append(elapsed_ms)

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        t0 = time.perf_counter()
        result = await self._inner.complete(messages, **kw)
        self._record((time.perf_counter() - t0) * 1000)
        return result

    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        try:
            async for chunk in self._inner.stream(messages, **kw):
                yield chunk
        finally:
            self._record((time.perf_counter() - t0) * 1000)

    @property
    def stats(self) -> dict[str, Any]:
        if not self.latencies_ms:
            return {"calls": 0, "mean_ms": 0.0, "p95_ms": 0.0, "per_turn_ms": []}
        s = sorted(self.latencies_ms)
        per_turn = [round(sum(b), 1) for b in self._turn_buckets]
        return {
            "calls": len(s),
            "mean_ms": round(sum(s) / len(s), 1),
            "p95_ms": round(s[int(len(s) * 0.95)], 1),
            "per_turn_ms": per_turn,   # index 0 = turn 1; total LLM ms per user turn
        }


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
        barrier_timeout: float = 4.0,
        hold_timeout: float | None = None,
        traversal_dir: str | Path | None = None,
        traversal_source: str = "",
        traversal_model: str = "",
    ) -> None:
        self._director_timer = _LLMTimer(director_llm)
        self._talker_timer = _LLMTimer(talker_llm)
        self.runtime = PlaybookRuntime(
            playbook,
            director_llm=self._director_timer,
            http=http,
            python_tools=python_tools,
        )
        self._talker = Talker(
            playbook,
            self._talker_timer,
            token_budget=token_budget,
            barrier_timeout=barrier_timeout,
            # explicit arg wins; else the playbook's policies decide
            hold_timeout=(
                hold_timeout
                if hold_timeout is not None
                else playbook.policies.hold_timeout
            ),
        )
        self._traversal_dir: Path | None = Path(traversal_dir) if traversal_dir else None
        self._traversal_source = traversal_source or (
            playbook.source_path and Path(playbook.source_path).name or ""
        )
        self._traversal_model = traversal_model or getattr(director_llm, "model_id", "")
        self._traversal_saved: bool = False
        self._started_at: datetime | None = None
        self._greeting_checkpoint: str | None = None
        self._playbook = playbook

    # ---- Agent Protocol -----------------------------------------------------

    async def turn(
        self,
        text: str,
        *,
        stream: bool = False,
    ) -> TurnResult | AsyncIterator[StreamChunk]:
        """Process one user turn; stream chunks live when ``stream=True``.

        When stream=True the coroutine returns an AsyncIterator — callers must
        ``await`` before iterating: ``async for chunk in await agent.turn(t, stream=True)``.
        Prefer ``agent.stream_turn(text)`` which requires no ``await``.
        """
        if stream:
            return self._stream_turn(text)
        final = Turn(text="")
        async for chunk in self._stream_turn(text):
            if chunk.done and chunk.turn is not None:
                final = chunk.turn
        return TurnResult(text=final.text, metadata=final.metadata)

    def stream_turn(self, text: str) -> AsyncIterator[StreamChunk]:
        """Stream one user turn — use as: ``async for chunk in agent.stream_turn(text)``.

        No ``await`` needed: returns the async iterator directly.
        """
        return self._stream_turn(text)

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
        self._traversal_saved = False

    # ---- internals ------------------------------------------------------------

    async def _stream_turn(self, text: str) -> AsyncIterator[StreamChunk]:
        """Talker chunks live, then pass-through, then the done chunk.

        Barge-in semantics: aborting this generator (host ``aclose()`` or
        cancellation mid-stream) interrupts SPEECH, not the state machine.
        The Talker stream stops, but the Director (``on_user_text``) runs to
        completion in a shielded background task — the user interrupted
        speech, so its decision still lands and the log stays quiescent.
        Cleanup is shielded too: partial talker speech is always logged
        exactly once and ``check_repairs`` always runs, even mid-abort.

        The Director runs as a detached ``asyncio`` task, NOT inside an
        ``anyio`` task group. A task group's cancel scope is task-affine —
        it must be exited in the task that entered it — but the host (or the
        async-generator finalizer) routinely ``aclose()``s this generator
        from a *different* task on barge-in, which would raise
        ``RuntimeError: Attempted to exit cancel scope in a different task``.
        An async generator must therefore never ``yield`` inside an anyio
        task group / cancel scope; the background task sidesteps that.
        """
        pass_through = await self._ensure_started()
        self._director_timer.begin_turn()
        self._talker_timer.begin_turn()
        quiescent = anyio.Event()

        async def run_director() -> None:
            # Shielded: a barge-in cancels the Talker, never the Director.
            # The shield is task-local to this background task — entered and
            # exited in the same task — so it is safe under a foreign abort.
            with anyio.CancelScope(shield=True):
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
        entry_cp = self.runtime.state.checkpoint_id

        # Start the Director concurrently so it is (usually) quiescent by the
        # time the Talker barriers at a hard gate. Nothing cancels this task;
        # it is shielded and always awaited in the finally below.
        director = asyncio.ensure_future(run_director())

        # First-turn double-greeting guard: if we are still at the checkpoint
        # where greet() spoke, wait briefly for the Director to advance before
        # snapshotting speak_state — otherwise the Talker would re-speak the
        # opening greeting from the same checkpoint a second time.
        if self._greeting_checkpoint and entry_cp == self._greeting_checkpoint:
            self._greeting_checkpoint = None
            with anyio.move_on_after(self._talker._hold_timeout):
                await quiescent.wait()

        # Snapshot AFTER the optional barrier: if Director advanced, Talker
        # speaks from the new checkpoint; if it timed out, speaks from entry_cp.
        speak_state = self.runtime.state
        talker_chunks: list[SpeechChunk] = []
        speech = self._talker.speak(speak_state, director_done=director_done)
        try:
            async for chunk in speech:
                talker_chunks.append(chunk)
                if chunk.text:
                    # GeneratorExit may be thrown here on barge-in: it unwinds
                    # straight to the finally (no task group to exit), so the
                    # foreign-task abort is clean.
                    yield StreamChunk(text=chunk.text)
        finally:
            # Runs on normal completion AND on GeneratorExit; shield so the
            # async cleanup survives the abort. Entered and exited within this
            # one finally, in whatever task drives the close — never spanning
            # a yield — so the cancel scope is task-consistent.
            with anyio.CancelScope(shield=True):
                self._director_timer.end_turn()
                self._talker_timer.end_turn()
                await speech.aclose()  # close the Talker's LLM stream now
                # Let the Director's decision land. It is shielded, so the only
                # cancellation that can reach it is loop teardown — tolerate
                # that; real Director errors still surface.
                try:
                    await director
                except (asyncio.CancelledError, anyio.get_cancelled_exc_class()):
                    pass
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
                if (
                    self._traversal_dir
                    and self.runtime.state.ended
                    and not self._traversal_saved
                ):
                    self._traversal_saved = True
                    await self._auto_save_traversal()
        # Reached only on normal completion — a GeneratorExit raised at the
        # yield above propagates out of the finally, so the host's aclose()
        # stays clean and the pass-through below is skipped on abort.
        for line in pass_through:
            yield StreamChunk(text=line)
        if suppress:
            full = " ".join(pass_through).strip()
        else:
            full = talker_text
            if pass_through:
                full = (talker_text + " " + " ".join(pass_through)).strip()
        yield StreamChunk(done=True, turn=Turn(text=full, metadata=self._metadata()))

    async def greet(self) -> AsyncIterator[StreamChunk]:
        """Speak and log the opening greeting (outbound-call: agent speaks first).

        Streams speech chunks from the initial checkpoint state, then logs the
        full text as an assistant utterance so the Director sees it next turn.
        """
        await self._ensure_started()
        state = self.runtime.state
        self._greeting_checkpoint = state.checkpoint_id
        talker_chunks: list[SpeechChunk] = []
        async for chunk in self._talker.speak(state):
            talker_chunks.append(chunk)
            if chunk.text:
                yield StreamChunk(text=chunk.text)
        text = "".join(c.text for c in talker_chunks).strip()
        if text:
            self.runtime.log.append(
                UtteranceEvent(
                    role="assistant",
                    text=text,
                    spoke_from_version=(
                        talker_chunks[-1].spoke_from_version if talker_chunks else 0
                    ),
                )
            )
        yield StreamChunk(done=True, turn=Turn(text=text, metadata=self._metadata()))

    async def _ensure_started(self) -> list[str]:
        """Start a never-started runtime; return its pass-through speech."""
        state = self.runtime.state
        if state.checkpoint_id is None and not state.ended:
            self._started_at = datetime.now(timezone.utc)
            return await self.runtime.start()
        return []

    async def _auto_save_traversal(self) -> None:
        """Save traversal JSON to _traversal_dir without blocking the event loop."""
        try:
            from .traversal import build_playbook_traversal, save_playbook_traversal

            traversal = build_playbook_traversal(
                self.runtime.log,
                self._playbook,
                source=self._traversal_source,
                model=self._traversal_model,
                started_at=self._started_at,
                latency={
                    "director": self._director_timer.stats,
                    "talker": self._talker_timer.stats,
                },
            )
            # File write is sync — offload to a thread so we don't block the loop.
            path = await anyio.to_thread.run_sync(
                lambda: save_playbook_traversal(traversal, self._traversal_dir)
            )
            logger.info("[PlaybookAgent] traversal saved: %s", path)
        except Exception:
            logger.warning("[PlaybookAgent] traversal save failed", exc_info=True)

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
