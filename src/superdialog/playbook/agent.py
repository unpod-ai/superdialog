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
from .director import AnchorMode, CompletesLLM
from .events import EventLog, SpeechCorrectionEvent, SummaryEvent, UtteranceEvent
from .models import Playbook
from .runtime import PlaybookRuntime
from .state import ConversationState
from .supervisor import Supervisor
from .talker import SpeechChunk, SpokenLine, StreamsLLM, Talker
from .toolexec import HttpFn, PythonToolFn

logger = logging.getLogger(__name__)


class _LLMTimer:
    """Wraps director/talker LLM, records latency per call and per user turn.

    Call begin_turn() / end_turn() around each user turn so per-turn totals
    can be reported alongside the overall mean/p95.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.latencies_ms: list[float] = []  # every call, flat
        self.ttft_ms: list[float] = []  # stream calls only: time to first chunk
        self._turn_buckets: list[list[float]] = []  # per turn, list of call durations
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

    async def stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        first_seen = False
        try:
            async for chunk in self._inner.stream(messages, **kw):
                if not first_seen:
                    first_seen = True
                    self.ttft_ms.append((time.perf_counter() - t0) * 1000)
                yield chunk
        finally:
            self._record((time.perf_counter() - t0) * 1000)

    @property
    def stats(self) -> dict[str, Any]:
        if not self.latencies_ms:
            return {"calls": 0, "mean_ms": 0.0, "p95_ms": 0.0, "per_turn_ms": []}
        s = sorted(self.latencies_ms)
        per_turn = [round(sum(b), 1) for b in self._turn_buckets]
        out: dict[str, Any] = {
            "calls": len(s),
            "mean_ms": round(sum(s) / len(s), 1),
            "p95_ms": round(s[int(len(s) * 0.95)], 1),
            "per_turn_ms": per_turn,  # index 0 = turn 1; total LLM ms per user turn
        }
        if self.ttft_ms:  # only when streams were timed; director stats unchanged
            t = sorted(self.ttft_ms)
            out["ttft_mean_ms"] = round(sum(t) / len(t), 1)
            out["ttft_p95_ms"] = round(t[int(len(t) * 0.95)], 1)
        return out


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
        settle_before_speak: bool = False,
        supervisor_llm: CompletesLLM | None = None,
        intercept_llm: CompletesLLM | None = None,
        filler: SpokenLine | None = None,
        hold_line: SpokenLine | None = None,
        allow_private_hosts: bool = False,
        # G37 slot-evidence anchor: shadow audits anchor_miss, enforce rejects
        # unanchored writes, off disables (see Director._anchor_ok).
        anchor: AnchorMode = "shadow",
    ) -> None:
        # Offline-eval knob: when True the Talker waits for the Director to
        # settle before speaking on EVERY turn (not just the greeting), so a
        # turn-based harness captures the reply from the checkpoint the Director
        # actually chose. Off in live voice, where the Talker speaks
        # speculatively and only barriers at hard gates.
        self._settle_before_speak = settle_before_speak
        self._director_timer = _LLMTimer(director_llm)
        self._talker_timer = _LLMTimer(talker_llm)
        self.runtime = PlaybookRuntime(
            playbook,
            director_llm=self._director_timer,
            http=http,
            python_tools=python_tools,
            intercept_llm=intercept_llm,
            allow_private_hosts=allow_private_hosts,
            anchor=anchor,
        )
        # Loop 2 (off the speech path): reviews the trajectory after a turn
        # completes, only when a trigger fires. An explicit ``supervisor_llm``
        # wins; else ``guidelines.supervisor`` decides (None resolves to on)
        # and reuses the Director model (raw, untimed — the call is off the
        # speech path, so it must not smear the Director's latency stats).
        _sup_flag = playbook.guidelines.supervisor
        _sup_on = _sup_flag if _sup_flag is not None else True
        sup_llm = supervisor_llm or (director_llm if _sup_on else None)
        self._supervisor = Supervisor(sup_llm, playbook) if sup_llm else None
        # Barrier lines: None keeps the Talker's built-in defaults. A host may
        # pass static text or a provider called with the live state at speak
        # time (language-aware fillers); explicit args win over the playbook's
        # own authored `policies.filler` / `policies.hold_line`.
        _line_kwargs: dict[str, SpokenLine] = {}
        _filler = filler if filler is not None else playbook.policies.filler
        if _filler is not None:
            _line_kwargs["filler"] = _filler
        _hold_line = hold_line if hold_line is not None else playbook.policies.hold_line
        if _hold_line is not None:
            _line_kwargs["hold_line"] = _hold_line
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
            **_line_kwargs,
        )
        self._traversal_dir: Path | None = (
            Path(traversal_dir) if traversal_dir else None
        )
        # The playbook's real file name wins over the host's static label:
        # production traversals from different agents all claimed the same
        # source because hosts pass one fixed traversal_source string.
        self._traversal_source = (
            (playbook.source_path and Path(playbook.source_path).name)
            or traversal_source
            or ""
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
        language: str | None = None,
    ) -> TurnResult | AsyncIterator[StreamChunk]:
        """Process one user turn; stream chunks live when ``stream=True``.

        When stream=True the coroutine returns an AsyncIterator — callers must
        ``await`` before iterating: ``async for chunk in await agent.turn(t, stream=True)``.
        Prefer ``agent.stream_turn(text)`` which requires no ``await``.

        ``language`` is the bridge-detected language of this turn; it is
        stamped on the user utterance so the reply adheres to the caller.
        """
        if stream:
            return self._stream_turn(text, language)
        final = Turn(text="")
        async for chunk in self._stream_turn(text, language):
            if chunk.done and chunk.turn is not None:
                final = chunk.turn
        return TurnResult(text=final.text, metadata=final.metadata)

    def stream_turn(
        self, text: str, language: str | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream one user turn — use as: ``async for chunk in agent.stream_turn(text)``.

        No ``await`` needed: returns the async iterator directly.
        """
        return self._stream_turn(text, language)

    def assist(self, text: str) -> None:
        """Push a system-level note into the log; takes effect next turn."""
        if not text:
            return
        self.runtime.log.append(UtteranceEvent(role="system", text=text))

    def mark_interrupted(self, heard_text: str | None = None) -> None:
        """Correct the last assistant utterance to what the caller actually heard.

        A barge-in cancels the Talker's SPEECH, but the shielded finally still
        logs the FULL generated reply as if spoken. That leaves the Director
        reasoning next turn against words the caller never heard — the root of
        "you already told me X" re-asks. The host calls this (from the SDK
        Session on a mid-stream ``UserInterruptEvent``) to append a
        ``SpeechCorrectionEvent`` truncating the transcript record to the heard
        prefix, tagged ``[interrupted by caller]``. Append-only: the original
        event keeps what was GENERATED; the fold's transcript shows what was
        DELIVERED.

        ``heard_text=None`` → keep the logged text, just tag it interrupted
        (best-effort when the worker sent no prefix). No-op when there is no
        assistant utterance to correct. Inert until a caller invokes it.
        Degenerate case: a second ``mark_interrupted(None)`` on the same
        utterance resets the record to full-text-plus-tag (the reverse scan
        reads the original event, not the prior correction) — acceptable
        because the SDK fires one interrupt per utterance.
        """
        for event in reversed(self.runtime.log.events):
            if isinstance(event, UtteranceEvent) and event.role == "user":
                # Nothing logged this turn (all-filler barge-in, or post-
                # terminal user turn): there is no utterance of THIS turn to
                # correct — correcting an older one would corrupt history.
                return
            if isinstance(event, UtteranceEvent) and event.role == "assistant":
                base = heard_text if heard_text else event.text
                # The append bumps log.version, so runtime.state refolds
                # naturally — no explicit cache invalidation needed.
                self.runtime.log.append(
                    SpeechCorrectionEvent(
                        utterance_version=event.version,
                        heard_text=f"{base} [interrupted by caller]",
                    )
                )
                return

    def apply_memory(self, summary: str) -> None:
        """Seed prior-call context for a returning caller; takes effect next turn.

        The cross-call digest is deployment-supplied; this is the framework hook
        that surfaces it to the Talker (rendered under '## Earlier in this
        conversation', guarded by the memory guidelines when
        ``guidelines.memory_enabled`` is set)."""
        if summary and summary.strip():
            self.runtime.log.append(SummaryEvent(text=summary.strip()))

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

    async def _stream_turn(
        self, text: str, language: str | None = None
    ) -> AsyncIterator[StreamChunk]:
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
        # Post-terminal short-circuit: once the session has ended, a further
        # user turn must NOT resurrect the call. Both production Westgate
        # transcripts show the flow speaking after it ended — the closing
        # replayed on every "Hello?", and a post-close utterance restarted the
        # pitch. Neither the Director (an LLM call per zombie turn) nor the
        # Talker (which re-speaks the terminal verbatim / re-greets) may run
        # here. Record the utterance for transcript/audit continuity, mark it,
        # and return silence so the host can disconnect cleanly.
        if self.runtime.state.ended:
            self.runtime.log.append(
                UtteranceEvent(role="user", text=text, language=language)
            )
            logger.info(
                "[PlaybookAgent] post-terminal turn ignored (ended); cp=%s",
                self.runtime.state.checkpoint_id,
            )
            for line in pass_through:  # normally empty; flush any pending speech
                yield StreamChunk(text=line)
            yield StreamChunk(done=True, turn=Turn(text="", metadata=self._metadata()))
            return
        self._director_timer.begin_turn()
        self._talker_timer.begin_turn()
        quiescent = anyio.Event()

        async def run_director() -> None:
            # Shielded: a barge-in cancels the Talker, never the Director.
            # The shield is task-local to this background task — entered and
            # exited in the same task — so it is safe under a foreign abort.
            with anyio.CancelScope(shield=True):
                try:
                    # The utterance is already in the log (appended before the
                    # snapshot below); record=False avoids a double-append.
                    pass_through.extend(
                        await self.runtime.on_user_text(text, record=False)
                    )
                except Exception as exc:  # noqa: BLE001 — loud, never silent
                    # A dead Director must not strand the Talker on its
                    # barrier every turn (it would re-speak the entry
                    # checkpoint forever with zero log evidence).
                    logger.error(
                        "[DIRECTOR] TURN FAILED %s cp=%s",
                        type(exc).__name__,
                        self.runtime.state.checkpoint_id,
                        exc_info=True,
                    )
                finally:
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
        self.runtime.log.append(
            UtteranceEvent(role="user", text=text, language=language)
        )
        entry_cp = self.runtime.state.checkpoint_id

        # Start the Director concurrently so it is (usually) quiescent by the
        # time the Talker barriers at a hard gate. Nothing cancels this task;
        # it is shielded and always awaited in the finally below.
        director = asyncio.ensure_future(run_director())

        # First-turn double-greeting guard: if we are still at the checkpoint
        # where greet() spoke, wait briefly for the Director to advance before
        # snapshotting speak_state — otherwise the Talker would re-speak the
        # opening greeting from the same checkpoint a second time.
        greeting_turn = bool(
            self._greeting_checkpoint and entry_cp == self._greeting_checkpoint
        )
        if greeting_turn:
            self._greeting_checkpoint = None
        # Wait for the Director to settle before snapshotting when either the
        # greeting guard fires or offline settle_before_speak is on. Otherwise
        # (live voice) skip the wait and let the Talker speak speculatively.
        if greeting_turn or self._settle_before_speak:
            with anyio.move_on_after(self._talker._hold_timeout):
                await quiescent.wait()

        # Snapshot AFTER the optional barrier: if Director advanced, Talker
        # speaks from the new checkpoint; if it timed out, speaks from entry_cp.
        speak_state = self.runtime.state
        talker_chunks: list[SpeechChunk] = []
        completed_normally = False
        speech = self._talker.speak(speak_state, director_done=director_done)
        try:
            async for chunk in speech:
                talker_chunks.append(chunk)
                if chunk.text:
                    # GeneratorExit may be thrown here on barge-in: it unwinds
                    # straight to the finally (no task group to exit), so the
                    # foreign-task abort is clean.
                    yield StreamChunk(text=chunk.text)
            completed_normally = True
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
                talker_text = "".join(
                    c.text for c in talker_chunks if not c.filler
                ).strip()
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
                # Loop 2: trajectory review. Trigger detection is pure (free);
                # an LLM verdict is spent only when derailed. Skipped on
                # barge-in abort so cleanup stays fast — the triggers persist
                # and the next completed turn picks them up.
                if self._supervisor is not None and completed_normally:
                    try:
                        decision = await self._supervisor.review(self.runtime)
                        if decision is not None:
                            # Loud like [DIRECTOR] verdict: a supervisor
                            # intervention is visible in eval/run logs (it
                            # otherwise lands only as silent events).
                            if decision.action != "none":
                                logger.info(
                                    "[SUPERVISOR] action=%s reason=%s cp=%s",
                                    decision.action,
                                    decision.reason or "-",
                                    self.runtime.state.checkpoint_id,
                                )
                            pass_through.extend(
                                await self._supervisor.apply(self.runtime, decision)
                            )
                    except Exception as exc:  # noqa: BLE001 — loud, never fatal
                        logger.error(
                            "[SUPERVISOR] FAILED %s",
                            type(exc).__name__,
                            exc_info=True,
                        )
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
