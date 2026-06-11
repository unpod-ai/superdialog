"""Talker: the fast path — one streaming call, tokens straight to TTS (§2/§2b)."""

from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

import anyio
from pydantic import BaseModel

from .models import Playbook
from .render import render_template, render_view
from .state import ConversationState

FILLER = "One moment, let me confirm that…"
HOLD_LINE = "I'm taking a little longer than usual — bear with me for a moment."
RECOVERY_LINE = "Sorry, could you say that again?"


class StreamsLLM(Protocol):
    """Anything that can stream plain-text tokens for a chat prompt."""

    def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]: ...


class SpeechChunk(BaseModel):
    """One streamed piece of speech, tagged with the state version spoken from."""

    text: str
    final: bool = False
    spoke_from_version: int = 0


class Talker:
    """One streaming LLM call per spoken turn; verbatim bypass; gated barrier."""

    def __init__(
        self,
        playbook: Playbook,
        llm: StreamsLLM,
        token_budget: int = 4000,
        barrier_timeout: float = 0.4,
        hold_timeout: float = 5.0,
        filler: str = FILLER,
        hold_line: str = HOLD_LINE,
        recovery_line: str = RECOVERY_LINE,
    ) -> None:
        self._pb = playbook
        self._llm = llm
        self._budget = token_budget
        self._barrier_timeout = barrier_timeout
        self._hold_timeout = hold_timeout
        self._filler = filler
        self._hold_line = hold_line
        self._recovery_line = recovery_line

    async def speak(
        self,
        state: ConversationState,
        director_done: Callable[[], Awaitable[ConversationState]] | None = None,
    ) -> AsyncIterator[SpeechChunk]:
        """Stream one spoken turn for ``state``, barriering at hard gates.

        ``director_done`` contract:

        * "Done" means the runtime is QUIESCENT — the Director's decision has
          been applied to state AND any hard-gate pipeline has completed (not
          merely that the Director's LLM call returned).
        * ``director_done`` is called up to twice; the first call's coroutine
          is cancelled when the barrier times out. It must therefore be
          idempotent and cancellation-safe (an Event-guarded result
          qualifies).
        * Callers at hard gates must supply ``director_done`` — without it
          the barrier is skipped entirely.
        """
        cp = self._pb.checkpoint(state.checkpoint_id) if state.checkpoint_id else None

        if cp is not None and cp.gate == "hard" and director_done is not None:
            fresh: ConversationState | None = None
            with anyio.move_on_after(self._barrier_timeout):
                fresh = await director_done()
            if fresh is None:
                # Filler is yielded HERE — between the expired barrier and the
                # second wait — so the listener hears it while the Director is
                # still pending. director_done() is called fresh below; the
                # coroutine above was cancelled by move_on_after.
                yield SpeechChunk(
                    text=self._filler + " ", spoke_from_version=state.version
                )
                with anyio.move_on_after(self._hold_timeout):
                    fresh = await director_done()
            if fresh is None:  # Director is down: degrade politely, never hang
                yield SpeechChunk(
                    text=self._hold_line, final=True, spoke_from_version=state.version
                )
                return
            state = fresh
            cp = (
                self._pb.checkpoint(state.checkpoint_id)
                if state.checkpoint_id
                else None
            )

        if cp is not None and cp.say_verbatim is not None:
            text = render_template(cp.say_verbatim, self._pb, state)
            yield SpeechChunk(
                text=text.strip(), final=True, spoke_from_version=state.version
            )
            return

        view = render_view(self._pb, state, token_budget=self._budget)
        # NOTE: a partial stream that fails midway replays from the start on
        # retry — acceptable for v1 (the retry targets connect-time failures;
        # mid-stream resume is a host concern).
        for attempt in (1, 2):
            try:
                stream = self._llm.stream(view.messages)
                try:
                    async for token in stream:
                        yield SpeechChunk(
                            text=token, spoke_from_version=view.spoke_from_version
                        )
                finally:
                    # Close the inner stream even when the host aborts speak()
                    # mid-stream (e.g. LiveKit barge-in calls aclose()) —
                    # otherwise the streaming HTTP response leaks per barge-in.
                    aclose = getattr(stream, "aclose", None)
                    if aclose is not None:
                        await aclose()
                yield SpeechChunk(
                    text="", final=True, spoke_from_version=view.spoke_from_version
                )
                return
            except Exception:
                if attempt == 2:
                    yield SpeechChunk(
                        text=self._recovery_line,
                        final=True,
                        spoke_from_version=view.spoke_from_version,
                    )
                    return
