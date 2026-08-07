"""Talker: the fast path — one streaming call, tokens straight to TTS (§2/§2b)."""

from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

import anyio

_log = logging.getLogger(__name__)
from pydantic import BaseModel

from .models import Checkpoint, Playbook
from .render import render_template, render_view
from .state import ConversationState

FILLER = "One moment, let me confirm that…"
HOLD_LINE = "I'm taking a little longer than usual — bear with me for a moment."
RECOVERY_LINE = "Sorry, could you say that again?"

# A spoken line is either static text or a provider called with the live
# ConversationState at speak time (e.g. language-aware fillers).
SpokenLine = str | Callable[[ConversationState], str]


def _excise(buf: str, folded: list[str]) -> str:
    """Remove every casefolded occurrence of each phrase from ``buf``."""
    low = buf.casefold()
    for p in folded:
        i = low.find(p)
        while i != -1:
            buf = buf[:i] + buf[i + len(p) :]
            low = buf.casefold()
            i = low.find(p)
    return buf


async def _filter_never_say(
    tokens: AsyncIterator[str], phrases: list[str]
) -> AsyncIterator[str]:
    """Deterministic never_say enforcement on the token stream.

    Buffers just enough text (longest phrase − 1 chars) to catch phrases split
    across chunk boundaries, excises any casefolded occurrence, and flushes
    the confirmed-clean prefix — so an authored phrase can never reach TTS,
    whatever the LLM emits. Prompt-only enforcement is probabilistic; this is
    the guarantee. Adds no LLM calls and microseconds per chunk.
    """
    folded = [p.casefold() for p in phrases if p]
    if not folded:
        async for token in tokens:
            yield token
        return
    tail = max(len(p) for p in folded) - 1
    buf = ""
    async for token in tokens:
        buf = _excise(buf + token, folded)
        if len(buf) > tail:
            yield buf[: len(buf) - tail]
            buf = buf[len(buf) - tail :]
    buf = _excise(buf, folded)
    if buf:
        yield buf


# A second, unrelated hallucination shape observed in production: instead of
# a <|tool_call>...<tool_call|> tag, the Talker sometimes speaks a raw JSON
# tool-call dump as the ENTIRE turn, e.g.:
#   {"tool_calls": [{"function": "slot_to_profile_check",
#                     "parameters": {"slot_id": "..."}}]}
# Same root cause as the tag variant (routing-only guidance text leaking
# into the Talker's own prompt), different syntax the model reaches for.
# Real spoken turns in this domain never start with a raw '{' or '[', so
# that's the trigger: buffer (don't yield) while tracking bracket depth:
#   - depth returns to 0 and the buffered span is valid JSON containing a
#     tool-call-shaped key -> discard outright. speak() tracks whether the
#     filtered stream produced any content at all and treats a clean-but-
#     empty stream the same as a failed one (retry, then recovery_line) --
#     turning a spoken JSON dump into a polite "could you say that again?"
#     instead of dead air.
#   - anything else (doesn't parse as JSON, parses but lacks those keys,
#     or never closes before the stream ends) is flushed as-is -- same
#     conservative principle as the tag filter below: never discard on
#     ambiguity, only on a confirmed match.
_STRUCTURED_LEAK_MARKER_RE = re.compile(
    r'"(tool_calls?|function_call|function|parameters)"\s*:', re.IGNORECASE
)


async def _filter_structured_output_leak(
    tokens: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Suppress a turn whose entire output is a hallucinated JSON tool-call
    dump instead of natural speech. See module comment above for the
    detection/discard rules."""
    buf = ""
    watching: bool | None = None
    depth = 0
    async for token in tokens:
        if watching is None:
            buf += token
            stripped = buf.lstrip()
            if not stripped:
                continue
            watching = stripped[0] in "{["
            if not watching:
                yield buf
                buf = ""
                continue
            for ch in buf:
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
        elif watching:
            buf += token
            for ch in token:
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
        else:
            yield token
            continue

        if watching and depth <= 0:
            flush = True
            if _STRUCTURED_LEAK_MARKER_RE.search(buf):
                try:
                    json.loads(buf.strip())
                    flush = False
                except ValueError:
                    pass
            if flush:
                yield buf
            buf = ""
            watching = None
    if buf:
        yield buf


# The Talker is never meant to emit tool/function-call syntax -- state
# transitions are the Director's job via a separate channel (compiler.py
# compiles edges into AdvanceRule objects the Director evaluates; they are
# never exposed to the Talker as callable tools). But checkpoint guidance
# text (rendered into the Talker's own prompt by render_view -- the same
# field the Director uses to decide when to route) often literally says
# "call X" as a routing instruction, and a weak Talker model can mimic
# that as a fake tool-call token instead of speaking naturally. Nothing
# upstream recognizes this shape, so without this filter it goes straight
# to TTS. Tolerant of the open/close tag being asymmetric or malformed
# (observed in production: "<|tool_call>...<tool_call|>") since the model
# that hallucinates this is also inconsistent about its own syntax.
_TOOL_CALL_OPEN_RE = re.compile(r"<\|?\s*tool_call\s*[|>]", re.IGNORECASE)
_TOOL_CALL_CLOSE_RE = re.compile(r"<\s*tool_call\s*\|?>", re.IGNORECASE)
# Longest plausible split open-tag fragment straddling a chunk boundary.
_TOOL_CALL_OPEN_TAIL = 16
# End-of-stream disambiguator for an unclosed open tag (see docstring): the
# body is a *confirmed* hallucination -- safe to discard, not flush -- only
# when it's nothing but a bare function-call expression with no natural
# language around it, e.g. `_hold_slot(slot_id='slot_..._1200')` (observed
# in production, reached TTS verbatim because the old rule always flushed
# unclosed spans). Real trailing speech after an open tag ("Got it -- one
# player, day after tomorrow at nine") never matches this -- it has spaces,
# punctuation, no parens -- so it still flushes exactly as before.
_TOOL_CALL_BARE_INVOCATION_RE = re.compile(r"\A[A-Za-z_][\w.]*\s*\([^{}\n]*\)\s*\Z")


async def _filter_tool_call_leakage(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Strip hallucinated tool-call-shaped spans from the token stream.

    Buffers from the first open-tag sighting until a matching close tag and
    discards the whole span -- safer to speak nothing here than a malformed
    fragment, for a *confirmed* tool-call span (both tags seen).

    An open tag with no close tag by end of stream is flushed as-is by
    default, not discarded. Every real production sighting of this
    hallucination had a matching close tag; an *unclosed* open tag is
    ambiguous -- it may mean the model kept talking normally afterward
    without ever emitting the literal close-tag string, in which case
    discarding would silently eat everything from the open tag to the end
    of the turn. That happened: a caller's own confirmed answer ("one
    player") got swallowed this way, indistinguishable from a genuinely
    empty completion except no error logs anywhere, because real content
    *was* streamed -- this filter just never let it out.

    The one exception: if the unclosed body is a bare function-call
    expression and nothing else (see ``_TOOL_CALL_BARE_INVOCATION_RE``),
    it's discarded instead -- that shape has no natural-language content to
    lose, so there's no swallowing risk, and flushing it would just speak
    raw call syntax to the caller (also observed in production). Any other
    unclosed body -- including one that mixes call-shaped text with real
    trailing speech -- still flushes, same conservative default as before.

    Ordinary text (no ``<``) is flushed immediately, same as if no filter
    were present -- this only ever holds text back starting from an actual
    ``<`` in the buffer, capped at ``_TOOL_CALL_OPEN_TAIL`` chars so a stray
    literal ``<`` that never becomes a tag doesn't stall the stream. Zero
    added latency for the common case, same as the pre-existing
    ``_filter_never_say`` behavior when its own phrase list is empty.
    """
    buf = ""
    in_tool_call = False
    async for token in tokens:
        buf += token
        while True:
            if not in_tool_call:
                m = _TOOL_CALL_OPEN_RE.search(buf)
                if m:
                    if m.start():
                        yield buf[: m.start()]
                    buf = buf[m.start() :]
                    in_tool_call = True
                    continue
                last_lt = buf.rfind("<")
                if last_lt == -1:
                    if buf:
                        yield buf
                        buf = ""
                    break
                if last_lt:
                    yield buf[:last_lt]
                    buf = buf[last_lt:]
                if len(buf) > _TOOL_CALL_OPEN_TAIL:
                    yield buf
                    buf = ""
                break
            else:
                m = _TOOL_CALL_CLOSE_RE.search(buf)
                if not m:
                    break
                buf = buf[m.end() :]
                in_tool_call = False
    # Unterminated open tag: flush rather than silently discard, EXCEPT the
    # narrow bare-function-call case, which discards instead (see docstring
    # for both). A confirmed span (in_tool_call False by now) was already
    # fully excised above; buf here is only ever the tail of ordinary text.
    if buf:
        if in_tool_call:
            open_match = _TOOL_CALL_OPEN_RE.match(buf)
            body = buf[open_match.end() :] if open_match else buf
            if _TOOL_CALL_BARE_INVOCATION_RE.match(body.strip()):
                buf = ""
        if buf:
            yield buf


# A third, distinct hallucination shape observed in production: a
# self-closing XML/HTML-style tag naming a playbook action directly, e.g.:
#   <call:collect_to_check_availability course_id="course_76e249e0"
#     date="2026-08-07" preferred_time="09:00" players=1 />
# Same root cause as the other two shapes (routing-only guidance text
# leaking into the Talker's own prompt), yet another syntax the model
# reaches for -- this time mimicking the *compiled action name itself*
# rather than a generic "tool_call" token, so it doesn't contain the
# literal "tool_call" substring _filter_tool_call_leakage requires, and it
# doesn't start with `{`/`[` so _filter_structured_output_leak never sees
# it either. Real spoken turns in this domain never contain literal
# angle-bracket markup at all, so the trigger here is deliberately broad:
# any `<` immediately followed by an identifier char (not "tool_call",
# that's the other filter's job via the negative lookahead below) starts
# buffering. Closes on a self-closing `/>` -- the only close shape actually
# observed for this variant. An open tag with no self-close by end of
# stream is flushed as-is, not discarded, for the same reason
# _filter_tool_call_leakage flushes rather than discards: an unclosed span
# is ambiguous, and silently eating a whole turn's worth of real trailing
# speech is a far worse failure than an occasional ugly tag fragment.
_GENERIC_TAG_OPEN_RE = re.compile(r"<(?!\|)\s*/?\s*[A-Za-z_][\w:.\-]*")
_GENERIC_TAG_SELFCLOSE_RE = re.compile(r"/\s*>")
# Longest plausible split open-tag fragment straddling a chunk boundary.
_GENERIC_TAG_OPEN_TAIL = 16


async def _filter_generic_tag_leak(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Strip hallucinated self-closing tag spans (``<name ...attrs.../>``)
    from the token stream. See module comment above for the detection/
    discard rules; structurally mirrors _filter_tool_call_leakage."""
    buf = ""
    in_tag = False
    async for token in tokens:
        buf += token
        while True:
            if not in_tag:
                m = _GENERIC_TAG_OPEN_RE.search(buf)
                if m:
                    if m.start():
                        yield buf[: m.start()]
                    buf = buf[m.start() :]
                    in_tag = True
                    continue
                last_lt = buf.rfind("<")
                if last_lt == -1:
                    if buf:
                        yield buf
                        buf = ""
                    break
                if last_lt:
                    yield buf[:last_lt]
                    buf = buf[last_lt:]
                if len(buf) > _GENERIC_TAG_OPEN_TAIL:
                    yield buf
                    buf = ""
                break
            else:
                m = _GENERIC_TAG_SELFCLOSE_RE.search(buf)
                if not m:
                    break
                buf = buf[m.end() :]
                in_tag = False
    if buf:
        yield buf


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
    # Canned barrier line (filler/hold), not conversational content: streamed
    # to the caller but excluded from the logged utterance so the transcript
    # the Director reasons over stays clean (audit finding E9).
    filler: bool = False


class Talker:
    """One streaming LLM call per spoken turn; verbatim bypass; gated barrier."""

    def __init__(
        self,
        playbook: Playbook,
        llm: StreamsLLM,
        token_budget: int = 4000,
        barrier_timeout: float = 0.4,
        hold_timeout: float = 4.0,
        filler: SpokenLine = FILLER,
        hold_line: SpokenLine = HOLD_LINE,
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

    def _is_gated(self, cp: Checkpoint) -> bool:
        """A turn is gated when the checkpoint is hard or any slot is hard."""
        if cp.gate == "hard":
            return True
        return any(s.gate == "hard" for s in cp.slots.values())

    @staticmethod
    def _resolve_line(line: SpokenLine, state: ConversationState, fallback: str) -> str:
        """Static lines pass through; a provider is called with the live state
        (language-aware fillers). A broken provider degrades to the default —
        a filler must never kill the turn."""
        if not callable(line):
            return line
        try:
            return str(line(state))
        except Exception:  # noqa: BLE001
            return fallback

    async def _await_director(
        self, director_done: Callable[[], Awaitable[ConversationState]]
    ) -> ConversationState | None:
        """Wait for the Director up to the barrier, then the hold budget. Returns
        the quiescent state, or None if the Director never settled."""
        fresh: ConversationState | None = None
        with anyio.move_on_after(self._barrier_timeout):
            fresh = await director_done()
        if fresh is None:
            with anyio.move_on_after(self._hold_timeout):
                fresh = await director_done()
        return fresh

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

        if cp is not None and director_done is not None and self._is_gated(cp):
            fresh: ConversationState | None = None
            with anyio.move_on_after(self._barrier_timeout):
                fresh = await director_done()
            if fresh is None:
                filler_text = self._resolve_line(self._filler, state, FILLER)
                folded = [p.casefold() for p in (cp.never_say or []) if p]
                if folded:
                    filler_text = _excise(filler_text, folded)
                # Skip a filler that excision reduced to punctuation/space —
                # speaking "…" alone is worse than silence.
                if re.sub(r"[\W_]+", "", filler_text, flags=re.UNICODE):
                    yield SpeechChunk(
                        text=filler_text + " ",
                        spoke_from_version=state.version,
                        filler=True,
                    )
                with anyio.move_on_after(self._hold_timeout):
                    fresh = await director_done()
            if fresh is None:  # Director is down: degrade politely, never hang
                yield SpeechChunk(
                    text=self._resolve_line(self._hold_line, state, HOLD_LINE),
                    final=True,
                    spoke_from_version=state.version,
                    filler=True,
                )
                return
            state = fresh
            cp = (
                self._pb.checkpoint(state.checkpoint_id)
                if state.checkpoint_id
                else None
            )

        if cp is not None and (cp.say_verbatim is not None or cp.strict):
            if cp.say_verbatim is not None:
                text = render_template(cp.say_verbatim, self._pb, state)
                yield SpeechChunk(
                    text=text.strip(), final=True, spoke_from_version=state.version
                )
            else:
                # strict but no verbatim authored: never improvise on a strict step.
                yield SpeechChunk(
                    text=self._recovery_line,
                    final=True,
                    spoke_from_version=state.version,
                )
            return

        view = render_view(self._pb, state, token_budget=self._budget)
        # NOTE: a partial stream that fails midway replays from the start on
        # retry — acceptable for v1 (the retry targets connect-time failures;
        # mid-stream resume is a host concern).
        never_say = list(cp.never_say) if cp is not None else []
        for attempt in (1, 2):
            try:
                stream = self._llm.stream(view.messages)
                produced_any = False
                try:
                    # never_say enforcement is deterministic, not prompt-hope:
                    # authored phrases are excised from the stream before TTS.
                    # Structured (JSON) and both tag-shaped leakage variants
                    # are all stripped first (see _filter_structured_output_leak
                    # / _filter_tool_call_leakage / _filter_generic_tag_leak)
                    # so none of them can smuggle a never_say phrase past the
                    # excision pass by splitting it across a tag/brace
                    # boundary. JSON detection runs outermost since it needs
                    # the stream's true first character, before any other
                    # filter touches it.
                    filtered = _filter_never_say(
                        _filter_generic_tag_leak(
                            _filter_tool_call_leakage(
                                _filter_structured_output_leak(stream)
                            )
                        ),
                        never_say,
                    )
                    async for token in filtered:
                        if token:
                            produced_any = True
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
                if not produced_any:
                    # A filter (or the LLM itself) suppressed/emitted nothing
                    # for this turn — e.g. checkpoint guidance told the
                    # Talker to say nothing / defer to a tool call, or a
                    # leaked JSON/tag dump got excised outright. This used to
                    # retry into a forced "say something" override, but that
                    # produced a confident, FALSE commitment in production
                    # ("Certainly, I will hold that slot for you" on a turn
                    # the Director had NOT actually advanced or held
                    # anything) — a wrong, specific claim is strictly worse
                    # than true silence for a transactional voice agent.
                    # Silence is accepted as-is here, never papered over with
                    # invented words.
                    _log.info(
                        "[talker] LLM stream produced no output (suppressed "
                        "by a filter or genuinely empty) -- accepting silence",
                    )
                yield SpeechChunk(
                    text="", final=True, spoke_from_version=view.spoke_from_version
                )
                return
            except Exception as _exc:
                _log.error(
                    "[talker] LLM stream attempt=%d failed: %s\n%s",
                    attempt,
                    _exc,
                    traceback.format_exc(),
                )
                if attempt == 2:
                    yield SpeechChunk(
                        text=self._recovery_line,
                        final=True,
                        spoke_from_version=view.spoke_from_version,
                    )
                    return
