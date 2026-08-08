import re
import textwrap

import anyio

from superdialog.playbook.events import AdvanceEvent, EventLog, UtteranceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.state import ConversationState
from superdialog.playbook.talker import FILLER, HOLD_LINE, RECOVERY_LINE, Talker
from tests.playbook.test_models import MINIMAL_YAML


class StreamLLM:
    def __init__(self, chunks: list[str], fail_times: int = 0) -> None:
        self.chunks = chunks
        self.fail_times = fail_times
        self.calls = 0

    async def stream(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("llm down")
        for c in self.chunks:
            yield c


class ClosingStreamLLM(StreamLLM):
    """StreamLLM whose generator records whether its finally block ran."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(chunks)
        self.closed = False

    async def stream(self, messages, **kwargs):
        self.calls += 1
        try:
            for c in self.chunks:
                yield c
        finally:
            self.closed = True


class MidstreamFailLLM:
    """Yields ``fail_after`` tokens then raises on call 1; succeeds on call 2."""

    def __init__(self, chunks: list[str], fail_after: int) -> None:
        self.chunks = chunks
        self.fail_after = fail_after
        self.calls = 0

    async def stream(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            for c in self.chunks[: self.fail_after]:
                yield c
            raise RuntimeError("mid-stream drop")
        for c in self.chunks:
            yield c


def _state(checkpoint: str) -> tuple[Playbook, ConversationState]:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint=checkpoint, rule="init")
    )
    log.append(UtteranceEvent(role="user", text="hello"))
    return pb, ConversationState.fold(log, playbook=pb)


async def test_streams_tokens_and_reports_version() -> None:
    pb, state = _state("booking.collect")
    llm = StreamLLM(["Sure", ", which", " city?"])
    talker = Talker(pb, llm)
    chunks = [c async for c in talker.speak(state)]
    assert "".join(c.text for c in chunks) == "Sure, which city?"
    assert chunks[-1].spoke_from_version == state.version
    assert sum(c.final for c in chunks) == 1


async def test_suppresses_json_tool_call_dump() -> None:
    # Exact production shape: the whole turn is a raw JSON tool-call dump,
    # split across several small stream chunks the way a real LLM streams.
    pb, state = _state("booking.collect")
    llm = StreamLLM(
        [
            '{\n  "tool_calls": [\n',
            '    {\n      "function": "slot_to_profile_check",\n',
            '      "parameters": {\n        "slot_id": "slot_x"\n',
            "      }\n    }\n  ]\n}",
        ]
    )
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    text = "".join(c.text for c in chunks)
    # Suppressed entirely -> filtered-empty triggers the same-prompt retry
    # (StreamLLM replays the identical dump on both calls, so it's filtered
    # to empty again) -> accepted as silence. Never a forced "say something"
    # override retry -- that previously produced confident but false
    # commitments in production; silence is strictly safer.
    assert text == ""
    assert llm.calls == 2


async def test_does_not_suppress_normal_speech_with_braces_midsentence() -> None:
    # Only a turn that OPENS with '{' or '[' is treated as suspect --
    # ordinary speech that happens to contain braces later is untouched.
    pb, state = _state("booking.collect")
    llm = StreamLLM(["The price is ", "12000", " rupees (approx {market rate})."])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert (
        "".join(c.text for c in chunks)
        == "The price is 12000 rupees (approx {market rate})."
    )


async def test_flushes_json_shaped_text_without_tool_call_markers() -> None:
    # Starts with '{' but isn't a tool-call dump (no recognized keys) --
    # conservative: flush rather than discard on ambiguity.
    pb, state = _state("booking.collect")
    llm = StreamLLM(['{"note": "not a real tool call"}'])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == '{"note": "not a real tool call"}'


async def test_strips_hallucinated_tool_call_span() -> None:
    pb, state = _state("booking.collect")
    llm = StreamLLM(
        [
            "Sure, ",
            "<|tool_call>",
            "call:golfai_teetime.collect_to_check_availability",
            "{course_id: 'x', date: '2026-08-06'}",
            "<tool_call|>",
            " one moment.",
        ]
    )
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == "Sure,  one moment."
    assert "tool_call" not in "".join(c.text for c in chunks)


async def test_strips_tool_call_span_with_asymmetric_tags() -> None:
    # Production observed a malformed variant: open tag has the pipe before
    # the angle bracket, close tag has it after -- both must be recognized.
    pb, state = _state("booking.collect")
    llm = StreamLLM(
        [
            "Ok, ",
            "<|tool_call>_call:course_pref_to_availability{players: 1}",
            "<tool_call|>",
        ]
    )
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == "Ok, "


async def test_flushes_unterminated_tool_call_span() -> None:
    # No close tag before the stream ends -- flushed as-is, not discarded.
    # Regression guard: an earlier version discarded this, which silently
    # ate a caller's real confirmed answer in production when the model
    # opened a tool-call tag then kept speaking normally without ever
    # emitting the literal close-tag string.
    pb, state = _state("booking.collect")
    llm = StreamLLM(["All set. ", "<|tool_call>call:whatever{incomplete"])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert (
        "".join(c.text for c in chunks)
        == "All set. <|tool_call>call:whatever{incomplete"
    )


async def test_does_not_swallow_real_speech_after_unclosed_open_tag() -> None:
    # The actual production failure mode: the model opens a tool-call tag
    # then keeps generating normal, meaningful speech afterward without
    # ever closing it. That trailing speech must reach the caller.
    pb, state = _state("booking.collect")
    llm = StreamLLM(
        [
            "<|tool_call>",
            "Got it -- ",
            "one player, ",
            "day after tomorrow at nine.",
        ]
    )
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    text = "".join(c.text for c in chunks)
    assert "one player" in text
    assert "day after tomorrow" in text


async def test_discards_unclosed_bare_function_call_invocation() -> None:
    # Exact production shape: <|tool_call>_hold_slot(slot_id='...') spoken
    # verbatim, no close tag, no trailing natural-language content -- just
    # a bare call expression. Unlike the mixed-content case above, there's
    # nothing here to lose by discarding, so this is the one case that
    # discards an unclosed span instead of flushing it. With the whole turn
    # suppressed, this is accepted as silence (no forced retry-into-speech).
    pb, state = _state("booking.collect")
    llm = StreamLLM(
        ["<|tool_call>_hold_slot(slot_id='slot_course_95f50dfb_2026-08-06_1200')"]
    )
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    text = "".join(c.text for c in chunks)
    assert text == ""
    assert "_hold_slot" not in text


async def test_tool_call_filter_does_not_touch_normal_speech() -> None:
    pb, state = _state("booking.collect")
    llm = StreamLLM(["Which ", "city ", "would you like?"])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == "Which city would you like?"


async def test_strips_self_closing_action_tag_leak() -> None:
    # Exact production shape: a self-closing tag naming the compiled action
    # directly, e.g. <call:collect_to_check_availability .../> -- contains
    # neither "tool_call" (misses _filter_tool_call_leakage) nor a leading
    # '{'/'[' (misses _filter_structured_output_leak). Suppressed entirely,
    # accepted as silence (no forced retry-into-speech).
    pb, state = _state("booking.collect")
    llm = StreamLLM(
        [
            '<call:collect_to_check_availability course_id="course_76e249e0" ',
            'date="2026-08-07" preferred_time="09:00" players=1 />',
        ]
    )
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    text = "".join(c.text for c in chunks)
    assert text == ""
    assert "<call:" not in text


async def test_generic_tag_filter_flushes_unterminated_tag() -> None:
    pb, state = _state("booking.collect")
    llm = StreamLLM(["<call:foo bar=", '"1" but then keeps talking normally'])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == (
        '<call:foo bar="1" but then keeps talking normally'
    )


async def test_generic_tag_filter_does_not_touch_normal_speech() -> None:
    pb, state = _state("booking.collect")
    llm = StreamLLM(["Is that ", "less ", "than ", "5 people?"])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == "Is that less than 5 people?"


async def test_say_verbatim_bypasses_llm() -> None:
    pb, state = _state("booking.confirm")
    llm = StreamLLM(["should not be called"])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == "Your booking is held."
    assert llm.calls == 0
    assert sum(c.final for c in chunks) == 1


async def test_failure_retries_once_then_recovers() -> None:
    pb, state = _state("booking.collect")
    flaky = Talker(pb, StreamLLM(["ok!"], fail_times=1))
    assert "".join([c.text async for c in flaky.speak(state)]) == "ok!"
    dead = Talker(pb, StreamLLM([], fail_times=99))
    assert RECOVERY_LINE in "".join([c.text async for c in dead.speak(state)])


async def test_clean_but_empty_stream_is_accepted_as_silence() -> None:
    # The LLM stream completes with no exception but yields nothing at all --
    # e.g. checkpoint guidance told it to say nothing / defer to a tool call,
    # or a leak got filtered to empty, or the model sampled an immediate stop
    # token on both tries. One retry with the SAME prompt is allowed (see
    # test_empty_stream_retries_same_prompt_and_speaks_if_nonempty below) --
    # but if it's still empty on the second attempt, that's accepted as
    # silence: a FORCED-OVERRIDE retry into speech previously produced a
    # confident but FALSE commitment in production ("Certainly, I will hold
    # that slot for you" while nothing had actually been held) -- silence is
    # strictly safer than an invented claim for a transactional voice agent.
    # No recovery line, no fabricated text, ever.
    pb, state = _state("booking.collect")
    empty = Talker(pb, StreamLLM([]))
    chunks = [c async for c in empty.speak(state)]
    assert "".join(c.text for c in chunks) == ""
    assert empty._llm.calls == 2


class EmptyThenRespondsLLM:
    """Empty on call 1 (mimics a checkpoint's own "say nothing" guidance
    being followed literally, or a one-off sampling glitch); speaks on the
    retry (call 2) -- and records the messages sent each time, so a test can
    confirm the retry reused the exact same prompt with no forced override."""

    def __init__(self, second_call_text: str) -> None:
        self.second_call_text = second_call_text
        self.calls_messages: list[list[dict[str, str]]] = []

    async def stream(self, messages, **kwargs):
        self.calls_messages.append(messages)
        if len(self.calls_messages) == 1:
            return
        yield self.second_call_text


async def test_empty_stream_retries_same_prompt_and_speaks_if_nonempty() -> None:
    # Root cause seen in production: checkpoint guidance instructs "say
    # NOTHING / zero words" once all slots are set, trusting a tool-calling
    # channel the Talker doesn't have -- but the SAME small model also just
    # sampled a stray immediate stop token on plenty of ordinary turns that
    # DID want real speech (e.g. a caller answering "one player" got dead
    # air), indistinguishable from the intentional-silence case from here. A
    # prior version of this code retried with a "you must speak" override
    # injected into the prompt, which improvised a false commitment ("I will
    # hold that slot for you") on a turn nothing had actually been held --
    # that forced override is still banned. But retrying the IDENTICAL
    # prompt (no override, no injected instruction) is safe: on a genuinely
    # say-nothing checkpoint the retry just reproduces empty again (see
    # test_clean_but_empty_stream_is_accepted_as_silence); on a one-off
    # sampling glitch it recovers real, correctly-grounded speech instead of
    # leaving the caller in dead air.
    pb, state = _state("booking.collect")
    llm = EmptyThenRespondsLLM("Got it, one player -- let me check that for you.")
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    text = "".join(c.text for c in chunks)
    assert text == "Got it, one player -- let me check that for you."
    assert len(llm.calls_messages) == 2
    assert llm.calls_messages[0] == llm.calls_messages[1]


async def test_hard_gate_filler_then_speech() -> None:
    # Hard gate: barrier BEFORE first token, filler on expiry.
    pb, state = _state("booking.confirm")
    event = anyio.Event()

    async def wait_director() -> ConversationState:
        await event.wait()
        return state

    talker = Talker(
        pb,
        StreamLLM([]),
        barrier_timeout=0.05,
        hold_timeout=10.0,
    )
    received: list[str] = []

    async def consume() -> None:
        async for c in talker.speak(state, director_done=wait_director):
            received.append(c.text)

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await anyio.sleep(0.2)  # past barrier_timeout, director still pending
        assert any(FILLER in t for t in received)  # filler already emitted
        event.set()
    assert "".join(received).endswith("Your booking is held.")


async def test_hard_gate_hold_line_when_director_never_comes() -> None:
    pb, state = _state("booking.confirm")

    async def never() -> ConversationState:
        await anyio.sleep(60)
        return state

    talker = Talker(pb, StreamLLM([]), barrier_timeout=0.02, hold_timeout=0.05)
    received = [c.text async for c in talker.speak(state, director_done=never)]
    assert any(HOLD_LINE in t for t in received)


async def test_abort_closes_inner_stream() -> None:
    """Host abort (aclose, e.g. LiveKit barge-in) must close the inner stream."""
    pb, state = _state("booking.collect")
    llm = ClosingStreamLLM(["a", "b", "c", "d"])
    gen = Talker(pb, llm).speak(state)
    await gen.__anext__()
    await gen.__anext__()
    assert not llm.closed  # stream still open mid-turn
    await gen.aclose()
    assert llm.closed  # inner generator's finally ran on abort


async def test_midstream_failure_replays() -> None:
    pb, state = _state("booking.collect")
    llm = MidstreamFailLLM(["a", "b", "c"], fail_after=2)
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    # Documented caveat: a partial stream that fails midway replays from the
    # start on retry, so the pre-failure prefix is duplicated.
    assert "".join(c.text for c in chunks) == "ababc"
    assert sum(c.final for c in chunks) == 1
    assert llm.calls == 2


async def test_fast_barrier_no_filler() -> None:
    pb, state = _state("booking.confirm")

    async def instant() -> ConversationState:
        return state

    talker = Talker(pb, StreamLLM([]))
    chunks = [c async for c in talker.speak(state, director_done=instant)]
    text = "".join(c.text for c in chunks)
    assert FILLER not in text
    assert text.endswith("Your booking is held.")


async def test_soft_gate_skips_barrier() -> None:
    pb, state = _state("booking.collect")

    async def never() -> ConversationState:
        await anyio.sleep(3600)
        return state

    talker = Talker(pb, StreamLLM(["hi"]))
    with anyio.fail_after(1):
        chunks = [c async for c in talker.speak(state, director_done=never)]
    assert "".join(c.text for c in chunks) == "hi"


async def test_custom_speech_lines() -> None:
    pb, state = _state("booking.confirm")

    async def never() -> ConversationState:
        await anyio.sleep(60)
        return state

    talker = Talker(
        pb,
        StreamLLM([]),
        barrier_timeout=0.02,
        hold_timeout=0.05,
        filler="एक पल रुकिए",
        hold_line="H",
        recovery_line="R",
    )
    received = [c.text async for c in talker.speak(state, director_done=never)]
    assert any("एक पल रुकिए" in t for t in received)
    assert any(t == "H" for t in received)  # hold-line path uses instance attr
    assert not any(FILLER in t or HOLD_LINE in t for t in received)

    pb2, state2 = _state("booking.collect")
    dead = Talker(pb2, StreamLLM([], fail_times=99), recovery_line="R")
    out = [c.text async for c in dead.speak(state2)]
    assert "R" in out  # recovery path uses instance attr
    assert RECOVERY_LINE not in "".join(out)


def test_default_hold_timeout_is_four_seconds() -> None:
    pb, _ = _state("booking.collect")
    talker = Talker(pb, StreamLLM([]))
    assert talker._hold_timeout == 4.0


async def test_strict_checkpoint_speaks_verbatim_without_llm() -> None:
    yaml_text = textwrap.dedent('''
        persona: "Assistant."
        journeys:
          j:
            checkpoints:
              - id: greet
                strict: true
                say_verbatim: "Welcome to Villa Raag."
                gate: hard
              - id: done
                terminal: true
    ''')
    pb = Playbook.from_yaml(yaml_text)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.greet", rule="init"))
    state = ConversationState.fold(log, playbook=pb)
    llm = StreamLLM(["should not be called"])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert "".join(c.text for c in chunks) == "Welcome to Villa Raag."
    assert llm.calls == 0


async def test_strict_without_verbatim_does_not_call_llm() -> None:
    # A strict checkpoint with NO say_verbatim must still not improvise via the
    # LLM; it falls back to the recovery line.
    yaml_text = textwrap.dedent('''
        persona: "Assistant."
        journeys:
          j:
            checkpoints:
              - id: greet
                strict: true
                guidance: "say hi"
              - id: done
                terminal: true
    ''')
    pb = Playbook.from_yaml(yaml_text)
    log = EventLog()
    log.append(AdvanceEvent(from_checkpoint=None, to_checkpoint="j.greet", rule="init"))
    state = ConversationState.fold(log, playbook=pb)
    llm = StreamLLM(["should not be called"])
    chunks = [c async for c in Talker(pb, llm).speak(state)]
    assert llm.calls == 0
    assert RECOVERY_LINE in "".join(c.text for c in chunks)


async def test_callable_filler_called_with_state() -> None:
    # A filler provider receives the live state (language-aware fillers) and
    # its text is spoken at barrier expiry.
    pb, state = _state("booking.confirm")
    event = anyio.Event()

    async def wait_director() -> ConversationState:
        await event.wait()
        return state

    seen: list[ConversationState] = []

    def provider(s: ConversationState) -> str:
        seen.append(s)
        return "एक second, main check करती हूँ…"

    talker = Talker(
        pb, StreamLLM([]), barrier_timeout=0.02, hold_timeout=10.0, filler=provider
    )
    received: list[str] = []

    async def consume() -> None:
        async for c in talker.speak(state, director_done=wait_director):
            received.append(c.text)

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await anyio.sleep(0.2)  # past barrier_timeout, director still pending
        assert any("एक second, main check करती हूँ…" in t for t in received)
        assert seen == [state]
        event.set()


async def test_callable_hold_line() -> None:
    pb, state = _state("booking.confirm")

    async def never() -> ConversationState:
        await anyio.sleep(60)
        return state

    talker = Talker(
        pb,
        StreamLLM([]),
        barrier_timeout=0.02,
        hold_timeout=0.05,
        hold_line=lambda s: "थोड़ा समय लग रहा है, एक moment…",
    )
    received = [c.text async for c in talker.speak(state, director_done=never)]
    assert any("थोड़ा समय लग रहा है, एक moment…" in t for t in received)


def _gated_never_say_state(phrase: str) -> tuple[Playbook, ConversationState]:
    """Playbook with a single hard-gate checkpoint declaring ``never_say``."""
    yaml_text = textwrap.dedent(f'''
        persona: "Assistant."
        journeys:
          j:
            checkpoints:
              - id: gatecp
                gate: hard
                guidance: "confirm the thing"
                never_say:
                  - "{phrase}"
              - id: done
                terminal: true
    ''')
    pb = Playbook.from_yaml(yaml_text)
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="j.gatecp", rule="init")
    )
    return pb, ConversationState.fold(log, playbook=pb)


async def test_filler_respects_never_say() -> None:
    """golf lists the engine's own filler in never_say; it must be excised."""
    pb, state = _gated_never_say_state("One moment, let me confirm that")

    async def director_done() -> ConversationState:
        await anyio.sleep(0.1)
        return state

    talker = Talker(
        pb, StreamLLM(["All", " set."]), barrier_timeout=0.01, hold_timeout=0.5
    )
    chunks = [c async for c in talker.speak(state, director_done=director_done)]
    assert all("One moment" not in c.text for c in chunks)
    assert "".join(c.text for c in chunks).endswith("All set.")


async def test_filler_excised_to_punctuation_is_skipped() -> None:
    """never_say covering the FULL filler (incl. ellipsis) leaves punctuation
    only — no filler chunk may be yielded at all."""
    pb, state = _gated_never_say_state(FILLER)  # full text incl. trailing "…"

    async def director_done() -> ConversationState:
        await anyio.sleep(0.1)
        return state

    talker = Talker(
        pb, StreamLLM(["All", " set."]), barrier_timeout=0.01, hold_timeout=0.5
    )
    chunks = [c async for c in talker.speak(state, director_done=director_done)]
    assert all("One moment" not in c.text for c in chunks)
    # No non-empty chunk consisting solely of non-word chars (e.g. "… ").
    assert not any(
        c.text and not re.sub(r"[\W_]+", "", c.text) for c in chunks
    )
    assert "".join(c.text for c in chunks).endswith("All set.")


async def test_broken_line_provider_degrades_to_defaults() -> None:
    # A raising provider must never kill the turn — defaults speak instead.
    pb, state = _state("booking.confirm")

    def broken(_s: ConversationState) -> str:
        raise RuntimeError("provider bug")

    async def never() -> ConversationState:
        await anyio.sleep(60)
        return state

    talker = Talker(
        pb,
        StreamLLM([]),
        barrier_timeout=0.02,
        hold_timeout=0.05,
        filler=broken,
        hold_line=broken,
    )
    received = [c.text async for c in talker.speak(state, director_done=never)]
    assert any(FILLER in t for t in received)
    assert any(HOLD_LINE in t for t in received)
