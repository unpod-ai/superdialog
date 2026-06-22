"""Playbook provider adapters emit per-call LLM usage for billing.

ProviderDirector.complete and ProviderTalker.stream fire ``on_llm_complete``
with an LLMUsageEvent (tokens + cache read/write) so the playbook engine — the
default engine — reports tokens to the same usage sink the graph engine feeds.
"""

from __future__ import annotations

import anyio

from superdialog.llm.provider import CompletionResult, StreamChunk
from superdialog.playbook.providers import (
    LLMUsageEvent,
    ProviderDirector,
    ProviderTalker,
)


class _DirectorProvider:
    model = "anthropic/claude-haiku-4-5"

    async def complete(self, messages, tools=None, **kw):
        return CompletionResult(
            text="verdict",
            tool_calls=[],
            metadata={
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "cache_read_tokens": 5000,
                "cache_write_tokens": 0,
            },
        )


class _TalkerProvider:
    model = "anthropic/claude-haiku-4-5"

    async def stream(self, messages, tools=None, **kw):
        yield StreamChunk(text="Hel", tool_call_delta=None, done=False)
        yield StreamChunk(text="lo", tool_call_delta=None, done=False)
        # trailing usage-only chunk (where cache tokens arrive)
        yield StreamChunk(
            text=None,
            tool_call_delta=None,
            done=True,
            usage={"prompt_tokens": 13, "completion_tokens": 4, "cache_write_tokens": 7000},
        )


def test_director_fires_usage_event():
    events: list[LLMUsageEvent] = []

    async def _cb(ev):
        events.append(ev)

    d = ProviderDirector(_DirectorProvider(), on_llm_complete=_cb)
    text = anyio.run(lambda: d.complete([{"role": "user", "content": "hi"}]))
    assert text == "verdict"
    assert len(events) == 1
    ev = events[0]
    assert ev.tokens_in == 20 and ev.tokens_out == 8
    assert ev.cached == 5000 and ev.cache_write == 0
    assert ev.model == "anthropic/claude-haiku-4-5"


def test_talker_fires_usage_event_from_trailing_chunk():
    events: list[LLMUsageEvent] = []

    async def _cb(ev):
        events.append(ev)

    t = ProviderTalker(_TalkerProvider(), on_llm_complete=_cb)

    async def _run():
        out = []
        async for tok in t.stream([{"role": "user", "content": "hi"}]):
            out.append(tok)
        return out

    out = anyio.run(_run)
    assert "".join(out) == "Hello"
    assert len(events) == 1
    assert events[0].tokens_in == 13
    assert events[0].cache_write == 7000


def test_no_callback_is_inert():
    d = ProviderDirector(_DirectorProvider())  # no callback
    assert anyio.run(lambda: d.complete([{"role": "user", "content": "hi"}])) == "verdict"


def test_dialog_machine_register_wires_playbook_adapters():
    """DialogMachine.register_llm_callback attaches the fn to already-built
    playbook provider adapters (the SDK delegates here for the default engine)."""
    from superdialog import DialogMachine

    machine = DialogMachine(
        source={"tasks": []}, llm="openai/gpt-4.1-mini", engine="playbook"
    )

    class _Adptr:
        on_llm_complete = None

    machine._pb_director = _Adptr()
    machine._pb_talker = _Adptr()

    async def _cb(ev):
        pass

    machine.register_llm_callback(_cb)
    assert machine._llm_callback is _cb
    assert machine._pb_director.on_llm_complete is _cb
    assert machine._pb_talker.on_llm_complete is _cb
