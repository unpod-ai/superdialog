"""Tests for Observer protocol, NullObserver, LangfuseObserver, TracingProvider."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from superdialog.observability import (
    LangfuseObserver,
    NullObserver,
    TracingProvider,
    build_observer,
)
from superdialog.llm.provider import CompletionResult


# ── NullObserver ──────────────────────────────────────────────────────────────

def test_null_observer_on_session_start_returns_session_id():
    obs = NullObserver()
    result = obs.on_session_start("session-1", {})
    assert result == "session-1"


def test_null_observer_on_generation_start_returns_string():
    obs = NullObserver()
    obs_id = obs.on_generation_start("trace-1", "turn", [{"role": "user", "content": "hi"}])
    assert isinstance(obs_id, str)


def test_null_observer_all_methods_dont_raise():
    obs = NullObserver()
    trace_id = obs.on_session_start("s1", {"source": "test"})
    obs_id = obs.on_generation_start(trace_id, "turn", [])
    obs.on_generation_end(obs_id, "hello", [], {})
    obs.on_tool_call(trace_id, "my_tool", {"x": 1}, "result")
    obs.on_flow_node(trace_id, "node-1", {"slot": "value"})
    obs.on_voice_turn(trace_id, {"ttfa_ms": 100.0, "asr_final_ms": 200.0, "tts_ttfb_ms": 50.0})
    obs.on_session_end(trace_id, "final output")


# ── LangfuseObserver ──────────────────────────────────────────────────────────

def _make_mock_trace():
    trace = MagicMock()
    trace.id = "trace-id-123"
    return trace


def _make_mock_generation():
    gen = MagicMock()
    gen.id = "gen-id-456"
    return gen


def test_langfuse_observer_on_session_start_calls_trace():
    client = MagicMock()
    mock_trace = _make_mock_trace()
    client.trace.return_value = mock_trace

    obs = LangfuseObserver(client)
    result = obs.on_session_start("session-abc", {"source": "superdialog"})

    client.trace.assert_called_once_with(
        id="session-abc",
        name="voice_session:session-abc",
        user_id="session-abc",
        session_id="session-abc",
        tags=["layer:superdialog", "mode:unknown", "agent:unknown-agent"],
        metadata={"source": "superdialog"},
    )
    assert result == "trace-id-123"


def test_langfuse_observer_on_generation_start_creates_generation():
    client = MagicMock()
    client.trace.return_value = _make_mock_trace()
    mock_gen = _make_mock_generation()
    client.generation.return_value = mock_gen

    obs = LangfuseObserver(client)
    messages = [{"role": "user", "content": "hi"}]
    obs_id = obs.on_generation_start("trace-id-123", "turn-1", messages)

    client.generation.assert_called_once_with(
        trace_id="trace-id-123",
        name="turn-1",
        input=messages,
    )
    assert obs_id == "gen-id-456"


def test_langfuse_observer_on_generation_end_calls_end_on_generation():
    client = MagicMock()
    client.trace.return_value = _make_mock_trace()
    mock_gen = _make_mock_generation()
    client.generation.return_value = mock_gen

    obs = LangfuseObserver(client)
    obs_id = obs.on_generation_start("trace-id-123", "turn-1", [])
    obs.on_generation_end(
        obs_id,
        output="hello world",
        tool_calls=[],
        metadata={"prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 100.0},
    )

    mock_gen.end.assert_called_once_with(
        output="hello world",
        usage={"input": 10, "output": 5},
        metadata={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "latency_ms": 100.0,
            "layer": "superdialog",
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
    )


def test_langfuse_observer_on_generation_end_unknown_id_is_noop():
    client = MagicMock()
    obs = LangfuseObserver(client)
    obs.on_generation_end("unknown-id", "text", [], {})  # must not raise


def test_langfuse_observer_on_tool_call_creates_span():
    client = MagicMock()
    obs = LangfuseObserver(client)
    obs.on_tool_call("trace-1", "search", {"query": "hello"}, "result text")

    client.span.assert_called_once()
    call_kwargs = client.span.call_args.kwargs
    assert call_kwargs["trace_id"] == "trace-1"
    assert call_kwargs["name"] == "tool:search"


def test_langfuse_observer_on_session_end_calls_update_and_flush():
    client = MagicMock()
    mock_trace = _make_mock_trace()
    client.trace.return_value = mock_trace

    obs = LangfuseObserver(client)
    obs.on_session_start("session-x", {})
    obs.on_session_end("trace-id-123", "final text")

    mock_trace.update.assert_called_once_with(output="final text")
    client.flush.assert_called_once()


def test_langfuse_observer_swallows_exceptions():
    client = MagicMock()
    client.trace.side_effect = RuntimeError("network error")
    obs = LangfuseObserver(client)
    result = obs.on_session_start("s1", {})  # must not raise
    assert result == "s1"


# ── TracingProvider ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tracing_provider_complete_calls_observer():
    """TracingProvider.complete() calls on_generation_start and on_generation_end."""
    from unittest.mock import AsyncMock, MagicMock, call
    from superdialog.observability import NullObserver, TracingProvider
    from superdialog.llm.provider import CompletionResult

    inner = MagicMock()
    inner.complete = AsyncMock(
        return_value=CompletionResult(
            text="hello",
            tool_calls=[],
            metadata={"prompt_tokens": 5, "completion_tokens": 3},
        )
    )

    # Spy on NullObserver methods
    observer = NullObserver()
    original_start = observer.on_generation_start
    original_end = observer.on_generation_end
    start_calls = []
    end_calls = []

    def spy_start(trace_id, name, input_messages, **kwargs):
        start_calls.append((trace_id, name))
        return original_start(trace_id, name, input_messages)

    def spy_end(obs_id, output, tool_calls, metadata):
        end_calls.append((obs_id, output))
        return original_end(obs_id, output, tool_calls, metadata)

    observer.on_generation_start = spy_start
    observer.on_generation_end = spy_end

    provider = TracingProvider(inner, observer, "trace-id-1")
    messages = [{"role": "user", "content": "hi"}]
    result = await provider.complete(messages)

    assert result.text == "hello"
    assert len(start_calls) == 1
    assert start_calls[0] == ("trace-id-1", "llm:complete")
    assert len(end_calls) == 1
    assert end_calls[0][1] == "hello"


@pytest.mark.asyncio
async def test_tracing_provider_complete_calls_langfuse_observer():
    client = MagicMock()
    mock_gen = _make_mock_generation()
    client.generation.return_value = mock_gen

    inner = MagicMock()

    async def _fake_complete(messages, tools=None, **opts):
        return CompletionResult(
            text="response",
            tool_calls=[],
            metadata={"prompt_tokens": 10, "completion_tokens": 7},
        )

    inner.complete = _fake_complete

    observer = LangfuseObserver(client)
    provider = TracingProvider(inner, observer, "trace-id-2")
    await provider.complete([{"role": "user", "content": "hello"}])

    client.generation.assert_called_once()
    mock_gen.end.assert_called_once_with(
        output="response",
        usage={"input": 10, "output": 7},
        metadata={
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "layer": "superdialog",
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
    )


# ── build_observer ────────────────────────────────────────────────────────────

def test_build_observer_returns_null_when_no_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    obs = build_observer()
    assert isinstance(obs, NullObserver)


def test_build_observer_returns_langfuse_when_keys_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    with patch("langfuse.Langfuse") as MockLangfuse:
        MockLangfuse.return_value = MagicMock()
        obs = build_observer()
    assert isinstance(obs, LangfuseObserver)


@pytest.mark.asyncio
async def test_tracing_provider_stream_calls_observer_with_metadata():
    """TracingProvider.stream() calls on_generation_end with latency_ms metadata."""
    from superdialog.llm.provider import StreamChunk

    async def _fake_stream(messages, tools=None, **opts):
        yield StreamChunk(text="hel", tool_call_delta=None, done=False)
        yield StreamChunk(text="lo", tool_call_delta=None, done=True)

    inner = MagicMock()
    inner.stream = _fake_stream

    end_calls = []
    observer = NullObserver()
    original_end = observer.on_generation_end
    def spy_end(obs_id, output, tool_calls, metadata):
        end_calls.append({"output": output, "metadata": metadata})
        return original_end(obs_id, output, tool_calls, metadata)
    observer.on_generation_end = spy_end

    provider = TracingProvider(inner, observer, "trace-stream-1")
    chunks = []
    async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert "".join(c.text for c in chunks if c.text) == "hello"
    assert len(end_calls) == 1
    assert end_calls[0]["output"] == "hello"
    assert "latency_ms" in end_calls[0]["metadata"]


# ── TracingProvider: optional model_uri / role tagging ─────────────────────────


def test_tracing_provider_accepts_model_uri_and_role():
    """Regression: PlaybookMachine.set_observer constructs TracingProvider with
    model_uri/role kwargs; the 5-arg form must not raise (was TypeError)."""
    inner = MagicMock()
    provider = TracingProvider(
        inner,
        NullObserver(),
        "trace-1",
        model_uri="openai/gpt-4.1-mini",
        role="talker",
    )
    assert provider is not None


def test_tracing_provider_backward_compatible_three_arg():
    """The original 3-arg construction stays valid and untagged."""
    inner = MagicMock()
    provider = TracingProvider(inner, NullObserver(), "trace-1")
    assert provider is not None


@pytest.mark.asyncio
async def test_tracing_provider_role_prefixes_generation_name():
    """role='talker' makes complete() open the generation as 'talker:complete'."""
    from unittest.mock import AsyncMock

    inner = MagicMock()
    inner.complete = AsyncMock(
        return_value=CompletionResult(text="hi", tool_calls=[], metadata={})
    )
    observer = NullObserver()
    names: list[str] = []
    original_start = observer.on_generation_start

    def spy_start(trace_id, name, input_messages, **kwargs):
        names.append(name)
        return original_start(trace_id, name, input_messages)

    observer.on_generation_start = spy_start

    provider = TracingProvider(inner, observer, "trace-1", role="talker")
    await provider.complete([{"role": "user", "content": "hi"}])

    assert names == ["talker:complete"]


@pytest.mark.asyncio
async def test_tracing_provider_injects_model_into_end_metadata():
    """model_uri is recorded as metadata['model'] on generation end so the
    trace shows which model produced the turn."""
    from unittest.mock import AsyncMock

    inner = MagicMock()
    inner.complete = AsyncMock(
        return_value=CompletionResult(text="hi", tool_calls=[], metadata={})
    )
    observer = NullObserver()
    metas: list[dict[str, Any]] = []
    original_end = observer.on_generation_end

    def spy_end(obs_id, output, tool_calls, metadata):
        metas.append(metadata)
        return original_end(obs_id, output, tool_calls, metadata)

    observer.on_generation_end = spy_end

    provider = TracingProvider(
        inner, observer, "trace-1", model_uri="openai/gpt-4.1-mini"
    )
    await provider.complete([{"role": "user", "content": "hi"}])

    assert metas and metas[0].get("model") == "openai/gpt-4.1-mini"


# ── LLMAgent + Observer ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_agent_calls_observer_on_turn():
    """LLMAgent.turn() triggers on_generation_start + on_generation_end."""
    from superdialog import LLMAgent

    client = MagicMock()
    mock_trace = _make_mock_trace()
    mock_gen = _make_mock_generation()
    client.trace.return_value = mock_trace
    client.generation.return_value = mock_gen

    # Patch the LLM provider so no real API calls happen
    with patch("superdialog.agents.llm_agent.resolve_llm") as mock_resolve:
        fake_provider = MagicMock()

        async def _complete(messages, tools=None, **opts):
            return CompletionResult(
                text="hi there", tool_calls=[], metadata={"prompt_tokens": 5, "completion_tokens": 3}
            )

        fake_provider.complete = _complete
        fake_provider.inner = fake_provider
        mock_resolve.return_value = fake_provider

        observer = LangfuseObserver(client)
        agent = LLMAgent(llm="openai/gpt-4.1-mini", system_prompt="you are helpful")
        agent.set_observer(observer, "trace-id-999")

        result = await agent.turn("hello")

    assert result.text == "hi there"
    client.generation.assert_called_once()
    mock_gen.end.assert_called_once()
    call_kwargs = client.generation.call_args.kwargs
    assert call_kwargs["trace_id"] == "trace-id-999"


# ── DialogMachine + Observer ──────────────────────────────────────────────────

def test_dialog_machine_set_observer_wraps_llm():
    """set_observer() wraps DialogMachine._llm with TracingProvider."""
    from superdialog.observability import TracingProvider, NullObserver

    with patch("superdialog.dialog_machine.resolve_llm") as mock_resolve:
        fake_provider = MagicMock()
        fake_provider.inner = fake_provider
        mock_resolve.return_value = fake_provider

        from superdialog import DialogMachine, Flow
        flow = MagicMock(spec=Flow)
        flow.name = "test_flow"
        flow.nodes = {}
        machine = DialogMachine(flow=flow, llm="openai/gpt-4.1-mini")

    observer = NullObserver()
    machine.set_observer(observer, "trace-abc")
    assert isinstance(machine._llm, TracingProvider)
    assert machine._llm._trace_id == "trace-abc"
    assert machine._llm._observer is observer


def test_dialog_machine_set_observer_playbook_engine_guard():
    """set_observer() on a playbook-engine DialogMachine does not raise."""
    from superdialog.observability import NullObserver

    # Create a playbook-engine instance (engine="playbook" + non-Flow source)
    # We need a source that isn't a Flow/FlowSet to trigger playbook mode
    with patch("superdialog.dialog_machine.resolve_llm") as mock_resolve:
        fake_provider = MagicMock()
        fake_provider.inner = fake_provider
        mock_resolve.return_value = fake_provider

        from superdialog import DialogMachine
        # Use a dict source with engine="playbook" to trigger playbook backend
        machine = DialogMachine(
            source={"tasks": []},
            llm="openai/gpt-4.1-mini",
            engine="playbook"
        )

    observer = NullObserver()
    # Must not raise; the guard should return early for playbook engine
    machine.set_observer(observer, "trace-playbook")
    # Verify the machine is still in playbook mode
    assert machine._engine == "playbook"
