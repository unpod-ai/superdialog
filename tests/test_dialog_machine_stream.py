"""Tests for ``DialogMachine.turn(stream=...)``."""

from __future__ import annotations

from pathlib import Path

from superdialog import DialogMachine, Flow, StreamChunk, Turn
from tests.scripted_toolcall import ScriptedToolProvider, route

FIXTURE = Path(__file__).parent / "fixtures" / "flow" / "kyc.json"


async def test_stream_text_yields_chunks_with_final_turn() -> None:
    machine = DialogMachine(flow=Flow.load(FIXTURE), llm="openai/gpt-4o-mini")
    machine._llm = ScriptedToolProvider(  # type: ignore[assignment]
        routes=[route("greet_to_name")], replies=["Welcome to Acme"]
    )

    chunks: list[StreamChunk] = []
    iterator = await machine.turn("hello", stream="text")
    async for chunk in iterator:
        chunks.append(chunk)

    assert chunks, "expected at least one chunk"
    assert chunks[-1].done is True
    assert isinstance(chunks[-1].turn, Turn)
    reassembled = "".join(c.text for c in chunks)
    assert reassembled == chunks[-1].turn.text
    assert chunks[-1].turn.metadata["to_node"] == "collect_name"


async def test_stream_bool_true_is_equivalent_to_text() -> None:
    machine = DialogMachine(flow=Flow.load(FIXTURE), llm="openai/gpt-4o-mini")
    machine._llm = ScriptedToolProvider(  # type: ignore[assignment]
        routes=[route("greet_to_name")], replies=["Hello there friend"]
    )
    iterator = await machine.turn("hi", stream=True)
    chunks = [c async for c in iterator]
    assert chunks[-1].done is True
    reassembled = "".join(c.text for c in chunks)
    assert reassembled == chunks[-1].turn.text
    assert reassembled  # non-empty


async def test_stream_done_chunk_carries_turn_metadata() -> None:
    machine = DialogMachine(flow=Flow.load(FIXTURE), llm="openai/gpt-4o-mini")
    machine._llm = ScriptedToolProvider(  # type: ignore[assignment]
        routes=[route(None, brief="please repeat")]
    )
    iterator = await machine.turn("hi", stream="text")
    chunks = [c async for c in iterator]
    assert chunks[-1].done is True
    assert chunks[-1].turn is not None
    assert chunks[-1].turn.metadata["outcome"] == "stay"
    # only the final chunk carries the Turn; earlier ones do not
    assert all(c.turn is None for c in chunks[:-1])
