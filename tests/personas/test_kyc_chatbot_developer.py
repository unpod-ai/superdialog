"""Persona 1: KYC chatbot developer.

A fintech developer building a Know-Your-Customer bot. They:
1. Load a pre-authored flow from JSON.
2. Wire DialogMachine with a model URI.
3. Drive a multi-turn conversation through all nodes (greet → name → dob → pan → done).
4. Check that state advances correctly, slots accumulate, and the machine
   reaches the final node.
5. Use .assist() to inject mid-conversation context.
6. Use streaming mode for one turn.
7. Switch between flows in a FlowSet.
8. Reset and start a fresh conversation on the same instance.

All LLM calls are stubbed via a scripted tool-calling provider so these tests
are fully hermetic — no API key, no network, no model nondeterminism.
"""

from __future__ import annotations

from pathlib import Path

from superdialog import DialogMachine, Flow, FlowSet, StreamChunk, Turn
from tests.scripted_toolcall import ScriptedToolProvider, route

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "flow"


def _dm(provider: ScriptedToolProvider, flow_path: str = "kyc.json") -> DialogMachine:
    dm = DialogMachine(
        flow=Flow.load(FIXTURE_DIR / flow_path), llm="openai/gpt-4o-mini"
    )
    # Injected before the first turn -> the ToolCallAdapter shares this provider.
    dm._llm = provider  # type: ignore[assignment]
    return dm


# ---------------------------------------------------------------------------
# Multi-turn: walk the KYC flow greet → name → dob → pan → done
# ---------------------------------------------------------------------------


async def test_full_kyc_journey_reaches_final_node() -> None:
    """Simulate a 4-turn KYC journey and verify the machine reaches 'done'."""
    dm = _dm(
        ScriptedToolProvider(
            routes=[
                route("greet_to_name"),
                route("name_to_dob", slots={"name": "Alice"}),
                route("dob_to_pan", slots={"dob": "1990-05-15"}),
                route("pan_to_done", slots={"pan": "ABCDE1234F"}),
            ]
        )
    )

    await dm.turn("Hello")
    assert dm.state["node_id"] == "collect_name"

    await dm.turn("My name is Alice")
    assert dm.state["node_id"] == "collect_dob"
    assert dm.state["slots"].get("name") == "Alice"

    await dm.turn("15 May 1990")
    assert dm.state["node_id"] == "collect_pan"

    r4 = await dm.turn("ABCDE1234F")
    assert dm.state["node_id"] == "done"
    assert isinstance(r4, Turn)
    assert r4.metadata["outcome"] == "transition"


async def test_stay_when_criteria_not_met() -> None:
    """LLM stays on the node → machine stays, with the brief reply spoken."""
    dm = _dm(
        ScriptedToolProvider(routes=[route(None, brief="Could you repeat your name?")])
    )
    result = await dm.turn("hmm")
    assert result.metadata["outcome"] == "stay"
    assert dm.state["node_id"] == "greet"
    assert "repeat" in result.text.lower()


# ---------------------------------------------------------------------------
# .assist() — mid-conversation context injection
# ---------------------------------------------------------------------------


async def test_assist_injects_system_message_before_next_turn() -> None:
    dm = _dm(ScriptedToolProvider(routes=[route("greet_to_name")]))
    dm.assist("Customer is VIP. Be extra polite.")
    assert len(dm._pending_system_messages) == 1
    await dm.turn("hi")
    # After turn, pending messages are consumed
    assert len(dm._pending_system_messages) == 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streaming_turn_yields_chunks_and_final_turn() -> None:
    dm = _dm(
        ScriptedToolProvider(
            routes=[route("greet_to_name")], replies=["Welcome to KYC."]
        )
    )
    stream = await dm.turn("hi", stream=True)
    chunks: list[StreamChunk] = [c async for c in stream]
    assert chunks, "expected chunks"
    assert chunks[-1].done is True
    assert isinstance(chunks[-1].turn, Turn)
    reassembled = "".join(c.text for c in chunks)
    assert reassembled == chunks[-1].turn.text


# ---------------------------------------------------------------------------
# FlowSet: switch between flows
# ---------------------------------------------------------------------------


async def test_switch_flow_resets_to_new_initial_node() -> None:
    kyc = Flow.load(FIXTURE_DIR / "kyc.json")
    appt = Flow.load(FIXTURE_DIR / "appointment.json")
    dm = DialogMachine(
        flow=FlowSet({"kyc": kyc, "appointment": appt}),
        llm="openai/gpt-4o-mini",
    )
    dm._llm = ScriptedToolProvider(routes=[route("greet_to_name")])  # type: ignore[assignment]

    await dm.turn("hello")
    assert dm.state["node_id"] == "collect_name"

    dm.switch_flow("appointment")
    assert dm.state["node_id"] == "intro"


# ---------------------------------------------------------------------------
# Reset: start fresh on same instance
# ---------------------------------------------------------------------------


async def test_reset_returns_to_initial_node_with_clean_slots() -> None:
    dm = _dm(
        ScriptedToolProvider(routes=[route("greet_to_name", slots={"name": "Alice"})])
    )
    await dm.turn("I'm Alice")
    assert dm.state["node_id"] == "collect_name"
    assert dm.state["slots"].get("name") == "Alice"

    dm.reset()
    assert dm.state["node_id"] == "greet"
    assert dm.state["slots"] == {}
