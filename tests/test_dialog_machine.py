"""Tests for the ``DialogMachine`` facade.

We exercise the spec-aligned API surface (``turn``, ``inject_system``,
``reset``, ``set_llm``, ``switch_flow``, ``state``) with a scripted tool-calling
``LLMProvider`` so the tests run hermetically without network calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superdialog import DialogMachine, Flow, FlowSet, Turn
from tests.scripted_toolcall import ScriptedToolProvider, route

FIXTURE = Path(__file__).parent / "fixtures" / "flow" / "kyc.json"


def _load_flow() -> Flow:
    return Flow.load(FIXTURE)


def _machine_with(
    provider: ScriptedToolProvider,
) -> tuple[DialogMachine, ScriptedToolProvider]:
    """Return a DialogMachine wired to a scripted provider.

    ``machine._llm`` is injected before the first turn; the ToolCallAdapter
    picks it up via ``set_provider`` when the machine is built.
    """
    machine = DialogMachine(flow=_load_flow(), llm="openai/gpt-4o-mini")
    machine._llm = provider  # type: ignore[assignment]
    return machine, provider


# ---------------------------------------------------------------------------
# turn() -- non-streaming
# ---------------------------------------------------------------------------


async def test_turn_returns_turn_and_advances_state() -> None:
    machine, _ = _machine_with(ScriptedToolProvider(routes=[route("greet_to_name")]))
    result = await machine.turn("Hello, my name is Alice")
    assert isinstance(result, Turn)
    assert result.text  # non-empty (response or generated_reply)
    assert result.metadata["from_node"] == "greet"
    assert result.metadata["to_node"] == "collect_name"
    assert result.metadata["outcome"] == "transition"
    assert machine.state["node_id"] == "collect_name"


async def test_turn_stays_when_no_edge_recommended() -> None:
    machine, _ = _machine_with(
        ScriptedToolProvider(routes=[route(None, brief="Could you repeat that?")])
    )
    result = await machine.turn("uh")
    assert result.metadata["outcome"] == "stay"
    assert result.metadata["from_node"] == result.metadata["to_node"] == "greet"
    assert "repeat" in result.text.lower()


# ---------------------------------------------------------------------------
# inject_system / reset
# ---------------------------------------------------------------------------


async def test_assist_flushes_into_history_on_next_turn() -> None:
    machine, provider = _machine_with(
        ScriptedToolProvider(routes=[route("greet_to_name")])
    )
    machine.assist("Caller is upset. Be empathetic.")
    await machine.turn("hi")
    routing = provider.routing_messages[0]
    assert any(
        m.get("role") == "system" and "empathetic" in m.get("content", "")
        for m in routing
    )


async def test_reset_clears_machine_and_memory() -> None:
    machine, _ = _machine_with(ScriptedToolProvider(routes=[route("greet_to_name")]))
    await machine.turn("hello")
    assert machine.state["node_id"] == "collect_name"
    machine.reset()
    # next turn rebuilds at the flow's initial node
    assert machine.state["node_id"] == "greet"


# ---------------------------------------------------------------------------
# set_llm / switch_flow
# ---------------------------------------------------------------------------


async def test_set_llm_swaps_provider_on_active_adapter() -> None:
    machine, _ = _machine_with(ScriptedToolProvider(routes=[route("greet_to_name")]))
    await machine.turn("hello")  # build the adapter
    second = ScriptedToolProvider(
        routes=[route("name_to_dob", slots={"name": "Alice"})]
    )
    machine.set_llm("openai/gpt-4o-mini")
    # set_llm rebuilt _llm via resolve; overwrite + re-inject for the hermetic test
    machine._llm = second  # type: ignore[assignment]
    assert machine._adapter is not None
    machine._adapter.set_provider(second)
    result = await machine.turn("Alice")
    assert result.metadata["to_node"] == "collect_dob"
    assert second.calls, "new provider should have been called"


async def test_switch_flow_routes_to_named_flow() -> None:
    flow_a = _load_flow()
    flow_b = _load_flow()
    machine = DialogMachine(
        flow=FlowSet({"main": flow_a, "alt": flow_b}),
        llm="openai/gpt-4o-mini",
    )
    machine._llm = ScriptedToolProvider(routes=[route("greet_to_name")])  # type: ignore[assignment]
    await machine.turn("hello")
    assert machine.state["node_id"] == "collect_name"
    machine.switch_flow("alt")
    # alt rebuilds clean
    assert machine.state["node_id"] == "greet"
    with pytest.raises(KeyError):
        machine.switch_flow("does_not_exist")


# ---------------------------------------------------------------------------
# state property
# ---------------------------------------------------------------------------


def test_state_before_first_turn_returns_initial_node() -> None:
    machine = DialogMachine(flow=_load_flow(), llm="openai/gpt-4o-mini")
    snapshot = machine.state
    assert snapshot == {"node_id": "greet", "slots": {}}
