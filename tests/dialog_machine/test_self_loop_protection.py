"""process_turn must enforce the self-loop ceiling (MAX_SELF_LOOPS)."""

from __future__ import annotations

import pytest

from superdialog.flow.models import ConversationFlow, Edge, FlowNode
from superdialog.machine.machine import DialogStateMachine
from superdialog.machine.testing.mock_adapter import MockAdapter


@pytest.mark.anyio
async def test_process_turn_enforces_self_loop_ceiling() -> None:
    flow = ConversationFlow(
        system_prompt="t",
        initial_node="s",
        nodes=[
            FlowNode(
                id="s",
                name="s",
                instruction="Loop.",
                edges=[Edge(id="edge_self", condition="loop", target_node_id="s")],
            )
        ],
    )
    # Judge keeps recommending the self-edge every turn.
    machine = await DialogStateMachine.from_flow(
        flow=flow, adapter=MockAdapter(["edge_self"] * 5),
    )
    last = None
    for _ in range(5):
        last = await machine.process_turn("again")

    # MAX_SELF_LOOPS default is 2 -> at most 2 self-transitions recorded.
    assert len(machine.context.transition_log) == 2
    assert machine.state == "s"

    # The ceiling-blocked turn must still return a usable stay reply, not a
    # crash or a silently dropped response.
    assert last is not None
    assert last.outcome == "stay"
    assert last.to_node == "s"
    assert last.response
