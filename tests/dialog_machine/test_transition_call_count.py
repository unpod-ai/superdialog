"""Spike A — the entered-node reply LLM call should be skipped only when the
auto-chain will provably discard it (prefilled / smart-skippable node), and must
STILL fire for ordinary transitions and on the voice (apply_transition) path.

Asserts adapter call COUNTS, since the engine's per-transition LLM-call count is
the contract under test (decoupled from any ToolCallAdapter extraction ladder).
"""

from __future__ import annotations

import pytest

from superdialog.flow.models import ConversationFlow, Edge, FlowNode
from superdialog.machine.machine import DialogStateMachine
from superdialog.machine.testing.mock_adapter import MockAdapter


class CountingAdapter(MockAdapter):
    """MockAdapter that counts routing (evaluate_criteria) and reply
    (generate_reply) LLM round-trips."""

    def __init__(self, edge_sequence: list[str]) -> None:
        super().__init__(edge_sequence)
        self.criteria_calls = 0
        self.reply_calls = 0

    async def evaluate_criteria(self, node, history, userdata):  # type: ignore[no-untyped-def]
        self.criteria_calls += 1
        return await super().evaluate_criteria(node, history, userdata)

    async def generate_reply(self, instruction, node, history=None, userdata=None):  # type: ignore[no-untyped-def]
        self.reply_calls += 1
        return await super().generate_reply(instruction, node, history, userdata)


_BAR_SCHEMA = {
    "type": "object",
    "properties": {"bar": {"type": "string"}},
    "required": ["bar"],
}


def _linear_fresh_flow() -> ConversationFlow:
    """A(instr) -> B(instr, fresh) ; B has a plain edge (no prefill)."""
    return ConversationFlow(
        system_prompt="t",
        initial_node="A",
        nodes=[
            FlowNode(id="A", name="A", instruction="Ask A.",
                     edges=[Edge(id="a_to_b", condition="go", target_node_id="B")]),
            FlowNode(id="B", name="B", instruction="Ask B.",
                     edges=[Edge(id="b_to_c", condition="next", target_node_id="C")]),
            FlowNode(id="C", name="C", instruction="Bye.", is_final=True),
        ],
    )


def _prefilled_flow() -> ConversationFlow:
    """A(instr) -> B(instr, prefilled via b_to_c.required=[bar]) -> C(final)."""
    return ConversationFlow(
        system_prompt="t",
        initial_node="A",
        nodes=[
            FlowNode(id="A", name="A", instruction="Ask A.",
                     edges=[Edge(id="a_to_b", condition="go", target_node_id="B")]),
            FlowNode(id="B", name="B", instruction="Ask B.",
                     edges=[Edge(id="b_to_c", condition="next", target_node_id="C",
                                 input_schema=_BAR_SCHEMA)]),
            FlowNode(id="C", name="C", instruction="Bye.", is_final=True),
        ],
    )


@pytest.mark.anyio
async def test_simple_transition_to_fresh_node_still_speaks() -> None:
    """Baseline / no-regression: a forward transition into a FRESH instruction
    node must still generate its reply (exactly one reply call)."""
    adapter = CountingAdapter(["a_to_b"])
    machine = await DialogStateMachine.from_flow(flow=_linear_fresh_flow(), adapter=adapter)

    result = await machine.process_turn("hi")

    assert adapter.criteria_calls == 1
    assert adapter.reply_calls == 1, "fresh entered node must speak"
    assert machine.state == "B"
    assert result.response, "entered node reply must not be silenced"


@pytest.mark.anyio
async def test_prefilled_entered_node_reply_is_suppressed() -> None:
    """THE WIN: when the entered node is prefilled, the auto-chain routes it
    onward and discards its reply — so the engine must NOT spend an LLM reply
    call on it. Only the final node's reply should be generated.

    RED on pre-fix code (reply_calls == 2); GREEN after the suppression."""
    adapter = CountingAdapter(["a_to_b", "b_to_c"])
    machine = await DialogStateMachine.from_flow(flow=_prefilled_flow(), adapter=adapter)
    # Pre-seed the slot b_to_c requires, so B is prefilled the moment it's entered.
    machine.context.userdata["bar"] = "ready"

    result = await machine.process_turn("hi")

    assert machine.is_complete, "chain should route through prefilled B to final C"
    assert machine.state == "C"
    # B's reply is discarded by the chain → only C (final) should generate one.
    assert adapter.reply_calls == 1, "prefilled B's reply must be suppressed"
    assert result.response, "final node must still speak (no silence regression)"


@pytest.mark.anyio
async def test_voice_path_apply_transition_always_speaks() -> None:
    """The fix is scoped to the auto-chain (process_turn) path. The voice/tool
    path (apply_transition, auto_chain=False) must ALWAYS generate the entered
    node's reply, even when that node is prefilled."""
    adapter = CountingAdapter([])
    machine = await DialogStateMachine.from_flow(flow=_prefilled_flow(), adapter=adapter)
    machine.context.userdata["bar"] = "ready"  # B prefilled

    await machine.apply_transition("a_to_b")

    assert machine.state == "B"
    assert adapter.reply_calls == 1, "voice path must speak the entered node"


@pytest.mark.anyio
async def test_self_loop_ceiling_boundary_still_speaks() -> None:
    """Boundary backstop: when the auto-chain self-loops into the MAX_SELF_LOOPS
    ceiling, the node it stops on must still speak — suppression must not leave a
    silenced (empty) response when the chain stops instead of routing onward."""
    flow = ConversationFlow(
        system_prompt="t",
        initial_node="s",
        nodes=[
            FlowNode(id="s", name="s", instruction="Loop.",
                     edges=[Edge(id="edge_self", condition="loop", target_node_id="s")]),
        ],
    )
    adapter = CountingAdapter(["edge_self"] * 6)
    machine = await DialogStateMachine.from_flow(flow=flow, adapter=adapter)

    responses = [(await machine.process_turn("again")).response for _ in range(5)]

    assert all(responses), f"no turn may be silenced at the ceiling: {responses!r}"
    # The audit log must also reflect the reply the caller heard — the boundary
    # backstop restores the record's bot_message that suppression had blanked,
    # so build_traversal does not report a silent step.
    log = machine.context.transition_log
    assert log and log[-1].bot_message, "boundary record must carry the spoken reply"
