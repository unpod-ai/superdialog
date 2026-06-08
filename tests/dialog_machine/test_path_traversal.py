"""Tier 2: Path traversal eval tests.

Tests end-to-end path completion via FlowEvaluator.eval_path().
Uses mock LLMs for deterministic testing.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

for _mod in [
    "livekit.agents",
    "livekit.agents.llm",
    "livekit.agents.llm.tool_context",
    "livekit.agents.voice",
    "livekit.api",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from superdialog.flow.models import ConversationFlow, Edge, FlowNode  # noqa: E402
from superdialog.machine.machine import DialogStateMachine  # noqa: E402
from superdialog.machine.testing.mock_adapter import MockAdapter  # noqa: E402

# The eval surface (FlowEvaluator, corpus models) was intentionally left
# behind in the slim superdialog port. Guard the import so this module still
# collects — the eval-dependent classes below are skipped when it is absent.
try:
    from superdialog.machine.eval.evaluator import FlowEvaluator  # noqa: E402
    from superdialog.machine.eval.models import (  # noqa: E402
        EdgeTest,
        EvalReport,
        PathStep,
        PathTest,
        TestCorpus,
    )

    _EVAL_AVAILABLE = True
except ModuleNotFoundError:
    _EVAL_AVAILABLE = False

_requires_eval = pytest.mark.skipif(
    not _EVAL_AVAILABLE,
    reason="superdialog.machine.eval not ported into this tree",
)


def _three_node_flow() -> ConversationFlow:
    """greeting -> collect_name -> goodbye."""
    return ConversationFlow(
        system_prompt="You are a friendly assistant.",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Say hello and ask for the user's name.",
                edges=[
                    Edge(
                        id="name_given",
                        condition="User provides name",
                        target_node_id="collect_name",
                    ),
                ],
            ),
            FlowNode(
                id="collect_name",
                name="Collect Name",
                instruction="Confirm the name and say goodbye.",
                edges=[
                    Edge(
                        id="confirmed",
                        condition="User confirms",
                        target_node_id="goodbye",
                    ),
                ],
            ),
            FlowNode(
                id="goodbye",
                name="Goodbye",
                static_text="Thank you! Goodbye.",
                is_final=True,
            ),
        ],
    )


def _make_sequenced_llm(edge_sequence: list[str]):
    """Create a mock LLM that returns edges in sequence."""
    idx = {"i": 0}

    async def llm(messages: list[dict]) -> str:
        sys_content = messages[0].get("content", "")
        if "evaluating" in sys_content:
            edge_id = edge_sequence[idx["i"]] if idx["i"] < len(edge_sequence) else None
            idx["i"] += 1
            return json.dumps(
                {
                    "all_required_met": True,
                    "recommended_edge_id": edge_id,
                    "reason": "mock",
                }
            )
        return "Mock reply"

    return llm


@_requires_eval
class TestPathTraversal:
    @pytest.mark.anyio
    async def test_complete_path_passes(self) -> None:
        flow = _three_node_flow()
        llm = _make_sequenced_llm(["name_given", "confirmed"])
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        path_test = PathTest(
            name="happy_path",
            description="Full path through the flow",
            steps=[
                PathStep(
                    utterance="Hi, I'm Alice",
                    expected_edge="name_given",
                    expected_node="collect_name",
                ),
                PathStep(
                    utterance="Yes that's correct",
                    expected_edge="confirmed",
                    expected_node="goodbye",
                ),
            ],
        )

        result = await evaluator.eval_path(path_test, "mock")
        assert result.completed is True
        assert len(result.steps) == 2
        assert all(s.passed for s in result.steps)

    @pytest.mark.anyio
    async def test_wrong_edge_in_path_fails(self) -> None:
        flow = _three_node_flow()
        # LLM returns wrong edge for second step
        llm = _make_sequenced_llm(["name_given", "name_given"])
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        path_test = PathTest(
            name="wrong_edge_path",
            steps=[
                PathStep(
                    utterance="I'm Bob",
                    expected_edge="name_given",
                    expected_node="collect_name",
                ),
                PathStep(
                    utterance="Yes",
                    expected_edge="confirmed",
                    expected_node="goodbye",
                ),
            ],
        )

        result = await evaluator.eval_path(path_test, "mock")
        assert result.completed is False
        assert result.steps[0].passed is True
        assert result.steps[1].passed is False


@_requires_eval
class TestEvalCorpus:
    @pytest.mark.anyio
    async def test_corpus_produces_report(self) -> None:
        flow = _three_node_flow()
        llm = _make_sequenced_llm(["name_given", "name_given", "confirmed"])
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        corpus = TestCorpus(
            flow_file="test.json",
            edge_tests=[
                EdgeTest(
                    node_id="greeting",
                    edge_id="name_given",
                    condition="User provides name",
                    utterances=["I'm Alice"],
                ),
            ],
            path_tests=[
                PathTest(
                    name="happy",
                    steps=[
                        PathStep(
                            utterance="I'm Bob",
                            expected_edge="name_given",
                            expected_node="collect_name",
                        ),
                        PathStep(
                            utterance="Yes",
                            expected_edge="confirmed",
                            expected_node="goodbye",
                        ),
                    ],
                ),
            ],
        )

        report = await evaluator.eval_corpus(corpus, ["mock"])
        assert isinstance(report, EvalReport)
        assert report.flow_file == "test.json"
        assert len(report.models) == 1
        assert report.models[0].model_id == "mock"
        assert report.models[0].edges_total == 1

    @pytest.mark.anyio
    async def test_corpus_with_negatives(self) -> None:
        """Corpus with negative utterances produces negative results."""
        flow = _three_node_flow()
        llm = _make_sequenced_llm(["name_given", "name_given"])
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        corpus = TestCorpus(
            flow_file="test.json",
            edge_tests=[
                EdgeTest(
                    node_id="greeting",
                    edge_id="name_given",
                    condition="User provides name",
                    utterances=["I'm Alice"],
                    negative_utterances=["I don't know"],
                ),
            ],
        )

        report = await evaluator.eval_corpus(corpus, ["mock"])
        ms = report.models[0]
        assert ms.negatives_total == 1
        # The mock always returns name_given, so negative should fail
        assert ms.negatives_passed == 0
        assert len(ms.negative_failures) == 1

    @pytest.mark.anyio
    async def test_report_summary(self) -> None:
        flow = _three_node_flow()
        llm = _make_sequenced_llm(["name_given"])
        evaluator = FlowEvaluator(
            flow=flow,
            llm_factory=lambda _: llm,
        )

        corpus = TestCorpus(
            flow_file="test.json",
            edge_tests=[
                EdgeTest(
                    node_id="greeting",
                    edge_id="name_given",
                    condition="User provides name",
                    utterances=["I'm Alice"],
                ),
            ],
        )

        report = await evaluator.eval_corpus(corpus, ["mock"])
        summary = report.summary()
        assert "test.json" in summary
        assert "mock" in summary
        assert "neg=" in summary


def _chain_flow() -> ConversationFlow:
    """A (instruction) -> R (router) -> C (final). One user turn, two hops."""
    return ConversationFlow(
        system_prompt="t",
        initial_node="A",
        nodes=[
            FlowNode(
                id="A",
                name="A",
                instruction="Ask the caller something.",
                edges=[
                    Edge(
                        id="edge_a_to_r", condition="caller replies", target_node_id="R"
                    )
                ],
            ),
            FlowNode(
                id="R",
                name="R",
                node_type="router",
                edges=[
                    Edge(id="edge_r_to_c", condition="route onward", target_node_id="C")
                ],
            ),
            FlowNode(id="C", name="C", instruction="Say goodbye.", is_final=True),
        ],
    )


@pytest.mark.anyio
async def test_router_chain_stamps_messages_on_each_record() -> None:
    machine = await DialogStateMachine.from_flow(
        flow=_chain_flow(),
        adapter=MockAdapter(["edge_a_to_r", "edge_r_to_c"]),
    )

    await machine.process_turn("hello there")

    log = machine.context.transition_log
    assert len(log) == 2, "one user turn should fire two transitions"

    # First hop: triggered by the real user utterance; enters a router (no speech).
    assert (log[0].from_node, log[0].to_node) == ("A", "R")
    assert log[0].user_message == "hello there"
    assert log[0].bot_message == ""

    # Chained hop: auto-routed, so NO new user input; carries its own reply.
    assert (log[1].from_node, log[1].to_node) == ("R", "C")
    assert log[1].user_message is None
    assert log[1].bot_message == "mock reply for C"
    assert machine.is_complete
