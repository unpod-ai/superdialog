"""Task 4.5 -- persona wiring, fallback eval, and slot assertion tests."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

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
from superdialog.machine.eval.evaluator import FlowEvaluator  # noqa: E402
from superdialog.machine.eval.models import (  # noqa: E402
    EdgeTest,
    ModelScore,
    PersonaConfig,
    PersonaResult,
    TestCorpus,
)
from superdialog.machine.models import TransitionRecord  # noqa: E402
from superdialog.machine.runner import FlowResult  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _flow_with_fallback() -> ConversationFlow:
    """2 nodes: greeting (max_turns=2, fallback edge) -> goodbye."""
    return ConversationFlow(
        system_prompt="Test bot.",
        initial_node="greeting",
        nodes=[
            FlowNode(
                id="greeting",
                name="Greeting",
                instruction="Say hello.",
                max_turns=2,
                edges=[
                    Edge(
                        id="name_given",
                        condition="User provides name",
                        target_node_id="goodbye",
                    ),
                    Edge(
                        id="fallback_edge",
                        condition="Fallback",
                        target_node_id="goodbye",
                        is_fallback=True,
                    ),
                ],
            ),
            FlowNode(
                id="goodbye",
                name="Goodbye",
                static_text="Bye!",
                is_final=True,
            ),
        ],
    )


def _dummy_llm_factory(model_id: str):
    """Return a no-op async callable as LLM stub."""

    async def _llm(prompt: str, **kwargs) -> str:  # type: ignore[override]
        return "ok"

    return _llm


def _make_flow_result(
    from_node: str,
    to_node: str,
    edge_id: str,
    *,
    is_complete: bool = False,
) -> FlowResult:
    """Build a FlowResult with a single transition."""
    return FlowResult(
        final_state=to_node,
        is_complete=is_complete,
        transitions=[
            TransitionRecord(
                from_node=from_node,
                to_node=to_node,
                edge_id=edge_id,
            )
        ],
    )


def _make_no_transition_result(final_state: str) -> FlowResult:
    """FlowResult with no transitions (edge not triggered)."""
    return FlowResult(
        final_state=final_state,
        is_complete=False,
        transitions=[],
    )


# ---------------------------------------------------------------------------
# TestEvalFallbackEdge
# ---------------------------------------------------------------------------


class TestEvalFallbackEdge:
    """Tests for FlowEvaluator.eval_fallback_edge."""

    @pytest.mark.anyio
    @patch(
        "superdialog.machine.eval.evaluator.run_flow_from_node",
        new_callable=AsyncMock,
    )
    async def test_fallback_triggers_on_max_turns_exceeded(
        self, mock_run: AsyncMock
    ) -> None:
        """Fallback edge is detected when the mock returns it."""
        flow = _flow_with_fallback()
        mock_run.return_value = _make_flow_result(
            "greeting", "goodbye", "fallback_edge"
        )

        evaluator = FlowEvaluator(flow, _dummy_llm_factory)
        result = await evaluator.eval_fallback_edge(
            node_id="greeting",
            expected_fallback_edge="fallback_edge",
            model_id="test-model",
        )

        assert result.passed is True
        assert result.actual_edge == "fallback_edge"
        assert result.error is None

    @pytest.mark.anyio
    async def test_fallback_node_not_found(self) -> None:
        """Non-existent node_id produces an error result."""
        flow = _flow_with_fallback()
        evaluator = FlowEvaluator(flow, _dummy_llm_factory)

        result = await evaluator.eval_fallback_edge(
            node_id="nonexistent",
            expected_fallback_edge="fallback_edge",
            model_id="test-model",
        )

        assert result.passed is False
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.anyio
    @patch(
        "superdialog.machine.eval.evaluator.run_flow_from_node",
        new_callable=AsyncMock,
    )
    async def test_fallback_sends_correct_number_of_messages(
        self, mock_run: AsyncMock
    ) -> None:
        """max_turns+1 messages are sent to trigger fallback."""
        flow = _flow_with_fallback()
        mock_run.return_value = _make_flow_result(
            "greeting", "goodbye", "fallback_edge"
        )

        evaluator = FlowEvaluator(flow, _dummy_llm_factory)
        await evaluator.eval_fallback_edge(
            node_id="greeting",
            expected_fallback_edge="fallback_edge",
            model_id="test-model",
        )

        # greeting node has max_turns=2, so 2+1=3 messages
        call_kwargs = mock_run.call_args
        messages = call_kwargs.kwargs.get(
            "user_messages", call_kwargs.args[3] if len(call_kwargs.args) > 3 else None
        )
        if messages is None:
            # positional: flow, llm_fn, start_node, user_messages
            messages = call_kwargs[1].get("user_messages", [])
        assert len(messages) == 3


# ---------------------------------------------------------------------------
# TestEvalNegativeEdge
# ---------------------------------------------------------------------------


class TestEvalNegativeEdge:
    """Tests for FlowEvaluator.eval_negative_edge."""

    @pytest.mark.anyio
    @patch(
        "superdialog.machine.eval.evaluator.run_flow_from_node",
        new_callable=AsyncMock,
    )
    async def test_negative_passes_when_edge_not_triggered(
        self, mock_run: AsyncMock
    ) -> None:
        """Negative test passes when the forbidden edge is NOT taken."""
        flow = _flow_with_fallback()
        mock_run.return_value = _make_no_transition_result("greeting")

        evaluator = FlowEvaluator(flow, _dummy_llm_factory)
        edge_test = EdgeTest(
            node_id="greeting",
            edge_id="name_given",
            utterances=["My name is Alice"],
            negative_utterances=["I like pizza"],
        )

        result = await evaluator.eval_negative_edge(
            edge_test, "I like pizza", "test-model"
        )

        assert result.passed is True
        assert result.actual_edge is None

    @pytest.mark.anyio
    @patch(
        "superdialog.machine.eval.evaluator.run_flow_from_node",
        new_callable=AsyncMock,
    )
    async def test_negative_fails_when_edge_triggered(
        self, mock_run: AsyncMock
    ) -> None:
        """Negative test fails when the forbidden edge IS taken."""
        flow = _flow_with_fallback()
        mock_run.return_value = _make_flow_result("greeting", "goodbye", "name_given")

        evaluator = FlowEvaluator(flow, _dummy_llm_factory)
        edge_test = EdgeTest(
            node_id="greeting",
            edge_id="name_given",
            utterances=["My name is Alice"],
            negative_utterances=["I like pizza"],
        )

        result = await evaluator.eval_negative_edge(
            edge_test, "I like pizza", "test-model"
        )

        assert result.passed is False
        assert result.actual_edge == "name_given"

    @pytest.mark.anyio
    async def test_negative_node_not_found(self) -> None:
        """Non-existent node_id produces an error."""
        flow = _flow_with_fallback()
        evaluator = FlowEvaluator(flow, _dummy_llm_factory)
        edge_test = EdgeTest(
            node_id="nonexistent",
            edge_id="name_given",
        )

        result = await evaluator.eval_negative_edge(edge_test, "hello", "test-model")

        assert result.passed is False
        assert result.error is not None
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# TestEvalCorpusPersonaWiring
# ---------------------------------------------------------------------------


class TestEvalCorpusPersonaWiring:
    """Tests for persona wiring in FlowEvaluator.eval_corpus."""

    @pytest.mark.anyio
    @patch(
        "superdialog.machine.eval.evaluator.run_flow_from_node",
        new_callable=AsyncMock,
    )
    async def test_persona_tests_are_simulated(self, mock_run: AsyncMock) -> None:
        """Persona tests in corpus are dispatched to UserSimulator."""
        flow = _flow_with_fallback()

        persona_result = PersonaResult(
            persona_name="friendly-user",
            model_id="",
            final_node="goodbye",
            expected_final_node="goodbye",
            reached_final=True,
            turns_taken=3,
        )

        mock_simulator_instance = MagicMock()
        mock_simulator_instance.simulate = AsyncMock(return_value=persona_result)

        corpus = TestCorpus(
            flow_file="test.yaml",
            edge_tests=[],
            path_tests=[],
            persona_tests=[
                PersonaConfig(
                    name="friendly-user",
                    traits="friendly, cooperative",
                    goal="Complete the greeting",
                    expected_final_node="goodbye",
                    max_turns=10,
                ),
            ],
        )

        with patch(
            "superdialog.machine.eval.user_simulator.UserSimulator",
            return_value=mock_simulator_instance,
        ):
            evaluator = FlowEvaluator(flow, _dummy_llm_factory)
            report = await evaluator.eval_corpus(corpus, ["test-model"])

        assert len(report.models) == 1
        score: ModelScore = report.models[0]
        assert len(score.persona_results) == 1
        assert score.persona_results[0].persona_name == "friendly-user"
        assert score.persona_results[0].reached_final is True

    @pytest.mark.anyio
    @patch(
        "superdialog.machine.eval.evaluator.run_flow_from_node",
        new_callable=AsyncMock,
    )
    async def test_persona_completion_calculated(self, mock_run: AsyncMock) -> None:
        """persona_completion is calculated as reached / total."""
        flow = _flow_with_fallback()

        reached = PersonaResult(
            persona_name="good-user",
            final_node="goodbye",
            expected_final_node="goodbye",
            reached_final=True,
            turns_taken=2,
        )
        not_reached = PersonaResult(
            persona_name="confused-user",
            final_node="greeting",
            expected_final_node="goodbye",
            reached_final=False,
            turns_taken=10,
        )

        call_count = 0

        async def _simulate_side_effect(persona):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return reached
            return not_reached

        mock_simulator_instance = MagicMock()
        mock_simulator_instance.simulate = AsyncMock(side_effect=_simulate_side_effect)

        corpus = TestCorpus(
            flow_file="test.yaml",
            edge_tests=[],
            path_tests=[],
            persona_tests=[
                PersonaConfig(
                    name="good-user",
                    goal="Complete",
                    expected_final_node="goodbye",
                ),
                PersonaConfig(
                    name="confused-user",
                    goal="Get confused",
                    expected_final_node="goodbye",
                ),
            ],
        )

        with patch(
            "superdialog.machine.eval.user_simulator.UserSimulator",
            return_value=mock_simulator_instance,
        ):
            evaluator = FlowEvaluator(flow, _dummy_llm_factory)
            report = await evaluator.eval_corpus(corpus, ["test-model"])

        score: ModelScore = report.models[0]
        assert len(score.persona_results) == 2
        # 1 out of 2 reached final => 0.5
        assert score.persona_completion == pytest.approx(0.5)
