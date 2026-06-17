from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from superdialog.flow.models import ConversationFlow
from superdialog.machine.criteria import CriteriaJudge
from superdialog.machine.eval.cache import ResponseCache
from superdialog.machine.eval.models import (
    EdgeResult,
    EdgeTest,
    EvalReport,
    ModelScore,
    NegativeEdgeResult,
    PathResult,
    PathStepResult,
    PathTest,
    PersonaResult,
    TestCorpus,
)
import superdialog.machine.eval.user_simulator as _user_simulator_module
from superdialog.machine.runner import FlowResult
from superdialog.machine.models import TransitionRecord

logger = logging.getLogger(__name__)

LLMFn = Callable[[list[dict[str, Any]]], Any]


async def run_flow_from_node(
    flow: ConversationFlow,
    llm_fn: LLMFn,
    start_node: str,
    user_messages: list[str],
) -> FlowResult:
    node_map = {n.id: n for n in flow.nodes}
    judge = CriteriaJudge(llm_fn=llm_fn)
    current_node_id = start_node
    current_node = node_map.get(start_node)
    transitions: list[TransitionRecord] = []
    history: list[dict[str, Any]] = []

    if current_node is None:
        return FlowResult(
            final_state=start_node,
            is_complete=False,
            transitions=[],
        )

    for msg in user_messages:
        if current_node is None or current_node.is_final:
            break
        history.append({"role": "user", "content": msg})
        result = await judge.evaluate(
            node=current_node,
            history=history,
            userdata={},
            system_prompt=flow.system_prompt or "",
        )
        if result.recommended_edge_id:
            edge = next(
                (e for e in current_node.edges if e.id == result.recommended_edge_id),
                None,
            )
            if edge and edge.target_node_id:
                record = TransitionRecord(
                    from_node=current_node_id,
                    to_node=edge.target_node_id,
                    edge_id=edge.id,
                    criteria_met=result.criteria_met,
                )
                transitions.append(record)
                current_node_id = edge.target_node_id
                current_node = node_map.get(current_node_id)

    is_complete = current_node is not None and current_node.is_final
    return FlowResult(
        final_state=current_node_id,
        is_complete=is_complete,
        transitions=transitions,
    )


def _cached_llm(
    llm_fn: LLMFn,
    cache: ResponseCache,
    model_id: str,
) -> LLMFn:
    async def wrapper(messages: list[dict[str, Any]]) -> Any:
        key = cache.hash_messages(messages)
        cached = cache.get_raw(model_id, key)
        if cached is not None:
            return cached
        result = await llm_fn(messages)
        cache.put_raw(model_id, key, result)
        return result

    return wrapper


class FlowEvaluator:
    def __init__(
        self,
        flow: ConversationFlow,
        llm_factory: Callable[[str], LLMFn],
        cache: ResponseCache | None = None,
    ) -> None:
        self._flow = flow
        self._llm_factory = llm_factory
        self._cache = cache
        self._node_map = {n.id: n for n in flow.nodes}

    def _make_llm(self, model_id: str) -> LLMFn:
        llm = self._llm_factory(model_id)
        if self._cache is not None:
            return _cached_llm(llm, self._cache, model_id)
        return llm

    async def eval_edge(
        self,
        edge_test: EdgeTest,
        utterance: str,
        model_id: str,
    ) -> EdgeResult:
        node = self._node_map.get(edge_test.node_id)
        if node is None:
            return EdgeResult(
                passed=False,
                expected_edge=edge_test.edge_id,
                error=f"Node '{edge_test.node_id}' not found in flow",
            )
        llm_fn = self._make_llm(model_id)
        judge = CriteriaJudge(llm_fn=llm_fn)
        try:
            result = await judge.evaluate(
                node=node,
                history=[{"role": "user", "content": utterance}],
                userdata={},
                system_prompt=self._flow.system_prompt or "",
            )
        except Exception as exc:
            return EdgeResult(
                passed=False,
                expected_edge=edge_test.edge_id,
                error=str(exc),
            )
        actual = result.recommended_edge_id
        return EdgeResult(
            passed=(actual == edge_test.edge_id),
            actual_edge=actual,
            expected_edge=edge_test.edge_id,
        )

    async def eval_negative_edge(
        self,
        edge_test: EdgeTest,
        utterance: str,
        model_id: str,
    ) -> NegativeEdgeResult:
        node = self._node_map.get(edge_test.node_id)
        if node is None:
            return NegativeEdgeResult(
                passed=False,
                error=f"Node '{edge_test.node_id}' not found in flow",
            )
        llm_fn = self._make_llm(model_id)
        try:
            flow_result = await run_flow_from_node(
                flow=self._flow,
                llm_fn=llm_fn,
                start_node=edge_test.node_id,
                user_messages=[utterance],
            )
        except Exception as exc:
            return NegativeEdgeResult(passed=False, error=str(exc))

        actual: str | None = None
        for tr in flow_result.transitions:
            if tr.from_node == edge_test.node_id:
                actual = tr.edge_id
                break

        return NegativeEdgeResult(
            passed=(actual != edge_test.edge_id),
            actual_edge=actual,
        )

    async def eval_fallback_edge(
        self,
        node_id: str,
        expected_fallback_edge: str,
        model_id: str,
    ) -> EdgeResult:
        node = self._node_map.get(node_id)
        if node is None:
            return EdgeResult(
                passed=False,
                expected_edge=expected_fallback_edge,
                error=f"Node '{node_id}' not found in flow",
            )
        max_turns = node.max_turns if node.max_turns is not None else 3
        vague_messages = ["hmm"] * (max_turns + 1)
        llm_fn = self._make_llm(model_id)
        flow_result = await run_flow_from_node(
            flow=self._flow,
            llm_fn=llm_fn,
            start_node=node_id,
            user_messages=vague_messages,
        )
        actual = (
            flow_result.transitions[-1].edge_id
            if flow_result.transitions
            else None
        )
        return EdgeResult(
            passed=(actual == expected_fallback_edge),
            actual_edge=actual,
            expected_edge=expected_fallback_edge,
        )

    async def eval_path(
        self,
        path_test: PathTest,
        model_id: str,
    ) -> PathResult:
        """Drive the flow from its initial node through each scripted step.

        Each step passes when the model takes the expected edge AND lands on the
        expected node; ``completed`` is true when the run ends on a final node.
        """
        llm_fn = self._make_llm(model_id)
        judge = CriteriaJudge(llm_fn=llm_fn)
        current_id = self._flow.initial_node
        current = self._node_map.get(current_id)
        history: list[dict[str, Any]] = []
        steps: list[PathStepResult] = []

        for step in path_test.steps:
            actual_edge: str | None = None
            if current is not None and not current.is_final:
                history.append({"role": "user", "content": step.utterance})
                result = await judge.evaluate(
                    node=current,
                    history=history,
                    userdata={},
                    system_prompt=self._flow.system_prompt or "",
                )
                actual_edge = result.recommended_edge_id
                edge = next(
                    (e for e in current.edges if e.id == actual_edge),
                    None,
                )
                if edge and edge.target_node_id:
                    current_id = edge.target_node_id
                    current = self._node_map.get(current_id)
            steps.append(
                PathStepResult(
                    utterance=step.utterance,
                    expected_edge=step.expected_edge,
                    actual_edge=actual_edge,
                    expected_node=step.expected_node,
                    actual_node=current_id,
                    passed=(
                        actual_edge == step.expected_edge
                        and current_id == step.expected_node
                    ),
                )
            )

        completed = current is not None and current.is_final
        return PathResult(name=path_test.name, completed=completed, steps=steps)

    async def eval_corpus(
        self,
        corpus: TestCorpus,
        model_ids: list[str],
    ) -> EvalReport:
        report = EvalReport(
            flow_file=corpus.flow_file,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        for model_id in model_ids:
            score = await self._eval_corpus_for_model(corpus, model_id)
            report.models.append(score)
        return report

    async def _eval_corpus_for_model(
        self,
        corpus: TestCorpus,
        model_id: str,
    ) -> ModelScore:
        score = ModelScore(model_id=model_id)
        llm_fn = self._make_llm(model_id)

        for edge_test in corpus.edge_tests:
            for utterance in edge_test.utterances:
                result = await self.eval_edge(edge_test, utterance, model_id)
                score.edge_results.append(result)
                if not result.passed:
                    score.failures.append(
                        f"edge '{edge_test.edge_id}' on '{edge_test.node_id}': "
                        f"utterance='{utterance}' → actual='{result.actual_edge}'"
                    )

        if score.edge_results:
            correct = sum(1 for r in score.edge_results if r.passed)
            score.edge_accuracy = correct / len(score.edge_results)

        for edge_test in corpus.edge_tests:
            for utterance in edge_test.negative_utterances:
                neg = await self.eval_negative_edge(edge_test, utterance, model_id)
                score.negative_results.append(neg)
                if not neg.passed:
                    score.negative_failures.append(
                        f"negative '{edge_test.edge_id}' on '{edge_test.node_id}': "
                        f"utterance='{utterance}' → actual='{neg.actual_edge}'"
                    )

        for path_test in corpus.path_tests:
            score.path_results.append(await self.eval_path(path_test, model_id))
        if score.path_results:
            done = sum(1 for r in score.path_results if r.completed)
            score.path_accuracy = done / len(score.path_results)

        persona_results: list[PersonaResult] = []
        for persona_config in corpus.persona_tests:
            simulator = _user_simulator_module.UserSimulator(
                flow=self._flow,
                system_llm_fn=llm_fn,
                persona_llm_fn=llm_fn,
            )
            persona_result = await simulator.simulate(persona_config)
            persona_result.model_id = model_id
            persona_results.append(persona_result)

        score.persona_results = persona_results
        if persona_results:
            reached = sum(1 for r in persona_results if r.reached_final)
            score.persona_completion = reached / len(persona_results)

        return score
