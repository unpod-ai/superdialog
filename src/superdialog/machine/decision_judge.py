"""DecisionCriteriaJudge -- node routing by the typed-decision router, LLM on doubt.

Drop-in for :class:`CriteriaJudge` (same ``evaluate`` / ``build_evaluation_messages``
surface) so :class:`LLMAdapter` uses it unchanged. One node evaluation fans out
in parallel to:

* ``choice`` over the node's edges plus ``stay`` (with the edges' input_schema
  fields as ``extract``),
* one ``noul`` per completion criterion,
* one ``noul`` for ``user_insisting`` when the node allows skipping.

The router never writes speech, so ``response`` is always None; the machine's STAY
path already falls back to ``adapter.generate_reply`` for that. Any router error
or a branch probability under ``client.min_prob`` hands the turn to the wrapped
LLM judge. In ``shadow`` mode the LLM decides and agreement is logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from superdialog.flow.models import Edge, FlowNode
from superdialog.llm.decision import (
    Decision,
    DecisionClient,
    DecisionError,
    Option,
    unique_id,
)
from superdialog.llm.provider import LLMProvider
from superdialog.machine.criteria import (
    _ROUTER_MAX_LIST_ITEMS,
    CriteriaJudge,
    _trim_userdata,
    classify_node_type,
)
from superdialog.machine.models import CriteriaResult

logger = logging.getLogger(__name__)

_HISTORY_TURNS = 8
_IST = timezone(timedelta(hours=5, minutes=30))


def _edge_fields(edge: Edge | None) -> dict[str, Any]:
    schema = edge.input_schema if edge else None
    if isinstance(schema, type):  # a pydantic model class
        schema = schema.model_json_schema()
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    return props if isinstance(props, dict) else {}


def _extract_spec(node: FlowNode) -> dict[str, dict[str, str]]:
    today = datetime.now(_IST).strftime("%Y-%m-%d")
    spec: dict[str, dict[str, str]] = {}
    for edge in node.edges:
        for key, prop in _edge_fields(edge).items():
            prop = prop if isinstance(prop, dict) else {}
            desc = prop.get("description", "") or key
            if prop.get("format") == "date" or "date" in key.lower():
                desc += f" (as YYYY-MM-DD; today is {today})"
            spec[key] = {"type": str(prop.get("type", "string")), "description": desc}
    return spec


def _render_state(history: list[dict[str, Any]], userdata: dict[str, Any]) -> str:
    lines = [
        f"{m.get('role')}: {m.get('content')}"
        for m in history[-_HISTORY_TURNS:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if userdata:
        # Last so tail-clipping drops old turns before the data routers key on.
        trimmed = _trim_userdata(userdata, _ROUTER_MAX_LIST_ITEMS)
        lines.append(f"[collected data] {json.dumps(trimmed, ensure_ascii=False, default=str)}")
    return "\n".join(lines)


class DecisionCriteriaJudge:
    def __init__(self, client: DecisionClient, fallback: CriteriaJudge) -> None:
        self._client = client
        self._fallback = fallback

    def set_fallback_llm(self, provider: LLMProvider) -> None:
        self._fallback = CriteriaJudge(llm=provider)

    def build_evaluation_messages(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        # Only used for call logging; the LLM prompt is the fallback's prompt.
        return self._fallback.build_evaluation_messages(*args, **kwargs)

    async def evaluate(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> CriteriaResult:
        if self._client.mode == "shadow":
            llm_result, routed = await asyncio.gather(
                self._fallback.evaluate(node, history, userdata, system_prompt, **kwargs),
                self._safe_route(node, history, userdata),
            )
            logger.info(
                "[decision-shadow] node=%s agree=%s llm=%s router=%s %s",
                node.id,
                routed is not None
                and routed.recommended_edge_id == llm_result.recommended_edge_id,
                llm_result.recommended_edge_id,
                routed.recommended_edge_id if routed else "<deferred>",
                routed.reason if routed else "",
            )
            return llm_result
        result = await self._safe_route(node, history, userdata)
        if result is not None:
            return result
        return await self._fallback.evaluate(node, history, userdata, system_prompt, **kwargs)

    async def _safe_route(
        self, node: FlowNode, history: list[dict[str, Any]], userdata: dict[str, Any]
    ) -> CriteriaResult | None:
        try:
            return await self._route(node, history, userdata)
        except DecisionError as exc:
            # Expected (timeout / HTTP / bad output): one line, no traceback.
            logger.warning("[decision] node=%s failed (%s), LLM decides", node.id, exc)
            return None
        except Exception:
            # Unexpected = a code bug: keep the traceback.
            logger.warning("[decision] node=%s failed, LLM decides", node.id, exc_info=True)
            return None

    async def _route(
        self, node: FlowNode, history: list[dict[str, Any]], userdata: dict[str, Any]
    ) -> CriteriaResult | None:
        if not node.edges or classify_node_type(node) == "final":
            return None
        state = _render_state(history, userdata)
        stay = unique_id("stay", {e.id for e in node.edges})
        goal = (node.instruction or node.static_text or node.name)[:400]
        options = [Option(e.id, e.condition or f"go to {e.target_node_id}") for e in node.edges]
        options.append(Option(stay, f"No transition applies yet; keep working on: {goal}"))
        question = f"The assistant is at step '{node.name}'. Which option fits the latest user turn?"

        criteria = node.completion_criteria or []
        calls = [self._client.choice(state, question, options, _extract_spec(node) or None)]
        calls += [self._client.noul(state, c.description) for c in criteria]
        if node.allow_skip and criteria:
            calls.append(
                self._client.noul(
                    state, "The user insists on moving on without giving the requested information."
                )
            )
        answers: list[Decision] = list(await asyncio.gather(*calls))
        choice, nouls = answers[0], answers[1:]
        logger.info(
            "[decision] node=%s mode=%s edge=%s@%.2f %s",
            node.id,
            self._client.mode,
            choice.branch,
            choice.prob,
            " ".join(f"{c.key}={n.branch}@{n.prob:.2f}" for c, n in zip(criteria, nouls)),
        )
        # Every answer gates: a shaky criterion or user_insisting silently
        # blocks or skips the node just like a shaky edge pick would.
        weakest = min(d.prob for d in answers)
        if weakest < self._client.min_prob:
            logger.info("[decision] node=%s low confidence p=%.2f, LLM decides", node.id, weakest)
            return None

        criteria_met = {c.key: n.branch == "true" for c, n in zip(criteria, nouls)}
        edge_id = None if choice.branch == stay else choice.branch
        wanted = _edge_fields(next((e for e in node.edges if e.id == edge_id), None))
        return CriteriaResult(
            node_id=node.id,
            criteria_met=criteria_met,
            all_required_met=all(criteria_met[c.key] for c in criteria if c.required),
            user_insisting=len(nouls) > len(criteria) and nouls[-1].branch == "true",
            recommended_edge_id=edge_id,
            reason=f"router p={choice.prob:.2f}",
            response=None,
            extracted_slots={
                k: v for k, v in choice.extracted.items() if k in wanted and v is not None
            },
        )
