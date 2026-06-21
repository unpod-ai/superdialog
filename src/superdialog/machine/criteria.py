"""CriteriaJudge -- LLM-based node completion evaluator."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from superdialog.flow.models import FlowNode
from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
from superdialog.llm.provider import LLMProvider
from superdialog.machine.models import CriteriaResult

logger = logging.getLogger(__name__)

# DEPRECATED: prefer ``LLMProvider`` (superdialog.llm.provider). Retained as a
# parallel-lives shim during Task 4 so callers can migrate incrementally.
LLMCallable = Callable[[list[dict[str, Any]]], Awaitable[str]]

# Fallback regex for flows that predate the node_type field.
# New flows should set node_type explicitly in the JSON instead.
_TOOL_ONLY_ROUTER_RE = re.compile(
    r"do not output any text|do not generate any speech|do not speak.*at all|"
    r"silent routing node|output only the tool call|"
    r"your only output must be a tool call|zero words.*before.*tool|just route.*no.*speech",
    re.IGNORECASE,
)


_ROUTER_MAX_LIST_ITEMS = 3
_INSTRUCTION_MAX_LIST_ITEMS = 15


def _trim_userdata(
    userdata: dict[str, Any], max_list_items: int, _depth: int = 0
) -> dict[str, Any]:
    """Truncate long lists inside userdata so the criteria prompt stays small.

    Router nodes only check scalar conditions (e.g. hold_result.success=True)
    so they use a small cap (3). Instruction nodes present data to users (slot
    lists, course names) so they use a larger cap (15) to preserve context.
    Lists longer than max_list_items are shortened to first N items plus a
    count sentinel so the LLM still understands the shape.
    """
    if _depth > 4:
        return userdata
    trimmed: dict[str, Any] = {}
    for k, v in userdata.items():
        if isinstance(v, list) and len(v) > max_list_items:
            trimmed[k] = v[:max_list_items] + [f"... ({len(v) - max_list_items} more items)"]
        elif isinstance(v, dict) and _depth < 4:
            trimmed[k] = _trim_userdata(v, max_list_items, _depth + 1)
        else:
            trimmed[k] = v
    return trimmed


def classify_node_type(node: FlowNode) -> str:
    """Classify a flow node into one of: static, instruction, router, final.

    Priority:
    1. node.is_final
    2. node.node_type explicit field (set in flow JSON — preferred)
    3. Structural signals (static_text, instruction)
    4. Regex fallback for legacy flows without node_type set
    """
    if node.is_final:
        return "final"
    if node.node_type:
        return node.node_type
    if node.static_text and not node.instruction:
        return "static"
    if node.instruction:
        # Legacy fallback: detect silent-router pattern from instruction text
        if (
            not node.static_text
            and node.edges
            and _TOOL_ONLY_ROUTER_RE.search(node.instruction)
        ):
            return "router"
        return "instruction"
    return "router"


def _node_type_instructions(node_type: str) -> str:
    """Return evaluation instructions specific to the node type."""
    if node_type == "final":
        return (
            "This is a final node. No transition evaluation needed. "
            "Set all_required_met to false and recommended_edge_id to null."
        )
    if node_type == "static":
        return (
            "This is a static-text node. The bot speaks scripted text. "
            "Focus on whether the user's response satisfies a transition "
            "condition. Do not generate guidance — the static text is the "
            "response. Set response to null."
        )
    if node_type == "router":
        return (
            "This is a SILENT router node. Your ONLY job is to pick the right edge.\n"
            "STRICT RULES:\n"
            "  1. response MUST be null — no exceptions. Zero speech. Zero words.\n"
            "  2. If the chosen edge has an input_schema, extract every field from "
            "the conversation and put them in extracted_slots as key-value pairs. "
            "Example: if edge input_schema has 'city', set extracted_slots={'city':'Delhi'}.\n"
            "  3. Never put edge payload data in response — it goes ONLY in extracted_slots.\n"
            "  4. Set all_required_met=true and pick the best recommended_edge_id."
        )
    # instruction
    return (
        "This is an instruction node. Evaluate whether the node's "
        "criteria are met. If not all met, generate a brief, natural "
        "response that acknowledges the user's input and guides them "
        "toward the remaining criteria. Keep it conversational."
    )


class CriteriaJudge:
    """Evaluates node completion criteria using an LLM.

    Prefer the ``llm: LLMProvider`` parameter (Task 4). The legacy
    ``llm_fn: LLMCallable`` keyword is kept as a deprecated shim and is
    wrapped into the provider protocol internally.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        llm_fn: LLMCallable | None = None,
    ) -> None:
        if llm is not None and llm_fn is not None:
            raise ValueError("Pass either `llm` or `llm_fn`, not both")
        self._llm: LLMProvider | None = llm
        self._llm_fn: LLMCallable | None = llm_fn

    async def _ask(self, messages: list[dict[str, Any]]) -> str:
        if self._llm is not None:
            result = await self._llm.complete(messages)
            return result.text
        assert self._llm_fn is not None  # guarded by evaluate()
        return await self._llm_fn(messages)

    def build_evaluation_messages(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        system_prompt: str = "",
        visit_count: int = 1,
        turns_in_node: int = 0,
        agent_language: str = "",
        agent_gender: str = "",
        node_slots: dict[str, Any] | None = None,
        previously_completed: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the LLM messages for criteria evaluation."""
        node_type = classify_node_type(node)

        def _edge_line(e: Any) -> str:
            line = f'  - id: "{e.id}" → condition: "{e.condition}"'
            schema = getattr(e, "input_schema", None)
            if schema and isinstance(schema, dict):
                props = schema.get("properties", {})
                if props:
                    fields = []
                    for k, v in props.items():
                        desc = v.get("description", "") if isinstance(v, dict) else ""
                        fields.append(f"{k}: {desc}" if desc else k)
                    line += f" | extract: {{{', '.join(fields)}}}"
            return line

        edges_desc = "\n".join(_edge_line(e) for e in node.edges)

        criteria_desc = ""
        has_criteria = bool(node.completion_criteria)
        if has_criteria:
            criteria_lines = []
            for c in node.completion_criteria:
                req = "REQUIRED" if c.required else "optional"
                criteria_lines.append(
                    f'  - key: "{c.key}", description: "{c.description}" ({req})'
                )
            criteria_desc = "\n\nNode completion criteria:\n" + "\n".join(
                criteria_lines
            )

        node_slots_desc = ""
        if node_slots:
            node_slots_desc = (
                f"\n\nSlots collected for this node so far:\n  {json.dumps(node_slots)}"
            )

        userdata_desc = ""
        if userdata:
            max_items = (
                _ROUTER_MAX_LIST_ITEMS
                if node_type == "router"
                else _INSTRUCTION_MAX_LIST_ITEMS
            )
            userdata_desc = f"\n\nCollected data so far:\n  {json.dumps(_trim_userdata(userdata, max_items))}"

        reentry_desc = ""
        if visit_count > 1:
            reentry_desc = (
                f"\n\nNOTE: This node has been visited {visit_count} "
                f"times. The user may be correcting previous input. "
                f"Re-evaluate criteria freshly — do not assume "
                f"previous data is still valid."
            )

        completed_desc = ""
        if previously_completed:
            completed_desc = (
                "\n\nNOTE: This node was previously completed "
                "(the conversation already transitioned out of it). "
                "Do not re-ask for information already collected. "
                "If the user is revisiting, accept updates gracefully."
            )

        turns_desc = ""
        if turns_in_node > 0:
            turns_desc = f"\n\nTurns in this node so far: {turns_in_node}"

        agent_desc = ""
        if agent_language:
            agent_desc += f"\nAgent language: {agent_language}"
        if agent_gender:
            agent_desc += f"\nAgent gender: {agent_gender}"

        node_type_desc = _node_type_instructions(node_type)

        # extracted_slots always present: populated when the recommended edge
        # has "extract: [...]" fields shown above — extract those values from
        # the conversation and put them here as key-value pairs.
        extracted_slots_schema = (
            '  "extracted_slots": {"<field>": "<value from conversation>", ...},\n'
        )

        _ist = timezone(timedelta(hours=5, minutes=30))
        today = datetime.now(_ist).strftime("%A, %d %B %Y")
        # Volatile, per-node/per-turn body (date, node, edges, collected data).
        _body = (
            f"Today's date is {today}. Use this year when resolving"
            " partial dates like '22 April' or 'next Monday'."
            " Store dates in YYYY-MM-DD format.\n\n"
            "You are evaluating whether a conversation node's"
            " objective is met.\n\n"
            f"Current node: {node.id} ({node.name})\n"
            f"Node type: {node_type}\n"
            f"Node instruction: {node.instruction or '(none)'}\n"
            f"\n{node_type_desc}\n"
            f"\nAvailable transitions:\n{edges_desc}"
            f"{criteria_desc}"
            f"{node_slots_desc}"
            f"{userdata_desc}"
            f"{reentry_desc}"
            f"{completed_desc}"
            f"{turns_desc}"
            f"{agent_desc}"
        )
        # Fixed JSON-response scaffolding. The edge-id examples reference the
        # node's own edges (which are "listed above" in ``_body``), so this
        # block stays immediately after ``_body``.
        _schema_block = (
            "\n\nIMPORTANT: recommended_edge_id MUST be one of the"
            " edge id values listed above (e.g. "
            + (
                ", ".join(f'"{e.id}"' for e in node.edges[:3])
                if node.edges
                else '"edge_id"'
            )
            + "). Do NOT use node names or target names."
            "\n\nextracted_slots RULE: When you set recommended_edge_id,"
            " look at that edge's 'extract: [...]' fields. Extract each"
            " field's value from the conversation history and add it to"
            " extracted_slots. Example: edge has 'extract: [city, date]',"
            " user said 'Delhi on Monday' → extracted_slots={'city':'Delhi',"
            " 'date':'2026-05-25'}. Leave extracted_slots={} if edge has no"
            " extract fields."
            "\n\nRespond with ONLY this JSON object, no markdown,"
            " no explanation:\n"
            "{\n"
            '  "criteria_met": {"<key>": true/false, ...},\n'
            f"{extracted_slots_schema}"
            '  "all_required_met": true/false,\n'
            '  "user_insisting": true/false,\n'
            '  "recommended_edge_id": "<edge_id or null>",\n'
            '  "reason": "<brief explanation>",\n'
            '  "response": "<natural language response to the user,'
            ' or null if not needed>"\n'
            "}"
        )

        # Hoist the flow's fixed system prompt to the FRONT so it forms a stable,
        # cacheable prefix (the persona is identical every turn, across nodes).
        # When there is no system prompt, fall back to its original placement at
        # the end of the body so the output stays byte-identical (nothing to
        # cache).
        stable_prefix = system_prompt.strip() if system_prompt else ""
        if stable_prefix:
            system = f"{stable_prefix}\n\n{_body}{_schema_block}"
        else:
            system = f"{_body}\n\n{system_prompt}{_schema_block}"

        sys_msg: dict[str, Any] = {"role": "system", "content": system}
        if stable_prefix and system.startswith(stable_prefix):
            sys_msg[CACHE_PREFIX_KEY] = stable_prefix
        messages: list[dict[str, Any]] = [sys_msg]
        messages.extend(history)
        return messages

    def parse_response(self, node_id: str, raw: str) -> CriteriaResult:
        """Parse LLM response into CriteriaResult.

        Raises on parse failure so the caller can retry.
        """
        text = raw.strip()
        if not text:
            raise json.JSONDecodeError("Empty response from LLM", text, 0)

        # Try markdown code block extraction
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if md_match:
            text = md_match.group(1).strip()

        # Try direct JSON parse first
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: find first { ... } block in the response
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                data = json.loads(brace_match.group(0))
            else:
                raise

        # Coerce criteria_met values to bool — LLMs sometimes
        # return strings ("internal") or ints instead of true/false.
        # Guard against bare bool (e.g. {"criteria_met": true}) from
        # router nodes that have no named criteria.
        raw_criteria = data.get("criteria_met", {})
        if not isinstance(raw_criteria, dict):
            raw_criteria = {}
        criteria_met: dict[str, bool] = {}
        for k, v in raw_criteria.items():
            if isinstance(v, bool):
                criteria_met[k] = v
            elif isinstance(v, str):
                criteria_met[k] = v.lower() in ("true", "yes", "1")
            else:
                criteria_met[k] = bool(v)

        return CriteriaResult(
            node_id=node_id,
            criteria_met=criteria_met,
            all_required_met=data.get("all_required_met", False),
            user_insisting=data.get("user_insisting", False),
            recommended_edge_id=data.get("recommended_edge_id"),
            reason=data.get("reason", ""),
            response=data.get("response"),
            extracted_slots=data.get("extracted_slots", {}),
        )

    async def evaluate(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        system_prompt: str = "",
        visit_count: int = 1,
        turns_in_node: int = 0,
        agent_language: str = "",
        agent_gender: str = "",
        node_slots: dict[str, Any] | None = None,
        previously_completed: bool = False,
    ) -> CriteriaResult:
        """Full evaluation with retry on parse failure."""
        if self._llm is None and self._llm_fn is None:
            return CriteriaResult(
                node_id=node.id,
                reason="No LLM function configured",
            )
        messages = self.build_evaluation_messages(
            node,
            history,
            userdata,
            system_prompt,
            visit_count,
            turns_in_node,
            agent_language=agent_language,
            agent_gender=agent_gender,
            node_slots=node_slots,
            previously_completed=previously_completed,
        )

        # First attempt
        try:
            raw = await self._ask(messages)
            return self.parse_response(node.id, raw)
        except (json.JSONDecodeError, TypeError, KeyError) as parse_exc:
            logger.warning("Criteria parse failed, retrying: %s", parse_exc)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise

        # Retry with stricter prompt
        retry_messages = list(messages)
        retry_messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Respond ONLY with a valid JSON object, "
                    "no markdown, no explanation, just the JSON."
                ),
            }
        )
        try:
            raw = await self._ask(retry_messages)
            return self.parse_response(node.id, raw)
        except Exception as exc:
            logger.error("LLM retry also failed: %s", exc)
            raise
