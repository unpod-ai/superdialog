"""DialogStateMachine -- runtime-agnostic conversation state machine."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import traceback
from collections.abc import Callable
from typing import Any

from transitions.extensions.asyncio import AsyncMachine

from superdialog.chat_context import ChatContext, ChatMessage
from superdialog.flow.models import (
    ActionTriggerType,
    ConversationFlow,
    CustomAction,
    Edge,
    FlowNode,
)
from superdialog.flow_state import FlowState
from superdialog.machine._lang_util import get_language_name, save_failed_execution_log
from superdialog.machine.actions import ActionExecutor
from superdialog.machine.extractor import VariableExtractor
from superdialog.machine.gate import TransitionGate
from superdialog.machine.gate import classify_node_type as _classify_node_type
from superdialog.machine.gate import is_auto_proceed as _is_auto_proceed
from superdialog.machine.hooks import MachineHooks
from superdialog.machine.models import (
    CriteriaResult,
    FlowContext,
    IntentFrame,
    NodeScope,
    ToolDefinition,
    ToolDescriptor,
    ToolResult,
    TransitionRecord,
    TransitionResult,
    TurnResult,
)
from superdialog.machine.tools import ToolRegistry
from superdialog.tools.base import Tool as ToolABC

logger = logging.getLogger(__name__)

# Regex to find Jinja2 template variables: {{ var_name }}
_TEMPLATE_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# System/time variables that should NOT be auto-captured from user input
_SYSTEM_VARS = frozenset(
    {
        "name",
        "current_time",
        "current_date",
        "current_time_Asia_Kolkata",
        "env",
        "userdata",
        "actions",
    }
)


def _is_api_result(value: Any) -> bool:
    """Return True if value looks like a stored API response."""
    return (
        isinstance(value, dict)
        and "data" in value
        and ("success" in value or "status" in value)
    )


def _build_api_dependency_map(
    action_map: dict[str, Any],
    env_var_names: frozenset[str] | None = None,
) -> dict[str, frozenset[str]]:
    """Build {store_response_as → frozenset of template vars it depends on}.

    Scans each action's URL and body for {{var}} references so stale
    invalidation can be surgical: only clear results whose inputs changed.
    env_var_names: set of flow-level environment variable keys (e.g.
    API_BASE_URL, ACCESS_TOKEN) to exclude — these never change mid-call.
    """
    _exclude = env_var_names or frozenset()
    _VAR_RE = re.compile(r"\{\{\s*(\w+)")
    dep_map: dict[str, frozenset[str]] = {}
    for action in action_map.values():
        store_key = getattr(action, "store_response_as", None)
        if not store_key:
            continue
        url = getattr(action, "url", "") or ""
        body = str(getattr(action, "body", "") or "")
        deps = frozenset(_VAR_RE.findall(url + " " + body)) - _exclude
        dep_map[store_key] = deps
    return dep_map


def _collect_stale_api_keys(
    userdata: dict[str, Any],
    collected_data: dict[str, Any],
    api_dep_map: dict[str, frozenset[str]] | None = None,
) -> list[str]:
    """Return API-result keys whose inputs actually changed.

    Surgical: only clears results that depend on the fields that changed.
    - api_dep_map maps store_response_as → set of template vars used in the
      action URL/body.  A result is stale only when a changed field appears
      in its dep set.
    - When api_dep_map is None (no action context available), falls back to
      clearing ALL API results (old behaviour — conservative).
    """
    changed_fields: set[str] = set()
    for key, new_value in collected_data.items():
        if new_value in (None, ""):
            continue
        old_value = userdata.get(key)
        if old_value not in (None, "") and old_value != new_value:
            changed_fields.add(key)

    if not changed_fields:
        return []

    if api_dep_map is None:
        return [k for k, v in userdata.items() if _is_api_result(v)]

    stale: list[str] = []
    for k, v in userdata.items():
        if not _is_api_result(v):
            continue
        deps = api_dep_map.get(k)
        if deps is not None and deps & changed_fields:
            stale.append(k)
    return stale


def _infer_input_schema(
    edge: Edge,
    target_node: FlowNode | None,
) -> dict[str, Any] | None:
    """Auto-detect input_schema from target node's instruction.

    Scans the target node's instruction for ``{{variable}}`` patterns
    and builds an input_schema so the LLM can extract data via tool params.

    Returns None if no capturable variables are found.
    """
    if edge.input_schema is not None:
        return None  # Already has explicit schema

    if target_node is None or not target_node.instruction:
        return None

    # Find all template variables in target node instruction
    vars_in_instruction = set(_TEMPLATE_VAR_RE.findall(target_node.instruction))

    # Also scan the edge condition for template variables
    if edge.condition:
        vars_in_instruction.update(_TEMPLATE_VAR_RE.findall(edge.condition))

    # Filter out system/time variables — capture everything else
    capturable = []
    for var in vars_in_instruction:
        if var in _SYSTEM_VARS:
            continue
        capturable.append(var)

    if not capturable:
        return None

    # Build input_schema
    properties: dict[str, Any] = {}
    for var in sorted(capturable):
        properties[var] = {
            "type": "string",
            "description": f"Value of '{var}' extracted from caller's response",
        }

    schema = {
        "type": "object",
        "properties": properties,
        "required": sorted(capturable),
    }
    logger.info(
        "[FLOW] auto-inferred input_schema for edge=%s vars=%s",
        edge.id,
        sorted(capturable),
    )
    return schema


class DialogStateMachine:
    """Runtime-agnostic conversation state machine.

    Supports two execution models:

    1. **Criteria-based** (text/batch): ``process_turn(user_input)``
       — CriteriaJudge evaluates and picks the edge.
    2. **Tool-call-based** (LiveKit voice): ``apply_transition(edge_id)``
       — external caller (LLM tool callback) tells which edge fired.

    Both share the same state management, action execution, and persistence.
    """

    # pytransitions uses `state` attribute on the model
    state: str = ""

    def __init__(
        self,
        flow: ConversationFlow,
        adapter: Any,
        machine: AsyncMachine,
        session_id: str | None = None,
        store: Any | None = None,
        tool_handlers: dict[str, Any] | None = None,
        hooks: MachineHooks | None = None,
        tool_registry: ToolRegistry | None = None,
        extractor: VariableExtractor | None = None,
        tools: list[ToolABC | Callable[..., Any]] | None = None,
        handler_registry: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self._flow = flow
        self._adapter = adapter
        self._machine = machine
        self._node_map: dict[str, FlowNode] = {n.id: n for n in flow.nodes}
        self._final_states: set[str] = {n.id for n in flow.nodes if n.is_final}
        self._action_map: dict[str, CustomAction] = {a.id: a for a in flow.actions}
        self._api_dep_map: dict[str, frozenset[str]] = _build_api_dependency_map(
            self._action_map,
            frozenset(flow.environment_variables.keys()),
        )
        self._global_edge_ids: set[str] = {e.id for e in flow.global_edges}
        self._global_edge_map: dict[str, Edge] = {e.id: e for e in flow.global_edges}
        self._session_id = session_id
        self._store = store
        self._gate = TransitionGate()
        self._action_executor = ActionExecutor(adapter, self._action_map)
        self._hooks = hooks or MachineHooks()
        self._tool_registry = tool_registry or ToolRegistry()
        self._extractor = extractor
        self._tool_def_map: dict[str, ToolDefinition] = {}
        for td in flow.tools:
            self._tool_def_map[td.id] = td
        for node in flow.nodes:
            for td in node.tools:
                self._tool_def_map[td.id] = td

        # Build the unified tools_by_id map (Task 4.8).
        # Sources (later overrides earlier):
        #   1. flow.tools + node.tools deserialized via Tool.from_dict
        #   2. ToolDefinition.handler — callable set when building flow in Python
        #   3. legacy tool_handlers dict wrapped as PythonTool (back-compat)
        #   4. explicit tools=[...] kwarg (Tool instances or plain callables)
        self._handler_registry: dict[str, Callable[..., Any]] = handler_registry or {}
        self._tools_by_id: dict[str, ToolABC] = {}
        for td in self._tool_def_map.values():
            # Inline callable reference on the ToolDefinition — highest priority
            # for python tools; overwrites anything built from spec/registry.
            if getattr(td, "handler", None) is not None:
                from superdialog.tools.python_tool import PythonTool as _PT

                self._tools_by_id[td.id] = _PT.of(td.handler)
                continue
            spec = self._tool_def_to_spec(td)
            # For python tools, only auto-build when we have a handler for them.
            # An unbound PythonTool would silently swallow the lookup and
            # produce a ToolResult(error=...) at execute time — legacy callers
            # expect a ValueError, so leave the entry out and let the legacy
            # tool_handlers path handle the error case.
            if td.type == "python":
                if td.id not in self._handler_registry and (
                    td.handler_id is None or td.handler_id not in self._handler_registry
                ):
                    continue
            try:
                self._tools_by_id[td.id] = ToolABC.from_dict(
                    spec, handler_registry=self._handler_registry
                )
            except (KeyError, ValueError):
                # Spec was incomplete (e.g. http tool without url declared in
                # flow JSON); fall through to legacy handler lookup at execute
                # time.
                pass
        # Legacy ``tool_handlers`` (signature ``handler(tool_id, args)``) are
        # NOT wrapped as PythonTool — their calling convention doesn't match
        # ``PythonTool.execute(args)``. They stay on the legacy path in
        # ``execute_tool`` and continue to raise ValueError when unbound.
        for tool in tools or []:
            if callable(tool) and not isinstance(tool, ToolABC):
                from superdialog.tools.python_tool import PythonTool as _PT

                tool = _PT.of(tool)
            self._tools_by_id[tool.id] = tool

        self._tool_handlers: dict[str, Any] = tool_handlers or {}
        self._language_callbacks: list[Callable[[str], None]] = []
        self.context = FlowContext(
            current_node_id=flow.initial_node,
            visit_count={flow.initial_node: 1},
            agent_language=flow.agent_language,
            agent_gender=flow.agent_gender,
        )

    @classmethod
    async def from_flow(
        cls,
        flow: ConversationFlow,
        adapter: Any = None,
        session_id: str | None = None,
        store: Any | None = None,
        tool_handlers: dict[str, Any] | None = None,
        hooks: MachineHooks | None = None,
        tool_registry: ToolRegistry | None = None,
        extractor: VariableExtractor | None = None,
        initial_userdata: dict[str, Any] | None = None,
        tools: list[ToolABC | Callable[..., Any]] | None = None,
        handler_registry: dict[str, Callable[..., Any]] | None = None,
        llm: Any = None,
    ) -> DialogStateMachine:
        """Factory: build machine from ConversationFlow + adapter.

        If ``adapter`` is omitted, ``llm`` must be provided -- a default
        :class:`~superdialog.machine.adapters.LLMAdapter` is constructed
        from it. This is the path the public :class:`DialogMachine`
        facade uses.
        """
        if adapter is None:
            if llm is None:
                raise ValueError(
                    "DialogStateMachine.from_flow requires either an "
                    "`adapter` or an `llm` provider"
                )
            from superdialog.machine.adapters.llm_adapter import LLMAdapter

            adapter = LLMAdapter(
                provider=llm,
                system_prompt=getattr(flow, "system_prompt", "") or "",
            )
        logger.info(
            "[FLOW] from_flow session=%s flow_nodes=%d global_edges=%d initial=%s",
            session_id,
            len(flow.nodes),
            len(flow.global_edges),
            flow.initial_node,
        )
        instance = cls.__new__(cls)

        # Build states
        states: list[dict[str, Any]] = [{"name": node.id} for node in flow.nodes]

        # Build transitions from node edges
        transitions: list[dict[str, Any]] = []
        for node in flow.nodes:
            for edge in node.edges:
                if edge.target_node_id:
                    transitions.append(
                        {
                            "trigger": edge.id,
                            "source": node.id,
                            "dest": edge.target_node_id,
                        }
                    )

        # Register global edges from every interruptible non-final node
        for gedge in flow.global_edges:
            if not gedge.target_node_id:
                continue
            for node in flow.nodes:
                if node.is_final or not node.interruptible:
                    continue
                # Skip if this node already has a local edge with same ID
                local_ids = {e.id for e in node.edges}
                if gedge.id in local_ids:
                    continue
                transitions.append(
                    {
                        "trigger": gedge.id,
                        "source": node.id,
                        "dest": gedge.target_node_id,
                    }
                )

        # Build AsyncMachine with instance as model
        machine = AsyncMachine(
            model=instance,
            states=states,
            transitions=transitions,
            initial=flow.initial_node,
            auto_transitions=False,
            queued=True,
        )

        instance.__init__(
            flow,
            adapter,
            machine,
            session_id,
            store,
            tool_handlers,
            hooks,
            tool_registry,
            extractor,
            tools=tools,
            handler_registry=handler_registry,
        )  # type: ignore[misc]

        # Restore context from store if available
        if session_id and store:
            saved = await store.load(session_id)
            if saved is not None:
                instance.context = saved
                # Sync pytransitions state with restored context
                if saved.current_node_id:
                    instance.state = saved.current_node_id
                logger.info(
                    "[traverse] restored session=%s node=%s visits=%d",
                    session_id,
                    saved.current_node_id,
                    len(saved.transition_log),
                )

        logger.info(
            "[traverse] init node=%s nodes=%d edges=%d global_edges=%d",
            flow.initial_node,
            len(flow.nodes),
            sum(len(n.edges) for n in flow.nodes),
            len(flow.global_edges),
        )

        # Inject initial userdata before ON_ENTER actions fire
        if initial_userdata:
            instance.context.userdata.update(initial_userdata)

        # Fire ON_ENTER actions for the initial node (no transition fires them)
        # Only if initial_userdata was provided (caller is ready for actions).
        # Production handlers set userdata after from_flow() and call
        # fire_initial_on_enter() explicitly.
        if initial_userdata:
            await instance.fire_initial_on_enter()

        return instance

    # ------------------------------------------------------------------
    async def fire_initial_on_enter(self) -> list[str]:
        """Fire ON_ENTER actions for the current (initial) node.

        Call this after setting userdata on the context so templates
        like ``{{phone}}`` resolve. Idempotent — tracks whether it
        already ran via ``_initial_on_enter_fired``.
        """
        if getattr(self, "_initial_on_enter_fired", False):
            return []
        self._initial_on_enter_fired = True

        node = self.current_node
        if not node or not node.actions:
            return []
        logger.info(
            "[FLOW] executing ON_ENTER actions for initial node=%s, action_count=%d",
            node.id,
            len(node.actions),
        )
        fired = await self._execute_actions(node.actions, ActionTriggerType.ON_ENTER)
        if fired:
            logger.info("[FLOW] initial ON_ENTER actions fired: %s", fired)
        return fired

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> str:
        """Current state ID (from pytransitions ``state`` attribute)."""
        return self.state

    @property
    def current_node(self) -> FlowNode:
        """Current FlowNode object."""
        return self._node_map[self.state]

    @property
    def is_complete(self) -> bool:
        """True if at a final node."""
        return self.state in self._final_states

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def has_trigger(self, trigger_name: str) -> bool:
        """Check if a trigger (edge ID) is registered in any state."""
        for node in self._flow.nodes:
            triggers = self._machine.get_triggers(node.id)
            if trigger_name in triggers:
                return True
        return False

    def on_language_change(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked when the agent language changes."""
        self._language_callbacks.append(callback)

    def set_language(self, lang: str) -> None:
        """Update agent language dynamically (e.g., after detecting user language).

        Fires all registered ``on_language_change`` callbacks after updating
        the context so downstream consumers (STT/TTS providers, conversation
        state) can react to the switch.
        """
        prev = self.context.agent_language
        self.context.agent_language = lang
        logger.info("[traverse] language updated to %s", lang)
        if lang != prev:
            for cb in self._language_callbacks:
                try:
                    cb(lang)
                except Exception:
                    logger.warning(
                        "[traverse] language callback failed",
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Session-layer integration: ChatContext + FlowState view/load methods
    # ------------------------------------------------------------------

    @property
    def chat_ctx(self) -> ChatContext:
        """Return current conversation history as a LiveKit-aligned ChatContext."""
        return ChatContext(
            items=[
                ChatMessage(role=m.get("role", ""), content=m.get("content", ""))
                for m in self.context.data.history
                if isinstance(m, dict)
            ]
        )

    def load_chat_ctx(self, ctx: ChatContext) -> None:
        """Replace conversation history from a ChatContext."""
        self.context.data.history = [
            {"role": m.role, "content": m.content} for m in ctx.items
        ]

    @property
    def flow_state(self) -> FlowState:
        """Return DM-specific runtime state as a FlowState snapshot."""
        return FlowState.from_flow_context(self.context)

    def load_flow_state(self, state: FlowState) -> None:
        """Apply a FlowState snapshot to this DM's context."""
        state.apply_to(self.context)
        # Keep pytransitions in sync with the restored node.
        if state.current_node_id:
            self.state = state.current_node_id

    def assist(self, text: str) -> None:
        """Push a system-level instruction; takes effect on the next turn."""
        if not text:
            return
        self.context.add_message("system", text)

    def inject_system(self, text: str) -> None:
        """Deprecated alias for :meth:`assist`."""
        import warnings

        warnings.warn(
            "DialogMachine.inject_system is deprecated; use .assist instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.assist(text)

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def push_language_trace(self, lang: str) -> None:
        """Store the latest detected language (single entry).

        Called on every STT turn regardless of whether it triggers a
        language switch. Stored on ``context.data.language_trace`` for
        end-to-end debugging visibility.
        """
        trace = self.context.data.language_trace
        trace.clear()
        trace.append(lang)
        logger.debug("[traverse] language_trace=%s", trace)

    def is_global_edge(self, edge_id: str) -> bool:
        """Check if an edge ID is a global edge."""
        return edge_id in self._global_edge_ids

    def get_tools_for_node(self, node: FlowNode | None = None) -> list[ToolDescriptor]:
        """Return provider-agnostic tool descriptors for a node's edges.

        Includes global edges if the node is interruptible and not final.
        """
        node = node or self.current_node
        descriptors: list[ToolDescriptor] = []
        logger.debug(
            "[FLOW] get_tools_for_node node=%s interruptible=%s is_final=%s",
            node.id,
            node.interruptible,
            node.is_final,
        )

        # Node's own edges
        from superdialog.machine.composer import render_template as _render_tpl

        for edge in node.edges:
            # Use explicit schema, or auto-infer from target node
            schema = edge.input_schema if isinstance(edge.input_schema, dict) else None
            if schema is None and edge.target_node_id:
                target = self._node_map.get(edge.target_node_id)
                schema = _infer_input_schema(edge, target)

            descriptors.append(
                ToolDescriptor(
                    id=edge.id,
                    description=(
                        _render_tpl(edge.condition, self) if edge.condition else edge.id
                    ),
                    is_data_collection=schema is not None,
                    is_global=False,
                    input_schema=schema,
                    target_node_id=edge.target_node_id,
                )
            )

        # Global edges (if interruptible and not final).
        # Router nodes have self-contained condition-based routing — presenting
        # global edges alongside their own edges causes the LLM to pick the wrong
        # global edge (e.g. global_token_expired) when it should follow the explicit
        # node routing logic.
        if (
            node.interruptible
            and not node.is_final
            and _classify_node_type(node) != "router"
        ):
            local_ids = {e.id for e in node.edges}
            for gedge in self._flow.global_edges:
                if gedge.id not in local_ids:
                    schema = (
                        gedge.input_schema
                        if isinstance(gedge.input_schema, dict)
                        else None
                    )
                    if schema is None and gedge.target_node_id:
                        target = self._node_map.get(gedge.target_node_id)
                        schema = _infer_input_schema(gedge, target)

                    descriptors.append(
                        ToolDescriptor(
                            id=gedge.id,
                            description=gedge.condition,
                            is_data_collection=schema is not None,
                            is_global=True,
                            input_schema=schema,
                            target_node_id=gedge.target_node_id,
                        )
                    )

        # Filter out self-targeting edges once the user has responded.
        # Interrupt edges that loop back to the same node are only useful
        # during TTS playback (to handle user barging in). After the user
        # has spoken, keeping them as tools causes the LLM to misuse them
        # as a "re-ask" mechanism, replaying static text endlessly.
        if self.context.user_spoke_in_node and self.context.node_spoken:
            descriptors = [d for d in descriptors if d.target_node_id != node.id]

        # Node-scoped custom tools
        for tool_def in node.tools:
            descriptors.append(
                ToolDescriptor(
                    id=tool_def.id,
                    description=tool_def.description,
                    is_custom=True,
                    is_data_collection=tool_def.input_schema is not None,
                    input_schema=tool_def.input_schema,
                    handler_id=tool_def.handler_id,
                )
            )

        # Flow-scoped custom tools (always available)
        node_tool_ids = {t.id for t in node.tools}
        for tool_def in self._flow.tools:
            if tool_def.id not in node_tool_ids:
                descriptors.append(
                    ToolDescriptor(
                        id=tool_def.id,
                        description=tool_def.description,
                        is_custom=True,
                        is_data_collection=tool_def.input_schema is not None,
                        input_schema=tool_def.input_schema,
                        handler_id=tool_def.handler_id,
                    )
                )

        tool_ids = [d.id for d in descriptors]
        logger.info(
            "[FLOW] tools_for_node node=%s tools=%s",
            node.id,
            tool_ids,
        )
        return descriptors

    @staticmethod
    def classify_node_type(node: FlowNode) -> str:
        """Classify a flow node for flow control instructions.

        Returns one of: 'final', 'static', 'instruction', 'router'.
        """
        return _classify_node_type(node)

    @staticmethod
    def _build_flow_control_block(node: FlowNode, edge_list: str) -> str:
        """Build per-node-type LLM instructions for flow control.

        Tells the LLM HOW to behave with tools for this node type:
        when to speak, when to wait, when to call tools.
        """
        node_type = DialogStateMachine.classify_node_type(node)
        auto_proceed = _is_auto_proceed(node)

        if node_type == "final":
            return (
                "## CONVERSATION FLOW INSTRUCTIONS\n"
                "This is the final step. Deliver your closing "
                "message and end the call."
            )

        if node_type == "static":
            if not node.edges:
                return (
                    "## CONVERSATION FLOW INSTRUCTIONS\n"
                    "The scripted message has already been spoken "
                    "via text-to-speech. Do NOT repeat it or "
                    "generate any additional speech."
                )
            return (
                "## CONVERSATION FLOW INSTRUCTIONS\n"
                "The scripted message has already been spoken "
                "via text-to-speech. Do NOT repeat it.\n"
                "Your ONLY job is to LISTEN for the caller's "
                "response and immediately call the best-matching "
                "transition tool. Do NOT generate speech — just "
                "route.\n"
                "Available transitions:\n" + edge_list
            )

        if node_type == "router":
            return (
                "## CONVERSATION FLOW INSTRUCTIONS\n"
                "Classify the user's intent and immediately call "
                "the best-matching transition tool. Do not generate "
                "a spoken response — just route.\n"
                "Available transitions:\n" + edge_list
            )

        # node_type == "instruction"
        # Check whether this node will use TTS-only (language markers
        # present) or generate_reply (no markers → LLM speaks).
        has_lang_markers = bool(
            node.instruction
            and re.search(r"^\[[A-Z]{2}\]\s", node.instruction, re.MULTILINE)
        )

        if has_lang_markers:
            # Speech is handled by session.say() — already spoken
            # before the LLM runs. LLM should only route.
            if auto_proceed:
                return (
                    "## CONVERSATION FLOW INSTRUCTIONS\n"
                    "The scripted message has already been spoken via "
                    "text-to-speech. Do NOT repeat it.\n"
                    "Immediately call the best-matching transition tool "
                    "— do NOT wait for the caller to respond.\n"
                    "Available transitions:\n" + edge_list
                )
            return (
                "## CONVERSATION FLOW INSTRUCTIONS\n"
                "You are following a conversation script. The scripted "
                "message for this step has already been spoken via "
                "text-to-speech. Do NOT repeat it or paraphrase it.\n"
                "Rules:\n"
                "1. WAIT for the caller to respond. Do NOT call any "
                "transition tool until the caller has spoken.\n"
                "2. After the caller responds, check which transition "
                "condition best matches and call that transition tool.\n"
                "3. If no tool perfectly matches, pick the closest one.\n"
                "Available transitions:\n" + edge_list
            )

        # No language markers — LLM generates speech via generate_reply().
        if auto_proceed:
            # Speak AND transition in the same response — no user input needed.
            return (
                "## CONVERSATION FLOW INSTRUCTIONS\n"
                "You are following a conversation script.\n"
                "Rules:\n"
                "1. FIRST, deliver the message described in your "
                "instruction. Speak naturally to the caller.\n"
                "2. In the SAME response, immediately call the "
                "best-matching transition tool. Do NOT wait for the "
                "caller to respond.\n"
                "Available transitions:\n" + edge_list
            )

        # Normal instruction node — speak first, wait for caller, then route.
        return (
            "## CONVERSATION FLOW INSTRUCTIONS\n"
            "You are following a conversation script. "
            "Follow these two phases exactly:\n"
            "PHASE 1 — Your first response (before the caller replies): "
            "Deliver the message in your instruction. Speak naturally. "
            "Do NOT call any transition tool in your first response.\n"
            "PHASE 2 — Your next response (after the caller has replied): "
            "STOP speaking. Call the single best-matching transition tool "
            "IMMEDIATELY. Do NOT generate any further speech — call the "
            "transition tool now.\n"
            "Available transitions:\n" + edge_list
        )

    def get_enriched_instructions(self, node: FlowNode | None = None) -> str:
        """Return node instruction enriched with state context."""
        from datetime import datetime, timedelta, timezone

        node = node or self.current_node
        parts: list[str] = []

        # Inject current date so the LLM knows the year — without this,
        # when the caller says "7th April" the LLM defaults to its
        # training-data year (2024) instead of the actual current year.
        _ist = timezone(timedelta(hours=5, minutes=30))
        parts.append(f"[Today: {datetime.now(_ist).strftime('%A, %d %B %Y')}]")

        # Mid-conversation context prefix so LLM doesn't restart
        if self.context.completed_nodes:
            completed = ", ".join(self.context.completed_nodes)
            parts.append(
                "[You are mid-conversation. Steps already completed: "
                f"{completed}. Continue naturally from where the "
                "conversation left off — do NOT re-greet or restart.]"
            )

        if node.instruction:
            parts.append(node.instruction)
        elif node.static_text:
            parts.append(node.static_text)

        # Build edge list and flow control instructions
        edge_list = ""
        if node.edges and not node.is_final:
            from superdialog.machine.composer import render_template as _render_tpl

            edge_lines: list[str] = []
            for e in node.edges:
                cond = _render_tpl(e.condition, self) if e.condition else ""
                line = f'  - "{e.id}": {cond}'
                # Include required data extraction hints from input_schema
                schema = e.input_schema if isinstance(e.input_schema, dict) else None
                if schema:
                    props = schema.get("properties", {})
                    required = schema.get("required", [])
                    if props:
                        fields = []
                        for k, v in props.items():
                            desc = (
                                v.get("description", "") if isinstance(v, dict) else ""
                            )
                            marker = " [REQUIRED]" if k in required else ""
                            fields.append(
                                f"{k}{marker}: {desc}" if desc else f"{k}{marker}"
                            )
                        line += "\n      EXTRACT: " + ", ".join(fields)
                edge_lines.append(line)
            edge_list = "\n".join(edge_lines)

        # Flow control block: tells the LLM WHEN to speak vs call tools
        parts.append(self._build_flow_control_block(node, edge_list))

        # Collected slots for this node
        node_slots = self.context.node_slots.get(node.id)
        if node_slots:
            parts.append(f"Collected data for this step: {node_slots}")

        vc = self.context.visit_count.get(node.id, 1)
        if vc > 1:
            parts.append(f"This step has been visited {vc} times.")

        if node.id in self.context.completed_nodes:
            # Replay previously collected slots so the LLM knows
            # what was already gathered and can handle corrections.
            prev_slots = self.context.node_slots.get(node.id)
            if prev_slots:
                parts.append(
                    "This step was previously completed. "
                    f"Previously collected: {prev_slots}. "
                    "The user may be correcting — ask only about "
                    "the specific detail they want to change."
                )
            else:
                parts.append(
                    "This step was previously completed. "
                    "Do not re-ask for information already collected."
                )

        if self.context.turns_in_node > 0:
            parts.append(f"Turns in this step so far: {self.context.turns_in_node}")

        if self.context.agent_language:
            language = get_language_name(self.context.agent_language)
            if self.context.agent_language != "en":
                parts.append(f"IMPORTANT: Speak in mix of English and {language}")
            else:
                parts.append(f"IMPORTANT: Speak in {language}")
        if self.context.agent_gender:
            parts.append(f"Agent's gender is: {self.context.agent_gender}")

        return "\n".join(parts)

    def is_node_completed(self, node_id: str) -> bool:
        """Return True if this node has been successfully transitioned out of."""
        return node_id in self.context.completed_nodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_target_to_edge(self, target_node_id: str) -> str | None:
        """If an LLM returns a target_node_id instead of an edge_id,
        find the edge on the current node that points to that target."""
        for edge in self.current_node.edges:
            if edge.target_node_id == target_node_id:
                return edge.id
        # Check global edges
        for gedge in self._flow.global_edges:
            if gedge.target_node_id == target_node_id:
                triggers = self._machine.get_triggers(self.state)
                if gedge.id in triggers:
                    return gedge.id
        return None

    def _get_edge(self, edge_id: str) -> Edge | None:
        """Find the Edge object by ID from the current node or global edges."""
        for edge in self.current_node.edges:
            if edge.id == edge_id:
                return edge
        return self._global_edge_map.get(edge_id)

    def _get_fallback_edge(self) -> Edge | None:
        """Find the first fallback edge from the current node."""
        for edge in self.current_node.edges:
            if edge.is_fallback and edge.target_node_id:
                return edge
        return None

    async def fire_initial_on_enter_actions(self) -> list[str]:
        """Fire ON_ENTER actions for the initial node.

        Called by SimpleFlowAgent.on_enter() for the initial node only,
        since _do_transition (which normally fires on_enter actions) is
        never called for the starting node.

        Delegates to fire_initial_on_enter() so the idempotency guard
        (_initial_on_enter_fired) prevents double-firing when from_flow()
        already ran the actions during initialization.
        """
        return await self.fire_initial_on_enter()

    async def _execute_actions(
        self,
        actions: list[Any],
        trigger_type: ActionTriggerType | None = None,
    ) -> list[str]:
        """Execute actions, return list of action IDs that fired."""
        return await self._action_executor.execute(actions, self.context, trigger_type)

    async def _generate_node_response(self, node: FlowNode) -> str:
        """Generate response for a node (on_enter behavior)."""
        if _classify_node_type(node) == "router":
            return ""
        if node.static_text:
            text = node.static_text
            if "{{" in text and hasattr(self._adapter, "_render"):
                userdata = dict(self.context.userdata)
                ctx = self._adapter._build_context(userdata)
                text = self._adapter._render(text, ctx)
            await self._adapter.speak(text, node)
            return text
        if node.instruction:
            history = list(self.context.data.history)
            reply = await self._adapter.generate_reply(
                node.instruction,
                node,
                history=history,
                userdata=dict(self.context.userdata),
            )
            return reply
        return ""

    def _should_persist_response_to_history(self) -> bool:
        """Whether adapter-generated response text belongs in history."""
        return not bool(getattr(self._adapter, "speech_passthrough", False))

    def _schedule_save(self) -> None:
        """Fire-and-forget context save (non-blocking)."""
        if not self._store or not self._session_id:
            return
        snapshot = self.context.model_copy(deep=True)
        sid = self._session_id

        async def _do_save() -> None:
            try:
                await self._store.save(sid, snapshot)
            except Exception as exc:
                logger.warning("Context save failed for %s: %s", sid, exc)
                await save_failed_execution_log(
                    task_id=(
                        (snapshot.userdata or {}).get("task_id") if snapshot else None
                    ),
                    step="flow_context_save_failed",
                    location="DialogStateMachine._schedule_save",
                    error_message=f"{type(exc).__name__}: {exc}".strip(),
                    data={"session_id": sid},
                    traceback_str=traceback.format_exc(),
                )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_save())
        except RuntimeError:
            pass

    def _push_intent_stack(self) -> None:
        """Push current node state to intent stack before an interrupt."""
        frame = IntentFrame(
            node_id=self.state,
            slots=dict(self.context.node_slots.get(self.state, {})),
            turns_spent=self.context.turns_in_node,
            suspended_at=time.time(),
        )
        self.context.intent_stack.append(frame)
        logger.info(
            "[traverse] intent_stack PUSH node=%s depth=%d",
            self.state,
            len(self.context.intent_stack),
        )

    async def _maybe_auto_return(self) -> TurnResult | None:
        """If intent stack is non-empty and current node is a return point,
        pop stack and restore to interrupted node.

        A node is a return point if it has no non-fallback outgoing edges
        (dead-end) or is_final.
        """
        if not self.context.intent_stack:
            return None

        node = self.current_node
        has_outgoing = any(e.target_node_id and not e.is_fallback for e in node.edges)
        if has_outgoing and not node.is_final:
            return None

        frame = self.context.intent_stack.pop()
        logger.info(
            "[traverse] intent_stack POP returning to node=%s depth=%d",
            frame.node_id,
            len(self.context.intent_stack),
        )

        # Restore to interrupted node
        self.state = frame.node_id
        self.context.current_node_id = frame.node_id
        self.context.node_slots.setdefault(frame.node_id, {}).update(frame.slots)
        self.context.turns_in_node = frame.turns_spent

        # Generate response for the restored node
        restored_node = self.current_node
        response = await self._generate_node_response(restored_node)
        if response and self._should_persist_response_to_history():
            self.context.add_assistant_message(response)

        self._schedule_save()
        return TurnResult(
            outcome="transition",
            from_node=self.state,
            to_node=self.state,
            response=response,
        )

    # ------------------------------------------------------------------
    # Entry point 1: criteria-based (text / batch)
    # ------------------------------------------------------------------

    async def process_turn(self, user_input: str) -> TurnResult:
        """Process one user turn — always returns a TurnResult with response."""
        current = self.current_node
        from_node = self.state

        # Early return for final nodes
        if self.is_complete:
            logger.debug(
                "[traverse] turn skipped — already at final node=%s",
                from_node,
            )
            return TurnResult(
                outcome="stay",
                from_node=from_node,
                to_node=from_node,
                response="",
            )

        # 0. Preprocess input hook
        user_input = self._hooks.apply_input(user_input, self.context)

        # 1. Record user input + increment turn counter
        self.context.add_user_message(user_input)
        self.context.turns_in_node += 1
        logger.info(
            "[traverse] turn node=%s turn_num=%d visit=%d input=%s",
            from_node,
            self.context.turns_in_node,
            self.context.visit_count.get(from_node, 1),
            user_input[:80],
        )

        # 2. Check max_turns → fallback edge
        if (
            current.max_turns is not None
            and self.context.turns_in_node > current.max_turns
        ):
            fallback = self._get_fallback_edge()
            if fallback:
                logger.info(
                    "[traverse] max_turns=%d exceeded — fallback edge=%s",
                    current.max_turns,
                    fallback.id,
                )
                return await self._do_transition(
                    edge_id=fallback.id,
                    edge=fallback,
                    criteria_met={},
                    skipped=True,
                    from_node=from_node,
                    user_message=user_input,
                )

        # 3. Evaluate criteria (with retry on failure)
        #    Inject flow metadata so adapters can forward to CriteriaJudge
        #    Skip if adapter doesn't support criteria (e.g. FlowActionRunner)
        if not getattr(self._adapter, "supports_criteria", True):
            logger.debug(
                "[traverse] skipping criteria eval — adapter does not "
                "support criteria (node=%s)",
                from_node,
            )
            return TurnResult(
                outcome="continue",
                response=(current.instruction or current.static_text or ""),
                from_node=from_node,
            )

        cur_node_slots = self.context.node_slots.get(self.state, {})
        eval_userdata = {
            **self.context.userdata,
            "_flow_meta": {
                "visit_count": self.context.visit_count.get(self.state, 1),
                "turns_in_node": self.context.turns_in_node,
                "agent_language": self.context.agent_language,
                "agent_gender": self.context.agent_gender,
                "node_slots": cur_node_slots,
                "previously_completed": self.state in self.context.completed_nodes,
            },
        }
        try:
            result: CriteriaResult = await self._adapter.evaluate_criteria(
                current,
                self.context.conversation_history,
                eval_userdata,
            )
        except Exception as exc:
            # ERROR path
            error_msg = str(exc)
            logger.error("Criteria evaluation failed: %s", error_msg)
            await save_failed_execution_log(
                task_id=(self.context.userdata or {}).get("task_id"),
                step="flow_criteria_eval_failed",
                location="DialogStateMachine.traverse_turn:evaluate_criteria",
                error_message=f"{type(exc).__name__}: {exc}".strip(),
                data={"node": from_node, "user_input": user_input[:120]},
                traceback_str=traceback.format_exc(),
            )
            try:
                recovery = await self._adapter.generate_recovery(current, error_msg)
            except Exception:
                recovery = (
                    current.instruction
                    or current.static_text
                    or "Could you repeat that?"
                )
            self.context.add_assistant_message(recovery)
            self._schedule_save()
            logger.warning(
                "[traverse] ERROR node=%s error=%s",
                from_node,
                error_msg[:120],
            )
            return TurnResult(
                outcome="error",
                from_node=from_node,
                to_node=from_node,
                response=recovery,
                error=error_msg,
            )

        # 3b. Merge extracted slots into node_slots
        if result.extracted_slots:
            self.context.node_slots.setdefault(self.state, {}).update(
                result.extracted_slots
            )
            self.context.userdata.update(result.extracted_slots)

        # 4. Route: transition or stay?
        if result.recommended_edge_id:
            # If the node has no completion criteria, the LLM's
            # all_required_met value is unreliable — treat it as True
            # since there are no requirements to gate the transition.
            has_criteria = bool(current.completion_criteria)
            all_met = result.all_required_met if has_criteria else True
            can_proceed = all_met or (result.user_insisting and current.allow_skip)

            edge_id = result.recommended_edge_id
            available_triggers = self._machine.get_triggers(self.state)
            edge_valid = edge_id in available_triggers

            # Auto-resolve: LLM sometimes returns target_node_id
            # instead of edge_id. Map it back to the correct edge.
            if not edge_valid:
                resolved = self._resolve_target_to_edge(edge_id)
                if resolved:
                    logger.info(
                        "Resolved target_node_id '%s' -> edge_id '%s'",
                        edge_id,
                        resolved,
                    )
                    edge_id = resolved
                    edge_valid = True

            if can_proceed and edge_valid:
                edge = self._get_edge(edge_id)

                # Push intent stack if this is a global edge
                if self.is_global_edge(edge_id):
                    self._push_intent_stack()

                turn_result = await self._do_transition(
                    edge_id=edge_id,
                    edge=edge,
                    criteria_met=result.criteria_met,
                    skipped=not result.all_required_met,
                    from_node=from_node,
                    user_message=user_input,
                )
                # Auto-chain: router nodes must not block on user input
                return await self._follow_router_chain(user_input, turn_result)

            if not edge_valid:
                logger.warning(
                    "Edge '%s' not valid from state '%s'. Available: %s",
                    edge_id,
                    self.state,
                    available_triggers,
                )

        # STAY path
        logger.info(
            "[traverse] STAY node=%s criteria=%s edge=%s reason=%s",
            from_node,
            result.criteria_met,
            result.recommended_edge_id,
            result.reason[:80] if result.reason else "",
        )
        # Router nodes are always silent — never use their response as speech
        node_type = self.classify_node_type(current)
        response = "" if node_type == "router" else (result.response or "")
        if not response and node_type != "router":
            # Fallback: generate a reply using node instruction
            if current.instruction:
                try:
                    response = await self._adapter.generate_reply(
                        current.instruction,
                        current,
                        history=list(self.context.data.history),
                        userdata=dict(self.context.userdata),
                    )
                except Exception:
                    response = current.instruction
            elif current.static_text:
                response = current.static_text
            else:
                response = "Could you tell me more?"

        if self._should_persist_response_to_history():
            self.context.add_assistant_message(response)
        self._schedule_save()
        return TurnResult(
            outcome="stay",
            from_node=from_node,
            to_node=from_node,
            response=response,
            criteria_snapshot=result.criteria_met,
        )

    # ------------------------------------------------------------------
    # Entry point 2: tool-call-based (LiveKit / external)
    # ------------------------------------------------------------------

    async def apply_transition(
        self,
        edge_id: str,
        user_input: str | None = None,
        collected_data: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Execute a transition triggered by an external caller.

        Use this when the routing decision is made outside the machine
        (e.g., by an LLM tool call in LiveKit).
        """
        print(
            f"[TRACK] apply_transition START - edge_id: {edge_id}, user_input: {user_input}, collected_data: {collected_data}"
        )
        from_node = self.state
        logger.info(
            "[FLOW] apply_transition edge=%s from_node=%s collected_data_keys=%s user_turns=%d",
            edge_id,
            from_node,
            list(collected_data.keys()) if collected_data else [],
            self.context.turns_in_node,
        )

        # Early return for final nodes
        if self.is_complete:
            logger.warning(
                "[FLOW] apply_transition called on final node=%s — ignoring", from_node
            )
            return TurnResult(
                outcome="stay",
                from_node=from_node,
                to_node=from_node,
                response="",
            )

        # Record user input and increment turn counter
        if user_input is not None:
            logger.info(
                "[FLOW] recording user_input for node=%s: %s",
                from_node,
                user_input[:100] + "..." if len(user_input) > 100 else user_input,
            )
            self.context.add_user_message(user_input)
            self.context.turns_in_node += 1

        # Check max_turns → fallback edge (mirrors process_turn behavior)
        current = self.current_node
        if (
            current.max_turns is not None
            and self.context.turns_in_node
            > current.max_turns  # TODO verify if the max_turns is correct or initialized
        ):
            fallback = self._get_fallback_edge()
            if fallback:
                logger.info(
                    "[FLOW] apply_transition max_turns=%d exceeded — fallback edge=%s",
                    current.max_turns,
                    fallback.id,
                )
                return await self._do_transition(
                    edge_id=fallback.id,
                    edge=fallback,
                    criteria_met={},
                    skipped=True,
                    from_node=from_node,
                    user_message=user_input,
                )

        # Merge collected data into node_slots and userdata
        if collected_data:
            logger.info(
                "[FLOW] collected_data merged into node=%s slots=%s",
                self.state,
                collected_data,
            )
            stale_keys = _collect_stale_api_keys(
                self.context.userdata, collected_data, self._api_dep_map
            )
            for key in stale_keys:
                self.context.userdata.pop(key, None)
            if stale_keys:
                logger.info(
                    "[FLOW] invalidated stale API state before merge: %s",
                    stale_keys,
                )
            print(
                f"[TRACK] apply_transition - MERGING collected_data into node_slots and userdata: {collected_data}"
            )
            self.context.node_slots.setdefault(self.state, {}).update(collected_data)
            self.context.userdata.update(collected_data)
            print(
                f"[TRACK] apply_transition - AFTER MERGE - node_slots[{self.state}]: {self.context.node_slots.get(self.state)}, userdata: {dict(list(self.context.userdata.items())[-5:])}"
            )
        else:
            print("[TRACK] apply_transition - NO collected_data to merge")

        # Validate edge
        available_triggers = self._machine.get_triggers(self.state)
        logger.info(
            "[FLOW] validating edge=%s from node=%s, available_triggers=%s",
            edge_id,
            self.state,
            available_triggers,
        )
        if edge_id not in available_triggers:
            msg = (
                f"Edge '{edge_id}' not valid from state '{self.state}'. "
                f"Available: {available_triggers}"
            )
            logger.error(
                "[FLOW] invalid edge %s from %s — available: %s",
                edge_id,
                self.state,
                available_triggers,
            )
            raise ValueError(msg)

        edge = self._get_edge(edge_id)

        # Push intent stack if this is a global edge
        if self.is_global_edge(edge_id):
            logger.info("[FLOW] global edge=%s fired — pushing intent stack", edge_id)
            self._push_intent_stack()

        return await self._do_transition(
            edge_id=edge_id,
            edge=edge,
            criteria_met={},
            skipped=False,
            from_node=from_node,
            user_message=user_input,
        )

    # ------------------------------------------------------------------
    # Entry point 3: custom tool execution
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_def_to_spec(td: ToolDefinition) -> dict[str, Any]:
        """Render a flow ToolDefinition as a Tool.from_dict spec."""
        spec: dict[str, Any] = {
            "type": td.type,
            "id": td.id,
            "name": td.name,
            "description": td.description,
            "input_schema": td.input_schema,
        }
        if td.type == "python":
            spec["handler_id"] = td.handler_id or td.id
        elif td.type == "http":
            spec["url"] = td.url
            if td.method:
                spec["method"] = td.method
        elif td.type == "mcp":
            spec["server"] = td.server
        return spec

    async def execute_tool(
        self,
        tool_id: str,
        args: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Execute a custom tool and merge result data.

        Looks up the tool's handler via ``tool_handlers``, invokes it,
        merges result data into ``node_slots`` / ``userdata``, and
        optionally triggers a transition if ``ToolResult.transition_edge_id``
        is set.
        """
        args = args or {}
        print(f"[TRACK] execute_tool START - tool_id: {tool_id}, args: {args}")
        from_node = self.state

        # Resolution order:
        #   1. New tools_by_id (Task 4 Tool ABC)
        #   2. Legacy tool_registry (MachineToolset)
        #   3. Legacy tool_handlers dict
        if tool_id in self._tools_by_id:
            abc_result = await self._tools_by_id[tool_id].execute(args)
            result = ToolResult(
                data=abc_result.data,
                transition_edge_id=abc_result.transition_edge_id,
            )
            if abc_result.error:
                logger.warning(
                    "[TOOL] %s returned error: %s", tool_id, abc_result.error
                )
        elif self._tool_registry.has(tool_id):
            result = await self._tool_registry.execute(tool_id, args, self.context)
        else:
            # Legacy handler path
            tool_def = self._tool_def_map.get(tool_id)
            handler_id = tool_def.handler_id if tool_def else tool_id

            if handler_id not in self._tool_handlers:
                raise ValueError(
                    f"No handler registered for '{handler_id}' (tool_id='{tool_id}')"
                )

            handler = self._tool_handlers[handler_id]
            raw_result = await handler(tool_id, args)

            # Normalize to ToolResult
            if isinstance(raw_result, ToolResult):
                result = raw_result
            elif isinstance(raw_result, dict):
                result = ToolResult(data=raw_result)
            else:
                result = ToolResult()

        # Merge data into node_slots and userdata
        if result.data:
            print(
                f"[TRACK] execute_tool - merging result.data: {result.data} into node_slots and userdata"
            )
            self.context.node_slots.setdefault(self.state, {}).update(result.data)
            self.context.userdata.update(result.data)
            print(
                f"[TRACK] execute_tool - AFTER MERGE - userdata keys: {list(self.context.userdata.keys())}"
            )
        else:
            print("[TRACK] execute_tool - NO result.data to merge")

        # Optionally trigger transition
        if result.transition_edge_id:
            print(
                f"[TRACK] execute_tool - triggering transition with edge_id: {result.transition_edge_id}, collected_data: {result.data}"
            )
            return await self.apply_transition(
                result.transition_edge_id,
                collected_data=result.data,
            )

        # Data-only: return stay result
        self._schedule_save()
        return TurnResult(
            outcome="stay",
            from_node=from_node,
            to_node=from_node,
            response="",
        )

    # ------------------------------------------------------------------
    # Router auto-chaining helpers
    # ------------------------------------------------------------------

    async def _evaluate_router(self, user_input: str) -> TurnResult | None:
        """Criteria eval for the current router node without incrementing turns.

        Called after transitioning into a router so it routes immediately
        rather than waiting for the next user message.
        Returns a TurnResult if a transition fired, None if routing failed.
        """
        current = self.current_node
        from_node = self.state
        cur_node_slots = self.context.node_slots.get(self.state, {})
        eval_userdata = {
            **self.context.userdata,
            "_flow_meta": {
                "visit_count": self.context.visit_count.get(self.state, 1),
                "turns_in_node": self.context.turns_in_node,
                "agent_language": self.context.agent_language,
                "agent_gender": self.context.agent_gender,
                "node_slots": cur_node_slots,
                "previously_completed": self.state in self.context.completed_nodes,
            },
        }
        # Router chain eval only needs recent turns — routing decisions are
        # based on API results in eval_userdata, not full conversation history.
        # Trimming to the last 6 messages cuts token count from 8000+ to ~3000
        # and prevents gpt-4.1-mini from returning empty responses on long prompts.
        _full_history = self.context.conversation_history
        chain_history = _full_history[-6:] if len(_full_history) > 6 else _full_history

        result: CriteriaResult | None = None
        for attempt in range(2):
            try:
                result = await self._adapter.evaluate_criteria(
                    current,
                    chain_history,
                    eval_userdata,
                )
            except Exception:
                logger.error(
                    "[traverse] router auto-eval failed node=%s attempt=%d",
                    from_node,
                    attempt,
                    exc_info=True,
                )
                continue
            if result.recommended_edge_id:
                break
            logger.warning(
                "[traverse] router node=%s produced no edge (attempt %d/2) — %s",
                from_node,
                attempt + 1,
                "retrying" if attempt == 0 else "stopping chain",
            )

        if result is None or not result.recommended_edge_id:
            return None

        if result.extracted_slots:
            self.context.node_slots.setdefault(self.state, {}).update(
                result.extracted_slots
            )
            self.context.userdata.update(result.extracted_slots)

        edge_id = result.recommended_edge_id
        available_triggers = self._machine.get_triggers(self.state)
        if edge_id not in available_triggers:
            resolved = self._resolve_target_to_edge(edge_id)
            if resolved:
                edge_id = resolved
            else:
                logger.warning(
                    "[traverse] router edge '%s' invalid from '%s' — stopping chain",
                    edge_id,
                    from_node,
                )
                return None

        edge = self._get_edge(edge_id)
        return await self._do_transition(
            edge_id=edge_id,
            edge=edge,
            criteria_met=result.criteria_met,
            skipped=False,
            from_node=from_node,
            user_message=None,
        )

    def _all_required_slots_present(self, node: FlowNode) -> bool:
        """Return True if ALL required fields for at least one outgoing edge
        are already populated in userdata.

        Used to detect pre-filled instruction nodes that should auto-evaluate
        criteria without waiting for user input (e.g. collect_booking_details
        when course/date/time/players were passed in via extracted_slots).
        """
        for edge in node.edges:
            schema = edge.input_schema
            if not isinstance(schema, dict):
                continue
            required = schema.get("required", [])
            if required and all(bool(self.context.userdata.get(k)) for k in required):
                return True
        return False

    async def _follow_router_chain(
        self, user_input: str, last_result: TurnResult, max_hops: int = 5
    ) -> TurnResult:
        """After a transition, auto-process nodes that must not block on user input:

        1. Router nodes — always evaluate and route immediately.
        2. Silent instruction nodes — LLM returned "" (all data present, 'ZERO speech').
        3. Auto-proceed nodes — speak their text then immediately fire their edge
           without waiting for user input. Their speech is accumulated and prepended
           to the next node's response so callers hear both in one turn.
        """
        hops = 0
        accumulated_speech: str = ""
        while not self.is_complete and hops < max_hops:
            node_type = _classify_node_type(self.current_node)
            is_router = node_type == "router"
            is_instruction = node_type == "instruction" and bool(
                self.current_node.edges
            )
            is_silent = is_instruction and not last_result.response
            is_auto = is_instruction and _is_auto_proceed(self.current_node)

            if not (is_router or is_silent or is_auto):
                break

            # Carry auto_proceed speech before transitioning away from this node
            if is_auto and last_result.response:
                accumulated_speech = (
                    accumulated_speech + " " + last_result.response
                ).strip()

            hops += 1
            reason = (
                "router"
                if is_router
                else ("auto-proceed" if is_auto else "silent-instruction")
            )
            logger.info(
                "[traverse] %s-chain hop=%d node=%s",
                reason,
                hops,
                self.state,
            )
            chained = await self._evaluate_router(user_input)
            if chained is None:
                break
            last_result = chained

        # Prepend any accumulated auto_proceed speech to the final response
        if accumulated_speech:
            final = (accumulated_speech + " " + (last_result.response or "")).strip()
            last_result = last_result.model_copy(update={"response": final})

        return last_result

    # ------------------------------------------------------------------
    # Shared transition logic
    # ------------------------------------------------------------------

    async def _do_transition(
        self,
        edge_id: str,
        edge: Edge | None,
        criteria_met: dict[str, bool],
        skipped: bool,
        from_node: str,
        user_message: str | None = None,
    ) -> TurnResult:
        """Execute a full transition: actions → trigger → record → response."""
        logger.info(
            "[FLOW] _do_transition START %s -[%s]-> ? skipped=%s",
            from_node,
            edge_id,
            skipped,
        )
        actions_fired: list[str] = []

        # 1. Execute ON_EXIT actions for current node
        current = self.current_node
        if current.actions:
            logger.info(
                "[FLOW] executing ON_EXIT actions for node=%s, action_count=%d",
                from_node,
                len(current.actions),
            )
            fired = await self._execute_actions(
                current.actions, ActionTriggerType.ON_EXIT
            )
            actions_fired.extend(fired)
            logger.info("[FLOW] ON_EXIT actions fired: %s", fired)

        # 2. Execute edge actions
        if edge and edge.actions:
            logger.info(
                "[FLOW] executing edge actions for edge=%s, action_count=%d",
                edge_id,
                len(edge.actions),
            )
            fired = await self._execute_actions(edge.actions)
            actions_fired.extend(fired)
            logger.info("[FLOW] edge actions fired: %s", fired)

        # 3. Fire pytransitions trigger
        logger.info("[FLOW] firing pytransitions trigger for edge=%s", edge_id)
        trigger_fn = getattr(self, edge_id)
        await trigger_fn()
        logger.info(
            "[traverse] TRANSITION %s -[%s]-> %s%s",
            from_node,
            edge_id,
            self.state,
            " (skipped)" if skipped else "",
        )

        # 4. Record transition
        record = TransitionRecord(
            from_node=from_node,
            to_node=self.state,
            edge_id=edge_id,
            criteria_met=criteria_met,
            skipped=skipped,
            timestamp=time.time(),
        )
        self.context.transition_log.append(record)
        self.context.completed_nodes.add(from_node)
        self.context.current_node_id = self.state
        logger.info(
            "[FLOW] transition recorded: %s -> %s, completed_nodes=%s",
            from_node,
            self.state,
            list(self.context.completed_nodes),
        )

        # 5. Update visit tracking
        self.context.turns_in_node = 0
        self.context.user_turns_in_node = 0
        vc = self.context.visit_count.get(self.state, 0)
        self.context.visit_count[self.state] = vc + 1
        logger.info(
            "[FLOW] visit tracking updated for node=%s, visit_count=%d",
            self.state,
            self.context.visit_count[self.state],
        )

        # Track consecutive self-loops (A: self-loop protection)
        if from_node == self.state:
            self.context.consecutive_self_loops += 1
            logger.info(
                "[FLOW] self-loop detected %s (count=%d/%d)",
                self.state,
                self.context.consecutive_self_loops,
                self.context.MAX_SELF_LOOPS,
            )
        else:
            self.context.consecutive_self_loops = 0

        # 5b. Variable extraction (Plan 1) — extract from exiting node
        if self._extractor:
            prev_node = self._node_map.get(from_node)
            extraction_vars = getattr(prev_node, "extraction_variables", None)
            if extraction_vars:
                logger.info(
                    "[FLOW] extracting variables from node=%s, var_count=%d",
                    from_node,
                    len(extraction_vars),
                )
                extracted = await self._extractor.extract(
                    extraction_vars,
                    self.context.conversation_history,
                    self.context.userdata,
                )
                if extracted:
                    logger.info("[FLOW] extracted variables: %s", extracted)
                    self.context.data.merge(
                        extracted,
                        source=f"extraction:{from_node}",
                    )

        # 5c. Fire transition hook (Plan 3)
        logger.info("[FLOW] firing transition hook")
        self._hooks.fire_transition(from_node, self.state, edge_id, self.context)

        # 6. Execute ON_ENTER actions for new node
        new_node = self.current_node
        if new_node.actions:
            logger.info(
                "[FLOW] executing ON_ENTER actions for node=%s, action_count=%d",
                self.state,
                len(new_node.actions),
            )
            fired = await self._execute_actions(
                new_node.actions, ActionTriggerType.ON_ENTER
            )
            actions_fired.extend(fired)
            logger.info("[FLOW] ON_ENTER actions fired: %s", fired)

        # Router nodes produce no speech — auto-mark as spoken so Gate 2
        # doesn't block their transitions.
        if _classify_node_type(new_node) == "router":
            self.mark_node_spoken(self.state)
            logger.debug("[FLOW] router node=%s auto-marked as spoken", self.state)

        # 7. Generate response for new node
        logger.info("[FLOW] generating response for new node=%s", self.state)
        response = await self._generate_node_response(new_node)
        if response:
            logger.info(
                "[FLOW] generated response for node=%s, length=%d",
                self.state,
                len(response),
            )
            if self._should_persist_response_to_history():
                self.context.add_assistant_message(response)

        # Attribute this turn's messages to the record (source of truth for
        # traversal; robust to router-chaining where one user turn = N records).
        record.user_message = user_message
        record.bot_message = response or ""

        # 8. End session if final
        if actions_fired:
            logger.info(
                "[FLOW] actions fired=%s after transition to node=%s",
                actions_fired,
                self.state,
            )
        if self.is_complete:
            logger.info(
                "[FLOW] FINAL node=%s total_transitions=%d completed_nodes=%s",
                self.state,
                len(self.context.transition_log),
                list(self.context.completed_nodes),
            )
            if self._store and self._session_id:
                await self._store.save(self._session_id, self.context)
            await self._adapter.end_session()
        else:
            logger.info(
                "[FLOW] now_at=%s intent_stack_depth=%d completed_nodes=%s",
                self.state,
                len(self.context.intent_stack),
                list(self.context.completed_nodes),
            )
            # 9. Check auto-return from intent stack
            auto_result = await self._maybe_auto_return()
            if auto_result:
                logger.info(
                    "[FLOW] auto_return triggered → back to node=%s",
                    auto_result.to_node,
                )
                return auto_result
            self._schedule_save()

        return TurnResult(
            outcome="transition",
            from_node=from_node,
            to_node=self.state,
            response=response,
            edge_id=edge_id,
            criteria_snapshot=criteria_met,
            actions_fired=actions_fired,
        )

    # ------------------------------------------------------------------
    # NodeScope builder — assembles everything an executor needs
    # ------------------------------------------------------------------

    def build_node_scope(self, node: FlowNode | None = None) -> NodeScope:
        """Assemble everything an executor needs to operate on a node.

        Returns a self-contained NodeScope with instructions, history,
        tools, data, and criteria. The executor should NOT reach back
        into the machine for any of this data.
        """
        node = node or self.current_node
        logger.info(
            "[FLOW] build_node_scope starting for node=%s, turns_in_node=%d",
            node.id,
            self.context.turns_in_node,
        )

        # Extract speech text if language markers exist
        speech_text: str | None = None
        try:
            from superdialog.machine.composer import extract_speech_text

            # resolve_active_language expects (state, machine)
            # but we can use the context language directly
            lang = self.context.agent_language or "en"
            if node.static_text:
                # static_text always takes precedence as the TTS greeting;
                # instruction governs LLM behavior after TTS fires
                speech_text = node.static_text
                logger.info(
                    "[FLOW] speech text set from static_text for node=%s, length=%d",
                    node.id,
                    len(speech_text),
                )
            elif node.instruction:
                speech_text = extract_speech_text(node.instruction, self, lang)
                logger.info(
                    "[FLOW] speech text extracted for node=%s, length=%d",
                    node.id,
                    len(speech_text) if speech_text else 0,
                )
        except Exception as ex:
            logger.debug("[FLOW] node_spoken marked for node=%s", str(ex))
            pass

        # Build criteria list from node config
        criteria_list: list[dict[str, Any]] = []
        if node.completion_criteria:
            logger.info(
                "[FLOW] building criteria list for node=%s, count=%d",
                node.id,
                len(node.completion_criteria),
            )
            for cc in node.completion_criteria:
                criteria_list.append(
                    {
                        "key": cc.key,
                        "description": cc.description,
                        "required": cc.required,
                    }
                )

        # Apply preprocessing hooks
        instruction = self.get_enriched_instructions(node)
        logger.info(
            "[FLOW] enriched instruction for node=%s, length=%d",
            node.id,
            len(instruction),
        )
        instruction = self._hooks.apply_prompt(instruction, self.context)

        history = list(self.context.conversation_history)
        logger.info(
            "[FLOW] conversation history for node=%s, messages=%d",
            node.id,
            len(history),
        )
        history = self._hooks.apply_history(history, self.context)

        # Merge registry tools with edge tools
        edge_tools = self.get_tools_for_node(node)
        edge_tools.extend(self._tool_registry.get_descriptors())
        logger.info(
            "[FLOW] tools assembled for node=%s, total_tools=%d",
            node.id,
            len(edge_tools),
        )

        scope = NodeScope(
            node_id=node.id,
            node_type=self.classify_node_type(node),
            is_final=node.is_final,
            is_initial=(node.id == self._flow.initial_node),
            is_self_loop=self.context.consecutive_self_loops > 0,
            auto_proceed=_is_auto_proceed(node),
            system_prompt=getattr(self._flow, "system_prompt", "") or "",
            node_instruction=instruction,
            speech_text=speech_text,
            language=self.context.agent_language or "en",
            conversation_history=history,
            completed_nodes=sorted(self.context.completed_nodes),
            turns_in_node=self.context.turns_in_node,
            visit_count=self.context.visit_count.get(node.id, 1),
            edge_tools=edge_tools,
            node_slots=dict(self.context.node_slots.get(node.id, {})),
            userdata=dict(self.context.userdata),
            completion_criteria=criteria_list,
            allow_skip=node.allow_skip,
            max_turns=node.max_turns,
        )

        logger.info(
            "[FLOW] node_scope built for node=%s, type=%s, tools=%d, history=%d",
            scope.node_id,
            scope.node_type,
            len(scope.edge_tools),
            len(scope.conversation_history),
        )
        return scope

    # ------------------------------------------------------------------
    # Gated transition — validates before executing
    # ------------------------------------------------------------------

    def mark_node_spoken(self, node_id: str | None = None) -> None:
        """Mark that the executor has spoken the current node's content."""
        nid = node_id or self.state
        self.context.node_spoken_flags[nid] = True
        logger.debug("[FLOW] node_spoken marked for node=%s", nid)

    async def request_transition(
        self,
        edge_id: str,
        collected_data: dict[str, Any] | None = None,
    ) -> TransitionResult:
        """Gated transition — validates before executing.

        Delegates gate checks to TransitionGate, then executes
        the transition and builds the new NodeScope if allowed.
        """
        print(
            f"[TRACK] request_transition START - edge_id: {edge_id}, collected_data: {collected_data}"
        )
        from_node = self.state
        node = self.current_node

        logger.info(
            "[FLOW] request_transition edge=%s from=%s "
            "collected=%s user_turns=%d auto_proceed=%s",
            edge_id,
            from_node,
            list((collected_data or {}).keys()),
            self.context.user_turns_in_node,
            _is_auto_proceed(node),
        )
        print(
            f"[TRACK] request_transition - from_node: {from_node}, collected_data_keys: {list((collected_data or {}).keys())}"
        )

        # Delegate validation to TransitionGate first. Criteria / user-spoke /
        # spoken-content gates run before the premature-final guard so callers
        # see the most specific reason a transition was denied (missing slots
        # over "call too short to conclude").
        logger.info("[FLOW] checking transition gate for edge=%s", edge_id)
        gate_result = await self._gate.check(
            edge_id=edge_id,
            node=node,
            context=self.context,
            available_triggers=self._machine.get_triggers(self.state),
            edge_obj=self._get_edge(edge_id),
            adapter=self._adapter,
            collected_data=collected_data,
        )

        # A3 guard: only fire after the gate accepts -- prevent premature jump
        # to a final node when the call is too short to conclude.
        # Auto-proceed source nodes bypass this -- their semantics is
        # "no caller response needed", so user-turn count is meaningless.
        if gate_result.allowed and not _is_auto_proceed(node):
            edge_obj_for_final_check = self._get_edge(edge_id)
            if (
                edge_obj_for_final_check is not None
                and edge_obj_for_final_check.target_node_id in self._final_states
            ):
                import os as _os

                min_turns = int(_os.getenv("MIN_TURNS_BEFORE_FINAL_NODE", "1"))
                total_user_turns = sum(
                    1
                    for m in self.context.conversation_history
                    if isinstance(m, dict) and str(m.get("role", "")).lower() == "user"
                )
                if total_user_turns < min_turns:
                    reason_msg = (
                        f"Too few user turns ({total_user_turns}/{min_turns}) "
                        f"to reach final node "
                        f"'{edge_obj_for_final_check.target_node_id}'."
                    )
                    logger.warning(
                        "[FLOW] gate DENIED (premature final): %s", reason_msg
                    )
                    return TransitionResult(
                        allowed=False,
                        reason=reason_msg,
                        correction_hint=(
                            "The conversation is too short to conclude. "
                            "Continue engaging the caller before ending the call."
                        ),
                    )

        if not gate_result.allowed:
            logger.info(
                "[FLOW] gate DENIED edge=%s: %s",
                edge_id,
                gate_result.reason or "No reason provided",
            )
            return gate_result

        # All gates passed — execute the transition
        logger.info("[FLOW] gate passed, executing transition for edge=%s", edge_id)
        print(
            f"[TRACK] request_transition - gate PASSED, executing transition with collected_data: {collected_data}"
        )
        turn_result = await self.apply_transition(
            edge_id, collected_data=collected_data
        )

        # Build new scope for the next node and enrich it once here so
        # the executor can reuse the same NodeScope without rebuilding
        # or re-enriching downstream.
        logger.info("[FLOW] building new node scope after transition")
        new_scope = self.build_node_scope()

        logger.info(
            "[FLOW] gate ALLOWED edge=%s %s -> %s",
            edge_id,
            from_node,
            self.state,
        )

        return TransitionResult(
            allowed=True,
            turn_result=turn_result,
            new_scope=new_scope,
        )
