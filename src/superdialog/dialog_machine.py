"""DialogMachine -- spec-aligned public facade over ``DialogStateMachine``.

The facade is the user-facing surface documented in
``docs/02-api-reference.md``. It hides the criteria-judge / adapter /
state-machine plumbing and offers a small, stable API:

* :meth:`turn` -- one turn, optionally streaming.
* :meth:`inject_system` -- queue a system message for the next turn.
* :meth:`reset` -- clear conversation memory.
* :meth:`set_llm` -- hot-swap the runtime model.
* :meth:`switch_flow` -- jump between flows in a :class:`FlowSet`.
* :attr:`state` -- read-only view of current node + slots.

Streaming policy (Task 5 v1): the underlying engine resolves the turn in
one shot via :meth:`DialogStateMachine.process_turn`. When the caller
requests ``stream="text"`` we chunk the assembled response into
whitespace-delimited fragments. This is honest token-shaped output
without a second LLM pass; true streaming inference lands in v0.4
(per ``docs/decisions.md`` roadmap).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from .flow.loader import FlowSet
from .flow.models import ConversationFlow as Flow
from .llm.provider import LLMProvider
from .llm.resolver import resolve_llm
from .machine.adapters.llm_adapter import LLMAdapter
from .machine.adapters.toolcall_adapter import ToolCallAdapter
from .machine.machine import DialogStateMachine
from .machine.store import ContextStore, InMemoryContextStore
from .stream import StreamChunk, ToolCall, Turn
from .tools.base import Tool

logger = logging.getLogger(__name__)

_SYSTEM_MARKER_KEY = "_pending_system_messages"


def _select_engine(source: Any, engine: str = "auto") -> Literal["graph", "playbook"]:
    """Resolve the backend engine from the constructor inputs (pure)."""
    from .playbook import Playbook

    is_flow = isinstance(source, (Flow, FlowSet))
    is_playbook = isinstance(source, Playbook)
    if engine == "flow":
        if is_playbook:
            raise ValueError(
                "engine='flow' but source is a Playbook (no graph runtime "
                "exists for it)"
            )
        return "graph"
    if engine == "playbook":
        return "playbook"
    if engine != "auto":
        raise ValueError(f"unknown engine: {engine!r}")
    # auto: a Flow object keeps the legacy graph engine (back-compat);
    # everything else (Playbook object, path string, parsed dict) runs Playbook.
    return "graph" if is_flow else "playbook"


def _python_tools_from(tools: list[Tool] | None) -> dict[str, Any]:
    """Bridge any Tool to the Playbook engine's PythonToolFn via execute()."""

    def _adapt(tool: Tool) -> Any:
        async def fn(args: dict[str, Any], state: Any) -> Any:
            return (await tool.execute(args)).data

        return fn

    return {t.id: _adapt(t) for t in (tools or [])}


class DialogMachine:
    """Spec-aligned public facade over :class:`DialogStateMachine`.

    Construct with a :class:`Flow` (or a :class:`FlowSet` for multi-flow
    apps) and a model URI; the underlying state machine is built lazily
    on first use so construction stays synchronous.
    """

    def __init__(
        self,
        source: Flow | FlowSet | Any = None,
        llm: str | None = None,
        tools: list[Tool] | None = None,
        memory: ContextStore | None = None,
        config: dict[str, Any] | None = None,
        traversal_dir: str | Path | None = None,
        adapter: str = "toolcall",
        *,
        flow: Flow | FlowSet | None = None,
        engine: str = "auto",
        director_llm: str | None = None,
    ) -> None:
        # Back-compat: the first param used to be `flow`; accept it by keyword.
        if source is None and flow is not None:
            source = flow
        self._engine: Literal["graph", "playbook"] = _select_engine(source, engine)
        if self._engine == "playbook":
            self._init_playbook(source, llm, tools, director_llm)
            return
        # graph engine: a path string under engine="flow" loads via Flow.load.
        flow = Flow.load(source) if isinstance(source, str) else source
        if llm is None:
            raise ValueError("DialogMachine needs an llm= for the graph engine")
        self._init_graph(flow, llm, tools, memory, config, traversal_dir, adapter)

    def _init_playbook(
        self,
        source: Any,
        llm: str | None,
        tools: list[Tool] | None,
        director_llm: str | None,
    ) -> None:
        """Wire the Playbook backend lazily; build it on first turn/start."""
        if not (llm or director_llm):
            raise ValueError("DialogMachine needs an llm= for the Playbook engine")
        self._pb_source = source
        self._llm_uri = llm
        self._director_uri = director_llm
        self._pb_tools = tools
        self._pb: Any = None
        # Test seams: inject scripted Talker/Director so tests stay offline.
        self._talker_override: Any = None
        self._director_override: Any = None

    def _init_graph(
        self,
        flow: Flow | FlowSet,
        llm: str,
        tools: list[Tool] | None,
        memory: ContextStore | None,
        config: dict[str, Any] | None,
        traversal_dir: str | Path | None,
        adapter: str,
    ) -> None:
        self._flowset: FlowSet = (
            flow if isinstance(flow, FlowSet) else FlowSet({"main": flow})
        )
        self._active_flow_name = next(iter(self._flowset.names()))
        self._llm_uri = llm
        self._llm: LLMProvider = resolve_llm(llm)
        self._tools = list(tools or [])
        self._memory: ContextStore = memory or InMemoryContextStore()
        self._config: dict[str, Any] = dict(config or {})
        self._pending_system_messages: list[str] = list(
            self._config.pop(_SYSTEM_MARKER_KEY, [])
        )
        self._adapter_mode: str = adapter
        self._machine: DialogStateMachine | None = None
        self._adapter: LLMAdapter | ToolCallAdapter | None = None
        # Session-layer rehydration: queued until first turn / explicit start.
        self._pending_chat_ctx: Any = None
        self._pending_flow_state: Any = None
        # Traversal tracking
        self._traversal_dir: Path | None = (
            Path(traversal_dir) if traversal_dir else None
        )
        self._chat_turns: list[dict[str, Any]] = []
        self._session_started_at: datetime | None = None
        self._traversal_saved: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> Any:
        """Build the wrapped PlaybookAgent lazily on first use."""
        if self._pb is not None:
            return self._pb
        from .playbook import Playbook, PlaybookAgent, httpx_http, provider_adapters

        if isinstance(self._pb_source, Playbook):
            pb = self._pb_source
        elif isinstance(self._pb_source, str):
            pb = Playbook.load(self._pb_source)
        else:
            pb = Playbook._from_doc(self._pb_source)
        director, talker = provider_adapters(resolve_llm(self._llm_uri or ""))
        if self._director_uri:
            director, _ = provider_adapters(resolve_llm(self._director_uri))
        self._pb = PlaybookAgent(
            playbook=pb,
            talker_llm=self._talker_override or talker,
            director_llm=self._director_override or director,
            http=httpx_http,
            python_tools=_python_tools_from(self._pb_tools),
        )
        return self._pb

    async def _ensure_machine(self) -> DialogStateMachine:
        if self._machine is not None:
            return self._machine
        active_flow = self._flowset[self._active_flow_name]
        if self._adapter_mode == "toolcall":
            self._adapter = ToolCallAdapter(
                model_id=self._llm_uri,
                system_prompt=getattr(active_flow, "system_prompt", "") or "",
                environment_variables=dict(
                    getattr(active_flow, "environment_variables", {}) or {}
                ),
            )
        else:
            self._adapter = LLMAdapter(
                provider=self._llm,
                system_prompt=getattr(active_flow, "system_prompt", "") or "",
                environment_variables=dict(
                    getattr(active_flow, "environment_variables", {}) or {}
                ),
            )
        self._machine = await DialogStateMachine.from_flow(
            flow=active_flow,
            adapter=self._adapter,
            store=self._memory,
            tools=self._tools,
        )
        if isinstance(self._adapter, ToolCallAdapter):
            self._adapter._machine = self._machine
        if self._pending_chat_ctx is not None:
            self._machine.load_chat_ctx(self._pending_chat_ctx)
            self._pending_chat_ctx = None
        if self._pending_flow_state is not None:
            self._machine.load_flow_state(self._pending_flow_state)
            self._pending_flow_state = None
        return self._machine

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    async def seed(self, userdata: dict) -> None:
        """Pre-seed userdata into machine context before start() is called.

        Use to inject dispatch-time variables (name, phone, inquiry, etc.)
        so Jinja2 templates in node instructions resolve without asking the caller.
        Must be called before start().
        """
        machine = await self._ensure_machine()
        machine.context.userdata.update(userdata)

    async def start(self) -> Turn:
        """Generate the initial greeting without adding a user message to history.

        Call instead of ``turn(" ")`` to bootstrap the first bot turn.
        Fires on_enter actions for the initial node (auth, data-preload, etc.)
        before generating the greeting.
        """
        if self._engine == "playbook":
            lines = await self._ensure_backend().runtime.start()
            return Turn(text=" ".join(lines).strip(), tool_calls=[], metadata={})
        machine = await self._ensure_machine()
        # Fire on_enter actions for the initial node (auth, preloads, etc.)
        # Mirrors SimpleFlowAgent.on_enter() for the voice path.
        await machine.fire_initial_on_enter()
        current = machine.current_node
        response = await machine._generate_node_response(current)
        if response and machine._should_persist_response_to_history():
            machine.context.add_assistant_message(response)
        # Record initial turn for traversal
        self._session_started_at = datetime.now(timezone.utc)
        self._chat_turns = [
            {
                "step": 1,
                "bot": response or "",
                "user": None,
                "node": current.id,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        ]
        self._traversal_saved = False
        return Turn(
            text=response,
            tool_calls=[],
            metadata={
                "from_node": current.id,
                "to_node": current.id,
                "outcome": "start",
                "edge_id": None,
                "actions_fired": [],
                "model": self._llm_uri,
            },
        )

    async def turn(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        stream: bool | Literal["text"] = False,
    ) -> Turn | AsyncIterator[StreamChunk]:
        """Run a single dialog turn.

        With ``stream=False`` (default) returns a complete :class:`Turn`.
        With ``stream=True`` or ``stream="text"`` returns an async
        iterator of :class:`StreamChunk` items; the final chunk carries
        the assembled :class:`Turn` on ``chunk.turn``.
        """
        if self._engine == "playbook":
            return await self._ensure_backend().turn(text, stream=bool(stream))
        if stream:
            return self._stream_turn(text, context)
        return await self._run_turn(text, context)

    async def _run_turn(
        self,
        text: str,
        context: dict[str, Any] | None,
    ) -> Turn:
        machine = await self._ensure_machine()
        self._consume_pending_system_messages(machine)
        if context:
            machine.context.userdata.update(context)
        # Record user input at the CURRENT node (before processing) so it is
        # paired with the node the user was responding to, not the node
        # the machine transitions into.
        if self._chat_turns:
            self._chat_turns[-1]["user"] = text

        result = await machine.process_turn(text)

        # Append a new record for the node we just transitioned INTO.
        # user=None will be filled on the next turn (or stay None at final nodes).
        self._chat_turns.append(
            {
                "step": len(self._chat_turns) + 1,
                "bot": result.response or "",
                "user": None,
                "node": result.to_node,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Auto-save traversal when session completes
        if self._traversal_dir and machine.is_complete and not self._traversal_saved:
            self._traversal_saved = True
            self._auto_save_traversal()
        return Turn(
            text=result.response,
            tool_calls=[],
            metadata={
                "from_node": result.from_node,
                "to_node": result.to_node,
                "outcome": result.outcome,
                "edge_id": result.edge_id,
                "actions_fired": list(result.actions_fired),
                "model": self._llm_uri,
            },
        )

    async def _stream_turn(
        self,
        text: str,
        context: dict[str, Any] | None,
    ) -> AsyncIterator[StreamChunk]:
        # Resolve the turn first, then surface the response as token-shaped
        # chunks. v0.4 will swap this for true provider.stream() delivery.
        turn = await self._run_turn(text, context)
        body = turn.text or ""
        if not body:
            yield StreamChunk(text="", done=True, turn=turn)
            return
        pieces = body.split(" ")
        last = len(pieces) - 1
        for idx, piece in enumerate(pieces):
            chunk_text = piece if idx == last else f"{piece} "
            is_last = idx == last
            yield StreamChunk(
                text=chunk_text,
                done=is_last,
                turn=turn if is_last else None,
            )

    # ------------------------------------------------------------------
    # Side-channel helpers (spec-aligned)
    # ------------------------------------------------------------------

    def assist(self, text: str) -> None:
        """Queue a system-level instruction for the next turn."""
        if self._engine == "playbook":
            self._ensure_backend().assist(text)
            return
        if not text:
            return
        self._pending_system_messages.append(text)

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
    # Agent Protocol surface
    # ------------------------------------------------------------------

    @property
    def chat_ctx(self):  # type: ignore[no-untyped-def]
        """Conversation history view (LiveKit-aligned ChatContext)."""
        from .chat_context import ChatContext, ChatMessage

        if self._engine == "playbook":
            return self._ensure_backend().chat_ctx
        if self._machine is None:
            # Pre-construction: synthesise from any queued system messages.
            items = [
                ChatMessage(role="system", content=m)
                for m in self._pending_system_messages
            ]
            return ChatContext(items=items)
        return self._machine.chat_ctx

    def load_chat_ctx(self, ctx) -> None:  # type: ignore[no-untyped-def]
        """Replace conversation history.

        If the underlying engine hasn't been built yet, the chat context is
        queued and applied on the first turn (the lazy bootstrap honours it).
        """
        if self._engine == "playbook":
            self._ensure_backend().load_chat_ctx(ctx)
            return
        self._pending_chat_ctx = ctx
        if self._machine is not None:
            self._machine.load_chat_ctx(ctx)

    @property
    def flow_state(self):  # type: ignore[no-untyped-def]
        """DM-specific runtime state (current node + slots)."""
        from .flow_state import FlowState

        if self._machine is None:
            return FlowState(
                current_node_id=self._flowset[self._active_flow_name].initial_node,
            )
        return self._machine.flow_state

    def load_flow_state(self, state) -> None:  # type: ignore[no-untyped-def]
        """Apply a FlowState snapshot.

        Like ``load_chat_ctx``, queued if the engine is not yet built.
        """
        self._pending_flow_state = state
        if self._machine is not None:
            self._machine.load_flow_state(state)

    def reset(self) -> None:
        """Drop the machine + memory; the next turn re-bootstraps."""
        self._machine = None
        self._adapter = None
        self._memory = InMemoryContextStore()
        self._pending_system_messages.clear()
        self._chat_turns = []
        self._session_started_at = None
        self._traversal_saved = False

    def set_llm(self, uri: str) -> None:
        """Hot-swap the runtime model URI."""
        self._llm_uri = uri
        self._llm = resolve_llm(uri)
        if self._adapter is not None:
            if isinstance(self._adapter, ToolCallAdapter):
                self._adapter._model_id = uri
            else:
                self._adapter.set_provider(self._llm)

    def switch_flow(self, name: str, preserve_memory: bool = False) -> None:
        """Switch to a named flow in the bound :class:`FlowSet`."""
        if name not in self._flowset:
            raise KeyError(f"Flow {name!r} not in FlowSet")
        self._active_flow_name = name
        self._machine = None
        self._adapter = None
        if not preserve_memory:
            self._memory = InMemoryContextStore()

    @property
    def state(self) -> dict[str, Any]:
        """Read-only view of the current node id and slot dictionary."""
        if self._machine is None:
            initial = self._flowset[self._active_flow_name].initial_node
            return {"node_id": initial, "slots": {}}
        return {
            "node_id": self._machine.context.current_node_id,
            "slots": dict(self._machine.context.userdata),
        }

    @property
    def is_complete(self) -> bool:
        """True if the machine is at a final node."""
        if self._machine is None:
            return False
        return self._machine.is_complete

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _auto_save_traversal(self) -> None:
        """Save traversal JSON to _traversal_dir. Swallows all errors."""
        try:
            from .traversal import build_traversal, save_traversal

            flow = self._flowset[self._active_flow_name]
            traversal = build_traversal(
                self,
                self._chat_turns,
                flow,
                source=self._active_flow_name,
                model=self._llm_uri,
                started_at=self._session_started_at or datetime.now(timezone.utc),
            )
            path = save_traversal(traversal, self._traversal_dir)
            logger.info("[DialogMachine] traversal saved: %s", path)
        except Exception:
            logger.warning("[DialogMachine] traversal save failed", exc_info=True)

    def _consume_pending_system_messages(self, machine: DialogStateMachine) -> None:
        """Flush queued system messages into the conversation history."""
        if not self._pending_system_messages:
            return
        for msg in self._pending_system_messages:
            machine.context.add_message("system", msg)
        logger.info(
            "[DialogMachine] injected %d system message(s)",
            len(self._pending_system_messages),
        )
        self._pending_system_messages.clear()


__all__ = ["DialogMachine", "Turn", "ToolCall", "StreamChunk"]
