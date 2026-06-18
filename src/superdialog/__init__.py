"""SuperDialog -- standalone dialog state machine framework."""

__version__ = "0.2.8"

from .agent import Agent, TurnResult
from .agents import LLMAgent
from .chat_context import ChatContext, ChatMessage
from .dialog_machine import DialogMachine
from .flow import Flow, FlowSet, create_dialog_flow
from .flow_state import FlowState
from .llm.registry import register_llm_provider
from .session import (
    AsyncioLockBackend,
    InMemorySessionStore,
    LockBackend,
    NullSessionStore,
    Session,
    SessionHandle,
    SessionRecord,
    SessionStore,
    SessionWorker,
)
from .stream import StreamChunk, ToolCall, Turn
from .tools import HttpTool, MCPTool, PythonTool, Tool, ToolResult
from .observability import (
    LangfuseObserver,
    NullObserver,
    Observer,
    TracingProvider,
    build_observer,
)

__all__ = [
    "Agent",
    "AsyncioLockBackend",
    "ChatContext",
    "ChatMessage",
    "DialogMachine",
    "Flow",
    "FlowSet",
    "FlowState",
    "HttpTool",
    "InMemorySessionStore",
    "LLMAgent",
    "LangfuseObserver",
    "LockBackend",
    "MCPTool",
    "NullObserver",
    "NullSessionStore",
    "Observer",
    "PythonTool",
    "Session",
    "SessionHandle",
    "SessionRecord",
    "SessionStore",
    "SessionWorker",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ToolResult",
    "TracingProvider",
    "Turn",
    "TurnResult",
    "build_observer",
    "create_dialog_flow",
    "register_llm_provider",
]
