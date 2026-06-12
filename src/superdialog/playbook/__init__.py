"""Playbook engine: declarative journeys behind the public Agent protocol."""

from .agent import PlaybookAgent
from .compiler import compile_flow, coverage_report
from .director import CompletesLLM
from .editable import Edit, FullDoc, MutationError, SimpleDoc, make_editable
from .eval_bridge import (
    EvalReport,
    PersonaSpec,
    SessionMetrics,
    run_eval,
    run_session,
)
from .events import EventLog
from .generate import generate_simple_playbook
from .models import Playbook
from .optimize import ObjectiveBreakdown, OptimizeReport, RoundTrace, optimize
from .personas import generate_personas, load_personas
from .providers import ProviderDirector, ProviderTalker, provider_adapters
from .replay import ReplayReport, replay
from .simple import is_simple_playbook, load_simple, simple_to_playbook
from .state import ConversationState
from .talker import StreamsLLM
from .toolexec import HttpFn, PythonToolFn, httpx_http

__all__ = [
    "CompletesLLM",
    "ConversationState",
    "Edit",
    "EvalReport",
    "EventLog",
    "FullDoc",
    "HttpFn",
    "MutationError",
    "ObjectiveBreakdown",
    "OptimizeReport",
    "PersonaSpec",
    "Playbook",
    "PlaybookAgent",
    "ProviderDirector",
    "ProviderTalker",
    "PythonToolFn",
    "ReplayReport",
    "RoundTrace",
    "SessionMetrics",
    "SimpleDoc",
    "StreamsLLM",
    "compile_flow",
    "coverage_report",
    "generate_personas",
    "generate_simple_playbook",
    "httpx_http",
    "is_simple_playbook",
    "load_personas",
    "load_simple",
    "make_editable",
    "optimize",
    "provider_adapters",
    "replay",
    "run_eval",
    "run_session",
    "simple_to_playbook",
]
