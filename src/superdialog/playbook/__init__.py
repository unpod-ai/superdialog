"""Playbook engine: declarative journeys behind the public Agent protocol."""

from .agent import PlaybookAgent
from .compiler import compile_flow, coverage_report
from .director import CompletesLLM
from .editable import Edit, FullDoc, MutationError, SimpleDoc, make_editable
from .eval import (
    AuditReport,
    CorpusGenerator,
    CorpusSpec,
    EdgeScenario,
    EvalCache,
    EvalReport,
    ModelScore,
    MultiModelReport,
    ObjectiveBreakdown,
    PersonaSpec,
    ScriptedUser,
    TimingLLM,
    SessionAuditor,
    SessionMetrics,
    SpeaksUser,
    cached_speaker,
    derive_default_persona,
    generate_personas,
    load_personas,
    load_traversal,
    persona_cache_path,
    run_eval,
    run_multi_model,
    run_session,
    save_personas,
    score_report,
    traversal_to_persona,
    traversal_to_scripted_user,
)
from .events import EventLog
from .generate import generate_simple_playbook
from .models import Playbook
from .optimize import OptimizeReport, RoundTrace, optimize
from .providers import ProviderDirector, ProviderTalker, provider_adapters
from .replay import ReplayReport, replay
from .generate import generate_simple_playbook
from .simple import is_simple_playbook, load_simple, simple_to_playbook
from .state import ConversationState
from .talker import StreamsLLM
from .toolexec import HttpFn, PythonToolFn, httpx_http
from .traversal import build_playbook_traversal, save_playbook_traversal

__all__ = [
    "AuditReport",
    "ScriptedUser",
    "TimingLLM",
    "load_traversal",
    "traversal_to_persona",
    "traversal_to_scripted_user",
    "CompletesLLM",
    "ConversationState",
    "CorpusGenerator",
    "CorpusSpec",
    "Edit",
    "EdgeScenario",
    "EvalCache",
    "EvalReport",
    "EventLog",
    "FullDoc",
    "HttpFn",
    "ModelScore",
    "MultiModelReport",
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
    "SessionAuditor",
    "SessionMetrics",
    "SimpleDoc",
    "SpeaksUser",
    "StreamsLLM",
    "cached_speaker",
    "compile_flow",
    "coverage_report",
    "derive_default_persona",
    "generate_personas",
    "generate_simple_playbook",
    "httpx_http",
    "is_simple_playbook",
    "load_personas",
    "load_simple",
    "make_editable",
    "optimize",
    "persona_cache_path",
    "provider_adapters",
    "replay",
    "run_eval",
    "run_multi_model",
    "run_session",
    "save_personas",
    "score_report",
    "simple_to_playbook",
    "build_playbook_traversal",
    "save_playbook_traversal",
]
