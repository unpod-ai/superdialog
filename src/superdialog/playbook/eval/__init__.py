# src/superdialog/playbook/eval/__init__.py
"""Playbook eval package: persona simulation, auditing, corpus generation, scoring."""

from .auditor import SessionAuditor
from .cache import EvalCache, cached_speaker
from .corpus import CorpusGenerator
from .models import (
    AuditReport,
    CorpusSpec,
    EdgeScenario,
    EvalReport,
    ModelScore,
    MultiModelReport,
    PersonaSpec,
    SessionMetrics,
)
from .personas import (
    derive_default_persona,
    generate_personas,
    load_personas,
    persona_cache_path,
    save_personas,
)
from .from_traversal import ScriptedUser, TimingLLM, load_traversal, traversal_to_persona, traversal_to_scripted_user
from .runner import SpeaksUser, run_eval, run_session
from .scorer import ObjectiveBreakdown, run_multi_model, score_report

__all__ = [
    "AuditReport",
    "ScriptedUser",
    "TimingLLM",
    "load_traversal",
    "traversal_to_persona",
    "traversal_to_scripted_user",
    "CorpusGenerator",
    "CorpusSpec",
    "EdgeScenario",
    "EvalCache",
    "EvalReport",
    "ModelScore",
    "MultiModelReport",
    "ObjectiveBreakdown",
    "PersonaSpec",
    "SessionAuditor",
    "SessionMetrics",
    "SpeaksUser",
    "cached_speaker",
    "derive_default_persona",
    "generate_personas",
    "load_personas",
    "persona_cache_path",
    "run_eval",
    "run_multi_model",
    "run_session",
    "save_personas",
    "score_report",
]