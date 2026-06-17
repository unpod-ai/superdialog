# src/superdialog/playbook/eval_bridge.py
"""Backward-compat shim: all logic now lives in superdialog.playbook.eval."""

from .eval.models import EvalReport, PersonaSpec, SessionMetrics
from .eval.runner import SpeaksUser, run_eval, run_session

__all__ = [
    "EvalReport",
    "PersonaSpec",
    "SessionMetrics",
    "SpeaksUser",
    "run_eval",
    "run_session",
]
