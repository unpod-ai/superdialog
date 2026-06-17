# src/superdialog/playbook/personas.py
"""Backward-compat shim: all logic now lives in superdialog.playbook.eval.personas."""

from .eval.personas import (
    derive_default_persona,
    generate_personas,
    load_personas,
    persona_cache_path,
    save_personas,
)

__all__ = [
    "derive_default_persona",
    "generate_personas",
    "load_personas",
    "persona_cache_path",
    "save_personas",
]
