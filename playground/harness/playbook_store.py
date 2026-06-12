"""Validation + persistence for playground playbooks.

Persistence sits behind a small seam: :class:`LocalDraftStore` writes a
git-ignored drafts overlay now; a future ``RemotePlaybookStore`` will target the
speech-service API under the user's account. Routes and the UI depend only on
this module, never on the backing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml
from pydantic import ValidationError
from superdialog.playbook.editable import make_editable


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a playbook YAML string."""

    valid: bool
    errors: list[str]
    steps: int
    journey: str


def _format_errors(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
    return [str(exc)]


def validate_yaml(text: str) -> ValidationResult:
    """Parse + validate ``text`` via superdialog; never raises."""
    try:
        playbook = make_editable(text).compile()
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        return ValidationResult(False, _format_errors(exc), 0, "")
    journey = next(iter(playbook.journeys), "")
    steps = len(playbook.journeys[journey].checkpoints) if journey else 0
    return ValidationResult(True, [], steps, journey)
