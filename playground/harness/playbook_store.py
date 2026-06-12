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

from playground.agents.playbooks import canonical_path, draft_path, effective_path


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


class LocalDraftStore:
    """Drafts overlay on the local filesystem (the seam's local impl).

    ``read`` prefers a draft; ``save_draft`` writes the overlay; ``publish``
    writes the canonical example and clears the draft. All path resolution is
    delegated to :mod:`playground.agents.playbooks` so the runner and this store
    agree on what "the playbook" is.
    """

    def has_draft(self, playbook_id: str) -> bool:
        return draft_path(playbook_id).exists()

    def read(self, playbook_id: str) -> str:
        path = effective_path(playbook_id)
        if path is None:
            raise KeyError(playbook_id)
        return path.read_text(encoding="utf-8")

    def save_draft(self, playbook_id: str, text: str) -> None:
        if canonical_path(playbook_id) is None:
            raise KeyError(playbook_id)
        dp = draft_path(playbook_id)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(text, encoding="utf-8")

    def publish(self, playbook_id: str, text: str) -> None:
        cp = canonical_path(playbook_id)
        if cp is None:
            raise KeyError(playbook_id)
        cp.write_text(text, encoding="utf-8")
        dp = draft_path(playbook_id)
        if dp.exists():
            dp.unlink()
