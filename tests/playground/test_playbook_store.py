"""validate_yaml + LocalDraftStore.

Run with: ``uv run --extra playground pytest tests/playground/test_playbook_store.py``
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("loguru")
pytest.importorskip("unpod")

from playground.agents.playbooks import canonical_path, playbook_registry
from playground.harness.playbook_store import validate_yaml


def _valid_text() -> str:
    infos = playbook_registry()
    return canonical_path(infos[0].id).read_text(encoding="utf-8")


def test_validate_accepts_a_real_playbook():
    vr = validate_yaml(_valid_text())
    assert vr.valid is True
    assert vr.errors == []
    assert vr.steps >= 1
    assert vr.journey  # non-empty journey name


def test_validate_rejects_broken_yaml():
    vr = validate_yaml("this: [is, not, a, playbook")  # unclosed flow seq
    assert vr.valid is False
    assert vr.errors


def test_validate_rejects_empty_playbook():
    vr = validate_yaml("goal: hi\nplaybook: []\n")  # min_length violation
    assert vr.valid is False
    assert vr.errors
