"""Draft-precedence path helpers in playbooks.py.

Run with: ``uv run --extra playground pytest tests/playground/test_playbook_paths.py``
"""

from __future__ import annotations

from pathlib import Path

from playground.agents import playbooks as pb


def _seed(pbdir: Path) -> str:
    """Copy a real, known-valid example playbook into pbdir as demo.yaml."""
    from playground.agents.playbooks import canonical_path, playbook_registry

    infos = playbook_registry()
    text = canonical_path(infos[0].id).read_text(encoding="utf-8")
    (pbdir / "demo.yaml").write_text(text, encoding="utf-8")
    return text


def test_effective_path_prefers_draft(tmp_path, monkeypatch):
    pbdir = tmp_path / "pb"
    pbdir.mkdir()
    draftdir = tmp_path / "drafts"
    text = _seed(pbdir)
    monkeypatch.setenv("PLAYBOOKS_DIR", str(pbdir))
    monkeypatch.setenv("PLAYBOOK_DRAFTS_DIR", str(draftdir))

    # No draft yet → effective == canonical.
    assert pb.effective_path("demo") == pbdir / "demo.yaml"
    assert pb.canonical_path("demo") == pbdir / "demo.yaml"
    assert pb.draft_path("demo") == draftdir / "demo.yaml"

    # Write a draft → effective switches to it.
    draftdir.mkdir(parents=True, exist_ok=True)
    (draftdir / "demo.yaml").write_text(text, encoding="utf-8")
    assert pb.effective_path("demo") == draftdir / "demo.yaml"

    # Unknown id → None.
    assert pb.canonical_path("nope") is None
    assert pb.effective_path("nope") is None
