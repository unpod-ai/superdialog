"""validate_yaml + LocalDraftStore.

Run with: ``uv run --extra playground pytest tests/playground/test_playbook_store.py``
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("loguru")
pytest.importorskip("unpod")

from playground.agents.playbooks import canonical_path, playbook_registry
from playground.harness.playbook_store import LocalDraftStore, validate_yaml


def _valid_text() -> str:
    infos = playbook_registry()
    return canonical_path(infos[0].id).read_text(encoding="utf-8")


def _seed(tmp_path, monkeypatch):
    pbdir = tmp_path / "pb"
    pbdir.mkdir()
    draftdir = tmp_path / "drafts"
    text = _valid_text()  # read from the default registry before env is patched
    (pbdir / "demo.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("PLAYBOOKS_DIR", str(pbdir))
    monkeypatch.setenv("PLAYBOOK_DRAFTS_DIR", str(draftdir))
    return pbdir, draftdir, text


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


def test_save_publish_roundtrip(tmp_path, monkeypatch):
    pbdir, draftdir, text = _seed(tmp_path, monkeypatch)
    store = LocalDraftStore()

    assert store.has_draft("demo") is False
    assert store.read("demo") == text

    store.save_draft("demo", text)
    assert store.has_draft("demo") is True
    assert (draftdir / "demo.yaml").exists()
    assert (pbdir / "demo.yaml").read_text(encoding="utf-8") == text  # canonical intact

    store.publish("demo", text)
    assert store.has_draft("demo") is False  # draft cleared
    assert (pbdir / "demo.yaml").read_text(
        encoding="utf-8"
    ) == text  # canonical written


def test_unknown_id_raises(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    store = LocalDraftStore()
    with pytest.raises(KeyError):
        store.read("nope")
    with pytest.raises(KeyError):
        store.publish("nope", "x")
