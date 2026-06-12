"""Playbook source/validate/save/publish/edit endpoints.

Run with: ``uv run --extra playground pytest tests/playground/test_playbook_endpoints.py``
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("loguru")
pytest.importorskip("unpod")

from starlette.testclient import TestClient

from playground.agents.playbooks import canonical_path, playbook_registry


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pbdir = tmp_path / "pb"
    pbdir.mkdir()
    text = canonical_path(playbook_registry()[0].id).read_text(encoding="utf-8")
    (pbdir / "demo.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("PLAYBOOKS_DIR", str(pbdir))
    monkeypatch.setenv("PLAYBOOK_DRAFTS_DIR", str(tmp_path / "drafts"))

    # Fake the LLM so /edit needs no keys/network.
    reply = f"Tweaked it.\n```yaml\n{text}\n```"

    class _Res:
        def __init__(self, body: str) -> None:
            self.text = body

    class _Provider:
        async def complete(self, messages, *a, **k):
            return _Res(reply)

    import playground.harness.api as api_mod

    monkeypatch.setattr(api_mod, "resolve_llm", lambda uri: _Provider())

    from playground.harness.api import build_app

    with TestClient(build_app()) as c:
        yield c, text


def test_get_source(client):
    c, text = client
    r = c.get("/playground/playbooks/demo/source")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["yaml"] == text and body["valid"] is True
    assert body["draft"] is False and body["steps"] >= 1


def test_get_source_unknown(client):
    c, _ = client
    assert c.get("/playground/playbooks/nope/source").json()["ok"] is False


def test_validate(client):
    c, _ = client
    r = c.post("/playground/playbooks/demo/validate", json={"yaml": "playbook: []"})
    body = r.json()
    assert body["valid"] is False and body["errors"]


def test_save_then_publish(client):
    c, text = client
    assert c.put("/playground/playbooks/demo/source", json={"yaml": text}).json()["ok"]
    assert c.get("/playground/playbooks/demo/source").json()["draft"] is True
    assert c.post("/playground/playbooks/demo/publish", json={}).json()["ok"]
    assert c.get("/playground/playbooks/demo/source").json()["draft"] is False


def test_save_rejects_invalid(client):
    c, _ = client
    body = c.put(
        "/playground/playbooks/demo/source", json={"yaml": "playbook: []"}
    ).json()
    assert body["ok"] is False and body["errors"]


def test_edit(client):
    c, _ = client
    body = c.post(
        "/playground/playbooks/demo/edit", json={"instruction": "tweak it"}
    ).json()
    assert body["ok"] and body["summary"] == "Tweaked it." and body["valid"] is True


def test_edit_requires_instruction(client):
    c, _ = client
    assert (
        c.post("/playground/playbooks/demo/edit", json={"instruction": ""}).json()["ok"]
        is False
    )
