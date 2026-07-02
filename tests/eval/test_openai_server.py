"""Sync FastAPI TestClient tests for the OpenAI-compatible playbook server.

The vanilla model id is used so ``InProcessVanilla`` (which accepts the
``FakeProvider`` returned by the patched ``resolve_llm``) is exercised without
needing a real playbook engine.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.eval.fakes import FakeProvider


def _write_playbook(tmp_path: Path) -> str:
    pb = tmp_path / "booking.yaml"
    pb.write_text("PERSONA: bot", encoding="utf-8")
    return str(pb)


def _build(monkeypatch, tmp_path, script):
    """Patch resolve_llm to a FakeProvider, then build the app."""
    monkeypatch.setattr(
        "superdialog.llm.resolver.resolve_llm",
        lambda uri: FakeProvider(script),
    )
    from superdialog.eval.server.openai_server import build_app

    return build_app(_write_playbook(tmp_path), agent_model="x")


def test_list_models(monkeypatch, tmp_path):
    """Both playbook and vanilla model ids are advertised."""
    app = _build(monkeypatch, tmp_path, {"*": "x"})
    resp = TestClient(app).get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert "superdialog/booking" in ids
    assert "superdialog-vanilla/booking" in ids


def test_vanilla_turn_returns_reply(monkeypatch, tmp_path):
    """A new session greets, then the user-turn reply is what's returned."""
    app = _build(monkeypatch, tmp_path, {"BEGIN": "Welcome!", "*": "noted"})
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "superdialog-vanilla/booking",
            "user": "s1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "noted"


def test_session_state_held_server_side(monkeypatch, tmp_path):
    """The same ``user`` reuses a server-side session; empty messages greet."""
    app = _build(monkeypatch, tmp_path, {"BEGIN": "Welcome!", "*": "noted"})
    client = TestClient(app)
    body = {
        "model": "superdialog-vanilla/booking",
        "user": "s2",
        "messages": [{"role": "user", "content": "hi"}],
    }
    first = client.post("/v1/chat/completions", json=body)
    second = client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200
    assert second.json()["choices"][0]["message"]["content"] == "noted"

    greet = client.post(
        "/v1/chat/completions",
        json={
            "model": "superdialog-vanilla/booking",
            "user": "fresh",
            "messages": [],
        },
    )
    assert greet.json()["choices"][0]["message"]["content"] == "Welcome!"


def test_streaming_sse(monkeypatch, tmp_path):
    """``stream=True`` emits SSE chunks carrying the reply then ``[DONE]``."""
    app = _build(monkeypatch, tmp_path, {"BEGIN": "Welcome!", "*": "noted"})
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "superdialog-vanilla/booking",
            "user": "s3",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    assert "chat.completion.chunk" in resp.text
    assert "noted" in resp.text
    assert "data: [DONE]" in resp.text
