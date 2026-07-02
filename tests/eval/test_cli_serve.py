"""`eval serve` builds the OpenAI-compatible app and hands it to uvicorn.

``uvicorn.run`` is patched to a capture so the server never binds a port; the
captured app is asserted to expose the ``/v1/chat/completions`` route.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from superdialog.eval.cli import cmd_serve
from tests.eval.fakes import FakeProvider


def test_cmd_serve_builds_and_launches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "superdialog.llm.resolver.resolve_llm",
        lambda uri: FakeProvider({"*": "noted"}),
    )
    captured: dict[str, Any] = {}

    def capture(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", capture)

    pb = tmp_path / "booking.yaml"
    pb.write_text("PERSONA: bot", encoding="utf-8")
    ns = argparse.Namespace(
        playbook=str(pb),
        agent_model="x",
        director_model=None,
        talker_model=None,
        port=8123,
    )

    assert cmd_serve(ns) == 0
    app = captured["app"]
    assert any(r.path == "/v1/chat/completions" for r in app.routes)
    assert captured["kwargs"]["port"] == 8123
