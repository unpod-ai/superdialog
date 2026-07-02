"""Remote ConversationEndpoint clients driven over httpx ASGITransport.

No network, no real LLM: the in-process OpenAI-compatible server is mounted
onto an ASGI transport so the OpenAI-compatible client speaks to it directly.
"""

import httpx
from httpx import ASGITransport

from superdialog.eval.endpoints.base import ConversationEndpoint
from superdialog.eval.endpoints.http import HttpEndpoint
from superdialog.eval.endpoints.openai_client import OpenAICompatEndpoint
from superdialog.eval.server.openai_server import build_app
from tests.eval.fakes import FakeProvider


async def test_openai_client_drives_server(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "superdialog.llm.resolver.resolve_llm",
        lambda uri: FakeProvider({"BEGIN": "Welcome!", "*": "noted"}),
    )
    pb = tmp_path / "booking.yaml"
    pb.write_text("PERSONA: bot")
    app = build_app(str(pb), agent_model="x")
    client = httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    ep = OpenAICompatEndpoint(
        "http://test", "superdialog-vanilla/booking", "s1", client=client
    )
    assert isinstance(ep, ConversationEndpoint)
    assert "Welcome" in await ep.start()
    assert await ep.turn("hi") == "noted"
    ep.reset()
    assert isinstance(await ep.turn("again"), str)
    await client.aclose()


async def test_http_endpoint_hits_turn():
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/turn")
    async def turn(body: dict) -> dict:
        return {"reply": f"echo:{body['text']}"}

    client = httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    ep = HttpEndpoint("http://test", client=client)
    assert isinstance(ep, ConversationEndpoint)
    assert await ep.start() == ""
    assert await ep.turn("hi") == "echo:hi"
    ep.reset()
    await client.aclose()
