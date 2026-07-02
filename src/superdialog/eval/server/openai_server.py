"""Expose a loaded playbook as an OpenAI-compatible chat endpoint.

Two model ids per playbook: ``superdialog/<name>`` (playbook engine) and
``superdialog-vanilla/<name>`` (playbook-as-text baseline). Sessions live
server-side, keyed by the OpenAI ``user`` field — a fresh id starts a new
conversation, which is exactly how the eval harness drives it.

NOTE: this module deliberately does NOT use ``from __future__ import
annotations``. FastAPI resolves route-handler annotations against the module
globals only, so a stringized ``req: ChatRequest`` (a class local to
``build_app``) would fail to resolve and degrade the body param to a missing
query param (HTTP 422). Eager annotations let the closure bind the real class.
"""

import json
import os
from collections import OrderedDict
from typing import Any

from superdialog.eval.endpoints.base import ConversationEndpoint


def build_app(
    playbook_path: str,
    *,
    agent_model: str,
    director_model: str | None = None,
    talker_model: str | None = None,
    max_sessions: int = 256,
) -> Any:
    """Build a FastAPI app serving ``playbook_path`` as OpenAI-compatible models."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    from superdialog.eval.endpoints.in_process import (
        InProcessPlaybook,
        InProcessVanilla,
    )
    from superdialog.llm.resolver import resolve_llm

    with open(playbook_path, encoding="utf-8") as fh:
        playbook_text = fh.read()
    name = os.path.splitext(os.path.basename(playbook_path))[0]
    sessions: OrderedDict[tuple[str, str], ConversationEndpoint] = OrderedDict()

    def _make(mode: str) -> ConversationEndpoint:
        if mode == "vanilla":
            return InProcessVanilla(playbook_text, resolve_llm(agent_model))
        return InProcessPlaybook(
            playbook_path,
            agent_model=agent_model,
            director_model=director_model,
            talker_model=talker_model,
        )

    class ChatMessage(BaseModel):
        role: str
        content: str = ""

    class ChatRequest(BaseModel):
        model: str
        messages: list[ChatMessage] = []
        user: str | None = None
        stream: bool = False

    app = FastAPI()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": f"superdialog/{name}",
                    "object": "model",
                    "owned_by": "superdialog",
                },
                {
                    "id": f"superdialog-vanilla/{name}",
                    "object": "model",
                    "owned_by": "superdialog",
                },
            ],
        }

    async def _reply(req: "ChatRequest") -> str:
        mode = "vanilla" if req.model.startswith("superdialog-vanilla") else "playbook"
        sid = req.user or "default"
        key = (mode, sid)
        greeting: str | None = None
        if key not in sessions:
            endpoint = _make(mode)
            sessions[key] = endpoint
            greeting = await endpoint.start()
            while len(sessions) > max_sessions:
                sessions.popitem(last=False)
        endpoint = sessions[key]
        sessions.move_to_end(key)
        users = [m for m in req.messages if m.role == "user"]
        if users:
            return await endpoint.turn(users[-1].content)
        return greeting if greeting is not None else ""

    def _completion(model: str, content: str) -> dict[str, Any]:
        return {
            "id": "chatcmpl-superdialog",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _sse(model: str, content: str) -> Any:
        first = {
            "id": "chatcmpl-superdialog",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(first)}\n\n"
        done = {
            "id": "chatcmpl-superdialog",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest) -> Any:
        content = await _reply(req)
        if req.stream:
            return StreamingResponse(
                _sse(req.model, content), media_type="text/event-stream"
            )
        return _completion(req.model, content)

    return app
