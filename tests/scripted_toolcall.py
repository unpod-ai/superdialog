"""Deterministic LLM provider speaking the ToolCallAdapter's tool-call protocol.

The flow engine's default ``ToolCallAdapter`` decides edges via tool-calling
(``result.tool_calls``), not JSON text. These tests inject a scripted provider
so they run hermetically (no API key, no network, no model nondeterminism).

A turn drives two kinds of provider call:

* **routing** — ``complete(messages, tools=...)``: returns a tool-call naming the
  chosen edge id (or ``__stay_on_node__``), with slots/brief in the arguments.
* **generation** — ``complete(messages)`` (no tools): returns reply text.

Both queues are lenient: an exhausted routing queue defaults to *stay* (so a
silent auto-chain pass never spuriously transitions), and an exhausted reply
queue returns ``"ok"``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from superdialog.llm.provider import CompletionResult, StreamChunk


def route(edge: str | None, *, slots: dict[str, Any] | None = None, brief: str = ""):
    """One scripted routing decision: an edge id (transition) or ``None`` (stay)."""
    return {"edge": edge, "slots": slots or {}, "brief": brief}


class ScriptedToolProvider:
    """Scripted ``LLMProvider`` for the tool-calling flow adapter."""

    def __init__(
        self,
        routes: list[dict[str, Any]] | None = None,
        replies: list[str] | None = None,
    ) -> None:
        self._routes = list(routes or [])
        self._replies = list(replies or [])
        self.calls: list[list[dict[str, Any]]] = []
        self.routing_messages: list[list[dict[str, Any]]] = []
        self.reply_messages: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> CompletionResult:
        self.calls.append(messages)
        if tools:
            self.routing_messages.append(messages)
            return self._route()
        self.reply_messages.append(messages)
        text = self._replies.pop(0) if self._replies else "ok"
        return CompletionResult(
            text=text,
            tool_calls=[],
            metadata={"prompt_tokens": 0, "completion_tokens": 0},
        )

    def _route(self) -> CompletionResult:
        spec = self._routes.pop(0) if self._routes else route(None)
        if spec["edge"] is None:
            name = "__stay_on_node__"
            args = json.dumps({"brief_response": spec["brief"]})
        else:
            name = spec["edge"]
            args = json.dumps(spec["slots"])
        return CompletionResult(
            text="",
            tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
            metadata={"prompt_tokens": 0, "completion_tokens": 0},
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamChunk]:
        result = await self.complete(messages, tools=tools, **opts)
        yield StreamChunk(text=result.text, tool_call_delta=None, done=True)


__all__ = ["ScriptedToolProvider", "route"]
