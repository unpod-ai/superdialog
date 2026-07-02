"""OpenAI-compatible ConversationEndpoint client (drives an `eval serve` server)."""

from __future__ import annotations

from typing import Any


class OpenAICompatEndpoint:
    """Drive a superdialog OpenAI-compatible server as a ConversationEndpoint.

    Session state lives server-side, keyed by the OpenAI ``user`` field; ``reset``
    rotates the id so the next turn starts a fresh server session.
    """

    def __init__(
        self, base_url: str, model: str, session_id: str, client: Any = None
    ) -> None:
        import httpx

        self._model = model
        self._base_session = session_id
        self._session = session_id
        self._rotations = 0
        self._client = client or httpx.AsyncClient(base_url=base_url)

    async def _post(self, messages: list[dict[str, str]]) -> str:
        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model,
                "user": self._session,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def start(self) -> str:
        return await self._post([])

    async def turn(self, text: str) -> str:
        return await self._post([{"role": "user", "content": text}])

    def reset(self) -> None:
        self._rotations += 1
        self._session = f"{self._base_session}-{self._rotations}"
