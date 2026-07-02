"""ConversationEndpoint over the existing FastAPI /turn adapter (best-effort)."""

from __future__ import annotations

from typing import Any


class HttpEndpoint:
    """Drive an agent served via ``adapters.fastapi.make_router`` (``/turn``).

    The legacy adapter has no greeting route and its ``/reset`` is async, so
    ``start`` returns "" and ``reset`` is a best-effort no-op. This transport is
    secondary — the in-process endpoints are the primary, fully-featured path.
    """

    def __init__(self, base_url: str, client: Any = None) -> None:
        import httpx

        self._client = client or httpx.AsyncClient(base_url=base_url)

    async def start(self) -> str:
        return ""

    async def turn(self, text: str) -> str:
        resp = await self._client.post("/turn", json={"text": text})
        resp.raise_for_status()
        return resp.json().get("reply", "")

    def reset(self) -> None:
        # ponytail: legacy /reset is async; the eval harness uses fresh in-process
        # sessions, so this best-effort no-op is acceptable for the secondary path.
        pass
