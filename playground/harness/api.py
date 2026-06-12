"""Playground backend — FastAPI control plane the web UI calls.

Routes:
  GET  /playground/agents               → catalog
  GET  /playground/config               → config (voices, flows, active LLM)
  POST /playground/sessions             → proxy supervoice POST /connect
  POST /playground/sessions/{id}/control→ live control (set_llm / switch_flow)
  WS   /playground/events               → side-channel: agent events → browser
  GET  /health                          → liveness

The agent (an SDK ``AgentRunner``) runs in-process as an asyncio task and
registers with the remote supervoice referenced by ``SUPERVOICE_URL``. The
browser talks audio directly to supervoice's ``/ws/audio``; transcript / flow /
metrics ride this server's ``/playground/events`` side-channel.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from playground.agents.catalog import CATALOG, get_agent, resolve_agent
from playground.agents.flows import flow_registry
from playground.agents.playbooks import playbook_registry
from playground.harness.control import ControlError, SessionRegistry
from playground.harness.events import EventBus
from playground.harness.playbook_edit import propose_edit
from playground.harness.playbook_store import LocalDraftStore, validate_yaml
from playground.harness.runner import build_runner
from superdialog.llm.resolver import resolve_llm
from unpod._base_url import ws_base

_HERE = Path(__file__).parent
_UI_DIST = _HERE.parent / "web" / "dist"


def _audio_ws_url(supervoice_url: str) -> str:
    """Return the browser audio WebSocket URL for a supervoice ws base."""
    return supervoice_url.rstrip("/") + "/ws/audio"


def _http_base_url(supervoice_url: str) -> str:
    """Convert ws(s):// → http(s):// base URL."""
    base = supervoice_url.rstrip("/")
    if base.startswith("wss://"):
        return "https://" + base[len("wss://") :]
    if base.startswith("ws://"):
        return "http://" + base[len("ws://") :]
    return base


def _connect_http_url(supervoice_url: str) -> str:
    """Return the HTTP(S) /connect URL for a supervoice ws(s) base."""
    return _http_base_url(supervoice_url) + "/connect"


def _edit_llm_uri() -> str:
    """A resolvable, provider-prefixed model URI for the AI builder.

    ``ACTIVE_LLM`` is used when it already names a provider (``openai/…``);
    otherwise pick a sensible default from whichever API key is present.
    """
    uri = os.getenv("ACTIVE_LLM", "")
    if "/" in uri:
        return uri
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-haiku-4-5-20251001"
    return "openai/gpt-4.1-mini"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the in-process AgentRunner for the configured agent."""
    agent_name = os.getenv("AGENT_NAME", "flows")
    spec = get_agent(agent_name)
    if spec is None:
        raise RuntimeError(f"unknown AGENT_NAME {agent_name!r}; see catalog")

    runner = build_runner(
        spec,
        app.state.bus,
        app.state.registry,
        base_url=app.state.supervoice_url,
        api_key=os.getenv("UNPOD_API_KEY", "dev-key"),
    )
    app.state.agent_spec = spec

    async def _run() -> None:
        await asyncio.sleep(1)  # let uvicorn bind first
        try:
            await runner.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[playground] agent exited: {exc}")

    task = asyncio.create_task(_run())
    logger.info(f"[playground] agent={spec.agent_id} → {app.state.supervoice_url}")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def build_app() -> FastAPI:
    """Build the playground FastAPI app."""
    app = FastAPI(title="unpod-playground", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.bus = EventBus()
    app.state.registry = SessionRegistry()
    app.state.store = LocalDraftStore()
    # Speech backend resolution (mirrors examples/browser_playground/run.py):
    # explicit SUPERVOICE_URL wins → else the hosted wss://<UNPOD_BASE_URL>
    # (so a fresh clone needs no local supervoice) → else the local dev service.
    app.state.supervoice_url = (
        os.getenv("SUPERVOICE_URL") or ws_base() or "ws://127.0.0.1:9000"
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/playground/agents")
    async def list_agents() -> dict[str, Any]:
        return {
            "agents": [
                {
                    "name": spec.name,
                    "agent_id": spec.agent_id,
                    "description": spec.description,
                }
                for spec in CATALOG.values()
            ]
        }

    @app.get("/playground/config")
    async def get_config() -> dict[str, Any]:
        llm = "claude-3-haiku" if os.getenv("ANTHROPIC_API_KEY") else "gpt-4.1-mini"
        voice_profiles: list[dict[str, Any]] = []
        try:
            base = _http_base_url(app.state.supervoice_url)
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{base}/voice-profiles")
                if resp.status_code == 200:
                    voice_profiles = [
                        {
                            "id": p["profile_id"],
                            "name": p.get("name", p["profile_id"]),
                            "stt": p.get("stt_provider", ""),
                            "tts": p.get("tts_provider", ""),
                            "description": p.get("description", ""),
                        }
                        for p in resp.json()
                    ]
        except Exception as exc:
            logger.warning(f"[playground] voice-profiles fetch failed: {exc}")
        return {
            "voice_profiles": voice_profiles,
            "flows": [
                {
                    "id": spec.agent_id,
                    "name": spec.name,
                    "description": spec.description,
                }
                for spec in CATALOG.values()
            ],
            "active_llm": os.getenv("ACTIVE_LLM", llm),
        }

    @app.get("/playground/flows")
    async def list_flows() -> dict[str, Any]:
        """Return the conversation flows the worker can switch between."""
        infos = flow_registry()
        return {
            "flows": [
                {
                    "id": f.id,
                    "label": f.label,
                    "nodes": f.nodes,
                    "initial_node": f.initial_node,
                    "description": f.description,
                }
                for f in infos
            ],
            "active": infos[0].id if infos else None,
        }

    @app.get("/playground/playbooks")
    async def list_playbooks() -> dict[str, Any]:
        """Return the runnable playbooks the worker can run (engine='playbook')."""
        infos = playbook_registry()
        return {
            "playbooks": [
                {
                    "id": p.id,
                    "label": p.label,
                    "goal": p.goal,
                    "journeys": p.journeys,
                    "checkpoints": p.checkpoints,
                    "initial": p.initial,
                    "description": p.description,
                    "draft": app.state.store.has_draft(p.id),
                }
                for p in infos
            ],
            "active": infos[0].id if infos else None,
        }

    @app.get("/playground/playbooks/{playbook_id}/source")
    async def get_source(playbook_id: str) -> dict[str, Any]:
        """Return the effective YAML (draft if present, else canonical) + verdict."""
        try:
            text = app.state.store.read(playbook_id)
        except KeyError:
            return {"ok": False, "error": f"unknown playbook {playbook_id!r}"}
        vr = validate_yaml(text)
        return {
            "ok": True,
            "id": playbook_id,
            "yaml": text,
            "draft": app.state.store.has_draft(playbook_id),
            "valid": vr.valid,
            "errors": vr.errors,
            "steps": vr.steps,
            "journey": vr.journey,
        }

    @app.post("/playground/playbooks/{playbook_id}/validate")
    async def validate_source(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Validate a candidate YAML without saving (drives the editor footer)."""
        vr = validate_yaml(str(body.get("yaml", "")))
        return {
            "valid": vr.valid,
            "errors": vr.errors,
            "steps": vr.steps,
            "journey": vr.journey,
        }

    @app.put("/playground/playbooks/{playbook_id}/source")
    async def save_source(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Validate, then persist as a draft (rejected if invalid)."""
        text = str(body.get("yaml", ""))
        vr = validate_yaml(text)
        if not vr.valid:
            return {"ok": False, "errors": vr.errors}
        try:
            app.state.store.save_draft(playbook_id, text)
        except KeyError:
            return {"ok": False, "errors": [f"unknown playbook {playbook_id!r}"]}
        return {"ok": True, "draft": True, "steps": vr.steps, "journey": vr.journey}

    @app.post("/playground/playbooks/{playbook_id}/publish")
    async def publish_source(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Promote the draft (or a supplied YAML) to the canonical example."""
        text = body.get("yaml")
        if text is None:
            try:
                text = app.state.store.read(playbook_id)
            except KeyError:
                return {"ok": False, "errors": [f"unknown playbook {playbook_id!r}"]}
        vr = validate_yaml(str(text))
        if not vr.valid:
            return {"ok": False, "errors": vr.errors}
        try:
            app.state.store.publish(playbook_id, str(text))
        except KeyError:
            return {"ok": False, "errors": [f"unknown playbook {playbook_id!r}"]}
        return {"ok": True, "published": True}

    @app.post("/playground/playbooks/{playbook_id}/edit")
    async def edit_playbook(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Rewrite the playbook from a natural-language instruction via the LLM."""
        instruction = str(body.get("instruction", "")).strip()
        if not instruction:
            return {"ok": False, "error": "instruction is required"}
        current = body.get("yaml")
        if current is None:
            try:
                current = app.state.store.read(playbook_id)
            except KeyError:
                return {"ok": False, "error": f"unknown playbook {playbook_id!r}"}
        try:
            provider = resolve_llm(_edit_llm_uri())
            proposal = await propose_edit(str(current), instruction, provider.complete)
        except Exception as exc:  # noqa: BLE001 — surface any LLM/resolve failure
            logger.warning(f"[playground] edit failed: {exc}")
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "yaml": proposal.yaml,
            "summary": proposal.summary,
            "valid": proposal.valid,
            "errors": proposal.errors,
            "steps": proposal.steps,
            "journey": proposal.journey,
        }

    @app.post("/playground/sessions")
    async def create_session(
        agent: str | None = None,
        voice_profile_id: str | None = None,
        mode: str | None = None,
        flow: str | None = None,
        playbook: str | None = None,
    ) -> dict[str, Any]:
        """Proxy supervoice /connect, return a transport descriptor.

        ``mode`` selects the conversation engine — ``"playbook"`` runs the chosen
        ``playbook`` file, ``"flow"`` (default) runs the selected ``flow``. The
        mode + file id ride the audio socket query so the worker builds the right
        engine for the call (see ``harness/runner.py``).
        """
        spec = resolve_agent(agent) if agent else app.state.agent_spec
        if spec is None:
            return {"error": f"unknown agent {agent!r}"}
        # Build the shared call metadata once: agent routing + engine selection.
        meta: dict[str, str] = {"agent_id": spec.agent_id}
        if voice_profile_id:
            meta["voice_profile_id"] = voice_profile_id
        if mode:
            meta["mode"] = mode
        if flow:
            meta["flow"] = flow
        if playbook:
            meta["playbook"] = playbook
        target = _connect_http_url(app.state.supervoice_url)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(target, params=meta)
            resp.raise_for_status()
        data = dict(resp.json())
        # Carry the same metadata on the audio socket so the dev speech service
        # routes the call to this worker and builds the chosen engine + file.
        data["ws_url"] = f"{_audio_ws_url(app.state.supervoice_url)}?{urlencode(meta)}"
        data["transport"] = "ws"
        data["agent"] = spec.name
        return data

    @app.post("/playground/sessions/{session_id}/control")
    async def control(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        action = body.get("action", "")
        params = body.get("params", {})
        try:
            app.state.registry.apply(session_id, action, params)
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}
        # Reflect the post-switch node to the UI immediately (switch_flow resets
        # the machine to the new flow's initial node — no turn fires on its own).
        if action == "switch_flow":
            try:
                adapter = app.state.registry.resolve(session_id).dialog_machine
                state = getattr(adapter, "state", None)
                if isinstance(state, dict):
                    app.state.bus.publish("flow_node_changed", **state)
            except Exception:  # best-effort echo; never fail the control call
                pass
        return {"ok": True}

    @app.websocket("/playground/events")
    async def events(ws: WebSocket) -> None:
        """Side-channel: stream agent-process events to the browser."""
        await ws.accept()
        bus: EventBus = app.state.bus
        with bus.subscribe() as queue:
            try:
                while True:
                    message = await queue.get()
                    await ws.send_json(message)
            except WebSocketDisconnect:
                pass

    @app.websocket("/{_path:path}")
    async def reject_unknown_ws(ws: WebSocket) -> None:
        """Cleanly close any websocket that matches no real WS route.

        The SPA is served by a ``StaticFiles`` mount at ``/`` (below), which
        asserts an HTTP scope. A websocket on any unrouted path would otherwise
        fall through to that mount and crash the ASGI app with ``AssertionError``.
        Real WS routes (``/playground/events``) are declared above and still win;
        this only catches strays (e.g. a misdirected audio socket — audio is
        meant to hit supervoice's ``/ws/audio`` directly, not the harness).
        """
        logger.debug(f"[playground] closing unmatched websocket: {ws.url.path}")
        await ws.close(code=1000)

    if _UI_DIST.exists():
        app.mount("/", StaticFiles(directory=_UI_DIST, html=True), name="ui")
    else:
        logger.warning("[playground] web/dist not found — run the Vite build")

    return app
