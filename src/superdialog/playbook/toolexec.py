"""Execute ToolSpecs: template rendering, run_once/when policies, env_updates.

Tool templates render over {slots, env, results}. Unlike the Talker renderer,
env IS visible here: tools run Director-side and their output is never shown
to the Talker. Templates still come from playbook artifacts, so rendering is
sandboxed and template errors degrade to a failed ToolResultEvent.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from jinja2 import TemplateError, Undefined
from jinja2.sandbox import SandboxedEnvironment

from .events import EnvWriteEvent, Event, ToolCallEvent, ToolResultEvent
from .expr import ExprError, evaluate
from .models import SlotSpec, ToolSpec
from .state import ConversationState

HttpFn = Callable[..., Awaitable[tuple[int, Any]]]

# Sandboxed: tool templates are playbook artifacts (optimizer-generated), so
# attribute-walking SSTI payloads must be blocked, not executed.
_jinja = SandboxedEnvironment(undefined=Undefined, autoescape=False)

_CASTS: dict[str, Callable[[Any], Any]] = {
    "int": int,
    "float": float,
    "bool": lambda v: str(v).lower() in ("1", "true", "yes"),
    "str": str,
}


class PythonToolFn(Protocol):
    async def __call__(self, args: dict[str, Any], state: ConversationState) -> Any: ...


def _template_ns(state: ConversationState) -> dict[str, Any]:
    return {
        "slots": {k: v.value for k, v in state.slots.items()},
        "env": dict(state.env),
        "results": {
            k: {"ok": r.ok, "status": r.status, "data": r.data, "error": r.error}
            for k, r in state.tool_results.items()
        },
    }


def _render(template: str, ns: dict[str, Any]) -> str:
    return _jinja.from_string(template).render(**ns)


_SECRET_KEY_RE = re.compile(
    r"secret|token|password|passwd|api[_-]?key|auth|credential|bearer|jwt"
    r"|signature|private[_-]?key|access[_-]?key|otp|pin",
    re.IGNORECASE,
)


def _redact(value: Any, key: str | None = None) -> Any:
    """Mask secret-like keys (recursively) before event-log recording."""
    if key is not None and _SECRET_KEY_RE.search(key):
        return "***"
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _redact_url(url: str) -> str:
    """Strip userinfo and mask secret-like query params before recording."""
    parts = urlsplit(url)
    netloc = parts.netloc.rsplit("@", 1)[-1]  # drop user:pass@ if present
    query = urlencode(
        [
            (k, "***" if _SECRET_KEY_RE.search(k) else v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
        ],
        safe="*",  # keep the mask literal, not %2A%2A%2A
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _dig(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def coerce_args(args: dict[str, Any], specs: dict[str, SlotSpec]) -> dict[str, Any]:
    """Cast incoming arg values to their declared SlotSpec types."""
    out = dict(args)
    for key, spec in specs.items():
        if key in out and spec.type in _CASTS:
            out[key] = _CASTS[spec.type](out[key])
    return out


class ToolExecutor:
    """Run a ToolSpec against state and return the events to append."""

    def __init__(
        self, http: HttpFn, python_tools: dict[str, PythonToolFn] | None = None
    ) -> None:
        self._http = http
        self._python_tools = python_tools or {}

    async def execute(
        self,
        spec: ToolSpec,
        state: ConversationState,
        args: dict[str, Any] | None = None,
    ) -> list[Event]:
        """Execute ``spec``; returns [] when run_once/when policies skip it."""
        if spec.run_once and state.tool_call_counts.get(spec.id, 0) > 0:
            return []
        if spec.when:
            try:
                if not evaluate(spec.when, state):
                    return []
            except ExprError:
                return []
        if args and spec.args:
            try:
                args = coerce_args(args, spec.args)
            except (TypeError, ValueError) as exc:
                return [
                    ToolCallEvent(tool=spec.id, args=args or {}),
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=False,
                        error=f"bad args: {exc}",
                    ),
                ]
        ns = _template_ns(state)
        events: list[Event] = []
        if spec.type == "python":
            fn = self._python_tools.get(spec.id)
            if fn is None:
                return [
                    ToolCallEvent(tool=spec.id, args=args or {}),
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=False,
                        error=f"python tool not registered: {spec.id}",
                    ),
                ]
            events.append(ToolCallEvent(tool=spec.id, args=args or {}))
            try:
                data = await fn(args or {}, state)
                events.append(
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=True,
                        data=data,
                    )
                )
            except Exception as exc:  # tool failure is data, not a crash
                events.append(
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            return events

        try:
            url = _render(spec.url, ns)
            headers = {k: _render(v, ns) for k, v in spec.headers.items()}
            body: Any = {
                k: _render(v, ns) if isinstance(v, str) else v
                for k, v in spec.body.items()
            } or None
        except TemplateError as exc:
            # Bad template (authoring typo or sandbox SecurityError) must not
            # crash the Director: record the attempt and a failed result.
            return [
                ToolCallEvent(tool=spec.id, args=args or {}),
                ToolResultEvent(
                    tool=spec.id,
                    store_as=spec.store_response_as,
                    ok=False,
                    error=f"template error: {exc}",
                ),
            ]
        # A compiler '_template' body is one whole Jinja-in-JSON document:
        # the rendered text IS the request body, so parse it into the real
        # structure (posting {"_template": "..."} literally would hand the
        # API a string instead of fields).
        if (
            isinstance(body, dict)
            and set(spec.body) == {"_template"}
            and isinstance(body.get("_template"), str)
        ):
            try:
                body = json.loads(body["_template"])
            except ValueError:
                return [
                    ToolCallEvent(tool=spec.id, args=args or {}),
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=False,
                        error="template body not valid JSON",
                    ),
                ]
        # Record a redacted url/body in the event log; the real url and body
        # still go to http. EnvWriteEvent values stay raw: env is never
        # rendered to the Talker, and export-time redaction is a later-task
        # concern.
        events.append(
            ToolCallEvent(
                tool=spec.id,
                args={"url": _redact_url(url), "body": _redact(body or {})},
            )
        )
        # Terminal trace of the REAL rendered request (event log url is redacted).
        # Dev visibility for tool/API calls during pg-stack runs.
        print(
            f"[tool] → {spec.id} {spec.method} {url}"
            + (f" body={body}" if body else ""),
            flush=True,
        )
        try:
            status, data = await self._http(
                method=spec.method,
                url=url,
                headers=headers,
                body=body,
                timeout=spec.timeout,
            )
        except Exception as exc:
            print(f"[tool] ✗ {spec.id} ERROR {type(exc).__name__}: {exc}", flush=True)
            events.append(
                ToolResultEvent(
                    tool=spec.id,
                    store_as=spec.store_response_as,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return events
        ok = 200 <= status < 300
        print(
            f"[tool] ← {spec.id} {status} {'ok' if ok else 'FAIL'} "
            f"→{spec.store_response_as or '-'} {str(data)[:300]}",
            flush=True,
        )
        result = ToolResultEvent(
            tool=spec.id,
            store_as=spec.store_response_as,
            ok=ok,
            status=status,
            data=data,
            error=None if ok else str(data),
        )
        events.append(result)
        if ok:
            for env_key, path in spec.env_updates.items():
                value = _dig(data, path)
                if value is not None:
                    events.append(EnvWriteEvent(key=env_key, value=str(value)))
        return events


async def httpx_http(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: Any,
    timeout: float,
) -> tuple[int, Any]:
    """Production HTTP callable backed by httpx."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(method, url, headers=headers, json=body)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"text": resp.text}
