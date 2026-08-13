"""Execute ToolSpecs: template rendering, run_once/when policies, env_updates.

Tool templates render over {slots, env, results}. Unlike the Talker renderer,
env IS visible here: tools run Director-side and their output is never shown
to the Talker. Templates still come from playbook artifacts, so rendering is
sandboxed and template errors degrade to a failed ToolResultEvent.

Retry amplification ceiling: a pathological step can chain the author-level
pipeline RetrySpec (<=11 rounds) x middleware replay (<=3 executions/step) x
transport retries (<=3 attempts) = up to ~99 HTTP attempts. Acceptable without
an extra cap because transport retries fire only on connection-level exceptions
(a dead host fails fast) and timeouts are never retried.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import anyio
from jinja2 import TemplateError, Undefined
from jinja2.sandbox import SandboxedEnvironment

from ._canon import canonical_json
from ._ssrf import validate_url
from .events import (
    EnvWriteEvent,
    Event,
    SlotWriteEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .expr import ExprError, evaluate
from .models import SAFE_HTTP_METHODS, SlotSpec, ToolSpec
from .state import ConversationState

HttpFn = Callable[..., Awaitable[tuple[int, Any]]]

_log = logging.getLogger(__name__)

# Per-tool timeout is clamped at execute time (not on the model) so a persisted
# playbook with an out-of-range `timeout:` still loads and merely runs clamped.
_MIN_TIMEOUT_S = 0.1
_MAX_TIMEOUT_S = 300.0

# Transport retry: raised connection-level exceptions only (see execute()); the
# author-level pipeline RetrySpec sits above this, so we never retry a non-2xx
# status here. Two retries, exponential backoff + small jitter.
_TRANSPORT_RETRIES = 2
_RETRY_BASE_S = 0.5

# Tool responses fold into ConversationState and serialize into every traversal
# export, so an unbounded body is a memory/log blowout. Capped in the production
# HTTP callable (httpx_http), enforced during the streamed read.
_MAX_RESPONSE_BYTES = 1 * 1024 * 1024  # 1 MiB

# Sandboxed: tool templates are playbook artifacts (optimizer-generated), so
# attribute-walking SSTI payloads must be blocked, not executed.
_jinja = SandboxedEnvironment(undefined=Undefined, autoescape=False)


def _backoff(attempt: int) -> float:
    """Exponential backoff (capped 5s) + up to 60ms jitter, per attempt (0-based)."""
    base = min(_RETRY_BASE_S * (2**attempt), 5.0)
    return base + random.uniform(0, 0.06)


def _is_timeout(exc: BaseException) -> bool:
    """True for a request timeout (terminal — retrying only multiplies the wait).

    Name-tolerant so it catches ``httpx.TimeoutException`` (which does NOT
    subclass the builtin ``TimeoutError``) without importing httpx here — the
    HTTP callable is injected, so the concrete exception type is host-defined.
    """
    return isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__


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
        # Attempt count per tool id; mirrors render.template_namespace so a
        # url/body template and its guidance see the same signal. A tool
        # skipped by its own `when:` guard leaves no ToolResultEvent, so
        # `results` alone cannot tell "returned nothing" from "never ran".
        "calls": defaultdict(int, state.tool_call_counts),
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
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
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


def _field_update_events(spec: ToolSpec, data: Any) -> list[Event]:
    """``env_updates``/``slot_updates`` events for a successful ``data`` payload.

    Shared by both the ``type: http`` and ``type: python`` execution paths —
    a python tool that never sets either mapping is unaffected (empty dicts,
    empty return), so this is purely additive: a python tool can now
    deterministically resolve a slot/env value (e.g. picking a candidate id
    out of data it already fetched) without needing a second, redundant
    tool call whose only purpose was to get onto the http path that used to
    be the only place these updates applied.
    """
    events: list[Event] = []
    for env_key, path in spec.env_updates.items():
        value = _dig(data, path)
        if value is not None:
            events.append(EnvWriteEvent(key=env_key, value=str(value)))
        else:
            # Path missed the response shape (e.g. `data.x` against a
            # flat body). env stays unset and downstream tools render
            # an empty header/arg — silently. Surface it.
            _log.warning(
                "[tool] ⚠ %s env_updates '%s': path '%s' not found "
                "in response — env unset",
                spec.id,
                env_key,
                path,
            )
    # slot_updates: write a resolved value straight into a slot (e.g. a
    # name->id lookup the Director can't resolve itself). Applied to the
    # log at pipeline end (state is an event-fold), so downstream nodes
    # read the fresh value; NOT folded mid-pipeline (see pipeline._refold)
    # so a later step in the same pipeline should read it from env, not
    # the slot. Status confirmed: a tool resolution is authoritative.
    for slot_key, path in spec.slot_updates.items():
        value = _dig(data, path)
        if value is not None:
            events.append(
                SlotWriteEvent(key=slot_key, value=value, status="confirmed", by="tool")
            )
        else:
            _log.warning(
                "[tool] ⚠ %s slot_updates '%s': path '%s' not found "
                "in response — slot unset",
                spec.id,
                slot_key,
                path,
            )
    return events


def _idempotency_key(spec: ToolSpec, url: str, body: Any) -> str:
    """Deterministic idempotency key for a side-effecting tool call.

    Keyed on the tool id + rendered request (method, url, body) so a retried or
    middleware-replayed call — the same logical operation — reuses the key and
    is de-duplicated server-side, while a materially different request gets a
    different key. Headers are excluded so a refreshed auth token (401 →
    refresh → replay) keeps the same key. canonical_json emits byte-identical
    output to the json.dumps call it replaced, so keys are stable across
    deploys.
    """
    payload = canonical_json([spec.id, spec.method.upper(), url, body])
    return hashlib.sha256(payload.encode()).hexdigest()


class ToolExecutor:
    """Run a ToolSpec against state and return the events to append."""

    def __init__(
        self,
        http: HttpFn,
        python_tools: dict[str, PythonToolFn] | None = None,
        *,
        allow_private_hosts: bool = False,
    ) -> None:
        self._http = http
        self._python_tools = python_tools or {}
        # Strict by default: block private/loopback/metadata targets in rendered
        # tool URLs. Local-dev hosts hitting localhost mocks opt out.
        self._allow_private_hosts = allow_private_hosts

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
                events.extend(_field_update_events(spec, data))
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
        # SSRF guard on the RENDERED url — the host may have arrived via
        # {{ env.X }} / {{ slots.Y }}, so validating spec.url would be bypassable.
        try:
            validate_url(url, allow_private_hosts=self._allow_private_hosts)
        except ValueError as exc:
            return [
                ToolCallEvent(tool=spec.id, args={"url": _redact_url(url)}),
                ToolResultEvent(
                    tool=spec.id,
                    store_as=spec.store_response_as,
                    ok=False,
                    error=str(exc),
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
        redacted_url = _redact_url(url)
        redacted_body = _redact(body or {})
        events.append(
            ToolCallEvent(
                tool=spec.id,
                args={"url": redacted_url, "body": redacted_body},
            )
        )
        # Idempotency: a retried or middleware-replayed side-effecting call is
        # the same logical operation, so give it a stable key the server can
        # de-dupe on (POST/PATCH/DELETE/PUT). An author-supplied key wins.
        if spec.method.upper() not in SAFE_HTTP_METHODS and not any(
            h.lower() == "idempotency-key" for h in headers
        ):
            headers["Idempotency-Key"] = _idempotency_key(spec, url, body)
        # Request trace uses the REDACTED url/body — the raw ones may carry
        # secrets in query params or body fields (and this line is captured by
        # the prod log pipeline, not just a dev terminal).
        _log.info(
            "[tool] → %s %s %s%s",
            spec.id,
            spec.method,
            redacted_url,
            f" body={redacted_body}" if body else "",
        )
        timeout = max(_MIN_TIMEOUT_S, min(spec.timeout, _MAX_TIMEOUT_S))
        # Transport retry: only RAISED transport exceptions (conn reset, DNS blip,
        # TLS) are retried. Timeouts and the >1MB ValueError from the HTTP callable
        # are terminal. A non-2xx status is RETURNED, not raised, so it stays
        # single-shot — the author owns it via pipeline `on:` branches. headers
        # (incl. the Idempotency-Key) were computed once above, so every retry
        # sends the byte-identical request and reuses the same key.
        status, data = 0, None
        last_exc: Exception | None = None
        for attempt in range(_TRANSPORT_RETRIES + 1):
            try:
                status, data = await self._http(
                    method=spec.method,
                    url=url,
                    headers=headers,
                    body=body,
                    timeout=timeout,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                terminal = _is_timeout(exc) or isinstance(exc, ValueError)
                if terminal or attempt == _TRANSPORT_RETRIES:
                    break
                _log.warning(
                    "[tool] retry %s attempt %d: %s",
                    spec.id,
                    attempt + 1,
                    type(exc).__name__,
                )
                await anyio.sleep(_backoff(attempt))
        if last_exc is not None:
            # Exception text can embed the full request URL incl. query secrets;
            # log only the type at WARNING, full detail at DEBUG.
            _log.warning("[tool] ✗ %s %s", spec.id, type(last_exc).__name__)
            _log.debug("[tool] ✗ %s detail", spec.id, exc_info=last_exc)
            events.append(
                ToolResultEvent(
                    tool=spec.id,
                    store_as=spec.store_response_as,
                    ok=False,
                    error=f"{type(last_exc).__name__}: {last_exc}",
                )
            )
            return events
        ok = 200 <= status < 300
        # Response body can hold freshly-minted access tokens (auth-refresh
        # tools) and customer PII; redact before logging.
        _log.info(
            "[tool] ← %s %s %s →%s %s",
            spec.id,
            status,
            "ok" if ok else "FAIL",
            spec.store_response_as or "-",
            str(_redact(data))[:300],
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
            events.extend(_field_update_events(spec, data))
        return events


#: One shared AsyncClient, built lazily, never closed — so an HTTPS tool call
#: does not pay a fresh TCP+TLS handshake every time. Tests inject a
#: MockTransport by setting this module global before calling httpx_http.
_client: "Any | None" = None


def _get_client() -> Any:
    global _client
    import httpx

    if _client is None:
        _client = httpx.AsyncClient()
    return _client


async def httpx_http(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: Any,
    timeout: float,
) -> tuple[int, Any]:
    """Production HTTP callable backed by a shared httpx client.

    Caps the response at 1 MiB: a content-length precheck plus a byte count
    during the streamed read (so a chunked hostile body can't OOM before the
    check runs). A breach raises ``ValueError`` — the executor turns that into a
    failed ToolResultEvent and does NOT retry it (an oversized body stays
    oversized). JSON is returned parsed; non-JSON as ``{"text": ...}``.
    """
    client = _get_client()
    async with client.stream(
        method, url, headers=headers, json=body, timeout=timeout
    ) as resp:
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > _MAX_RESPONSE_BYTES:
            raise ValueError(f"response too large: {declared} bytes (>1MiB cap)")
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf.extend(chunk)
            if len(buf) > _MAX_RESPONSE_BYTES:
                raise ValueError("response too large (>1MiB cap)")
    try:
        return resp.status_code, json.loads(bytes(buf))
    except ValueError:
        return resp.status_code, {"text": bytes(buf).decode(errors="replace")}
