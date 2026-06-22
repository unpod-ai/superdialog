"""LLMAdapter -- adapter backed by an ``LLMProvider``.

Default adapter used by ``DialogMachine`` when the caller does not pass an
explicit adapter. Wraps an :class:`~superdialog.llm.provider.LLMProvider`
and synthesises every method the dialog machine expects:

* :meth:`evaluate_criteria` delegates to :class:`CriteriaJudge`.
* :meth:`generate_reply` calls ``provider.complete``.
* :meth:`speak` is a no-op (text mode -- static_text is returned by the
  machine and surfaced through ``TurnResult.response``).
* :meth:`generate_recovery` returns a short fallback or, if available, a
  one-shot LLM regen of the recovery instruction.
* :meth:`end_session` is a no-op.
"""

from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

from jinja2 import BaseLoader, ChainableUndefined, Environment

from superdialog.llm.provider import LLMProvider
from superdialog.machine.criteria import CriteriaJudge
from superdialog.machine.models import CriteriaResult

if TYPE_CHECKING:
    from superdialog.flow.models import CustomAction, FlowNode

logger = logging.getLogger(__name__)


@dataclass
class LLMCallData:
    node_id: str
    model: str
    call_type: Literal["routing", "generate_reply"]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    prompt_messages: list[dict]
    response_json: dict
    edge_id: str | None
    cached: int = 0  # prompt-cache READ tokens (billable as llm_cached_tokens)
    cache_write: int = 0  # prompt-cache WRITE/creation tokens (llm_cache_write_tokens)


def _coerce_numeric_strings(d: dict, skip_fields: set[str]) -> dict:
    """Coerce string values that look like ints/floats — Jinja2 renders everything as str."""
    for k, v in d.items():
        if k in skip_fields:
            continue
        if isinstance(v, str):
            try:
                d[k] = int(v)
                continue
            except ValueError:
                pass
            try:
                d[k] = float(v)
            except ValueError:
                pass
        elif isinstance(v, dict):
            _coerce_numeric_strings(v, skip_fields)
    return d


class LLMAdapter:
    """Provider-backed adapter for the dialog state machine.

    The adapter is intentionally thin -- all routing intelligence lives
    in :class:`CriteriaJudge`. Callers that need bespoke behaviour
    (custom recovery, action execution, speech passthrough) should
    subclass this or roll their own adapter implementing the same
    duck-typed surface.
    """

    supports_criteria: bool = True
    speech_passthrough: bool = False

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "",
        criteria_judge: CriteriaJudge | None = None,
        environment_variables: dict[str, str] | None = None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._judge = criteria_judge or CriteriaJudge(llm=provider)
        self.responses: list[str] = []
        self.session_ended: bool = False
        # HTTP action execution state
        self._env_vars: dict[str, str] = dict(environment_variables or {})
        self._jinja_env = Environment(loader=BaseLoader(), undefined=ChainableUndefined)
        self._on_llm_complete: Callable[[LLMCallData], Awaitable[None]] | None = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_provider(self, provider: LLMProvider) -> None:
        """Hot-swap the underlying provider (used by ``set_llm``)."""
        self._provider = provider
        self._judge = CriteriaJudge(llm=provider)

    # ------------------------------------------------------------------
    # Adapter surface (duck-typed; called by DialogStateMachine)
    # ------------------------------------------------------------------

    async def speak(self, text: str, node: FlowNode) -> None:
        """Record scripted (static_text) output. No external side effects."""
        self.responses.append(text)

    async def generate_reply(
        self,
        instruction: str,
        node: FlowNode,
        history: list[dict[str, Any]] | None = None,
        userdata: dict[str, Any] | None = None,
    ) -> str:
        """Generate an LLM reply from the node's enriched instruction.

        Renders Jinja2 templates in ``instruction`` using ``userdata`` so the
        LLM sees concrete API values (e.g. actual course lists) instead of raw
        ``{{city_courses_result.data.courses}}`` placeholders.
        """
        if userdata and "{{" in instruction:
            ctx = self._build_context(userdata)
            instruction = self._render(instruction, ctx)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{self._system_prompt}\n\n{instruction}"
                    if self._system_prompt
                    else instruction
                ),
            }
        ]
        if history:
            messages.extend(history)
        t0 = time.perf_counter()
        result = await self._provider.complete(messages)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.responses.append(result.text)
        if self._on_llm_complete is not None:
            await self._on_llm_complete(
                LLMCallData(
                    node_id=node.id,
                    model=getattr(self._provider, "model", "unknown"),
                    call_type="generate_reply",
                    latency_ms=latency_ms,
                    tokens_in=int(result.metadata.get("prompt_tokens", 0) or 0),
                    tokens_out=int(result.metadata.get("completion_tokens", 0) or 0),
                    cached=int(result.metadata.get("cache_read_tokens", 0) or 0),
                    cache_write=int(result.metadata.get("cache_write_tokens", 0) or 0),
                    prompt_messages=messages,
                    response_json={"text": result.text},
                    edge_id=None,
                )
            )
        return result.text

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        silent: bool = False,
    ) -> CriteriaResult:
        """Delegate criteria evaluation to :class:`CriteriaJudge`."""
        meta = userdata.get("_flow_meta", {})
        clean_userdata = {k: v for k, v in userdata.items() if k != "_flow_meta"}

        # Pre-render Jinja2 templates in node instruction so CriteriaJudge
        # sees concrete values (e.g. all_slots=[]) instead of raw placeholders.
        # Without this, the LLM ignores guards like EMPTY SLOTS GUARD because
        # it can't evaluate {{availability_result.data.slots|default([])}}.
        eval_node = node
        if node.instruction and "{{" in node.instruction:
            ctx = self._build_context(clean_userdata)
            rendered = self._render(node.instruction, ctx)
            if rendered != node.instruction:
                eval_node = node.model_copy(update={"instruction": rendered})

        eval_messages = self._judge.build_evaluation_messages(
            node=node,
            history=history,
            userdata=clean_userdata,
            system_prompt=self._system_prompt,
            visit_count=meta.get("visit_count", 1),
            turns_in_node=meta.get("turns_in_node", 0),
            agent_language=meta.get("agent_language", ""),
            agent_gender=meta.get("agent_gender", ""),
            node_slots=meta.get("node_slots"),
            previously_completed=meta.get("previously_completed", False),
        )
        t0 = time.perf_counter()
        result = await self._judge.evaluate(
            node=eval_node,
            history=history,
            userdata=clean_userdata,
            system_prompt=self._system_prompt,
            visit_count=meta.get("visit_count", 1),
            turns_in_node=meta.get("turns_in_node", 0),
            agent_language=meta.get("agent_language", ""),
            agent_gender=meta.get("agent_gender", ""),
            node_slots=meta.get("node_slots"),
            previously_completed=meta.get("previously_completed", False),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if self._on_llm_complete is not None:
            await self._on_llm_complete(
                LLMCallData(
                    node_id=node.id,
                    model=getattr(self._provider, "model", "unknown"),
                    call_type="routing",
                    latency_ms=latency_ms,
                    tokens_in=0,
                    tokens_out=0,
                    prompt_messages=eval_messages,
                    response_json={
                        "criteria_met": result.criteria_met,
                        "recommended_edge_id": result.recommended_edge_id,
                        "response": result.response,
                    },
                    edge_id=result.recommended_edge_id,
                )
            )
        return result

    def register_llm_callback(self, fn: Any) -> None:
        self._on_llm_complete = fn

    async def generate_recovery(self, node: FlowNode, error: str) -> str:
        """Return a recovery line when criteria evaluation fails."""
        instruction = node.instruction or node.static_text or ""
        fallback = "I didn't quite catch that. Could you say that again?"
        if not instruction:
            return fallback
        try:
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        f"{self._system_prompt}\n\n"
                        "The previous response failed to process. Generate "
                        "a brief, natural recovery message that re-engages "
                        f"the caller with the current task: {instruction}"
                    ),
                }
            ]
            result = await self._provider.complete(messages)
            return result.text or fallback
        except Exception as exc:
            logger.warning("LLMAdapter recovery generation failed: %s", exc)
            return fallback

    def _render(self, template_str: str, context: dict[str, Any]) -> str:
        try:
            return self._jinja_env.from_string(template_str).render(**context)
        except Exception as exc:
            logger.warning("LLMAdapter: template render failed for %r: %s", template_str[:60], exc)
            return template_str

    def _build_context(self, userdata: dict[str, Any]) -> dict[str, Any]:
        """Merge env_vars + userdata into a flat Jinja2 context."""
        ctx: dict[str, Any] = {}
        ctx.update(self._env_vars)
        ctx.update(userdata)
        return ctx

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute an HTTP action with Jinja2 template rendering."""
        import httpx
        import re as _re

        print(f"[TRACK] LLMAdapter.execute_action START - action_id: {action.id}, userdata keys: {list(userdata.keys()) if userdata else 'None'}")

        # run_once: return cached result if already succeeded this session
        if action.run_once and action.store_response_as:
            cached = userdata.get(action.store_response_as, {})
            if isinstance(cached, dict) and cached.get("success"):
                print(f"[TRACK] LLMAdapter - run_once HIT for action={action.id}, returning cached result")
                return cached

        ctx = self._build_context(userdata)
        print(f"[TRACK] LLMAdapter - context keys: {list(ctx.keys())}")

        # condition guard
        if action.condition:
            condition_result = self._render(action.condition, ctx)
            if not condition_result.strip():
                print(f"[TRACK] LLMAdapter - action={action.id} SKIPPED (condition empty after render)")
                return None

        url = self._render(action.url, ctx)
        headers = {k: self._render(v, ctx) for k, v in action.headers.items()}

        # Log rendered URL and unresolved template vars
        print(f"[TRACK] LLMAdapter - rendered URL: {url}  (template: {action.url})")
        method = action.method.value if hasattr(action.method, "value") else str(action.method)
        print(f"[TRACK] LLMAdapter - method: {method}")
        if "{{" in action.url:
            for var in _re.findall(r"\{\{([^}]+)\}\}", action.url):
                key = var.strip().split("|")[0].strip()
                print(f"[TRACK] LLMAdapter - template var [{key}] = {ctx.get(key, 'NOT_FOUND')}")

        body: Any = None
        if action.body_template:
            body_str = self._render(action.body_template, ctx)
            print(f"[TRACK] LLMAdapter - rendered body: {body_str}  (template: {action.body_template})")
            try:
                body = json.loads(body_str)
                if isinstance(body, dict):
                    body = _coerce_numeric_strings(body, set(action.string_fields))
            except json.JSONDecodeError:
                body = body_str

        try:
            async with httpx.AsyncClient(timeout=float(action.timeout)) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, dict) else None,
                    content=body.encode() if isinstance(body, str) else None,
                )
                result: dict[str, Any] = {
                    "status": response.status_code,
                    "success": response.status_code < 400,
                    "headers": dict(response.headers),
                }
                try:
                    result["data"] = response.json()
                except Exception:
                    result["data"] = response.text

                result["_rendered_url"] = url
                result["_method"] = method

                status_tag = "OK" if response.status_code < 400 else "FAILED"
                print(f"[TRACK] LLMAdapter - action={action.id} {status_tag} status={response.status_code}")
                print(f"[TRACK] LLMAdapter - response data: {json.dumps(result.get('data'), default=str)[:500]}")

                if action.store_response_as:
                    print(f"[TRACK] LLMAdapter - storing result as: {action.store_response_as}")

                # Apply env_updates (e.g. store ACCESS_TOKEN for subsequent actions)
                for update in action.env_updates:
                    value: Any = result
                    try:
                        for key in update.result_path.split("."):
                            value = value[key]
                        self._env_vars[update.env_key] = str(value)
                        print(f"[TRACK] LLMAdapter - env_update: {update.env_key} = {str(value)[:80]}")
                    except (KeyError, TypeError, IndexError):
                        print(f"[TRACK] LLMAdapter - env_update FAILED: could not resolve {update.result_path} for {update.env_key}")

                return result

        except Exception as exc:
            print(f"[TRACK] LLMAdapter - action={action.id} HTTP error: {exc}")
            error_result: dict[str, Any] = {
                "success": False,
                "error": str(exc),
                "status": 0,
                "_rendered_url": url,
                "_method": method,
            }
            return error_result

    async def end_session(self) -> None:
        """Mark the session as ended (idempotent)."""
        self.session_ended = True
