"""Vanilla-mode session driver: the run_session shape, for a flat-prompt LLMAgent.

run_session (runner.py) is typed to PlaybookAgent and reads .runtime.start()/
.runtime.state.ended — APIs a vanilla LLMAgent doesn't have. This mirrors its
turn-taking loop against LLMAgent.turn() directly.

Vanilla now gets REAL tool-calling: the playbook's own ``tools:`` are exposed
to the LLM as native OpenAI function schemas (see ``tool_schemas``); the LLM
decides args itself (there's no ConversationState/slot store in vanilla mode
to pull them from) and a real HTTP call is made. Events are recorded in the
SAME dict shape playbook mode's EventLog uses (see ``events.py``) so
evidence_gated_task_success/pass_at_1/failure_recovery_rate work unchanged
against ``VanillaSessionMetrics.events`` -- no tool call ever happening (no
``tools`` passed in) means an empty events list, same as before this change.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, Field

from .._ssrf import validate_url
from ..models import ToolSpec
from ..toolexec import httpx_http
from .models import PersonaSpec
from .runner import SpeaksUser

_JINJA = SandboxedEnvironment()

_JSON_SCHEMA_TYPE = {
    "str": "string", "int": "integer", "float": "number", "bool": "boolean",
    "date": "string", "time": "string", "enum": "string",
    "array": "array", "object": "object",
}


class TranscriptEntry(BaseModel):
    role: str  # "user" | "assistant"
    text: str


class VanillaSessionMetrics(BaseModel):
    """The vanilla-side counterpart to SessionMetrics — no slots, no
    checkpoints (a flat LLMAgent has neither), but real tool-call events when
    the LLM used a native tool."""

    persona: str
    turns: int
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


class TurnsText(Protocol):
    """Duck-typed: anything with an async turn(text) -> object-with-.text."""

    async def turn(self, text: str) -> Any: ...


def tool_schemas(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """OpenAI function-calling schemas for a playbook's declared tools.

    Parameters come from each ToolSpec's ``args:`` block. A tool that
    declares none gets an empty-object schema (the LLM can still call it
    with ``{}`` and any ``{{slots.x|default(...)}}`` fallbacks render)."""
    schemas = []
    for spec in tools:
        props: dict[str, Any] = {}
        required: list[str] = []
        for key, slot in spec.args.items():
            prop: dict[str, Any] = {"type": _JSON_SCHEMA_TYPE.get(slot.type, "string")}
            if slot.description:
                prop["description"] = slot.description
            if slot.type == "enum" and slot.values:
                prop["enum"] = slot.values
            props[key] = prop
            if slot.required:
                required.append(key)
        schemas.append({
            "type": "function",
            "function": {
                "name": spec.id,
                "description": f"Call the {spec.id} action.",
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return schemas


def tool_call_guidance(tools: list[ToolSpec]) -> str:
    """Generic natural-language nudge appended to vanilla's system prompt.

    A playbook's own trigger for a tool (e.g. ``on_enter:`` on a checkpoint)
    is Director-only runtime semantics -- when the raw playbook YAML is
    dumped as vanilla's system prompt, that trigger reads as inert
    structural config, not an instruction. The function schema alone makes
    the model mechanically CAPABLE of calling a tool but never tells it
    WHEN to, so it talks its way through the whole call without ever
    calling one. This is playbook-agnostic (built from each ToolSpec's own
    id/args), not hardcoded to any one playbook's tools.
    """
    if not tools:
        return ""
    lines = [
        "\n\nYou have real callable tools (see the function list) -- when the "
        "conversation reaches the point where a human agent following this "
        "playbook would actually act (e.g. once you have the details a tool "
        "needs), CALL that tool for real. Do not just describe, promise, or "
        "narrate the action in your reply instead of calling it.",
    ]
    for t in tools:
        args = ", ".join(t.args) if t.args else "no arguments"
        lines.append(f"- {t.id}: call once you have {args}.")
    return "\n".join(lines)


def _render(template: str, args: dict[str, Any]) -> str:
    return _JINJA.from_string(template).render(slots=args, env=os.environ)


async def _execute_tool_call(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    """Real HTTP call for one vanilla tool invocation.

    No ConversationState exists in vanilla mode, so rendering uses the LLM's
    own supplied arguments as the template namespace's ``slots`` -- the LLM
    decides the values itself, unlike the Director which pulls from
    pre-extracted, confirmed slots. Same SSRF guard as production tool calls.
    """
    try:
        url = _render(spec.url, args)
        validate_url(url)
        headers = {k: _render(v, args) for k, v in spec.headers.items()}
        body = {
            k: (_render(v, args) if isinstance(v, str) else v)
            for k, v in spec.body.items()
        }
        status, data = await httpx_http(
            method=spec.method, url=url, headers=headers, body=body,
            timeout=spec.timeout,
        )
        return {"ok": 200 <= status < 300, "status": status, "data": data}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


async def run_vanilla_session(
    agent: TurnsText,
    persona: PersonaSpec,
    user_llm: SpeaksUser,
    *,
    tools: list[ToolSpec] | None = None,
) -> VanillaSessionMetrics:
    """Drive one persona session against a flat vanilla agent; measure it."""
    tools_by_name = {t.id: t for t in (tools or [])}
    transcript: list[TranscriptEntry] = []
    events: list[dict[str, Any]] = []
    user_text = persona.opening
    turns = 0
    while turns < persona.max_turns:
        transcript.append(TranscriptEntry(role="user", text=user_text))
        events.append({"type": "utterance", "role": "user", "text": user_text})
        result = await agent.turn(user_text)
        reply = result.text if hasattr(result, "text") else str(result)
        transcript.append(TranscriptEntry(role="assistant", text=reply))
        events.append({"type": "utterance", "role": "assistant", "text": reply})
        for tc in getattr(result, "tool_calls", None) or []:
            spec = tools_by_name.get(tc.name)
            if spec is None:
                continue
            events.append({"type": "tool_call", "tool": tc.name, "args": tc.arguments})
            outcome = await _execute_tool_call(spec, tc.arguments)
            events.append({
                "type": "tool_result", "tool": tc.name,
                "ok": outcome["ok"], "status": outcome.get("status"),
            })
            assist = getattr(agent, "assist", None)
            if callable(assist):
                assist(f"[Tool '{tc.name}' result: {json.dumps(outcome)[:1500]}]")
        turns += 1
        if turns >= persona.max_turns:
            break
        messages = _persona_messages(persona, transcript)
        user_text = (await user_llm.complete(messages)).strip()
        if not user_text:
            break
    return VanillaSessionMetrics(
        persona=persona.name, turns=turns, transcript=transcript, events=events,
    )


def _persona_messages(
    persona: PersonaSpec, transcript: list[TranscriptEntry]
) -> list[dict[str, str]]:
    # Same fix as runner.py's copy: give the persona its own ground-truth
    # details so it answers with the SAME values slot_accuracy checks it
    # against, instead of inventing a random name/language every time.
    details = "; ".join(f"{k}={v}" for k, v in persona.ground_truth_slots.items())
    your_details = (
        f" Your details (use these EXACT values when asked, never invent "
        f"different ones): {details}."
        if details
        else ""
    )
    system = (
        f"You are role-playing a caller. Traits: {persona.traits}. "
        f"Your goal: {persona.goal}.{your_details} Reply with ONLY the "
        "caller's next utterance, 1-2 sentences."
    )
    messages = [{"role": "system", "content": system}]
    for entry in transcript[-10:]:
        role = "assistant" if entry.role == "user" else "user"
        messages.append({"role": role, "content": entry.text})
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "(silence)"})
    return messages
