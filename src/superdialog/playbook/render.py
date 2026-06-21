"""Render the Talker's view: priority-packed, token-budgeted (design doc §3).

The system block (persona, guidance, steering note, slots, computed views,
summary) is protected; only the recent transcript (packed newest-first) is
droppable under budget pressure. Env lane is NEVER rendered.
Guidance/say_verbatim are Jinja templates over {slots, views, results}.
"""

from __future__ import annotations

from typing import Any

from jinja2 import ChainableUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, Field

from .expr import ExprError, evaluate
from .models import Playbook
from .state import ConversationState

# Sandboxed: templates come from playbook artifacts (optimizer-generated),
# so attribute-walking SSTI payloads must be blocked, not executed.
# ChainableUndefined: compiled flows chain through possibly-missing results
# ({{ results.x.data.name|default('there') }}) — attribute access on a
# missing root must defer to the |default filter, not raise mid-speech.
_jinja = SandboxedEnvironment(undefined=ChainableUndefined, autoescape=False)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 UTF-8 bytes per token, floor of 1.

    Byte-based so non-Latin scripts (e.g. Devanagari at 3 bytes/char,
    which also tokenizes worse) don't blow past the budget.
    TODO: wire the real Talker tokenizer in Task 9+.
    """
    return max(1, len(text.encode("utf-8")) // 4)


class RenderedView(BaseModel):
    """The Talker's packed prompt plus the state version it rendered."""

    messages: list[dict[str, str]] = Field(default_factory=list)
    spoke_from_version: int = 0


def template_namespace(pb: Playbook, state: ConversationState) -> dict[str, Any]:
    """Build the {slots, views, results} namespace for Jinja templates."""
    views: dict[str, Any] = {}
    for name, expr in pb.views.items():
        try:
            # env is never renderer-visible: shadow it so a view expr like
            # "env.ACCESS_TOKEN" evaluates to None and is dropped by the
            # Reference-data filter. Tool when:/url templates (Director-side)
            # still access env legitimately.
            views[name] = evaluate(expr, state, extra={"env": None})
        except ExprError:
            views[name] = None
    return {
        "slots": {k: v.value for k, v in state.slots.items()},
        "views": views,
        "results": {
            k: {"ok": r.ok, "status": r.status, "data": r.data, "error": r.error}
            for k, r in state.tool_results.items()
        },
    }


def render_template(
    text: str,
    pb: Playbook,
    state: ConversationState,
    ns: dict[str, Any] | None = None,
) -> str:
    """Render a Jinja template over {slots, views, results}.

    Callers rendering repeatedly for the same state should pass a
    precomputed ``ns=`` to avoid re-evaluating views on every call.
    """
    namespace = ns if ns is not None else template_namespace(pb, state)
    try:
        return _jinja.from_string(text).render(**namespace)
    except TemplateError:
        # An authoring typo (undefined name, broken syntax) or a sandbox
        # SecurityError (a TemplateError subclass) must never crash the
        # speaking path: degrade to the raw text, which surfaces the
        # un-rendered template in transcripts as the debugging signal.
        return text


def _system_block(pb: Playbook, state: ConversationState) -> str:
    ns = template_namespace(pb, state)
    cp = pb.checkpoint(state.checkpoint_id) if state.checkpoint_id else None
    parts: list[str] = [pb.persona.strip()]
    # "Direction" (steer) notes appear before step guidance so the guidance
    # text — which is more specific and may explicitly override the note —
    # lands later in the context and takes precedence with the LLM.
    # "Correction" (repair) notes appear after guidance: they are high-priority
    # fixes (e.g. "you already have name=X; acknowledge it") that must override.
    if state.steering_note and state.steering_kind == "steer":
        parts.append(f"## Direction from supervisor\n{state.steering_note}")
    if cp:
        guidance = render_template(cp.guidance, pb, state, ns=ns)
        parts.append(f"## Current step: {cp.id}\nGoal: {cp.goal}\n{guidance}".strip())
        missing = [
            k for k, s in cp.slots.items() if s.required and k not in state.slots
        ]
        if missing:
            parts.append("Still needed: " + ", ".join(missing))
        if cp.never_say:
            parts.append("Never say: " + "; ".join(cp.never_say))
    if state.steering_note and state.steering_kind != "steer":
        label = "Correction" if state.steering_kind == "repair" else "Direction"
        parts.append(f"## {label} from supervisor\n{state.steering_note}")
    if state.slots:
        slot_lines = "\n".join(f"- {k}: {v.value}" for k, v in state.slots.items())
        parts.append("## Known information\n" + slot_lines)
    view_lines = "\n".join(
        f"- {k}: {v}" for k, v in ns["views"].items() if v not in (None, [], {})
    )
    if view_lines:
        parts.append("## Reference data\n" + view_lines)
    if state.summary:
        parts.append("## Earlier in this conversation\n" + state.summary)
    # Global knowledge base: answer off-flow questions in-context, then resume
    # the current step (the drift fix). Rendered through the same Jinja sandbox,
    # so an authoring typo degrades to raw text and never crashes the speaking
    # path. When the KB is empty the prompt is byte-identical to before, so
    # existing playbooks are unaffected.
    kb = (
        render_template(pb.knowledge_base, pb, state, ns=ns).strip()
        if pb.knowledge_base
        else ""
    )
    if kb:
        parts.append("## Knowledge base\n" + kb)
        parts.append(
            "If the caller asks something covered by the Knowledge base but "
            "outside the current step's goal, answer briefly from the Knowledge "
            "base, then steer back to the current step's goal; do not abandon "
            "the current step."
        )
        fact_sources = "Known information, Reference data, or the Knowledge base"
    else:
        fact_sources = "Known information or Reference data"
    parts.append(
        f"Only state facts present in {fact_sources}; "
        "if asked something not there, say you are checking."
    )
    return "\n\n".join(p for p in parts if p)


def render_view(
    pb: Playbook, state: ConversationState, token_budget: int = 4000
) -> RenderedView:
    """Pack the Talker's view into ``token_budget`` estimated tokens."""
    system = _system_block(pb, state)
    used = estimate_tokens(system)
    chat: list[dict[str, str]] = []
    # newest-first packing of transcript, then reverse to chronological
    for entry in reversed(state.transcript):
        cost = estimate_tokens(entry.text) + 4
        if used + cost > token_budget:
            break
        chat.append({"role": entry.role, "content": entry.text})
        used += cost
    chat.reverse()
    return RenderedView(
        messages=[{"role": "system", "content": system}, *chat],
        spoke_from_version=state.version,
    )
