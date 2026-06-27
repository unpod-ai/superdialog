"""Render the Talker's view: priority-packed, token-budgeted (design doc §3).

The system block (persona, guidance, steering note, slots, computed views,
summary) is protected; only the recent transcript (packed newest-first) is
droppable under budget pressure. Env lane is NEVER rendered.
Guidance/say_verbatim are Jinja templates over {slots, views, results}.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jinja2 import ChainableUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, Field

from ..llm.prompt_cache import CACHE_PREFIX_KEY
from ._guidelines import DATE_DISCIPLINE, compose_guidelines, datetime_anchor_line
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


def _system_block(pb: Playbook, state: ConversationState) -> tuple[str, str]:
    ns = template_namespace(pb, state)
    cp = pb.checkpoint(state.checkpoint_id) if state.checkpoint_id else None
    blocks = compose_guidelines(
        pb.guidelines,
        has_summary=bool(state.summary),
        # Checkpoint.handover lands in a later task; getattr default keeps this a
        # safe no-op until then.
        handover=bool(cp and getattr(cp, "handover", False)),
    )
    # state.now is session-frozen via runtime.start(); the wall-clock fallback is
    # the degraded test path (not cache-stable).
    now = state.now or datetime.now(timezone.utc)
    anchor = datetime_anchor_line(now)
    # Opt-in runtime trace (SUPERDIALOG_GUIDELINES_DEBUG=1): confirms the framework
    # voice-guideline block + date anchor are actually fed into the Talker prompt,
    # and flags double-injection when a persona already carries a guideline block
    # (e.g. the playground's own _with_default_guidelines append).
    # Always-on console trace so guideline feeding is visible in every run.
    # print (not logging) so it shows even when the host (uvicorn/playground)
    # hasn't configured an INFO-level root logger. `fed` enumerates every
    # guideline chunk active this turn — the default voice spine sections plus
    # the conditionals (memory/handover/date-discipline/strict) — so it's clear
    # which planned guidelines are feeding and which are gated off.
    fed = list(blocks.static and blocks.sections or [])
    if blocks.memory_guard:
        fed.append("memory_guard")
    if blocks.handover:
        fed.append("handover")
    fed.append("date_discipline")  # always fed to the Talker (cache prefix)
    if cp and (cp.say_verbatim is not None or getattr(cp, "strict", False)):
        fed.append("strict_verbatim")
    print(
        f"[guidelines] checkpoint={state.checkpoint_id} "
        f"channel={pb.guidelines.channel} anchor_from_log={state.now is not None} "
        f"persona_already_has_guidelines={'DEFAULT VOICE GUIDELINES' in pb.persona} "
        f"fed={fed}",
        flush=True,
    )
    # persona + static guideline block + date anchor are session-constant, so
    # together they form the stable cacheable prefix. Persona stays FIRST so the
    # existing persona-leads invariant holds.
    prefix_parts = [pb.persona.strip()]
    if blocks.static:
        prefix_parts.append(blocks.static)
    prefix_parts.append(anchor)
    # DATE_DISCIPLINE feeds the Talker unconditionally: the anchor line alone
    # doesn't stop weaker models computing ages/durations from their training-prior
    # year (born-2019 → "3–4 yrs"). It's session-constant, so the cache prefix holds.
    prefix_parts.append(DATE_DISCIPLINE.strip())
    cache_prefix = "\n\n".join(p for p in prefix_parts if p)

    parts: list[str] = [cache_prefix]
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
    if blocks.handover:
        parts.append(blocks.handover)
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
        if blocks.memory_guard:
            parts.append(blocks.memory_guard)
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
    return "\n\n".join(p for p in parts if p), cache_prefix


def render_view(
    pb: Playbook, state: ConversationState, token_budget: int = 4000
) -> RenderedView:
    """Pack the Talker's view into ``token_budget`` estimated tokens."""
    system, cache_prefix = _system_block(pb, state)
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
    # Some providers (Anthropic) require at least one non-system message.
    # When the transcript is empty (opening greeting) or the entire history
    # was truncated by the token budget, inject a minimal placeholder so the
    # message list is always valid for all providers.
    if not chat:
        chat = [{"role": "user", "content": "[start]"}]
    # Mark the stable prompt prefix (persona + static guideline block + anchor,
    # all session-constant) so the provider seam can cache it. The private key
    # is stripped (or split into cache blocks) at the provider seam; ``content``
    # itself stays a bare string here.
    sys_msg: dict[str, str] = {"role": "system", "content": system}
    if cache_prefix and system.startswith(cache_prefix):
        sys_msg[CACHE_PREFIX_KEY] = cache_prefix
    return RenderedView(
        messages=[sys_msg, *chat],
        spoke_from_version=state.version,
    )
