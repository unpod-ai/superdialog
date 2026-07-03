"""Simple authoring format -> Playbook compiler.

The simple format is a human-friendly surface for authoring playbooks: prose
steps, a nested persona, and reference data (facts/objections/boundaries/
fallbacks). `simple_to_playbook` lowers it to the validated `Playbook` runtime
artifact, the same way `compile_flow` lowers legacy flows.

Facts, objections, boundaries, fallbacks, and the closing line are folded into
ONE rich `persona` string. The Talker sees `persona` every turn but the `env`
lane is never rendered to it, so this reference material must live in persona —
NOT env — to stay visible during speech.
"""

from __future__ import annotations

import json
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .models import (
    AdvanceRule,
    Checkpoint,
    GuidelineConfig,
    InterruptSpec,
    Journey,
    Playbook,
    SlotSpec,
)


class SimplePersona(BaseModel):
    name: str = ""
    # A language name ("English"), an ISO 639-1 code ("hi"), or a list of
    # either — first entry is the default, the rest are also spoken.
    language: str | list[str] = ""
    voice_style: str = ""
    identity: str = ""


class SimpleStep(BaseModel):
    id: str
    purpose: str = ""
    say: str = ""
    collect: list[str] = Field(default_factory=list)
    done_when: str = ""
    # Which person this step collects for (multi-entity). Defaults to "caller"
    # so single-entity playbooks are unchanged.
    entity: str = "caller"
    # User turns before the runtime steers "wrap this step up" (default 4).
    turn_budget: int | None = None
    # Inject the knowledge base on this step. None = legacy heuristic (say
    # mentions 'knowledge_base'); set false on steps that merely reference it.
    kb: bool | None = None


class SimpleObjection(BaseModel):
    trigger: str
    handle: str


class SimpleInterrupt(BaseModel):
    id: str = ""
    when: str
    to: str
    resume: bool = False  # return to the step we left once the detour is done


class SimplePlaybook(BaseModel):
    name: str = ""
    goal: str = ""
    persona: SimplePersona = Field(default_factory=SimplePersona)
    opening: str = ""
    closing: str = ""
    playbook: list[SimpleStep] = Field(min_length=1)
    facts: dict[str, Any] = Field(default_factory=dict)
    objections: list[SimpleObjection] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    fallback_actions: dict[str, str] = Field(default_factory=dict)
    interrupts: list[SimpleInterrupt] = Field(default_factory=list)
    channel: str = "voice"
    tone: str = "professional"
    call_type: str | None = None
    timezone: str = "UTC"
    memory_enabled: bool = False
    followup_enabled: bool = False
    # Opt-in: scope slot storage/lookups per step entity. Off ⇒ unchanged.
    multi_entity: bool = False


def is_simple_playbook(doc: Any) -> bool:
    """True when ``doc`` is a simple playbook: top-level ``playbook`` is a list."""
    return (
        isinstance(doc, dict)
        and isinstance(doc.get("playbook"), list)
        and len(doc["playbook"]) > 0
    )


# ISO 639-1 -> readable name for 59 common languages.
# NOTE: in YAML, quote the Norwegian code ("no") — unquoted it parses as a
# boolean under yaml.safe_load, which the simple format uses.
_LANG_NAMES = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "ml": "Malayalam",
    "mr": "Marathi",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh",
}


def _language_line(language: str | list[str]) -> str:
    """Fold language(s) into one persona line; codes map to readable names."""
    raw = [language] if isinstance(language, str) else list(language)
    names = [_LANG_NAMES.get(s.strip().lower(), s.strip()) for s in raw if s.strip()]
    if not names:
        return ""
    line = f"Default conversation language: {names[0]}."
    if len(names) > 1:
        line += " Also speaks: " + ", ".join(names[1:]) + "."
    return line


def _build_persona(sp: SimplePlaybook) -> str:
    parts: list[str] = []
    if sp.persona.identity.strip():
        parts.append(sp.persona.identity.strip())
    name = sp.persona.name.strip()
    if name and name.lower() not in sp.persona.identity.lower():
        parts.append(f"Your name is {name}.")
    language_line = _language_line(sp.persona.language)
    if language_line:
        parts.append(language_line)
    if sp.persona.voice_style.strip():
        parts.append(f"Voice & manner: {sp.persona.voice_style.strip()}")
    if sp.goal.strip():
        parts.append(f"Overall goal: {sp.goal.strip()}")
    if sp.facts:
        # A `knowledge_base` facts key is routed to Playbook.knowledge_base (the
        # engine's KB-answer / drift-fix path); keep it out of the persona dump
        # so it is not duplicated in the prompt.
        persona_facts = {k: v for k, v in sp.facts.items() if k != "knowledge_base"}
        if persona_facts:
            dumped = yaml.safe_dump(persona_facts, sort_keys=False, allow_unicode=True)
            parts.append(
                "## Reference facts (never invent beyond these)\n" + dumped.strip()
            )
    if sp.objections:
        # Skip objections whose handle just restates a hard boundary — the
        # boundary block already renders every turn; duplicating it costs
        # persona tokens on every single Talker call for zero signal.
        blows = [b.strip().casefold() for b in sp.boundaries]
        kept = [
            o
            for o in sp.objections
            if not any(
                o.handle.strip().casefold() in b or b in o.handle.strip().casefold()
                for b in blows
            )
        ]
        if kept:
            bullets = "\n".join(f"- If {o.trigger} -> {o.handle.strip()}" for o in kept)
            parts.append("## Objection handling\n" + bullets)
    if sp.boundaries:
        bullets = "\n".join(f"- {b}" for b in sp.boundaries)
        parts.append("## Hard boundaries\n" + bullets)
    if sp.fallback_actions:
        bullets = "\n".join(f"- {k}: {v}" for k, v in sp.fallback_actions.items())
        parts.append("## Fallback actions\n" + bullets)
    if sp.closing.strip():
        parts.append("## Closing line\n" + sp.closing.strip())
    return "\n\n".join(parts)


# Past this many user turns on one step, the runtime steers "wrap this step
# up" (no on_failure is compiled, so it never force-advances — steer only).
_DEFAULT_TURN_BUDGET = 4


def _step_to_checkpoint(
    step: SimpleStep, next_id: str | None, opening: str
) -> Checkpoint:
    guidance = step.say.strip() or opening.strip()
    # gate="soft" on the SLOT (not the checkpoint): requires below then demands
    # filled, not confirmed. Hard inheritance would demand confirmation that
    # verdict writes can't self-provide (they land provisional at hard gates by
    # anti-injection design), blocking advance after the user answered and
    # causing the re-asking this compiler previously avoided via requires=[].
    slots = {
        c: SlotSpec(type="str", description="", required=True, gate="soft")
        for c in step.collect
    }
    if next_id is None:
        return Checkpoint(
            id=step.id,
            goal=step.purpose,
            guidance=guidance,
            slots=slots,
            entity=step.entity,
            terminal=True,
            outcome="closed",
            uses_kb=step.kb,
        )
    # requires=collect: the Director may not advance past unfilled slots (a
    # done_when verdict alone could skip capture on terse callers); a blocked
    # advance emits the "still need: X" steer and the Talker's "Still needed"
    # hint, so the agent circles back instead of moving on.
    rules = []
    if step.collect and step.entity == "caller":
        # Companion expr rule ahead of the llm rule: all-slots-filled advances
        # deterministically with ZERO LLM cost. It fires on the same-turn
        # quiesce hop when a verdict wrote the slots but forgot "advance"
        # (otherwise a whole extra user round-trip = 3 LLM calls is burned).
        # Caller-entity only: the expr `slots.*` namespace is caller-scoped.
        expr = " and ".join(f"slots.{c} is not None" for c in step.collect)
        rules.append(AdvanceRule(when=expr, judge="expr", to=next_id))
    rules.append(
        AdvanceRule(
            when=step.done_when.strip() or "step complete",
            judge="llm",
            to=next_id,
            requires=list(step.collect),
        )
    )
    # Hard gate on ALL steps: Talker barriers on the Director so it always
    # speaks from post-advance state.  The opening greeting is spoken via
    # PlaybookAgent.greet() which passes director_done=None, bypassing the
    # barrier; the first user utterance then barriers and advances normally.
    return Checkpoint(
        id=step.id,
        goal=step.purpose,
        guidance=guidance,
        slots=slots,
        entity=step.entity,
        advance_when=rules,
        gate="hard",
        turn_budget=step.turn_budget or _DEFAULT_TURN_BUDGET,
        uses_kb=step.kb,
    )


def simple_to_playbook(doc: dict[str, Any]) -> Playbook:
    """Compile a simple-format dict into a validated ``Playbook``."""
    sp = SimplePlaybook.model_validate(doc)
    checkpoints: list[Checkpoint] = []
    for i, step in enumerate(sp.playbook):
        is_last = i == len(sp.playbook) - 1
        next_id = None if is_last else f"main.{sp.playbook[i + 1].id}"
        opening = sp.opening if i == 0 else ""
        checkpoints.append(_step_to_checkpoint(step, next_id, opening))
    interrupts = [
        InterruptSpec(
            id=intr.id or f"interrupt_{i}",
            when=intr.when,
            to=intr.to,
            resume=intr.resume,
        )
        for i, intr in enumerate(sp.interrupts)
    ]
    guidelines = GuidelineConfig(
        channel=sp.channel,
        tone=sp.tone,
        language=sp.persona.language or "en",
        call_type=sp.call_type,
        timezone=sp.timezone,
        memory_enabled=sp.memory_enabled,
        followup_enabled=sp.followup_enabled,
    )
    kb = sp.facts.get("knowledge_base") if isinstance(sp.facts, dict) else None
    knowledge_base = (
        yaml.safe_dump(kb, sort_keys=False, allow_unicode=True).strip() if kb else ""
    )
    return Playbook(
        persona=_build_persona(sp),
        multi_entity=sp.multi_entity,
        journeys={"main": Journey(checkpoints=checkpoints)},
        interrupts=interrupts,
        guidelines=guidelines,
        knowledge_base=knowledge_base,
    )


def load_simple(path: str) -> Playbook:
    """Load a simple-format file (YAML or JSON) and compile it to a Playbook."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    doc = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    return simple_to_playbook(doc)
