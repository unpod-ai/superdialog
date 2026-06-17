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

from .models import AdvanceRule, Checkpoint, InterruptSpec, Journey, Playbook, SlotSpec


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


class SimpleObjection(BaseModel):
    trigger: str
    handle: str


class SimpleInterrupt(BaseModel):
    id: str = ""
    when: str
    to: str


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
        dumped = yaml.safe_dump(sp.facts, sort_keys=False, allow_unicode=True)
        parts.append(
            "## Reference facts (never invent beyond these)\n" + dumped.strip()
        )
    if sp.objections:
        bullets = "\n".join(
            f"- If {o.trigger} -> {o.handle.strip()}" for o in sp.objections
        )
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


def _step_to_checkpoint(
    step: SimpleStep, next_id: str | None, opening: str
) -> Checkpoint:
    guidance = step.say.strip() or opening.strip()
    slots = {c: SlotSpec(type="str", description="") for c in step.collect}
    if next_id is None:
        return Checkpoint(
            id=step.id,
            goal=step.purpose,
            guidance=guidance,
            slots=slots,
            terminal=True,
            outcome="closed",
        )
    # No requires: Director advances based solely on done_when condition.
    # Requiring slots blocks advance until the Director's previous-turn note
    # ("still need: X") is cleared, which bleeds into the next Talker turn
    # and causes re-asking. Slots are still extracted independently.
    rule = AdvanceRule(
        when=step.done_when.strip() or "step complete",
        judge="llm",
        to=next_id,
        requires=[],
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
        advance_when=[rule],
        gate="hard",
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
        )
        for i, intr in enumerate(sp.interrupts)
    ]
    return Playbook(
        persona=_build_persona(sp),
        journeys={"main": Journey(checkpoints=checkpoints)},
        interrupts=interrupts,
    )


def load_simple(path: str) -> Playbook:
    """Load a simple-format file (YAML or JSON) and compile it to a Playbook."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    doc = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    return simple_to_playbook(doc)
