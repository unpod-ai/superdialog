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
import warnings
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

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


class SimpleBranch(BaseModel):
    when: str
    to: str
    requires: list[str] = Field(default_factory=list)

    @field_validator("when")
    @classmethod
    def _when_not_blank(cls, v: str) -> str:
        # An empty condition renders an empty rule into the Director prompt.
        if not v.strip():
            raise ValueError("branch `when` cannot be empty")
        return v

    @field_validator("to")
    @classmethod
    def _strip_to(cls, v: str) -> str:
        return v.strip()


class SimpleStep(BaseModel):
    id: str
    purpose: str = ""
    say: str = ""
    collect: list[str] = Field(default_factory=list)
    done_when: str = ""
    # Step to advance to when done_when holds. Default: the next step in list
    # order (the original linear-chain behavior).
    then: str = ""
    # Marks this step as a call ending. Compiles to a terminal checkpoint —
    # previously only the LAST list element could end the call, which made
    # fallback steps (whatsapp/callback/DNC) structurally unclosable.
    terminal: bool = False
    outcome: str = "closed"  # recorded on SessionEnd when terminal
    # Which person this step collects for (multi-entity). Defaults to "caller"
    # so single-entity playbooks are unchanged.
    entity: str = "caller"
    # User turns before the runtime steers "wrap this step up" (default 4).
    turn_budget: int | None = None
    # Inject the knowledge base on this step. None = legacy heuristic (say
    # mentions 'knowledge_base'); set false on steps that merely reference it.
    kb: bool | None = None
    # Slots the Director must see filled before advancing. None = auto: all of
    # `collect` for focused capture steps (<=2 slots), NONE for branchy steps
    # collecting many per-path slots (requiring all would deadlock the step).
    require: list[str] | None = None
    # Talker sync barrier. None = "hard" (Talker waits for the Director, so it
    # speaks from post-advance state — the safe default). Set "soft" on
    # pure-talk steps (no collect) to speak immediately: saves the Director's
    # settle time (~1-2s p50) per turn; harmless where nothing is captured.
    gate: Literal["hard", "soft"] | None = None
    # Optional multi-way routing, judged by the Director in author order and
    # AHEAD of the done_when default. Each compiles to one llm AdvanceRule.
    branches: list[SimpleBranch] = Field(default_factory=list)
    # Line to deliver while advancing out of this step (post-capture pitch).
    then_say: str = ""

    @field_validator("then")
    @classmethod
    def _strip_then(cls, v: str) -> str:
        # `then: " close"` would otherwise compile to unknown "main. close".
        return v.strip()


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
    # Opt-in: enable the trajectory-level Supervisor (loop 2 — recovery/redirect
    # meta-agent). Off ⇒ unchanged. See GuidelineConfig.supervisor.
    supervisor: bool = False
    # Continuity v2 escape hatch: true restores pre-v2 semantics (no junk-slot
    # rejection, no churn dampener, no uncorroborated-advance steer, supervisor
    # stays opt-in). New playbooks get v2 by default. See Playbook.legacy_continuity.
    legacy_continuity: bool = False


def is_simple_playbook(doc: Any) -> bool:
    """True when ``doc`` is a simple playbook: top-level ``playbook`` is a list."""
    return (
        isinstance(doc, dict)
        and isinstance(doc.get("playbook"), list)
        and len(doc["playbook"]) > 0
    )


# ISO 639-1 -> readable name for the common languages.
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
# up". No on_failure is compiled here, so _TURN_BUDGET_GRACE turns later
# PlaybookRuntime._apply_turn_budget force-advances to the journey-order
# successor (Playbook.next_checkpoint_id) rather than steering forever.
_DEFAULT_TURN_BUDGET = 4


def _step_to_checkpoint(
    step: SimpleStep, next_id: str | None, opening: str
) -> Checkpoint:
    guidance = step.say.strip() or opening.strip()
    # Which collected slots actually gate the advance: an explicit `require:`
    # wins; else all of `collect` for focused capture steps, none for branchy
    # steps collecting many per-path alternatives (e.g. a 14-slot category
    # qualifier) — requiring all of those would block the advance forever.
    required = (
        step.require
        if step.require is not None
        else (step.collect if len(step.collect) <= 2 else [])
    )
    # gate="soft" on the SLOT (not the checkpoint): requires below then demands
    # filled, not confirmed. Hard inheritance would demand confirmation that
    # verdict writes can't self-provide (they land provisional at hard gates by
    # anti-injection design), blocking advance after the user answered and
    # causing the re-asking this compiler previously avoided via requires=[].
    slots = {
        c: SlotSpec(type="str", description="", required=c in required, gate="soft")
        for c in step.collect
    }
    if next_id is None:
        # Terminal checkpoints carry no advance rules, so branches here would
        # silently compile to nothing — reject instead of losing routing.
        if step.branches:
            raise ValueError(
                f"step {step.id!r}: terminal steps (including a last step "
                "without then:) cannot have branches"
            )
        # A terminal step never advances OUT, so its exit_say would never fire.
        if step.then_say:
            raise ValueError(
                f"step {step.id!r}: then_say is never spoken on a terminal step"
            )
        return Checkpoint(
            id=step.id,
            goal=step.purpose,
            guidance=guidance,
            slots=slots,
            entity=step.entity,
            terminal=True,
            outcome=step.outcome,
            uses_kb=step.kb,
        )
    # requires=collect: the Director may not advance past unfilled slots (a
    # done_when verdict alone could skip capture on terse callers); a blocked
    # advance emits the "still need: X" steer and the Talker's "Still needed"
    # hint, so the agent circles back instead of moving on.
    rules = []
    if required and step.entity == "caller":
        # Companion expr rule ahead of the llm rule: all-required-filled
        # advances deterministically with ZERO LLM cost. It fires on the
        # same-turn quiesce hop when a verdict wrote the slots but forgot
        # "advance" (otherwise a whole extra user round-trip = 3 LLM calls).
        # Caller-entity only: the expr `slots.*` namespace is caller-scoped.
        expr = " and ".join(f"slots.{c} is not None" for c in required)
        rules.append(AdvanceRule(when=expr, judge="expr", to=next_id))
    # Branch rules ahead of the done_when default: the first llm rule whose
    # target the Director's verdict names wins, in author order.
    for b in step.branches:
        rules.append(
            AdvanceRule(
                when=b.when,
                judge="llm",
                to=f"main.{b.to}",
                requires=list(b.requires),
            )
        )
    rules.append(
        AdvanceRule(
            when=step.done_when.strip() or "step complete",
            judge="llm",
            to=next_id,
            requires=list(required),
        )
    )
    # Hard gate by default: Talker barriers on the Director so it always
    # speaks from post-advance state.  The opening greeting is spoken via
    # PlaybookAgent.greet() which passes director_done=None, bypassing the
    # barrier; the first user utterance then barriers and advances normally.
    # A step may opt into gate: soft (pure-talk steps) to skip the barrier.
    return Checkpoint(
        id=step.id,
        goal=step.purpose,
        guidance=guidance,
        slots=slots,
        entity=step.entity,
        advance_when=rules,
        gate=step.gate or "hard",
        turn_budget=step.turn_budget or _DEFAULT_TURN_BUDGET,
        uses_kb=step.kb,
        exit_say=step.then_say,
    )


def _unknown_keys(doc: dict[str, Any]) -> list[str]:
    """Dotted paths of keys the simple format does not recognize.

    pydantic's default extra="ignore" would silently drop these — an authored
    top-level section (e.g. ``language_lock``) or a step typo (``done_wehn``)
    never reaches the runtime while the author believes it is configured.
    model_fields keys are field names; the simple format uses no aliases, so
    direct membership is correct.
    """
    # str(k): a non-string YAML key (e.g. `2024:`) must report cleanly,
    # not TypeError inside ', '.join.
    bad = [str(k) for k in doc if k not in SimplePlaybook.model_fields]
    steps = doc.get("playbook")
    if isinstance(steps, list):
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            bad += [
                f"playbook[{i}].{k}" for k in step if k not in SimpleStep.model_fields
            ]
            # One level further into branches: a typo like `require` would
            # silently drop the branch slot gate — the routing-critical case.
            branches = step.get("branches")
            if isinstance(branches, list):
                for j, br in enumerate(branches):
                    if isinstance(br, dict):
                        bad += [
                            f"playbook[{i}].branches[{j}].{k}"
                            for k in br
                            if k not in SimpleBranch.model_fields
                        ]
    return bad


def simple_to_playbook(doc: dict[str, Any], strict: bool = True) -> Playbook:
    """Compile a simple-format dict into a validated ``Playbook``.

    Unknown keys raise (``strict=True``, default) or warn (``strict=False``)
    instead of being silently dropped by pydantic.
    """
    unknown = _unknown_keys(doc)
    if unknown:
        msg = (
            "simple playbook has keys the format does not support "
            f"(they would be silently dropped): {', '.join(unknown)}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    sp = SimplePlaybook.model_validate(doc)
    checkpoints: list[Checkpoint] = []
    for i, step in enumerate(sp.playbook):
        is_last = i == len(sp.playbook) - 1
        # A self-then livelocks the quiesce loop (expr rule re-fires
        # every hop), so reject it at compile time.
        if step.then and step.then == step.id:
            raise ValueError(f"step {step.id!r}: then cannot target itself")
        if any(b.to == step.id for b in step.branches):
            raise ValueError(f"step {step.id!r}: branch cannot target itself")
        if step.then:
            next_id = f"main.{step.then}"
        else:
            next_id = None if is_last else f"main.{sp.playbook[i + 1].id}"
        if step.terminal:
            next_id = None
        # Non-default outcome on a non-terminal step is silently ignored at
        # runtime (only SessionEnd records it) — reject so authors notice.
        if next_id is not None and step.outcome != "closed":
            raise ValueError(
                f"step {step.id!r}: outcome is only used on terminal steps"
            )
        seen_targets: set[str] = set()
        for b in step.branches:
            if b.to in seen_targets:
                raise ValueError(f"step {step.id!r}: duplicate branch target {b.to!r}")
            seen_targets.add(b.to)
            # The Director resolves verdicts by target and the first llm rule
            # wins, so a branch aliasing the default advance target shadows
            # the done_when rule's slot gate. done_when already covers that
            # path — reject the redundant (and gate-bypassing) branch.
            if next_id == f"main.{b.to}":
                raise ValueError(
                    f"step {step.id!r}: branch targets the default next step "
                    f"{b.to!r}; use done_when for that path"
                )
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
        supervisor=sp.supervisor,
    )
    kb = sp.facts.get("knowledge_base") if isinstance(sp.facts, dict) else None
    knowledge_base = (
        yaml.safe_dump(kb, sort_keys=False, allow_unicode=True).strip() if kb else ""
    )
    return Playbook(
        persona=_build_persona(sp),
        multi_entity=sp.multi_entity,
        legacy_continuity=sp.legacy_continuity,
        journeys={"main": Journey(checkpoints=checkpoints)},
        interrupts=interrupts,
        guidelines=guidelines,
        knowledge_base=knowledge_base,
    )


def load_simple(path: str, strict: bool = True) -> Playbook:
    """Load a simple-format file (YAML or JSON) and compile it to a Playbook."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    doc = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    return simple_to_playbook(doc, strict=strict)
