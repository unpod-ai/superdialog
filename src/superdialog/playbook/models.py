"""Playbook: the authored, optimizable conversation artifact (design doc §1/§1b)."""

from __future__ import annotations

import json
import logging
import re
import warnings
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


_log = logging.getLogger(__name__)


class _YamlLoader(yaml.SafeLoader):
    """YAML 1.2-style booleans: only true/false; on/off/yes/no stay strings."""


_YamlLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


class GuidelineConfig(BaseModel):
    """Voice-guideline knobs. Every default reproduces pre-feature behavior."""

    channel: Literal["voice", "text"] = "voice"
    tone: Literal["professional", "casual"] = "professional"
    # A language name ("English"), an ISO 639-1 code ("hi"), or a list.
    language: str | list[str] = "en"
    call_type: Literal["sales", "support", "booking"] | None = None
    timezone: str = "UTC"  # IANA name; a model field, NOT an env var.
    memory_enabled: bool = False
    followup_enabled: bool = False
    # Speaker gender — the single source of truth for gendered grammar. Wired
    # from the selected voice profile's gender so the agent's verb/adjective
    # forms (करूँगी vs करूँगा) match the voice. "neutral" emits no gender block.
    gender: Literal["male", "female", "neutral"] = "neutral"
    # DEPRECATED: superseded by the top-level ``Playbook.llm`` block, which a
    # host actually reads (these two never were — see ``LLMConfig`` below).
    # Kept parseable for one release so an in-flight draft doesn't hard-fail;
    # setting either without an ``llm`` block warns rather than silently
    # dropping the author's intent.
    director_model: str | None = None
    talker_model: str | None = None
    # Loop 2: the trajectory-level Supervisor (recovery/redirect meta-agent).
    # None = on (explicit false turns the supervisor off). When on,
    # PlaybookAgent runs it off the speech path, reusing the Director model
    # unless the host passes an explicit ``supervisor_llm``. See
    # PlaybookAgent.__init__.
    supervisor: bool | None = None


class LLMRoleConfig(BaseModel):
    """A single role's provider/model pair (used for the ``director`` override)."""

    provider: str
    model: str

    @property
    def uri(self) -> str:
        return f"{self.provider}/{self.model}"


class LLMConfig(BaseModel):
    """Playbook-declared LLM: the model a host should actually run this on.

    ``director`` is optional — unset, the Director shares the talker's
    provider/model (today's implicit behavior made explicit). This replaces
    ``GuidelineConfig.director_model``/``talker_model``, which no host ever
    read back.
    """

    provider: str
    model: str
    director: LLMRoleConfig | None = None

    @property
    def uri(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def director_uri(self) -> str:
        return self.director.uri if self.director else self.uri


class ResolveFrom(BaseModel):
    """Where the Director sources candidate values for a slot it cannot extract
    from raw speech (e.g. an opaque id the user never utters verbatim).

    The verdict prompt renders ``"<name>" -> <id>`` pairs pulled from
    ``state.tool_results[result].data[list_field]`` so the LLM maps the spoken
    name to the canonical id itself against the live list — no fuzzy code and no
    domain alias table in the engine. Field names are supplied by the playbook,
    keeping this generic across any name->id lookup.
    """

    result: str  # tool_results key holding the candidate list
    list_field: str = "items"  # path under .data holding the candidate list
    name_field: str = "name"  # human-spoken field
    id_field: str = "id"  # canonical id field to output


class SlotSpec(BaseModel):
    type: Literal[
        "str", "int", "float", "bool", "date", "time", "enum", "array", "object"
    ] = "str"
    required: bool = False
    values: list[str] | None = None  # enum members
    # Candidate source for id-like slots the user never speaks verbatim. When
    # set, the Director is given the live list and resolves the spoken value to
    # the matching id (see ResolveFrom).
    resolve_from: ResolveFrom | None = None
    # only the Director may write; never asserted unless present
    authoritative: bool = False
    invalidates: list[str] = Field(default_factory=list)
    description: str = ""
    # Per-slot confirmation gate (risk class). ``None`` inherits the
    # checkpoint's ``gate`` (backward compatible — unannotated slots behave as
    # today). ``"hard"`` marks an intrinsically risky slot (phone/email/payment/
    # routing) that must be Director-confirmed before it gates advance or is
    # spoken; ``"soft"`` lets it advance on a provisional fill even inside a
    # hard-gated checkpoint. See capability ``dialogue-gate-policy``.
    gate: Literal["soft", "hard"] | None = None
    # Continuity v2's junk-value guard (director.py's _is_junk) rejects ""/
    # "none"/"n/a" etc. as a failed extraction, not a real answer -- correct
    # for most slots, but wrong for one whose OWN legitimate confirmed value
    # is "nothing" (e.g. special_requests when the caller declines: '' IS the
    # answer, not a missing one). Set true to exempt "" specifically from
    # junk rejection for this slot; every other junk value is still rejected.
    allow_empty: bool = False
    # `type: date` only: reject a resolved date earlier than the call anchor.
    # The Director invents years under latency pressure -- a caller saying
    # "this Saturday, the 10th of June" was extracted as "2024-06-10", two
    # years before the call, and the availability request went out for it. A
    # well-formed ISO date is not necessarily a bookable one.
    #
    # OPT-IN, not default. Rejecting the past by default was tried and is
    # wrong: date_of_birth is a date slot whose whole purpose is a past value,
    # and it silently stopped advancing. Only the author knows which of their
    # date slots is forward-looking, so they declare it.
    future_only: bool = False
    # Per-slot override of the session's G37 anchor mode. "inherit" (default)
    # uses the Director's own setting, so nothing changes unless an author asks.
    #
    # Exists because the anchor is only meaningful per slot. Flipping the whole
    # session to "enforce" was tried and was actively harmful: it rejects
    # DERIVED slots, which the caller never utters. "8 AM" legitimately becomes
    # time_from=07:00 / time_to=09:00 (a +/-1h search window), and "07:00" is
    # not in "8 am", so those correct writes were dropped, the availability call
    # never fired, and the Talker -- starved of data -- invented MORE prices
    # than before. Set "enforce" only on CALLER-STATED slots, where the Director
    # can supply a real span (see Director._anchor_ok: for a non-date/time slot
    # a valid span anchors the write whatever its normalized form, so
    # preferred_time="08:00" from the words "8 AM" still lands, while an
    # invented "afternoon" -- which has no words to point at -- does not).
    anchor: Literal["inherit", "off", "shadow", "enforce"] = "inherit"

    @model_validator(mode="after")
    def _date_defaults_hard(self) -> "SlotSpec":
        # A date/time slot drives a booking; confirm-before-action is realized
        # by a hard gate. Default it to hard so a date slot is hard even inside
        # an explicitly-soft collection checkpoint. An author may still set
        # `gate: soft` on the slot to override.
        if self.type == "date" and self.gate is None:
            self.gate = "hard"
        return self


class AdvanceRule(BaseModel):
    when: str
    judge: Literal["llm", "expr"] = "llm"
    to: str
    requires: list[str] = Field(default_factory=list)
    set: dict[str, Any] = Field(default_factory=dict)

    @property
    def rule_id(self) -> str:
        return f"{self.judge}:{self.to}"


_ENTITY_RE = re.compile(r"^[a-z_][a-z0-9_]*\Z")


class Checkpoint(BaseModel):
    id: str
    goal: str = ""
    # Which person this checkpoint's slots describe; safe as a storage-key
    # prefix. Defaults to "caller" so single-entity playbooks are unchanged.
    entity: str = "caller"
    slots: dict[str, SlotSpec] = Field(default_factory=dict)
    guidance: str = ""  # may contain Jinja over {slots, views, results}
    say_verbatim: str | None = None  # same Jinja namespace; bypasses the Talker LLM
    never_say: list[str] = Field(default_factory=list)
    # Spoken on the turn that LEAVES this checkpoint via a director rule:
    # rendered and injected as a one-shot steer so capture-then-pitch steps
    # work (the expr companion advance otherwise skips any post-capture talk).
    exit_say: str = ""
    advance_when: list[AdvanceRule] = Field(default_factory=list)
    gate: Literal["soft", "hard"] = "hard"
    auto: bool = False  # speak verbatim once, then advance without user input
    strict: bool = (
        False  # speak say_verbatim word-for-word; never paraphrase via the LLM
    )
    handover: bool = False  # inject the handover summary instruction at this step
    pipeline: str | None = None
    on_enter: list[str] = Field(default_factory=list)  # tool ids
    on_failure: str | None = None  # checkpoint id
    terminal: bool = False
    outcome: str | None = None
    turn_budget: int | None = None
    # Inject the knowledge base into this step's Talker prompt. None = legacy
    # heuristic (guidance mentions 'knowledge_base'); set explicitly to keep
    # the (large) KB off hot steps whose guidance merely references it.
    uses_kb: bool | None = None
    # Conversation-history strategy for this step. ``None`` inherits
    # ``policies.context`` (itself defaulting to "append", i.e. no change).
    #
    # The author's test for setting ``reset`` is NOT "is this a new topic" --
    # it is "is this a topic start the caller is NOT coming back from". A
    # ``resume: true`` interrupt target is topic-sized and still wants
    # ``append``: reset on the way in hides the aside that triggered it, and
    # again on the way out makes the caller repeat the task they were mid-way
    # through. Good candidates are a close-then-new-request boundary or a
    # switch to an unrelated task (booking -> cancellation).
    context: ContextStrategy | None = None
    # Only read when the effective strategy is ``reset_with_summary``: what the
    # compactor should emphasise when folding the pre-entry turns into the
    # summary (a booking step wants order details, an objection detour wants
    # the caller's concerns). Empty uses compact.py's default instruction.
    summary_prompt: str = ""

    @field_validator("entity")
    @classmethod
    def _entity_is_safe_key_prefix(cls, v: str) -> str:
        if not _ENTITY_RE.match(v):
            raise ValueError(
                f"entity {v!r} must match [a-z_][a-z0-9_]* (safe key prefix)"
            )
        return v


class Journey(BaseModel):
    checkpoints: list[Checkpoint] = Field(min_length=1)


class DispatchEntry(BaseModel):
    intent: str
    to: str
    requires: list[str] = Field(default_factory=list)


class RetrySpec(BaseModel):
    # Capped: an unbounded retry from a buggy compiler would become an HTTP
    # hot loop inside a live call (middleware can triple the call count).
    retry: int = Field(0, ge=0, le=10)
    on_exhaust: str | None = None  # checkpoint id


StepOutcome = Union[str, RetrySpec]  # "continue" | checkpoint id | RetrySpec


class PipelineStep(BaseModel):
    tool: str
    on: dict[str, StepOutcome] = Field(default_factory=dict)  # ok|failed|http_<code>


class PipelineSpec(BaseModel):
    id: str
    steps: list[PipelineStep]


#: HTTP methods that are safe/idempotent by spec (RFC 9110).
SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


class ToolSpec(BaseModel):
    id: str
    type: Literal["http", "python"] = "http"
    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)
    store_response_as: str | None = None
    env_updates: dict[str, str] = Field(default_factory=dict)  # env key -> result path
    slot_updates: dict[str, str] = Field(
        default_factory=dict
    )  # slot key -> result path
    run_once: bool = False
    when: str | None = None  # expr over state; skip when falsy
    timeout: float = 30.0
    ttl_seconds: float | None = None  # reserved — TTL scheduling is deferred
    on_expire: str | None = None  # reserved — handler id
    args: dict[str, SlotSpec] = Field(default_factory=dict)
    # Reversibility tier (Shepherd-style effect classes). Governs conversation
    # rewind and pre-materialization interception:
    # - reversible: pure state (reads); a rewind undoes it structurally.
    # - compensable: has a real-world side effect that ``compensate`` (a tool
    #   id) can undo — rewinding across it requires caller confirmation, then
    #   runs the compensation tool.
    # - irreversible: cannot be undone; a rewind across it is refused.
    # ``None`` (default) infers conservatively at rewind time (safe methods →
    # reversible, writes → compensable-without-compensate, i.e. rewind refuses)
    # and — deliberately — never activates the interception guard, so
    # unannotated playbooks behave exactly as before.
    tier: Literal["reversible", "compensable", "irreversible"] | None = None
    compensate: str | None = None  # tool id that undoes an ok result on rewind

    @property
    def effective_tier(self) -> str:
        """Tier used by rewind safety: explicit tier, else a conservative guess."""
        if self.tier:
            return self.tier
        if self.type == "http" and self.method.upper() in SAFE_HTTP_METHODS:
            return "reversible"
        return "compensable"  # unannotated writes/python tools: assume effects


class MiddlewareSpec(BaseModel):
    on_status: int = 401
    refresh_with: str = ""  # tool id
    then: Literal["replay"] = "replay"


class HandlerSpec(BaseModel):
    id: str
    on: str  # "webhook.<name>" | "timer.<name>"
    pipeline: str


class InterruptSpec(BaseModel):
    id: str
    when: str
    judge: Literal["llm", "event"] = "llm"
    to: str
    resume: bool = False  # resume=True restoration is deferred; golf needs False only


class SilencePolicy(BaseModel):
    max_prompts: int = 2
    prompts: list[str] = Field(default_factory=list)
    then: str = ""  # checkpoint id


#: How a checkpoint treats the conversation history it inherits.
#:
#: * ``append`` -- keep it (today's behavior, and the default everywhere).
#: * ``reset`` -- show only what was said since this checkpoint was entered.
#: * ``reset_with_summary`` -- as ``reset``, plus the pre-entry turns folded
#:   into the protected ``state.summary`` by the off-path compactor.
#:
#: Pipecat Flows exposes the same three per node. Their nodes are coarse (one
#: per topic), so per-node reset ~ per-topic reset. Ours are fine -- a topic is
#: typically 4-7 checkpoints -- so a blanket reset would fire several times
#: INSIDE one task and make the caller repeat themselves. Hence: default
#: ``append``, opt in at real boundaries.
ContextStrategy = Literal["append", "reset", "reset_with_summary"]


class Policies(BaseModel):
    silence: SilencePolicy | None = None
    # Playbook-wide default strategy; a checkpoint's own ``context`` wins.
    # Mirrors Pipecat's global FlowManager config + per-node override, so a
    # playbook wanting "reset everywhere except three places" doesn't have to
    # annotate every checkpoint.
    context: ContextStrategy = "append"
    # Max post-filler wait for the Director before the hold line is spoken;
    # short enough that a caller doesn't feel disengaged.
    hold_timeout: float = Field(default=4.0, gt=0)
    # Extra wait AFTER the hold line for a hard-gated pipeline that's still
    # working (not down) -- 0 (default) ends the turn on the hold line exactly
    # as before. A playbook whose checkpoints chain several external HTTP
    # calls (routinely 3-8s total) sets this so the Director resolving inside
    # this window still flows into real spoken content instead of being
    # orphaned: the turn already ended on the hold line with nothing left to
    # speak the pipeline's result once it lands.
    extended_timeout: float = Field(default=0.0, ge=0)
    # Author-facing barrier lines (spoken at hard-gated checkpoints while the
    # Director settles). None keeps the Talker's built-in English defaults —
    # a persona whose call runs in another language authors these here
    # instead of a host wiring per-playbook Python overrides.
    filler: str | None = None
    hold_line: str | None = None


class PronunciationSpec(BaseModel):
    """An authored pronunciation rule (additive; backward compatible).

    Mirrors the runtime ``PronunciationRule`` shape so authored entries are
    directly consumable by the voice runtime's ``PronunciationManager``.
    Respelling-first (``respelling`` → runtime ``replacement``); IPA is advisory.
    """

    word: str
    language: str = "en"
    respelling: str | None = None
    ipa: str | None = None
    provider: str | None = None
    context: Literal["auto", "force", "skip"] = "auto"
    enabled: bool = True

    def to_rule(self) -> dict[str, Any]:
        """Map to the runtime ``PronunciationRule`` field names."""
        return {
            "word": self.word,
            "language": self.language,
            "provider": self.provider,
            "replacement": self.respelling,
            "ipa": self.ipa,
            "enabled": self.enabled,
            "context": self.context,
        }


class Playbook(BaseModel):
    persona: str = ""
    # Opt-in: scope slot storage/lookups per checkpoint entity. Off ⇒ today.
    multi_entity: bool = False
    # DEPRECATED and ignored: v3 semantics are the only semantics. The field
    # stays declared so existing playbook YAML still parses; setting it True
    # logs a deprecation warning (see _warn_deprecated_llm_fields).
    legacy_continuity: bool = False
    guidelines: GuidelineConfig = Field(default_factory=GuidelineConfig)
    # The model a host should actually run this playbook on. ``None`` (the
    # default) preserves today's behavior byte-for-byte: the host's caller-
    # supplied session model applies to both roles, unvalidated.
    llm: LLMConfig | None = None
    # Authored pronunciation rules (additive; empty preserves prior behavior).
    pronunciations: list[PronunciationSpec] = Field(default_factory=list)
    journeys: dict[str, Journey] = Field(min_length=1)
    dispatch: list[DispatchEntry] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    pipelines: list[PipelineSpec] = Field(default_factory=list)
    handlers: list[HandlerSpec] = Field(default_factory=list)
    interrupts: list[InterruptSpec] = Field(default_factory=list)
    policies: Policies = Field(default_factory=Policies)
    middleware: MiddlewareSpec | None = None
    env: dict[str, str] = Field(default_factory=dict)
    views: dict[str, str] = Field(default_factory=dict)  # name -> expr
    knowledge_base: str = ""  # global KB text (Jinja-renderable); injected into
    # every Talker prompt so off-flow questions are answered in-context, then the
    # flow resumes. Empty (the default) leaves the prompt byte-identical to before.
    initial: str | None = None  # defaults to first checkpoint of first journey
    source_path: str = ""  # set by Playbook.load(); empty when built from text

    # -- lookups ------------------------------------------------------------
    def checkpoint(self, ref: str) -> Checkpoint:
        journey, _, cp_id = ref.partition(".")
        for cp in self.journeys[journey].checkpoints:
            if cp.id == cp_id:
                return cp
        raise KeyError(ref)

    def context_for(self, cp: "Checkpoint | None") -> ContextStrategy:
        """Effective history strategy: checkpoint override, else the default.

        ``None`` for the checkpoint (no current step) resolves to the playbook
        default too, so a caller never has to special-case it.
        """
        if cp is not None and cp.context is not None:
            return cp.context
        return self.policies.context

    def next_checkpoint_id(self, ref: str) -> str | None:
        """Journey-order successor of ``ref``; None if it is the last one.

        The turn-budget backstop routes here when a checkpoint declares no
        ``on_failure``: authored list order is the happy-path sequence, so the
        successor is where the flow was heading next anyway.
        """
        # ponytail: list order == happy path (branch steps live at the tail);
        # authors who interleave branches should set on_failure explicitly.
        journey, _, cp_id = ref.partition(".")
        checkpoints = self.journeys[journey].checkpoints
        for i in range(len(checkpoints) - 1):
            if checkpoints[i].id == cp_id:
                return f"{journey}.{checkpoints[i + 1].id}"
        return None

    @property
    def initial_checkpoint_id(self) -> str:
        if self.initial:
            return self.initial
        journey = next(iter(self.journeys))
        return f"{journey}.{self.journeys[journey].checkpoints[0].id}"

    def checkpoint_ids(self) -> set[str]:
        return {
            f"{jname}.{cp.id}"
            for jname, j in self.journeys.items()
            for cp in j.checkpoints
        }

    def tool(self, tool_id: str) -> ToolSpec:
        for t in self.tools:
            if t.id == tool_id:
                return t
        raise KeyError(tool_id)

    def pipeline(self, pipeline_id: str) -> PipelineSpec:
        for p in self.pipelines:
            if p.id == pipeline_id:
                return p
        raise KeyError(pipeline_id)

    def slot_spec(self, key: str) -> SlotSpec | None:
        for j in self.journeys.values():
            for cp in j.checkpoints:
                if key in cp.slots:
                    return cp.slots[key]
        return None

    def llm_uri(self) -> str | None:
        """The talker model URI this playbook declares, or ``None`` if unset."""
        return self.llm.uri if self.llm else None

    def director_llm_uri(self) -> str | None:
        """The director model URI: ``llm.director`` if set, else the talker's."""
        return self.llm.director_uri if self.llm else None

    async def resolve_llm_providers(
        self, *, override: str | None = None
    ) -> tuple[Any, Any]:
        """Resolve ready-to-use ``(talker_llm, director_llm)`` providers.

        One call does everything a host script needs: priority (an explicit
        ``override`` wins, with a warning if this playbook ALSO declares
        ``llm:`` — it's being shadowed; else the playbook's own ``llm:``
        block; neither present raises), live-API validation per model (see
        :func:`superdialog.llm.check_model_available` — a confirmed-invalid
        model raises, an unverifiable one only warns), and resolution to
        actual provider instances via :func:`superdialog.llm.resolve_llm`.
        Self-contained: no host framework (e.g. a playground) required.
        """
        from ..llm import check_model_available, resolve_llm

        declared_talker = self.llm_uri()
        declared_director = self.director_llm_uri()

        if override:
            if declared_talker:
                warnings.warn(
                    f"playbook declares llm={declared_talker!r} but an "
                    f"explicit override {override!r} takes priority",
                    UserWarning,
                    stacklevel=2,
                )
            talker_uri = director_uri = override
        elif declared_talker:
            talker_uri = declared_talker
            director_uri = declared_director or declared_talker
        else:
            raise ValueError(
                "No LLM defined: pass override=, or add an "
                "`llm: {provider: ..., model: ...}` block to the playbook YAML."
            )

        for uri in {talker_uri, director_uri}:
            verdict = await check_model_available(uri)
            if verdict is False:
                raise ValueError(
                    f"Model {uri!r} isn't recognized by its provider's API — "
                    "check the provider/model spelling."
                )
            if verdict is None:
                warnings.warn(
                    f"couldn't verify {uri!r} against its provider (no API "
                    "key found) — proceeding unverified",
                    UserWarning,
                    stacklevel=2,
                )

        talker_llm = resolve_llm(talker_uri)
        director_llm = (
            talker_llm if director_uri == talker_uri else resolve_llm(director_uri)
        )
        return talker_llm, director_llm

    # -- validation ----------------------------------------------------------
    @model_validator(mode="after")
    def _warn_deprecated_llm_fields(self) -> "Playbook":
        if self.legacy_continuity:
            # logging (not warnings.warn): survives default warning filters on
            # server hosts, so operators actually see it in production logs.
            logging.getLogger(__name__).warning(
                "legacy_continuity is deprecated and ignored; v3 semantics apply"
            )
        deprecated = [
            name
            for name in ("director_model", "talker_model")
            if getattr(self.guidelines, name, None)
        ]
        if deprecated and self.llm is None:
            warnings.warn(
                "guidelines."
                + " / guidelines.".join(deprecated)
                + " is deprecated and no longer read by any host — set the"
                " top-level `llm:` block instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _check_references(self) -> "Playbook":
        def need_unique(seen: set[str], item_id: str, ctx: str) -> None:
            if item_id in seen:
                raise ValueError(f"{ctx}: duplicate id {item_id!r}")
            seen.add(item_id)

        for jname, j in self.journeys.items():
            if "." in jname:
                raise ValueError(f"journey name must not contain '.': {jname!r}")
            cp_seen: set[str] = set()
            for cp in j.checkpoints:
                need_unique(cp_seen, cp.id, f"journey {jname!r}")
        tool_seen: set[str] = set()
        for t in self.tools:
            need_unique(tool_seen, t.id, "tools")
            # "pipeline" is the runtime's reserved result key gating the
            # pipeline.ok/pipeline.failed expr namespace — never clobber it.
            if t.store_response_as == "pipeline":
                raise ValueError(
                    f"tool {t.id!r}: store_response_as 'pipeline' is reserved"
                )
            if t.compensate:
                if t.tier != "compensable":
                    raise ValueError(
                        f"tool {t.id!r}: compensate requires tier 'compensable'"
                    )
                if t.compensate == t.id:
                    raise ValueError(f"tool {t.id!r}: compensate cannot be itself")
        pipe_seen: set[str] = set()
        for p in self.pipelines:
            need_unique(pipe_seen, p.id, "pipelines")

        ids = self.checkpoint_ids()
        pipeline_ids = {p.id for p in self.pipelines}
        tool_ids = {t.id for t in self.tools}

        if self.middleware and self.middleware.refresh_with not in tool_ids:
            raise ValueError(
                "middleware.refresh_with: unknown tool "
                f"{self.middleware.refresh_with!r}"
            )

        for t in self.tools:
            if t.compensate and t.compensate not in tool_ids:
                raise ValueError(
                    f"tool {t.id!r}: unknown compensate tool {t.compensate!r}"
                )

        def need_cp(ref: str, ctx: str) -> None:
            if ref not in ids:
                raise ValueError(f"{ctx}: unknown checkpoint {ref!r}")

        # A typo'd requires key at a hard gate would deadlock the checkpoint:
        # every key must be declared on some checkpoint or set by its own rule.
        declared_slots = {
            key
            for j in self.journeys.values()
            for cp in j.checkpoints
            for key in cp.slots
        }
        for jname, j in self.journeys.items():
            for cp in j.checkpoints:
                for rule in cp.advance_when:
                    need_cp(rule.to, f"{jname}.{cp.id} advance_when")
                    for req in rule.requires:
                        if req not in declared_slots and req not in rule.set:
                            raise ValueError(
                                f"{jname}.{cp.id} advance_when: requires key "
                                f"{req!r} is not declared in any checkpoint's "
                                "slots nor set by the rule"
                            )
                if cp.pipeline and cp.pipeline not in pipeline_ids:
                    raise ValueError(
                        f"{jname}.{cp.id}: unknown pipeline {cp.pipeline!r}"
                    )
                if cp.on_failure:
                    need_cp(cp.on_failure, f"{jname}.{cp.id} on_failure")
                for t in cp.on_enter:
                    if t not in tool_ids:
                        raise ValueError(f"{jname}.{cp.id}: unknown tool {t!r}")
        for d in self.dispatch:
            need_cp(d.to, "dispatch")
        for itr in self.interrupts:
            need_cp(itr.to, f"interrupt {itr.id}")
        for h in self.handlers:
            if h.pipeline not in pipeline_ids:
                raise ValueError(f"handler {h.id}: unknown pipeline {h.pipeline!r}")
        for p in self.pipelines:
            for step in p.steps:
                if step.tool not in tool_ids:
                    raise ValueError(f"pipeline {p.id}: unknown tool {step.tool!r}")
                for outcome in step.on.values():
                    if isinstance(outcome, str) and outcome != "continue":
                        need_cp(outcome, f"pipeline {p.id}")
                    elif isinstance(outcome, RetrySpec) and outcome.on_exhaust:
                        need_cp(outcome.on_exhaust, f"pipeline {p.id}")
        if self.policies.silence and self.policies.silence.then:
            need_cp(self.policies.silence.then, "policies.silence.then")
        if self.initial:
            need_cp(self.initial, "initial")
        return self

    # -- io -------------------------------------------------------------------
    @classmethod
    def _from_doc(cls, doc: Any) -> "Playbook":
        """Validate a parsed document, lowering other formats first.

        Three authoring surfaces land here: simple-format docs (top-level
        ``playbook`` list) and legacy flow docs (``nodes`` +
        ``initial_node``) are compiled; full-format docs validate directly.
        """
        # Lazy imports: both frontends import this module's models.
        from .simple import is_simple_playbook, simple_to_playbook

        if is_simple_playbook(doc):
            return cls._warn_if_kb_oversized(simple_to_playbook(doc))
        if isinstance(doc, dict) and "nodes" in doc and "initial_node" in doc:
            from superdialog.flow.models import ConversationFlow

            from .compiler import compile_flow

            return cls._warn_if_kb_oversized(
                compile_flow(ConversationFlow.model_validate(doc))
            )
        return cls._warn_if_kb_oversized(cls.model_validate(doc))

    @staticmethod
    def _warn_if_kb_oversized(pb: "Playbook") -> "Playbook":
        """Surface an oversized knowledge_base at LOAD, not mid-call.

        render.render_view truncates an over-cap KB and logs a warning -- but
        that only fires once a caller is already on the line and the step that
        needed the missing facts is the step that lost them. Authoring-time is
        where this is cheap to fix, so say it when the playbook is read.

        This measures the RAW field, which is an upper bound: the KB is a Jinja
        template, so a playbook that scopes sections with {% if %} may render
        smaller per checkpoint than this. A warning here means "verify your
        scoping", not necessarily "you are truncating today".

        Never raises -- an over-cap KB still runs (truncated), and refusing to
        load would take a live deployment down over a prompt-packing problem.
        """
        # Lazy: render imports this module, so a top-level import would cycle.
        from .render import _KB_MAX_TOKENS, estimate_tokens

        if not pb.knowledge_base:
            return pb
        size = estimate_tokens(pb.knowledge_base)
        if size > _KB_MAX_TOKENS:
            _log.warning(
                "[playbook] knowledge_base is ~%d estimated tokens, over the "
                "%d cap -- it will be truncated from the tail on every "
                "uses_kb checkpoint. Scope the content per step (move "
                "objection scripts and topic FAQs onto the checkpoints that "
                "answer them) instead of relying on truncation.",
                size,
                _KB_MAX_TOKENS,
            )
        return pb

    @classmethod
    def from_yaml(cls, text: str) -> "Playbook":
        return cls._from_doc(yaml.load(text, Loader=_YamlLoader))

    @classmethod
    def from_json(cls, text: str) -> "Playbook":
        return cls._from_doc(json.loads(text))

    @classmethod
    def load(cls, path: str) -> "Playbook":
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        pb = (
            cls.from_yaml(text)
            if path.endswith((".yaml", ".yml"))
            else cls.from_json(text)
        )
        return pb.model_copy(update={"source_path": path})
