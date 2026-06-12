"""Playbook: the authored, optimizable conversation artifact (design doc §1/§1b)."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, model_validator


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


class SlotSpec(BaseModel):
    type: Literal["str", "int", "float", "bool", "date", "enum", "array", "object"] = (
        "str"
    )
    required: bool = False
    values: list[str] | None = None  # enum members
    # only the Director may write; never asserted unless present
    authoritative: bool = False
    invalidates: list[str] = Field(default_factory=list)
    description: str = ""


class AdvanceRule(BaseModel):
    when: str
    judge: Literal["llm", "expr"] = "llm"
    to: str
    requires: list[str] = Field(default_factory=list)
    set: dict[str, Any] = Field(default_factory=dict)

    @property
    def rule_id(self) -> str:
        return f"{self.judge}:{self.to}"


class Checkpoint(BaseModel):
    id: str
    goal: str = ""
    slots: dict[str, SlotSpec] = Field(default_factory=dict)
    guidance: str = ""  # may contain Jinja over {slots, views, results}
    say_verbatim: str | None = None  # same Jinja namespace; bypasses the Talker LLM
    never_say: list[str] = Field(default_factory=list)
    advance_when: list[AdvanceRule] = Field(default_factory=list)
    gate: Literal["soft", "hard"] = "soft"
    auto: bool = False  # speak verbatim once, then advance without user input
    pipeline: str | None = None
    on_enter: list[str] = Field(default_factory=list)  # tool ids
    on_failure: str | None = None  # checkpoint id
    terminal: bool = False
    outcome: str | None = None
    turn_budget: int | None = None


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


class ToolSpec(BaseModel):
    id: str
    type: Literal["http", "python"] = "http"
    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)
    store_response_as: str | None = None
    env_updates: dict[str, str] = Field(default_factory=dict)  # env key -> result path
    run_once: bool = False
    when: str | None = None  # expr over state; skip when falsy
    timeout: float = 30.0
    ttl_seconds: float | None = None  # reserved — TTL scheduling is deferred
    on_expire: str | None = None  # reserved — handler id
    args: dict[str, SlotSpec] = Field(default_factory=dict)


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


class Policies(BaseModel):
    silence: SilencePolicy | None = None
    # Max post-filler wait for the Director before the hold line is spoken;
    # short enough that a caller doesn't feel disengaged.
    hold_timeout: float = Field(default=4.0, gt=0)


class Playbook(BaseModel):
    persona: str = ""
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
    initial: str | None = None  # defaults to first checkpoint of first journey

    # -- lookups ------------------------------------------------------------
    def checkpoint(self, ref: str) -> Checkpoint:
        journey, _, cp_id = ref.partition(".")
        for cp in self.journeys[journey].checkpoints:
            if cp.id == cp_id:
                return cp
        raise KeyError(ref)

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

    # -- validation ----------------------------------------------------------
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
            return simple_to_playbook(doc)
        if isinstance(doc, dict) and "nodes" in doc and "initial_node" in doc:
            from superdialog.flow.models import ConversationFlow

            from .compiler import compile_flow

            return compile_flow(ConversationFlow.model_validate(doc))
        return cls.model_validate(doc)

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
        if path.endswith((".yaml", ".yml")):
            return cls.from_yaml(text)
        return cls.from_json(text)
