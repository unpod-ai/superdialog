"""Compile legacy ConversationFlow graphs into Playbooks (design doc §6)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from superdialog.flow.enums import ActionTriggerType
from superdialog.flow.models import ConversationFlow, CustomAction, Edge, FlowNode
from superdialog.playbook.models import (
    AdvanceRule,
    Checkpoint,
    DispatchEntry,
    HandlerSpec,
    InterruptSpec,
    Journey,
    MiddlewareSpec,
    PipelineSpec,
    PipelineStep,
    Playbook,
    Policies,
    SilencePolicy,
    SlotSpec,
    ToolSpec,
)

NodeKind = Literal["conversational", "computational", "system"]


class FlowIndex:
    """Degree/classification index over a legacy flow graph.

    Builds an indegree map (counting both per-node edges and
    ``global_edges``) and a reverse-edge map, then classifies each node:

    - **system**: indegree 0 and not the initial node — only reachable
      via out-of-band triggers (webhooks, timers).
    - **computational**: routers or ``auto_proceed`` nodes — silent
      steps that never wait for the caller.
    - **conversational**: everything else — nodes that speak/listen.
    """

    def __init__(self, flow: ConversationFlow) -> None:
        self.flow = flow
        self._nodes: dict[str, FlowNode] = {n.id: n for n in flow.nodes}
        self.indegree: dict[str, int] = {n.id: 0 for n in flow.nodes}
        self.reverse_edges: dict[str, list[tuple[str, str]]] = {
            n.id: [] for n in flow.nodes
        }
        for node in flow.nodes:
            for edge in node.edges:
                target = edge.target_node_id
                if target in self.indegree:
                    self.indegree[target] += 1
                    self.reverse_edges[target].append((node.id, edge.id))
        for ge in flow.global_edges:
            if ge.target_node_id in self.indegree:
                self.indegree[ge.target_node_id] += 1

    def node(self, node_id: str) -> FlowNode:
        """Return the node with ``node_id`` (KeyError if absent)."""
        return self._nodes[node_id]

    def classify(self, node: FlowNode) -> NodeKind:
        """Classify a node as conversational, computational, or system."""
        if self.indegree.get(node.id, 0) == 0 and node.id != self.flow.initial_node:
            return "system"
        if node.node_type == "router":
            return "computational"
        # auto_proceed nodes with on-enter tools are computational: their job is
        # running tools, not speaking. Only tool-free auto_proceed nodes that
        # carry spoken content (instruction / static_text) are promoted to
        # conversational so their text is not silently dropped during folding.
        if node.auto_proceed:
            has_content = bool(node.instruction or node.static_text)
            has_tools = bool(_on_enter_ids(node))
            if not has_content or has_tools:
                return "computational"
        return "conversational"


# -- edge condition → AdvanceRule ---------------------------------------------

# Legacy flows spell equality as "==", "=", or prose "is".
_EQ = r"(?:==|=|\bis\b)"
_SUCCESS_RE = re.compile(rf"^\s*(\w+)\.success\s*{_EQ}\s*(true|false)\s*$", re.I)
_NOT_SUCCESS_RE = re.compile(r"^\s*not\s+(\w+)\.success\s*$", re.I)
_STATUS_RE = re.compile(rf"^\s*(\w+)\.status\s*{_EQ}\s*(\d{{3}})\s*$", re.I)
# Only glosses that are pure ROUTING NARRATION may be stripped (allowlist —
# anything else might qualify the predicate and must keep the llm judge).
_NARRATION_GLOSS_RE = re.compile(
    r"^(route(s)?\s+(to|back)|go(es)?\s+to|proceed(s|ing)?\b|"
    r"fall(s|ing)?\s+back|then\b|retry\b|continue(s)?\b)",
    re.I,
)


def _translate_predicate(text: str, store_keys: set[str]) -> str | None:
    """Translate one anchored data predicate to a runtime expr, or None."""
    if m := _SUCCESS_RE.match(text):
        key, value = m.group(1), m.group(2).lower()
        if key in store_keys:
            ok = f"results.{key}.ok"
            return ok if value == "true" else f"not {ok}"
    if m := _NOT_SUCCESS_RE.match(text):
        if m.group(1) in store_keys:
            return f"not results.{m.group(1)}.ok"
    if m := _STATUS_RE.match(text):
        if m.group(1) in store_keys:
            return f"results.{m.group(1)}.status == {m.group(2)}"
    return None


def compile_edge_condition(
    condition: str, store_keys: set[str], target: str
) -> AdvanceRule:
    """Compile a legacy edge condition into an AdvanceRule.

    Single-clause deterministic data predicates over known
    ``store_response_as`` keys become ``judge: expr`` rules:

    - ``X.success == true``  → ``results.X.ok``
    - ``X.success == false`` / ``not X.success`` → ``not results.X.ok``
    - ``X.status == NNN``    → ``results.X.status == NNN``

    Equality may be spelled ``==``, ``=``, or prose ``is`` (the legacy
    golf flow writes "X.success is true"). A trailing em-dash gloss is
    stripped before matching ONLY when it is pure routing narration
    ("route to retry", "fall back to ..."): the safe allowlist. Any
    other gloss might qualify the predicate ("— unless the caller
    already paid"), so the whole condition stays with the llm judge.

    Compound conditions ("A and B"), unknown keys, and anything else not
    confidently translatable stay ``judge: llm`` with the prose passed
    through verbatim — lossless beats clever; the Director can judge data
    conditions too, just slower.
    """
    expr = _translate_predicate(condition, store_keys)
    if expr is None:
        head, dash, gloss = condition.partition(" — ")
        if dash and _NARRATION_GLOSS_RE.match(gloss.strip()):
            expr = _translate_predicate(head, store_keys)
    if expr is not None:
        return AdvanceRule(when=expr, judge="expr", to=target)
    return AdvanceRule(when=condition, judge="llm", to=target)


# -- edge input_schema → slot union -------------------------------------------

_JSON_TYPE: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "array",
    "object": "object",
}


def _slot_spec_from_property(prop: Any) -> SlotSpec:
    """Map one JSON-Schema property to an optional SlotSpec."""
    if not isinstance(prop, dict):
        return SlotSpec()
    enum = prop.get("enum")
    description = prop.get("description") or ""
    if isinstance(enum, list):
        return SlotSpec(
            type="enum", values=[str(v) for v in enum], description=description
        )
    json_type = _JSON_TYPE.get(prop.get("type", ""), "str")
    # The keys of _JSON_TYPE are exactly SlotSpec's literal members.
    return SlotSpec(type=json_type, description=description)  # type: ignore[arg-type]


def union_slot_schemas(
    node: FlowNode,
) -> tuple[dict[str, SlotSpec], dict[str, list[str]]]:
    """Union a node's edge ``input_schema`` properties into optional slots.

    Returns ``(slots, requires_by_edge)``:

    - ``slots``: every property declared by any edge schema, as an
      OPTIONAL ``SlotSpec`` (``required=False``) — per-branch requirements
      live on the rule, not the slot. On conflicting redeclarations the
      first declaration wins (consistent with ``Playbook.slot_spec``).
    - ``requires_by_edge``: edge id → that schema's ``required`` list,
      for every edge that has a (non-empty) schema.
    """
    slots: dict[str, SlotSpec] = {}
    requires_by_edge: dict[str, list[str]] = {}
    for edge in node.edges:
        schema = edge.input_schema
        if not isinstance(schema, dict) or not schema:
            continue
        properties = schema.get("properties") or {}
        for key, prop in properties.items():
            if key not in slots:
                slots[key] = _slot_spec_from_property(prop)
        requires_by_edge[edge.id] = list(schema.get("required") or [])
    return slots, requires_by_edge


# -- template rewriting --------------------------------------------------------

# Jinja syntax words and literals that must never be namespaced.
_JINJA_RESERVED = frozenset(
    {
        "if",
        "else",
        "elif",
        "endif",
        "for",
        "endfor",
        "in",
        "and",
        "or",
        "not",
        "is",
        "defined",
        "undefined",
        "none",
        "None",
        "true",
        "false",
        "True",
        "False",
        "null",
        "loop",
        "range",
        "set",
        "endset",
    }
)
_BLOCK_RE = re.compile(r"\{\{(.*?)\}\}|\{%(.*?)%\}", re.S)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SUCCESS_ATTR_RE = re.compile(r"\.success\b")


def _prev_nonspace(text: str, i: int) -> str:
    for j in range(i - 1, -1, -1):
        if not text[j].isspace():
            return text[j]
    return ""


def _next_nonspace2(text: str, i: int) -> str:
    for j in range(i, len(text)):
        if not text[j].isspace():
            return text[j : j + 2]
    return ""


def _rewrite_expr_body(body: str, env_keys: set[str], result_keys: set[str]) -> str:
    """Namespace the head identifiers of one Jinja expression body."""
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in "'\"":  # string literal: copy verbatim
            end = body.find(ch, i + 1)
            end = len(body) - 1 if end == -1 else end
            out.append(body[i : end + 1])
            i = end + 1
            continue
        m = _IDENT_RE.match(body, i)
        if not m:
            out.append(ch)
            i += 1
            continue
        name = m.group(0)
        nxt = _next_nonspace2(body, m.end())
        head = (
            name not in _JINJA_RESERVED
            # attribute access (x.name) and filter names (x|name) stay
            and _prev_nonspace(body, i) not in {".", "|"}
            # kwarg (attribute='b') stays; `==` comparison does not
            and not (nxt.startswith("=") and not nxt.startswith("=="))
        )
        if not head:
            out.append(name)
            i = m.end()
        elif name in env_keys:
            out.append(f"env.{name}")
            i = m.end()
        elif name in result_keys:
            out.append(f"results.{name}")
            i = m.end()
            # the executor stores results as {ok, status, data, error}:
            # the legacy ".success" field is ".ok" in the new namespace
            if _SUCCESS_ATTR_RE.match(body, i):
                out.append(".ok")
                i += len(".success")
        else:
            out.append(f"slots.{name}")
            i = m.end()
    return "".join(out)


def _rewrite_template(text: str, env_keys: set[str], result_keys: set[str]) -> str:
    """Rewrite bare legacy template names into the executor namespace.

    Legacy flow templates reference bare names ({{ACCESS_TOKEN}},
    {{availability_result.data.slots}}, {{city}}); the new executor and
    renderer expose namespaced lanes instead. Inside every ``{{ ... }}``
    and ``{% ... %}`` block, each head identifier (not an attribute, not
    a filter name, not a kwarg, not a Jinja keyword) is rewritten:

    - env var names (declared env + ``env_updates`` keys) -> ``env.NAME``
    - ``store_response_as`` keys -> ``results.KEY`` (and a following
      ``.success`` becomes ``.ok``, matching the stored result shape)
    - everything else -> ``slots.NAME``

    String literals, filters, and kwargs are preserved verbatim. Text
    outside template blocks is untouched.
    """

    def _sub(m: re.Match[str]) -> str:
        if m.group(1) is not None:
            return "{{" + _rewrite_expr_body(m.group(1), env_keys, result_keys) + "}}"
        return "{%" + _rewrite_expr_body(m.group(2), env_keys, result_keys) + "%}"

    return _BLOCK_RE.sub(_sub, text)


# -- global_actions → ToolSpecs ------------------------------------------------

_SIMPLE_WHEN_RE = re.compile(r"^\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$")
_OUTER_MUSTACHE_RE = re.compile(r"^\{\{(.*)\}\}$", re.S)


def _compile_tool_when(
    condition: str, env_keys: set[str], result_keys: set[str]
) -> str:
    """Compile a legacy action ``condition`` template into a ``when`` expr.

    ``{{X}}`` binds to the namespaced name: ``env.X`` when X is an env
    key, ``results.X`` for a store key, else ``slots.X``. Anything more
    complex is namespaced via :func:`_rewrite_template` and unwrapped
    from its outer mustache (best effort; the expr judge evaluates it).
    """
    m = _SIMPLE_WHEN_RE.match(condition.strip())
    if m:
        name = m.group(1)
        if name in env_keys:
            return f"env.{name}"
        if name in result_keys:
            return f"results.{name}"
        return f"slots.{name}"
    rewritten = _rewrite_template(condition, env_keys, result_keys).strip()
    outer = _OUTER_MUSTACHE_RE.match(rewritten)
    return outer.group(1).strip() if outer else rewritten


def _rewrite_body_value(value: Any, env_keys: set[str], result_keys: set[str]) -> Any:
    if isinstance(value, str):
        return _rewrite_template(value, env_keys, result_keys)
    if isinstance(value, dict):
        return {
            k: _rewrite_body_value(v, env_keys, result_keys) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_body_value(v, env_keys, result_keys) for v in value]
    return value


def _compile_tool(
    action: CustomAction,
    env_keys: set[str],
    result_keys: set[str],
    notes: list[str],
) -> ToolSpec:
    """Compile one legacy CustomAction into a ToolSpec (1:1, lossless).

    Templates are namespaced via :func:`_rewrite_template`; env_updates
    flatten to ``{env_key: result_path}``; ``string_fields`` become typed
    ``str`` args. A JSON-parseable body template becomes a body dict with
    rewritten string values; a non-JSON template (raw Jinja-in-JSON) is
    kept whole under the ``_template`` key and noted in the coverage
    report — the executor renders it, then JSON-parses the rendered text
    into the real request body.
    """
    body: dict[str, Any] = {}
    if action.body_template:
        try:
            parsed: Any = json.loads(action.body_template)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            body = _rewrite_body_value(parsed, env_keys, result_keys)
        else:
            body = {
                "_template": _rewrite_template(
                    action.body_template, env_keys, result_keys
                )
            }
            note = (
                f"tool {action.id}: non-JSON body template kept under "
                "'_template' (rendered then JSON-parsed by the executor)"
            )
            if note not in notes:
                notes.append(note)
    return ToolSpec(
        id=action.id,
        type="http",
        method=str(action.method.value),
        url=_rewrite_template(action.url, env_keys, result_keys),
        headers={
            k: _rewrite_template(v, env_keys, result_keys)
            for k, v in action.headers.items()
        },
        body=body,
        store_response_as=action.store_response_as,
        env_updates={u.env_key: u.result_path for u in action.env_updates},
        run_once=action.run_once,
        when=(
            _compile_tool_when(action.condition, env_keys, result_keys)
            if action.condition
            else None
        ),
        timeout=float(action.timeout),
        args={f: SlotSpec(type="str") for f in action.string_fields},
    )


# -- coverage ------------------------------------------------------------------


class CoverageReport(BaseModel):
    """Lossless-compilation audit: what mapped where, and what did not.

    ``dropped`` buckets are informational: silence_policy/middleware/
    handler list constructs absorbed by policies; computational_chains
    and hubs list silent nodes folded into advance rules or compiled to
    intermediate pipeline checkpoints. Anything in the ``unmapped_*``
    lists is a compiler bug.
    """

    unmapped_nodes: list[str] = Field(default_factory=list)
    unmapped_edges: list[str] = Field(default_factory=list)
    unmapped_actions: list[str] = Field(default_factory=list)
    orphans: list[str] = Field(default_factory=list)
    dropped: dict[str, list[str]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


@dataclass
class _CompileTrace:
    """Compile-time provenance, re-derived on demand by coverage_report."""

    mapped_nodes: set[str] = field(default_factory=set)
    mapped_edges: set[tuple[str, str]] = field(default_factory=set)
    mapped_global_edges: set[str] = field(default_factory=set)
    orphans: list[str] = field(default_factory=list)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def drop(self, bucket: str, item: str) -> None:
        items = self.dropped.setdefault(bucket, [])
        if item not in items:
            items.append(item)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


@dataclass
class _Landing:
    """One checkpoint reached by walking a computational chain."""

    node_id: str
    condition: str | None  # None -> use the source edge's own condition
    requires: list[str]  # extra requires picked up from hub edge schemas
    slots: dict[str, SlotSpec]  # extra slot declarations (hub schemas)


@dataclass
class _ChainPlan:
    """A linearized tool chain: pipeline steps plus landing routes.

    ``success``/``failure`` are checkpoint refs ("main.X") or None when
    the chain did not surface one (the caller falls back to the source
    checkpoint). ``extra_rules`` preserve branches the pipeline cannot
    route deterministically (kept as llm rules on the intermediate).
    ``goal_names`` are cosmetic and excluded from the dedup signature.
    """

    steps: list[PipelineStep] = field(default_factory=list)
    goal_names: list[str] = field(default_factory=list)
    success: str | None = None
    failure: str | None = None
    extra_rules: list[AdvanceRule] = field(default_factory=list)


# -- full compilation ----------------------------------------------------------

_SILENCE_RE = re.compile(r"silence|no\s+response|not\s+respond", re.I)
_TOKEN_EDGE_RE = re.compile(r"\b401\b|token", re.I)
_STATUS_CODE_RE = re.compile(r"\b([45]\d{2})\b")
# Failure-shaped branch conditions ("X.success is false", "API returned an
# error", "creation failed"): routable as the pipeline step's `failed` exit.
_FAILURE_COND_RE = re.compile(
    rf"\.success\s*{_EQ}\s*false|\bnot\s+\w+\.success\b|\berror\b|\bfail",
    re.I,
)
_HUB_MIN_EDGES = 4
_MAX_CHAIN_DEPTH = 10


def _on_enter_ids(node: FlowNode) -> list[str]:
    return [
        t.action_id
        for t in node.actions
        if t.trigger_type == ActionTriggerType.ON_ENTER
    ]


def _handler_trigger(instruction: str) -> str | None:
    """Map a system node's instruction to an external trigger name.

    Keyword detection: an instruction mentioning "webhook" becomes
    ``webhook.<name>`` (name derived from an "<x>.<y> webhook" mention,
    fallback ``payment_captured``); one mentioning "timer" becomes
    ``timer.<name>`` (derived from "<x> expires", fallback
    ``hold_expired``). The fallbacks encode the two known golf handlers.
    """
    low = instruction.lower()
    if "webhook" in low:
        m = re.search(r"(\w+)\.(\w+)\s+webhook", low)
        name = f"{m.group(1)}_{m.group(2)}" if m else "payment_captured"
        return f"webhook.{name}"
    if "timer" in low:
        m = re.search(r"(\w+)\s+expires", low)
        name = f"{m.group(1)}_expired" if m else "hold_expired"
        return f"timer.{name}"
    return None


class _Compiler:
    """Single-journey ("main") lossless flow → playbook compilation.

    Journey splitting is an authoring concern; the compiler stays
    lossless and boring. Computational nodes never become checkpoints
    themselves: tool-free chains fold into their conversational sources'
    advance rules, while tool-bearing chains compile to a PipelineSpec
    owned by a synthetic INTERMEDIATE checkpoint — tools run as pipeline
    steps and route on their results, never as on_enter side effects of
    landing checkpoints.
    """

    def __init__(self, flow: ConversationFlow) -> None:
        self.flow = flow
        self.idx = FlowIndex(flow)
        self.trace = _CompileTrace()
        self.node_ids = {n.id for n in flow.nodes}
        self.edges = {(n.id, e.id): e for n in flow.nodes for e in n.edges}
        self.kinds: dict[str, NodeKind] = {
            n.id: self.idx.classify(n) for n in flow.nodes
        }
        self.env_keys = set(flow.environment_variables) | {
            u.env_key for a in flow.actions for u in a.env_updates
        }
        self.result_keys = {
            a.store_response_as for a in flow.actions if a.store_response_as
        }
        self.silence = self._find_silence_nodes()
        self.handler_triggers = self._find_handler_triggers()
        self.middleware_edge, self.middleware_tool = self._find_middleware()
        self.middleware_nodes: set[str] = set()
        if (
            self.middleware_edge is not None
            and self.middleware_edge.target_node_id in self.node_ids
        ):
            self.middleware_nodes.add(self.middleware_edge.target_node_id)
        self.hubs = {
            n.id
            for n in flow.nodes
            if self.kinds[n.id] == "computational"
            and n.id not in self.middleware_nodes
            and len(n.edges) >= _HUB_MIN_EDGES
        }
        self.hub_schemas = {
            hub: union_slot_schemas(self.idx.node(hub)) for hub in self.hubs
        }
        self.checkpoints: dict[str, Checkpoint] = {}
        self.intermediates: list[str] = []  # synthetic checkpoint ids, in order
        self.chain_pipelines: list[PipelineSpec] = []
        self._chain_by_sig: dict[str, str] = {}  # plan signature -> checkpoint id
        self._chain_active: set[str] = set()  # nodes on the build stack (cycles)

    def _rw(self, text: str) -> str:
        return _rewrite_template(text, self.env_keys, self.result_keys)

    # -- discovery ----------------------------------------------------------

    def _find_silence_nodes(self) -> set[str]:
        """Conversational nodes whose EVERY inbound edge is a silence cue."""
        out: set[str] = set()
        for node in self.flow.nodes:
            if self.kinds[node.id] != "conversational":
                continue
            inbound = self.idx.reverse_edges.get(node.id, [])
            if inbound and all(
                _SILENCE_RE.search(self.edges[(src, eid)].condition)
                for src, eid in inbound
            ):
                out.add(node.id)
        return out

    def _find_handler_triggers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for node in self.flow.nodes:
            if self.kinds[node.id] != "system":
                continue
            trigger = _handler_trigger(node.instruction or "")
            if trigger:
                out[node.id] = trigger
        return out

    def _find_middleware(self) -> tuple[Edge | None, str | None]:
        """Find the token-expiry global edge and the auth-refresh tool."""
        ge = next(
            (g for g in self.flow.global_edges if _TOKEN_EDGE_RE.search(g.condition)),
            None,
        )
        if ge is None:
            return None, None
        tool = next(
            (
                a.id
                for a in self.flow.actions
                if {"ACCESS_TOKEN", "REFRESH_TOKEN"}
                <= {u.env_key.upper() for u in a.env_updates}
            ),
            None,
        ) or next(
            (
                a.id
                for a in self.flow.actions
                if "refresh" in a.id.lower() or "refresh" in a.name.lower()
            ),
            None,
        )
        return (ge, tool) if tool else (None, None)

    # -- checkpoints ----------------------------------------------------------

    def _build_checkpoints(self) -> None:
        for node in self.flow.nodes:
            kind = self.kinds[node.id]
            is_orphan = kind == "system" and node.id not in self.handler_triggers
            if is_orphan:
                # unreachable from the dialog graph, but preserved: lossless
                self.trace.orphans.append(node.id)
            keep = is_orphan or (
                kind == "conversational"
                and node.id not in self.silence
                and node.id not in self.middleware_nodes
            )
            if not keep:
                continue
            if any(t.trigger_type != ActionTriggerType.ON_ENTER for t in node.actions):
                self.trace.note(
                    f"node {node.id}: non-on_enter action triggers not carried"
                )
            slots, _ = union_slot_schemas(node)
            self.checkpoints[node.id] = Checkpoint(
                id=node.id,
                goal=node.name,
                slots=slots,
                guidance=self._rw(node.instruction or ""),
                say_verbatim=(self._rw(node.static_text) if node.static_text else None),
                auto=bool(node.auto_proceed) and bool(node.instruction or node.static_text),
                on_enter=_on_enter_ids(node),
                terminal=node.is_final,
                outcome=node.id if node.is_final else None,
                turn_budget=node.max_turns,
            )
            self.trace.mapped_nodes.add(node.id)

    # -- edges → rules (with computational-chain walking) ---------------------

    def _drop_edge(self, source_id: str, edge: Edge, bucket: str) -> None:
        self.trace.mapped_edges.add((source_id, edge.id))
        self.trace.drop(bucket, f"{source_id}:{edge.id}")

    def _resolve_edge(self, source_id: str, edge: Edge) -> list[_Landing]:
        """Resolve one edge to checkpoint landings, walking comp. chains."""
        target = edge.target_node_id
        if target is None or target not in self.node_ids:
            self._drop_edge(source_id, edge, "dangling")
            return []
        if target in self.silence:
            self._drop_edge(source_id, edge, "silence_policy")
            return []
        if target in self.middleware_nodes:
            self._drop_edge(source_id, edge, "middleware")
            return []
        self.trace.mapped_edges.add((source_id, edge.id))
        if target in self.checkpoints:
            return [_Landing(target, None, [], {})]
        if self.kinds.get(target) != "computational":
            self._drop_edge(source_id, edge, "unsupported_target")
            return []
        origin = f"main.{source_id}" if source_id in self.checkpoints else None
        if target not in self.hubs and self._chain_has_tools(target):
            cp_id = self._chain_checkpoint(source_id, edge.id, target, origin)
            if cp_id is not None:
                return [_Landing(cp_id, None, [], {})]
        out: list[_Landing] = []
        self._walk(
            node_id=target,
            condition=None,
            requires=[],
            slots={},
            visited={source_id, target},
            seen=set(),
            out=out,
            depth=1,
            origin=origin,
        )
        return out

    def _walk(
        self,
        node_id: str,
        condition: str | None,
        requires: list[str],
        slots: dict[str, SlotSpec],
        visited: set[str],
        seen: set[str],
        out: list[_Landing],
        depth: int,
        origin: str | None,
    ) -> None:
        """DFS a tool-free computational chain, collecting landings.

        The first edge of a multi-exit node is the DEFAULT path and
        inherits the current condition (the source edge's, on the pure
        default path); other exits re-condition on their own edge. Hub
        nodes (≥ ``_HUB_MIN_EDGES`` exits) re-condition EVERY exit and
        contribute their edge schemas' slots/requires. Tool-bearing
        chains never fold here: any computational target whose chain
        declares on_enter tools lands on a pipeline intermediate
        instead (see :meth:`_chain_checkpoint`), so walked paths carry
        no side effects — except hubs, which are noted if they declare
        tools of their own.
        """
        node = self.idx.node(node_id)
        self.trace.mapped_nodes.add(node_id)
        is_hub = node_id in self.hubs
        self.trace.drop("hubs" if is_hub else "computational_chains", node_id)
        if node.static_text:
            self.trace.note(
                f"chain node {node_id}: static_text not carried "
                "(folded node has no speaking turn)"
            )
        if _on_enter_ids(node):  # only hubs can reach here with tools
            self.trace.note(
                f"chain node {node_id}: on_enter tools not carried by the "
                "rule fold (hub tools are not pipeline-owned)"
            )
        if depth > _MAX_CHAIN_DEPTH:
            self.trace.note(
                f"chain walk depth bound ({_MAX_CHAIN_DEPTH}) hit at {node_id}"
            )
            return
        multi = len(node.edges) > 1
        for i, edge in enumerate(node.edges):
            target = edge.target_node_id
            if target is None or target not in self.node_ids:
                self._drop_edge(node_id, edge, "dangling")
                continue
            if target in self.silence:
                self._drop_edge(node_id, edge, "silence_policy")
                continue
            if target in self.middleware_nodes:
                self._drop_edge(node_id, edge, "middleware")
                continue
            self.trace.mapped_edges.add((node_id, edge.id))
            if is_hub:
                hub_slots, hub_reqs = self.hub_schemas[node_id]
                b_cond: str | None = edge.condition
                b_req = [*requires, *hub_reqs.get(edge.id, [])]
                b_slots = {**slots, **hub_slots}
            elif multi and i > 0:
                b_cond, b_req, b_slots = edge.condition, requires, slots
            else:
                b_cond, b_req, b_slots = condition, requires, slots
            if target in visited:
                # Cycle back into the walk (often to the source checkpoint
                # itself): the edge is accounted for, but a self-landing
                # would loop the rule fold, so it is suppressed and
                # surfaced as a note instead.
                self.trace.note(
                    f"chain loop suppressed: {node_id}:{edge.id} returns to {target}"
                )
                continue
            if target in self.checkpoints:
                if target not in seen:
                    seen.add(target)
                    out.append(_Landing(target, b_cond, b_req, b_slots))
                continue
            if self.kinds.get(target) == "computational":
                if target not in self.hubs and self._chain_has_tools(target):
                    cp_id = self._chain_checkpoint(node_id, edge.id, target, origin)
                    if cp_id is not None and cp_id not in seen:
                        seen.add(cp_id)
                        out.append(_Landing(cp_id, b_cond, b_req, b_slots))
                    continue
                self._walk(
                    target,
                    b_cond,
                    b_req,
                    b_slots,
                    visited | {target},
                    seen,
                    out,
                    depth + 1,
                    origin,
                )
            else:
                self._drop_edge(node_id, edge, "unsupported_target")

    # -- tool chains → intermediate pipeline checkpoints ------------------------

    def _chain_has_tools(self, entry: str) -> bool:
        """True when the chain reachable from ``entry`` declares any tool.

        Reachability follows computational non-hub nodes only: a hub
        re-conditions every exit and is folded by :meth:`_walk`, which
        re-probes each of its computational targets.
        """
        seen: set[str] = set()
        stack = [entry]
        while stack:
            nid = stack.pop()
            if nid in seen or nid not in self.node_ids:
                continue
            seen.add(nid)
            if (
                self.kinds.get(nid) != "computational"
                or nid in self.hubs
                or nid in self.middleware_nodes
            ):
                continue
            node = self.idx.node(nid)
            if _on_enter_ids(node):
                return True
            stack.extend(e.target_node_id for e in node.edges if e.target_node_id)
        return False

    def _chainable(self, node_id: str) -> bool:
        return (
            self.kinds.get(node_id) == "computational"
            and node_id not in self.hubs
            and node_id not in self.middleware_nodes
        )

    def _branch_status(self, condition: str) -> int | None:
        """HTTP status when the branch condition is a pure status predicate."""
        for text in (condition, condition.partition(" — ")[0]):
            m = _STATUS_RE.match(text)
            if m and m.group(1) in self.result_keys:
                gloss = condition.partition(" — ")[2].strip()
                if text == condition or _NARRATION_GLOSS_RE.match(gloss):
                    return int(m.group(2))
        return None

    def _chain_checkpoint(
        self, src: str, edge_id: str, entry: str, fallback: str | None
    ) -> str | None:
        """Create (or dedup-reuse) the intermediate checkpoint owning a chain.

        The chain entered at ``entry`` compiles to a PipelineSpec plus one
        synthetic soft checkpoint that runs it on entry and routes on the
        result: ``pipeline.ok`` → the chain's default conversational
        landing, ``pipeline.failed`` → its failure landing (or ``fallback``,
        the source checkpoint). Identical plans (same steps, routes, and
        extra rules) share one intermediate. Returns the bare checkpoint
        id, or None when the chain cannot be linearized.
        """
        if entry in self._chain_active:
            self.trace.note(
                f"chain cycle at {entry}: branch {src}:{edge_id} not routed"
            )
            return None
        plan = self._build_chain_plan(entry, fallback)
        success = plan.success or fallback
        if success is None:
            self.trace.note(
                f"chain at {entry} (via {src}:{edge_id}) has no landing and "
                "no source fallback; folded without a pipeline"
            )
            return None
        failure = plan.failure or fallback or success
        sig = json.dumps(
            {
                "steps": [s.model_dump(mode="json") for s in plan.steps],
                "success": success,
                "failure": failure,
                "extra": [r.model_dump(mode="json") for r in plan.extra_rules],
            },
            sort_keys=True,
        )
        existing = self._chain_by_sig.get(sig)
        if existing is not None:
            return existing
        if plan.failure is None and fallback is not None:
            self.trace.note(
                f"chain at {entry}: no failure branch in the legacy flow; "
                f"pipeline.failed routes to {fallback}"
            )
        cp_id = f"{src}__{edge_id}"
        pipe = PipelineSpec(id=f"{cp_id}_pipe", steps=plan.steps)
        self.chain_pipelines.append(pipe)
        self.checkpoints[cp_id] = Checkpoint(
            id=cp_id,
            goal="(auto) " + " → ".join(plan.goal_names),
            pipeline=pipe.id,
            gate="soft",
            advance_when=[
                AdvanceRule(when="pipeline.ok", judge="expr", to=success),
                AdvanceRule(when="pipeline.failed", judge="expr", to=failure),
                *plan.extra_rules,
            ],
        )
        self.intermediates.append(cp_id)
        self._chain_by_sig[sig] = cp_id
        return cp_id

    def _build_chain_plan(self, entry: str, fallback: str | None) -> _ChainPlan:
        """Linearize the chain at ``entry`` into pipeline steps + routes.

        The trunk follows each node's FIRST viable exit (the legacy
        default path) through computational nodes until it reaches a
        conversational landing — the chain's ``success``. Branch exits
        route off the node's last pipeline step: a pure status predicate
        becomes ``http_<code>``, a failure-shaped condition becomes
        ``failed``, and anything else is preserved as an llm rule on the
        intermediate (noted: the pipeline routes ok/failed only). A
        branch into another tool-bearing chain routes to that chain's
        own intermediate, built recursively.
        """
        plan = _ChainPlan()
        cur: str | None = entry
        visited: set[str] = set()
        while cur is not None and len(visited) <= _MAX_CHAIN_DEPTH:
            node = self.idx.node(cur)
            visited.add(cur)
            self._chain_active.add(cur)
            self.trace.mapped_nodes.add(cur)
            self.trace.drop("computational_chains", cur)
            plan.goal_names.append(node.name)
            if node.static_text:
                self.trace.note(
                    f"chain node {cur}: static_text not carried "
                    "(folded node has no speaking turn)"
                )
            for tool_id in _on_enter_ids(node):
                plan.steps.append(PipelineStep(tool=tool_id))
            branch_step = plan.steps[-1] if plan.steps else None
            next_trunk: str | None = None
            trunk_taken = False
            for edge in node.edges:
                target = edge.target_node_id
                if target is None or target not in self.node_ids:
                    self._drop_edge(cur, edge, "dangling")
                    continue
                if target in self.silence:
                    self._drop_edge(cur, edge, "silence_policy")
                    continue
                if target in self.middleware_nodes:
                    self._drop_edge(cur, edge, "middleware")
                    continue
                self.trace.mapped_edges.add((cur, edge.id))
                if not trunk_taken:  # first viable exit: the default path
                    trunk_taken = True
                    if target in self.checkpoints:
                        plan.success = f"main.{target}"
                    elif self._chainable(target) and target not in visited:
                        next_trunk = target
                    else:
                        self.trace.note(
                            f"chain trunk stops at {cur}:{edge.id} "
                            f"(target {target} not linearizable)"
                        )
                    continue
                ref = self._branch_ref(cur, edge, target, fallback)
                if ref is None:
                    continue
                status = self._branch_status(edge.condition)
                if branch_step is not None and status is not None:
                    branch_step.on[f"http_{status}"] = ref
                elif (
                    branch_step is not None
                    and _FAILURE_COND_RE.search(edge.condition)
                    and "failed" not in branch_step.on
                ):
                    branch_step.on["failed"] = ref
                    if plan.failure is None:
                        plan.failure = ref
                else:
                    plan.extra_rules.append(
                        AdvanceRule(when=edge.condition, judge="llm", to=ref)
                    )
                    self.trace.note(
                        f"chain branch {cur}:{edge.id} not deterministically "
                        "routable; kept as an llm rule on the intermediate "
                        "(the pipeline routes ok/failed)"
                    )
            cur = next_trunk
        for nid in visited:
            self._chain_active.discard(nid)
        return plan

    def _branch_ref(
        self, src: str, edge: Edge, target: str, fallback: str | None
    ) -> str | None:
        """Resolve a chain branch edge to a checkpoint ref ("main.X")."""
        if target in self.checkpoints:
            return f"main.{target}"
        if self._chainable(target):
            sub = self._chain_checkpoint(src, edge.id, target, fallback)
            return f"main.{sub}" if sub is not None else None
        self.trace.note(
            f"chain branch {src}:{edge.id}: target {target} not mappable "
            "to a checkpoint; branch dropped"
        )
        return None

    def _compile_edges(self, node: FlowNode) -> None:
        cp = self.checkpoints[node.id]
        _, requires_by_edge = union_slot_schemas(node)
        store_keys = set(self.result_keys)
        for edge in node.edges:
            base_requires = requires_by_edge.get(edge.id, [])
            for landing in self._resolve_edge(node.id, edge):
                cond = (
                    landing.condition
                    if landing.condition is not None
                    else edge.condition
                )
                rule = compile_edge_condition(
                    cond, store_keys, f"main.{landing.node_id}"
                )
                requires = list(dict.fromkeys([*base_requires, *landing.requires]))
                if requires:
                    rule = rule.model_copy(update={"requires": requires})
                if not any(
                    r.when == rule.when
                    and r.to == rule.to
                    and r.requires == rule.requires
                    for r in cp.advance_when
                ):
                    cp.advance_when.append(rule)
                for key, spec in landing.slots.items():
                    cp.slots.setdefault(key, spec)

    # -- hubs → dispatch -------------------------------------------------------

    def _build_dispatch(self) -> list[DispatchEntry]:
        out: list[DispatchEntry] = []
        for node in self.flow.nodes:
            if node.id not in self.hubs:
                continue
            self.trace.mapped_nodes.add(node.id)
            _, hub_reqs = self.hub_schemas[node.id]
            for edge in node.edges:
                landings = self._resolve_edge(node.id, edge)
                if not landings:
                    continue
                out.append(
                    DispatchEntry(
                        intent=edge.condition,
                        to=f"main.{landings[0].node_id}",
                        requires=hub_reqs.get(edge.id, []),
                    )
                )
        return out

    # -- silence nodes → policy ------------------------------------------------

    def _build_silence_policy(self) -> SilencePolicy | None:
        if not self.silence:
            return None
        ordered = [n.id for n in self.flow.nodes if n.id in self.silence]
        entry = next(
            (
                nid
                for nid in ordered
                if any(
                    src not in self.silence for src, _ in self.idx.reverse_edges[nid]
                )
            ),
            ordered[0],
        )
        chain: list[str] = []
        cur: str | None = entry
        while cur is not None and cur not in chain:
            chain.append(cur)
            cur = next(
                (
                    e.target_node_id
                    for e in self.idx.node(cur).edges
                    if e.target_node_id in self.silence
                ),
                None,
            )
        chain += [nid for nid in ordered if nid not in chain]
        for nid in self.silence:
            self.trace.mapped_nodes.add(nid)
            self.trace.drop("silence_policy", nid)
            for edge in self.idx.node(nid).edges:
                self._drop_edge(nid, edge, "silence_policy")
        then = ""
        for edge in self.idx.node(chain[-1]).edges:
            target = edge.target_node_id
            if target in self.checkpoints and _SILENCE_RE.search(edge.condition):
                then = f"main.{target}"
                break
        prompts = [self._rw(self.idx.node(nid).static_text or "") for nid in chain]
        return SilencePolicy(max_prompts=len(prompts), prompts=prompts, then=then)

    # -- system nodes → handlers ------------------------------------------------

    def _build_handlers(self) -> tuple[list[HandlerSpec], list[PipelineSpec]]:
        handlers: list[HandlerSpec] = []
        pipelines: list[PipelineSpec] = []
        for node in self.flow.nodes:
            trigger = self.handler_triggers.get(node.id)
            if trigger is None:
                continue
            pipe = PipelineSpec(
                id=f"{node.id}_pipe",
                steps=[PipelineStep(tool=t) for t in _on_enter_ids(node)],
            )
            pipelines.append(pipe)
            handlers.append(HandlerSpec(id=node.id, on=trigger, pipeline=pipe.id))
            self.trace.mapped_nodes.add(node.id)
            self.trace.drop("handlers", node.id)
            for edge in node.edges:
                self._drop_edge(node.id, edge, "handlers")
        return handlers, pipelines

    # -- global edges → interrupts + middleware ---------------------------------

    def _build_interrupts(self) -> list[InterruptSpec]:
        out: list[InterruptSpec] = []
        for ge in self.flow.global_edges:
            if self.middleware_edge is not None and ge.id == self.middleware_edge.id:
                continue
            self.trace.mapped_global_edges.add(ge.id)
            if ge.target_node_id not in self.checkpoints:
                self.trace.drop("dangling", f"global:{ge.id}")
                continue
            out.append(
                InterruptSpec(
                    id=ge.id,
                    when=ge.condition,
                    judge="llm",
                    to=f"main.{ge.target_node_id}",
                    resume=False,
                )
            )
        return out

    def _apply_middleware(self) -> MiddlewareSpec | None:
        if self.middleware_edge is None or self.middleware_tool is None:
            return None
        self.trace.mapped_global_edges.add(self.middleware_edge.id)
        for nid in self.middleware_nodes:
            self.trace.mapped_nodes.add(nid)
            self.trace.drop("middleware", nid)
            for edge in self.idx.node(nid).edges:
                self._drop_edge(nid, edge, "middleware")
        m = _STATUS_CODE_RE.search(self.middleware_edge.condition)
        status = int(m.group(1)) if m else 401
        return MiddlewareSpec(on_status=status, refresh_with=self.middleware_tool)

    # -- assembly ---------------------------------------------------------------

    def compile(self) -> tuple[Playbook, _CompileTrace]:
        tools = [
            _compile_tool(a, self.env_keys, self.result_keys, self.trace.notes)
            for a in self.flow.actions
        ]
        self._build_checkpoints()
        for node in self.flow.nodes:
            if node.id in self.checkpoints:
                self._compile_edges(node)
        dispatch = self._build_dispatch()
        silence_policy = self._build_silence_policy()
        handlers, pipelines = self._build_handlers()
        interrupts = self._build_interrupts()
        middleware = self._apply_middleware()
        self.trace.note(
            "tool-bearing computational chains compile to intermediate "
            "pipeline checkpoints: chain tools run as pipeline steps and "
            "route on results, never as landing-checkpoint on_enter"
        )
        self.trace.note(
            "dispatch is compile-time organization in v1: the Director "
            "judges per-checkpoint advance rules; hub routes are merged "
            "into each inbound checkpoint's advance_when"
        )
        pb = Playbook(
            persona=self._rw(self.flow.system_prompt),
            journeys={
                "main": Journey(
                    checkpoints=[
                        self.checkpoints[n.id]
                        for n in self.flow.nodes
                        if n.id in self.checkpoints
                    ]
                    + [self.checkpoints[cp_id] for cp_id in self.intermediates]
                )
            },
            dispatch=dispatch,
            tools=tools,
            pipelines=[*self.chain_pipelines, *pipelines],
            handlers=handlers,
            interrupts=interrupts,
            policies=Policies(silence=silence_policy),
            middleware=middleware,
            env=dict(self.flow.environment_variables),
            initial=f"main.{self.flow.initial_node}",
        )
        return pb, self.trace


def _compile_internal(flow: ConversationFlow) -> tuple[Playbook, _CompileTrace]:
    return _Compiler(flow).compile()


def compile_flow(flow: ConversationFlow) -> Playbook:
    """Compile a legacy ConversationFlow into a single-journey Playbook.

    Lossless by construction — every legacy construct lands somewhere:

    - conversational nodes → checkpoints in journey "main"
    - tool-free computational nodes → folded into their sources' advance
      rules
    - tool-bearing computational chains → a PipelineSpec plus a synthetic
      intermediate checkpoint that runs it on entry and routes on
      ``pipeline.ok`` / ``pipeline.failed`` (status/failure branch edges
      become step ``on:`` routes; the rest stay as llm rules)
    - hub routers (≥4-exit computational) → dispatch entries + rules
      merged into every inbound checkpoint
    - silence nodes → ``policies.silence`` (prompts in chain order)
    - token-expiry global edge + refresh node → ``middleware``
    - other global edges → interrupts
    - webhook/timer system nodes → handlers with single-step pipelines
    - orphan system nodes → normal (unreachable) checkpoints
    - global_actions → tools 1:1, templates rewritten to the
      {env, slots, results} namespace

    Use :func:`coverage_report` to audit the mapping.
    """
    return _compile_internal(flow)[0]


def coverage_report(flow: ConversationFlow, pb: Playbook) -> CoverageReport:
    """Audit a compiled playbook against its source flow.

    Re-derives compile-time provenance by re-running the compilation
    (the trace is never attached to the Playbook artifact) and lists
    any node, edge, or action that did not map anywhere. Tool coverage
    is checked against ``pb`` so a hand-edited playbook is audited as
    given.
    """
    _, trace = _compile_internal(flow)
    tool_ids = {t.id for t in pb.tools}
    unmapped_edges = [
        f"{n.id}:{e.id}"
        for n in flow.nodes
        for e in n.edges
        if (n.id, e.id) not in trace.mapped_edges
    ]
    unmapped_edges += [
        f"global:{ge.id}"
        for ge in flow.global_edges
        if ge.id not in trace.mapped_global_edges
    ]
    return CoverageReport(
        unmapped_nodes=[n.id for n in flow.nodes if n.id not in trace.mapped_nodes],
        unmapped_edges=unmapped_edges,
        unmapped_actions=[a.id for a in flow.actions if a.id not in tool_ids],
        orphans=trace.orphans,
        dropped=trace.dropped,
        notes=trace.notes,
    )
