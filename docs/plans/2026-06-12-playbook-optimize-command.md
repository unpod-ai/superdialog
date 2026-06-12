# `superdialog optimize` — Reflective Prompt Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship `superdialog optimize` — a reflective prose optimizer that runs persona
self-play against a playbook, asks a candidate LLM for **targeted prose edits**, and keeps
only edits that win a **paired** evaluation. Output: an improved playbook **in the source
format** (full or simple), plus a per-round metric trace.

**Architecture:** (per the validated design `2026-06-12-playbook-optimize-design.md` —
read it first). Three new modules in `superdialog.playbook`: `editable.py` (an
`EditableDoc` abstraction — FullDoc/SimpleDoc — enforcing a prose-only whitelist **by
construction**), `optimize.py` (pure scoring, the reflect step, the paired-round loop,
an informational Pareto frontier), `personas.py` (load/save/generate persona suites).
Each round: REFLECT (worst sessions + source YAML → JSON edit list) → APPLY (whitelist +
recompile + Jinja check) → PAIR-EVAL (incumbent AND candidate evaluated fresh, same round)
→ ACCEPT iff candidate objective strictly beats incumbent's same-round objective. The
written artifact is the final incumbent. v1 never mutates structure.

**Tech Stack:** Python ≥3.10, pydantic v2, jinja2 (already a dependency),
`superdialog.playbook` (eval_bridge, models, simple, director, providers), pytest
(`asyncio_mode = "auto"`), uv, ruff, pyrefly.

**Shipped substrate (verified file:line — cite these, they are real):**
- `src/superdialog/playbook/eval_bridge.py` — `PersonaSpec(name, traits, goal,
  max_turns=12, opening="Hello", ground_truth_slots)` (:33), `SessionMetrics` (:45,
  fields: persona, completed, outcome, turns, turns_per_checkpoint, slot_accuracy,
  slot_diffs, repair_count, degraded_count, event_log_jsonl), `EvalReport` (:61,
  properties `.completion_rate`, `.mean_slot_accuracy` — **no** smoothness/repair
  aggregates exist), `SpeaksUser` protocol (:81), `run_eval(playbook_factory, personas,
  user_llm, n=1)` (:112) — **the first param is an AGENT factory**
  `Callable[[], PlaybookAgent]`, despite its name.
- `src/superdialog/playbook/models.py` — `Playbook.from_yaml` (:294) uses a custom
  `_YamlLoader` (:13, YAML-1.2 booleans: `on`/`off`/`yes`/`no` stay strings — required
  for pipeline `on:` keys), `Playbook.load` (:302), `checkpoint(ref)` (:162),
  `initial_checkpoint_id` (:169). `Checkpoint` (:52): `goal` (:54), `guidance` (:56),
  `say_verbatim` (:57), `never_say` (:58), `advance_when` (:59). `AdvanceRule` (:40):
  `when`, `judge: Literal["llm","expr"]` (default `"llm"`), `to`, `requires`, `set`.
  `SlotSpec.description` (:37). Reference validation raises `ValueError` inside
  `model_validate` (surfaced as pydantic `ValidationError`, a `ValueError` subclass).
- `src/superdialog/playbook/simple.py` — `SimplePlaybook` (:45: name, goal, persona
  {identity, voice_style, name, language}, opening, closing, playbook: list[SimpleStep],
  facts, objections, boundaries, fallback_actions), `SimpleStep` (:32: id, purpose, say,
  collect, done_when), `is_simple_playbook(doc)` (:58), `simple_to_playbook(doc)` (:125),
  `load_simple(path)` (:140, parses with `yaml.safe_load`). Facts/objections/boundaries/
  fallbacks are folded into ONE persona string by `_build_persona` (:67).
- `src/superdialog/playbook/agent.py` — `PlaybookAgent(playbook, talker_llm,
  director_llm, http, ...)` (:44).
- `src/superdialog/playbook/director.py` — `CompletesLLM` protocol (:16):
  `async complete(messages, **kwargs) -> str`. Structurally identical to `SpeaksUser`.
- `src/superdialog/playbook/providers.py` — `provider_adapters(provider) ->
  (ProviderDirector, ProviderTalker)` (:44). `ProviderDirector` satisfies `CompletesLLM`
  (and therefore `SpeaksUser`).
- `src/superdialog/cli/main.py` — subcommands today: `chat`, `flow` only (no `eval`).
  `_looks_like_simple_playbook(path)` (:69), `_chat_playbook` (:200, the
  pre-flight/clean-error pattern to mirror), `_build_parser` (:343), `http=httpx_http`
  (:130). `superdialog.playbook.__init__` exports `httpx_http`.
- `src/superdialog/llm/resolver.py` — `resolve_llm(uri)` (:10).
- Test fakes (cross-import them — the repo convention is
  `from tests.playbook.test_X import Y`; `tests/` has `__init__.py` throughout):
  `CannedLLM(payload: dict)` — test_director.py:17; `StreamLLM(chunks: list[str])` —
  test_talker.py:10; `FakeHttp(responses: list[tuple[int, dict]])` — test_toolexec.py:12;
  `ScriptedUser(lines: list[str])` — test_eval_bridge.py:31 (pops lines, repeats last
  forever, returns "" when constructed empty); `MINIMAL_YAML` — test_models.py:8;
  `SIMPLE` — test_simple.py:13.

**Conventions for every task:**
- Branch: `feat/playbook-engine` (continue; never commit to `main`).
- TDD: write the failing test, run it, implement, run again, commit.
- Run `uv run pytest <test file> -v` after each test/impl step.
- Before each commit: `uv run ruff format . && uv run ruff check . --fix &&
  uv run pyrefly check` — fix what they flag (88-char lines, type hints, explicit
  None checks).
- `asyncio_mode = "auto"`: plain `async def` tests, no markers, no anyio fixtures.
- **No network in tests.** Scripted fakes only.
- New modules import only from within `superdialog.playbook` + stdlib/pydantic/yaml/
  jinja2 — never from `superdialog.machine`.
- `scripts/run_tests.sh` does not exist in this repo — skip the global register-tests
  instruction.
- Commit style: `feat(playbook): …` / `feat(cli): …` / `docs(playbook): …`.

---

## Phase 1 — Pure scoring

### Task 1: `score_report` — objective + breakdown (pure, no LLM)

Smoothness is computed over **completed sessions only** (design §5: fail-fast sessions
must not game the smoothness mean; incomplete sessions are penalized via completion).

**Files:**
- Create: `src/superdialog/playbook/optimize.py`
- Create: `tests/playbook/test_optimize.py`

**Step 1: Write the failing test**

`tests/playbook/test_optimize.py`:
```python
"""Tests for the optimize loop: scoring, reflection, paired rounds."""

from superdialog.playbook.eval_bridge import EvalReport, SessionMetrics
from superdialog.playbook.optimize import ObjectiveBreakdown, score_report


def _session(**kw) -> SessionMetrics:
    base = dict(
        persona="p", completed=True, outcome="confirmed", turns=4,
        turns_per_checkpoint={"booking.collect": 2, "booking.confirm": 2},
        slot_accuracy=1.0, slot_diffs={}, repair_count=0, degraded_count=0,
        event_log_jsonl="",
    )
    base.update(kw)
    return SessionMetrics(**base)


def test_breakdown_dimensions_match_metrics() -> None:
    report = EvalReport(sessions=[_session(), _session(completed=False, outcome=None)])
    b = score_report(report)
    assert isinstance(b, ObjectiveBreakdown)
    assert b.completion_rate == 0.5
    assert b.slot_accuracy == 1.0
    # smoothness proxy: mean turns/checkpoint over COMPLETED sessions only
    assert b.mean_turns_per_checkpoint == 2.0
    assert b.repair_rate == 0.0


def test_scalar_objective_is_weighted_sum_in_unit_range() -> None:
    good = score_report(EvalReport(sessions=[_session()]))
    bad = score_report(EvalReport(sessions=[
        _session(completed=False, outcome=None, slot_accuracy=0.0,
                 repair_count=3, turns_per_checkpoint={"a": 8}),
    ]))
    assert 0.0 <= bad.objective < good.objective <= 1.0


def test_empty_report_scores_zero() -> None:
    b = score_report(EvalReport(sessions=[]))
    assert b.objective == 0.0
    assert b.completion_rate == 0.0


def test_smoothness_rewards_fewer_turns_per_checkpoint() -> None:
    smooth = score_report(EvalReport(sessions=[
        _session(turns_per_checkpoint={"a": 1, "b": 1})]))
    bumpy = score_report(EvalReport(sessions=[
        _session(turns_per_checkpoint={"a": 6, "b": 6})]))
    assert smooth.objective > bumpy.objective


def test_incomplete_sessions_earn_no_smoothness_credit() -> None:
    # A fail-fast incomplete session must not raise the smoothness term.
    failing = score_report(EvalReport(sessions=[
        _session(completed=False, outcome=None, slot_accuracy=0.0,
                 turns_per_checkpoint={"a": 1})]))
    assert failing.mean_turns_per_checkpoint == 0.0  # nothing completed
```

**Step 2: Run it — verify it fails**

Run: `uv run pytest tests/playbook/test_optimize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'superdialog.playbook.optimize'`.

**Step 3: Implement**

`src/superdialog/playbook/optimize.py`:
```python
"""Reflective prose optimizer: scoring, reflection, paired-round loop."""

from __future__ import annotations

from statistics import mean

from pydantic import BaseModel

from .eval_bridge import EvalReport

W_COMPLETION = 0.4
W_SLOT = 0.3
W_SMOOTHNESS = 0.2
W_REPAIR = 0.1


class ObjectiveBreakdown(BaseModel):
    """Scalar objective plus its per-dimension breakdown."""

    objective: float
    completion_rate: float
    slot_accuracy: float
    mean_turns_per_checkpoint: float
    repair_rate: float


def _smoothness(mean_turns_per_checkpoint: float) -> float:
    """Map mean turns/checkpoint to [0, 1]; 1 turn -> 1.0, more -> less."""
    return 1.0 / (1.0 + max(0.0, mean_turns_per_checkpoint - 1.0))


def score_report(report: EvalReport) -> ObjectiveBreakdown:
    """Score an eval report. Pure: no LLM, no I/O.

    Smoothness is averaged over completed sessions only, so fail-fast
    incomplete sessions cannot game the mean (they pay via completion).
    """
    if not report.sessions:
        return ObjectiveBreakdown(
            objective=0.0, completion_rate=0.0, slot_accuracy=0.0,
            mean_turns_per_checkpoint=0.0, repair_rate=0.0,
        )
    per_completed = [
        mean(s.turns_per_checkpoint.values())
        for s in report.sessions
        if s.completed and s.turns_per_checkpoint
    ]
    mean_tpc = mean(per_completed) if per_completed else 0.0
    total_turns = sum(s.turns for s in report.sessions)
    total_repairs = sum(s.repair_count for s in report.sessions)
    repair_rate = total_repairs / total_turns if total_turns else 0.0
    smooth = _smoothness(mean_tpc) if per_completed else 0.0
    objective = (
        W_COMPLETION * report.completion_rate
        + W_SLOT * report.mean_slot_accuracy
        + W_SMOOTHNESS * smooth
        + W_REPAIR * (1.0 - min(1.0, repair_rate))
    )
    return ObjectiveBreakdown(
        objective=objective,
        completion_rate=report.completion_rate,
        slot_accuracy=report.mean_slot_accuracy,
        mean_turns_per_checkpoint=mean_tpc,
        repair_rate=repair_rate,
    )
```

**Step 4: Run it — verify it passes**

Run: `uv run pytest tests/playbook/test_optimize.py -v` — 5 PASS.

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add src/superdialog/playbook/optimize.py tests/playbook/test_optimize.py
git commit -m "feat(playbook): objective scoring for the optimize loop"
```

---

## Phase 2 — EditableDoc: prose-only by construction

### Task 2: `FullDoc` — whitelist enumeration, targeted apply, faithful emit

The reflector never returns YAML; it returns `{address, new_text}` edits. `FullDoc`
enumerates the whitelist, applies edits to the **parsed source dict** (key order and
authored defaults survive; PyYAML comments do not), and re-validates by compiling.

**Whitelist (design §3):** `persona`; per checkpoint `guidance`, `goal`,
`never_say` (entries may be edited or added, never removed), `say_verbatim` **only
where already present**, `slots.<name>.description`, `advance_when[<i>].when` **only
where `judge == "llm"`** (the default). Everything else — including `expr` whens,
dispatch intents, interrupt whens, silence prompts — is unreachable.

**Files:**
- Create: `src/superdialog/playbook/editable.py`
- Create: `tests/playbook/test_editable.py`

**Step 1: Write the failing test**

`tests/playbook/test_editable.py`:
```python
"""Tests for the EditableDoc abstraction (FullDoc / SimpleDoc)."""

import pytest

from superdialog.playbook.editable import Edit, FullDoc, MutationError
from tests.playbook.test_models import MINIMAL_YAML

_GUIDANCE = "journeys.booking.checkpoints.collect.guidance"


def test_fields_enumerates_exactly_the_whitelist() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    addrs = {f.address for f in doc.fields()}
    assert "persona" in addrs
    assert _GUIDANCE in addrs
    assert "journeys.booking.checkpoints.collect.goal" in addrs
    assert "journeys.booking.checkpoints.collect.slots.city.description" in addrs
    # the collect rule is llm-judged -> editable
    assert "journeys.booking.checkpoints.collect.advance_when[0].when" in addrs
    # confirm's rules are expr-judged -> frozen
    assert "journeys.booking.checkpoints.confirm.advance_when[0].when" not in addrs
    # say_verbatim editable only where present
    assert "journeys.booking.checkpoints.confirm.say_verbatim" in addrs
    assert "journeys.booking.checkpoints.collect.say_verbatim" not in addrs
    # structure is unreachable
    assert "journeys.booking.checkpoints.confirm.gate" not in addrs


def test_apply_returns_new_doc_and_compiles() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    edited = doc.apply([Edit(address=_GUIDANCE, new_text="Collect warmly.")])
    assert edited.compile().checkpoint("booking.collect").guidance == "Collect warmly."
    # the original is untouched (apply is functional)
    assert doc.compile().checkpoint("booking.collect").guidance == "Collect naturally."


def test_emit_diff_touches_only_the_edited_line() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    edited = doc.apply([Edit(address=_GUIDANCE, new_text="Collect warmly.")])
    before = doc.emit().splitlines()
    after = edited.emit().splitlines()
    assert len(before) == len(after)
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == 1
    assert "Collect warmly." in changed[0][1]


def test_apply_rejects_non_whitelisted_addresses() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    for bad in (
        "journeys.booking.checkpoints.confirm.gate",            # structure
        "journeys.booking.checkpoints.confirm.advance_when[0].when",  # expr
        "journeys.booking.checkpoints.collect.say_verbatim",    # absent -> no add
        "journeys.booking.checkpoints.nope.guidance",           # unknown checkpoint
        "tools",                                                # structure
    ):
        with pytest.raises(MutationError):
            doc.apply([Edit(address=bad, new_text="x")])


def test_never_say_entries_may_be_added_but_not_removed() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    addr = "journeys.booking.checkpoints.collect.never_say"
    grown = doc.apply([Edit(address=addr, new_text=["never promise refunds"])])
    cp = grown.compile().checkpoint("booking.collect")
    assert cp.never_say == ["never promise refunds"]
    with pytest.raises(MutationError):
        grown.apply([Edit(address=addr, new_text=[])])  # shrinking is removal


def test_string_field_requires_string_payload() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    with pytest.raises(MutationError):
        doc.apply([Edit(address=_GUIDANCE, new_text=["not", "a", "string"])])


def test_pipeline_on_keys_survive_the_round_trip() -> None:
    # MINIMAL_YAML's pipeline uses an `on:` key; YAML 1.1 would load it as a
    # boolean. FullDoc must parse with the models loader, not yaml.safe_load.
    doc = FullDoc.from_text(MINIMAL_YAML)
    reparsed = FullDoc.from_text(doc.emit())
    assert reparsed.compile().pipeline("confirm_and_hold").steps[0].on
```

**Step 2: Run it — verify it fails**

Run: `uv run pytest tests/playbook/test_editable.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'superdialog.playbook.editable'`.

**Step 3: Implement**

`src/superdialog/playbook/editable.py`:
```python
"""EditableDoc: prose-only mutation surface over playbook documents.

The optimizer's reflect step returns targeted ``{address, new_text}`` edits.
Applying them through this module enforces the prose whitelist by
construction: a non-whitelisted address raises ``MutationError`` and the
document is recompiled (re-validated) after every apply.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import yaml
from pydantic import BaseModel

from .models import Playbook, _YamlLoader
from .simple import is_simple_playbook, simple_to_playbook


class MutationError(ValueError):
    """An edit addressed a frozen field or carried a bad payload."""


class Edit(BaseModel):
    """One targeted prose edit, addressed into the source document."""

    address: str
    new_text: str | list[str]


class FieldRef(BaseModel):
    """An editable field: its address and current text."""

    address: str
    text: str | list[str]


_RULE_RE = re.compile(r"^advance_when\[(\d+)\]\.when$")


class FullDoc:
    """Editable view over a full-format playbook document."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self.compile()  # validate eagerly; raises on a broken document

    @classmethod
    def from_text(cls, text: str) -> "FullDoc":
        """Parse full-format YAML/JSON text (models' YAML-1.2 loader)."""
        return cls(yaml.load(text, Loader=_YamlLoader))

    def compile(self) -> Playbook:
        """Validate and return the runtime Playbook."""
        return Playbook.model_validate(self._doc)

    def emit(self) -> str:
        """Serialize the document back to YAML, preserving key order."""
        return yaml.safe_dump(self._doc, sort_keys=False, allow_unicode=True)

    def fields(self) -> list[FieldRef]:
        """Enumerate every whitelisted (editable) field with current text."""
        refs = [FieldRef(address="persona", text=self._doc.get("persona", ""))]
        for jname, journey in self._doc.get("journeys", {}).items():
            for cp in journey.get("checkpoints", []):
                base = f"journeys.{jname}.checkpoints.{cp['id']}"
                refs.append(
                    FieldRef(address=f"{base}.guidance", text=cp.get("guidance", ""))
                )
                refs.append(FieldRef(address=f"{base}.goal", text=cp.get("goal", "")))
                refs.append(
                    FieldRef(
                        address=f"{base}.never_say", text=cp.get("never_say", [])
                    )
                )
                if cp.get("say_verbatim") is not None:
                    refs.append(
                        FieldRef(
                            address=f"{base}.say_verbatim", text=cp["say_verbatim"]
                        )
                    )
                for slot_name, spec in (cp.get("slots") or {}).items():
                    refs.append(
                        FieldRef(
                            address=f"{base}.slots.{slot_name}.description",
                            text=(spec or {}).get("description", ""),
                        )
                    )
                for i, rule in enumerate(cp.get("advance_when", [])):
                    if rule.get("judge", "llm") == "llm":
                        refs.append(
                            FieldRef(
                                address=f"{base}.advance_when[{i}].when",
                                text=rule.get("when", ""),
                            )
                        )
        return refs

    def apply(self, edits: list[Edit]) -> "FullDoc":
        """Return a new FullDoc with edits applied; reject frozen addresses."""
        allowed = {f.address: f.text for f in self.fields()}
        new = copy.deepcopy(self._doc)
        for edit in edits:
            if edit.address not in allowed:
                raise MutationError(f"address not editable: {edit.address}")
            _check_payload(edit, allowed[edit.address])
            _set_full(new, edit.address, edit.new_text)
        return FullDoc(new)


def _check_payload(edit: Edit, current: str | list[str]) -> None:
    """List fields take grow-only lists of strings; others take strings."""
    if isinstance(current, list):
        if not isinstance(edit.new_text, list) or not all(
            isinstance(s, str) for s in edit.new_text
        ):
            raise MutationError(f"{edit.address}: expected a list of strings")
        if len(edit.new_text) < len(current):
            raise MutationError(f"{edit.address}: entries may not be removed")
    elif not isinstance(edit.new_text, str):
        raise MutationError(f"{edit.address}: expected a string")


def _set_full(doc: dict[str, Any], address: str, value: str | list[str]) -> None:
    """Write `value` at a (pre-validated) FullDoc address inside the dict."""
    if address == "persona":
        doc["persona"] = value
        return
    parts = address.split(".")
    # journeys.<j>.checkpoints.<id>.<rest...>
    journey = doc["journeys"][parts[1]]
    cp = next(c for c in journey["checkpoints"] if c["id"] == parts[3])
    rest = parts[4:]
    if rest[0] == "slots":
        cp["slots"][rest[1]]["description"] = value
        return
    rule_match = _RULE_RE.match(".".join(rest))
    if rule_match:
        cp["advance_when"][int(rule_match.group(1))]["when"] = value
        return
    cp[rest[0]] = value
```

Note: `_YamlLoader` is a private name imported from a sibling module in the same
package — acceptable intra-package use; do not re-export it. It subclasses
`yaml.SafeLoader` (models.py:13) and only overrides boolean resolution, so
`yaml.load(text, Loader=_YamlLoader)` is a safe load — it cannot construct
arbitrary Python objects. This mirrors what `Playbook.from_yaml` already does.

Note: `advance_when` entries in MINIMAL_YAML are flow-style mappings
(`{when: ..., judge: llm, ...}`) — after parsing they are plain dicts, so
`rule.get("judge", "llm")` handles both explicit and defaulted judges.

The `slots.<name>.description` address has dots inside `parts[4:]` only at fixed
positions (`slots`, `<name>`, `description`), so positional split is safe. Slot names
and checkpoint ids containing dots are not supported (they aren't today either:
`Playbook.checkpoint` uses dotted journey refs).

**Step 4: Run it — verify it passes**

Run: `uv run pytest tests/playbook/test_editable.py -v` — 7 PASS.

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add src/superdialog/playbook/editable.py tests/playbook/test_editable.py
git commit -m "feat(playbook): FullDoc editable document with prose whitelist"
```

---

### Task 3: `SimpleDoc` + `make_editable` — simple format round-trips

Simple-origin playbooks stay in their format. Facts/objections/boundaries/fallbacks
are **not** whitelisted; each compile refolds them into the persona, so reference
facts are protected by construction (design §3).

**Whitelist:** per step `say`, `done_when`, `purpose`; top-level `opening`, `closing`,
`persona.identity`, `persona.voice_style`.

**Files:**
- Modify: `src/superdialog/playbook/editable.py`
- Modify: `tests/playbook/test_editable.py`

**Step 1: Write the failing test**

Append to `tests/playbook/test_editable.py`:
```python
import yaml as _yaml

from superdialog.playbook.editable import SimpleDoc, make_editable
from tests.playbook.test_simple import SIMPLE


def test_simple_fields_enumerate_step_and_persona_prose() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    addrs = {f.address for f in doc.fields()}
    assert {"opening", "closing", "persona.identity", "persona.voice_style",
            "steps.collect.say", "steps.collect.done_when",
            "steps.collect.purpose"} <= addrs
    # reference data is frozen
    assert not any(a.startswith(("facts", "objections", "boundaries",
                                 "fallback_actions")) for a in addrs)


def test_simple_apply_recompiles_and_emits_simple_format() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    edited = doc.apply(
        [Edit(address="steps.collect.say", new_text="Warmly ask their name.")]
    )
    cp = edited.compile().checkpoint("main.collect")
    assert cp.guidance == "Warmly ask their name."
    out = _yaml.safe_load(edited.emit())
    assert "playbook" in out and "journeys" not in out  # still simple format


def test_simple_facts_survive_prose_edits() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    edited = doc.apply(
        [Edit(address="persona.voice_style", new_text="Bubbly and quick.")]
    )
    persona = edited.compile().persona
    assert "₹400" in persona            # canonical pricing intact
    assert "NEVER invent prices" in persona  # boundaries intact


def test_simple_apply_rejects_frozen_addresses() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    for bad in ("facts.canonical_pricing.haircut", "boundaries",
                "steps.collect.collect", "steps.nope.say"):
        with pytest.raises(MutationError):
            doc.apply([Edit(address=bad, new_text="x")])


def test_make_editable_routes_by_format() -> None:
    assert isinstance(make_editable(SIMPLE), SimpleDoc)
    assert isinstance(make_editable(MINIMAL_YAML), FullDoc)
```

**Step 2: Run it — verify it fails**

Run: `uv run pytest tests/playbook/test_editable.py -v`
Expected: FAIL — `ImportError: cannot import name 'SimpleDoc'`.

**Step 3: Implement**

Append to `src/superdialog/playbook/editable.py`:
```python
class SimpleDoc:
    """Editable view over a simple-format playbook document."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self.compile()  # validate eagerly

    @classmethod
    def from_text(cls, text: str) -> "SimpleDoc":
        """Parse simple-format YAML/JSON text (same loader as load_simple)."""
        return cls(yaml.safe_load(text))

    def compile(self) -> Playbook:
        """Lower to the validated runtime Playbook."""
        return simple_to_playbook(self._doc)

    def emit(self) -> str:
        """Serialize back to simple-format YAML, preserving key order."""
        return yaml.safe_dump(self._doc, sort_keys=False, allow_unicode=True)

    def fields(self) -> list[FieldRef]:
        """Enumerate the simple-format prose whitelist."""
        persona = self._doc.get("persona") or {}
        refs = [
            FieldRef(address="opening", text=self._doc.get("opening", "")),
            FieldRef(address="closing", text=self._doc.get("closing", "")),
            FieldRef(address="persona.identity", text=persona.get("identity", "")),
            FieldRef(
                address="persona.voice_style", text=persona.get("voice_style", "")
            ),
        ]
        for step in self._doc.get("playbook", []):
            base = f"steps.{step['id']}"
            refs.append(FieldRef(address=f"{base}.say", text=step.get("say", "")))
            refs.append(
                FieldRef(address=f"{base}.done_when", text=step.get("done_when", ""))
            )
            refs.append(
                FieldRef(address=f"{base}.purpose", text=step.get("purpose", ""))
            )
        return refs

    def apply(self, edits: list[Edit]) -> "SimpleDoc":
        """Return a new SimpleDoc with edits applied; reject frozen addresses."""
        allowed = {f.address: f.text for f in self.fields()}
        new = copy.deepcopy(self._doc)
        for edit in edits:
            if edit.address not in allowed:
                raise MutationError(f"address not editable: {edit.address}")
            _check_payload(edit, allowed[edit.address])
            _set_simple(new, edit.address, edit.new_text)
        return SimpleDoc(new)


def _set_simple(doc: dict[str, Any], address: str, value: str | list[str]) -> None:
    """Write `value` at a (pre-validated) SimpleDoc address inside the dict."""
    if address.startswith("steps."):
        _, step_id, field = address.split(".")
        step = next(s for s in doc["playbook"] if s["id"] == step_id)
        step[field] = value
    elif address.startswith("persona."):
        doc.setdefault("persona", {})[address.split(".", 1)[1]] = value
    else:
        doc[address] = value


EditableDoc = FullDoc | SimpleDoc


def make_editable(text: str) -> EditableDoc:
    """Detect the source format and wrap it in the matching editable doc."""
    probe = yaml.safe_load(text)
    if is_simple_playbook(probe):
        return SimpleDoc(probe)
    return FullDoc.from_text(text)
```

**Step 4: Run it — verify it passes**

Run: `uv run pytest tests/playbook/test_editable.py -v` — 12 PASS.

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git commit -am "feat(playbook): SimpleDoc round-trips simple-format playbooks"
```

---

## Phase 3 — Reflect, trace, loop

### Task 4: `propose_edits` — reflect prompt, JSON parse, validate, retry

Wraps the candidate `CompletesLLM`: prompt = source YAML + enumerated editable fields
+ worst-k session evidence; response = JSON edit array. Validation before any eval
spend: whitelist (via `apply`), recompile (construction), and a Jinja **syntax** parse
on edited `guidance`/`say`/`say_verbatim` (broken templates pass model validation and
only fail at runtime — render.py renders guidance through jinja2). On any failure,
retry up to `max_attempts`; exhausted → `None`.

Returns `tuple[EditableDoc, list[Edit]] | None` — the loop records the edit list in
its trace (it *is* the human-readable diff).

**Files:**
- Modify: `src/superdialog/playbook/optimize.py`
- Modify: `tests/playbook/test_optimize.py`

**Step 1: Write the failing test**

Append to `tests/playbook/test_optimize.py`:
```python
import json

from superdialog.playbook.editable import FullDoc
from superdialog.playbook.optimize import propose_edits
from tests.playbook.test_models import MINIMAL_YAML

_GUIDANCE = "journeys.booking.checkpoints.collect.guidance"


class CannedEditsLLM:
    """Candidate LLM: pops scripted outputs, repeating the last one forever."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


def _edit_json(address: str = _GUIDANCE,
               new_text: str = "Ask for the city first, warmly.") -> str:
    return json.dumps([{"address": address, "new_text": new_text}])


def _report(**kw) -> EvalReport:
    base = dict(
        persona="p", completed=False, outcome=None, turns=6,
        turns_per_checkpoint={"booking.collect": 6}, slot_accuracy=0.0,
        slot_diffs={"city": ("Pune", None)}, repair_count=2, degraded_count=0,
        event_log_jsonl='{"type":"utterance","version":1,"role":"user","text":"uh"}',
    )
    base.update(kw)
    return EvalReport(sessions=[SessionMetrics(**base)])


async def test_propose_returns_doc_and_edits() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM([_edit_json()])
    proposal = await propose_edits(doc, _report(), llm, max_attempts=3)
    assert proposal is not None
    cand, edits = proposal
    assert (cand.compile().checkpoint("booking.collect").guidance
            == "Ask for the city first, warmly.")
    assert edits[0].address == _GUIDANCE
    # the prompt showed current prose, the editable address, and the evidence
    prompt = " ".join(m["content"] for m in llm.calls[0])
    assert "Collect naturally." in prompt
    assert _GUIDANCE in prompt
    assert "city" in prompt


async def test_fenced_json_is_accepted() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM(["```json\n" + _edit_json() + "\n```"])
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is not None


async def test_invalid_json_retries_then_falls_back() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM(["not json at all", "{\"also\": \"not a list\"}"])
    assert await propose_edits(doc, _report(), llm, max_attempts=2) is None
    assert len(llm.calls) == 2


async def test_frozen_address_is_rejected() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM([_edit_json(
        address="journeys.booking.checkpoints.confirm.gate", new_text="soft")])
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is None


async def test_broken_jinja_is_rejected() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM([_edit_json(new_text="Hello {{ slots.city ")])
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is None


async def test_empty_edit_list_is_rejected() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM(["[]"])
    assert await propose_edits(doc, _report(), llm, max_attempts=1) is None
```

**Step 2: Run it — verify it fails**

Run: `uv run pytest tests/playbook/test_optimize.py -v`
Expected: FAIL — `ImportError: cannot import name 'propose_edits'`.

**Step 3: Implement**

Append to `src/superdialog/playbook/optimize.py`:
```python
import json

from jinja2 import Environment, TemplateSyntaxError

from .director import CompletesLLM
from .editable import Edit, EditableDoc
from .eval_bridge import SessionMetrics

_JINJA = Environment()
_EVENT_LOG_CAP = 4000  # chars of event log shown per worst session
_JINJA_CHECKED_SUFFIXES = (".guidance", ".say_verbatim", ".say")

_REFLECT_RULES = """\
You improve conversational playbook prose. You will see the current playbook,
the exact list of editable field addresses, and evidence from the worst
self-play sessions.

Return ONLY a JSON array of edits: [{"address": "...", "new_text": "..."}].
Rules:
- Use only addresses from the EDITABLE FIELDS list, verbatim.
- new_text is a string (or a list of strings for never_say-style fields;
  never remove existing entries).
- Do not alter factual claims, prices, or hard boundaries anywhere.
- Propose at least one edit. No commentary, no markdown fences.
"""


def _worst_sessions(report: EvalReport, k: int = 3) -> list[SessionMetrics]:
    """The k weakest sessions: incomplete, then inaccurate, then repair-heavy."""
    ranked = sorted(
        report.sessions,
        key=lambda s: (s.completed, s.slot_accuracy, -s.repair_count),
    )
    return ranked[:k]


def _reflect_messages(
    doc: EditableDoc, report: EvalReport, k: int = 3
) -> list[dict[str, str]]:
    """Build the candidate-LLM prompt from the doc and the failing evidence."""
    fields_block = "\n".join(
        f"- {f.address}: {f.text!r}" for f in doc.fields()
    )
    sessions: list[str] = []
    for s in _worst_sessions(report, k):
        sessions.append(
            f"persona={s.persona} completed={s.completed} outcome={s.outcome}\n"
            f"slot_diffs={s.slot_diffs} repair_count={s.repair_count}\n"
            f"turns_per_checkpoint={s.turns_per_checkpoint}\n"
            f"log:\n{s.event_log_jsonl[:_EVENT_LOG_CAP]}"
        )
    user = (
        f"PLAYBOOK:\n{doc.emit()}\n\n"
        f"EDITABLE FIELDS:\n{fields_block}\n\n"
        f"WORST SESSIONS:\n" + "\n---\n".join(sessions)
    )
    return [
        {"role": "system", "content": _REFLECT_RULES},
        {"role": "user", "content": user},
    ]


def _parse_edits(raw: str) -> list[Edit]:
    """Parse the candidate's JSON edit array; raise ValueError when malformed."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    if not isinstance(data, list) or not data:
        raise ValueError("expected a non-empty JSON array of edits")
    return [Edit.model_validate(item) for item in data]


def _check_jinja(edits: list[Edit]) -> None:
    """Syntax-parse template-bearing edits; broken Jinja fails at runtime."""
    for edit in edits:
        if edit.address.endswith(_JINJA_CHECKED_SUFFIXES) and isinstance(
            edit.new_text, str
        ):
            try:
                _JINJA.parse(edit.new_text)
            except TemplateSyntaxError as exc:
                raise ValueError(f"{edit.address}: broken Jinja: {exc}") from exc


async def propose_edits(
    doc: EditableDoc,
    report: EvalReport,
    candidate_llm: CompletesLLM,
    *,
    max_attempts: int = 3,
) -> tuple[EditableDoc, list[Edit]] | None:
    """Ask the candidate LLM for prose edits; validate; retry; None on failure.

    The candidate output is untrusted text: it is parsed and validated, never
    executed. ValidationError, MutationError and JSONDecodeError are all
    ValueError subclasses, so one except clause covers every reject path.
    """
    messages = _reflect_messages(doc, report)
    for _ in range(max_attempts):
        raw = await candidate_llm.complete(messages)
        try:
            edits = _parse_edits(raw)
            _check_jinja(edits)
            candidate = doc.apply(edits)  # whitelist + recompile validation
        except ValueError:
            continue
        return candidate, edits
    return None
```

**Step 4: Run it — verify it passes**

Run: `uv run pytest tests/playbook/test_optimize.py -v` — all PASS.

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git commit -am "feat(playbook): reflect step proposes validated targeted edits"
```

---

### Task 5: `RoundTrace` + informational `ParetoFrontier`

The frontier never picks the output (design §2); it reports which rounds traded off
completion vs slot-accuracy vs smoothness. Field is `round_no` (not `round` — avoid
shadowing the builtin).

**Files:**
- Modify: `src/superdialog/playbook/optimize.py`
- Modify: `tests/playbook/test_optimize.py`

**Step 1: Write the failing test**

Append to `tests/playbook/test_optimize.py`:
```python
from superdialog.playbook.optimize import ParetoFrontier, RoundTrace


def _breakdown(completion: float, slot: float, turns: float) -> ObjectiveBreakdown:
    return ObjectiveBreakdown(
        objective=completion, completion_rate=completion, slot_accuracy=slot,
        mean_turns_per_checkpoint=turns, repair_rate=0.0)


def _trace(round_no: int, completion: float, slot: float,
           turns: float) -> RoundTrace:
    return RoundTrace(
        round_no=round_no, accepted=True,
        incumbent_breakdown=_breakdown(0.1, 0.1, 9.0),
        candidate_breakdown=_breakdown(completion, slot, turns))


def test_frontier_keeps_non_dominated() -> None:
    f = ParetoFrontier()
    f.consider(_trace(1, completion=0.9, slot=0.5, turns=2.0))
    f.consider(_trace(2, completion=0.5, slot=0.9, turns=2.0))  # trades off
    f.consider(_trace(3, completion=0.4, slot=0.4, turns=3.0))  # dominated
    assert sorted(t.round_no for t in f.members) == [1, 2]


def test_frontier_drops_newly_dominated_member() -> None:
    f = ParetoFrontier()
    f.consider(_trace(1, completion=0.6, slot=0.6, turns=2.0))
    f.consider(_trace(2, completion=0.9, slot=0.9, turns=1.0))  # dominates #1
    assert [t.round_no for t in f.members] == [2]


def test_frontier_ignores_rounds_without_a_candidate() -> None:
    f = ParetoFrontier()
    f.consider(RoundTrace(
        round_no=1, accepted=False,
        incumbent_breakdown=_breakdown(0.5, 0.5, 2.0),
        detail="no valid candidate"))
    assert f.members == []
```

**Step 2: Run it — verify it fails**

Run: `uv run pytest tests/playbook/test_optimize.py -v` — ImportError.

**Step 3: Implement**

Append to `src/superdialog/playbook/optimize.py`:
```python
from pydantic import Field


class RoundTrace(BaseModel):
    """One optimization round: same-round paired scores plus the edit list."""

    round_no: int
    accepted: bool
    incumbent_breakdown: ObjectiveBreakdown
    candidate_breakdown: ObjectiveBreakdown | None = None
    edits: list[Edit] = Field(default_factory=list)
    detail: str = ""


class ParetoFrontier(BaseModel):
    """Non-dominated candidate rounds over completion/slot/smoothness.

    Informational only: the loop never picks its output from the frontier
    (cross-round scores come from different eval runs).
    """

    members: list[RoundTrace] = Field(default_factory=list)

    @staticmethod
    def _vector(t: RoundTrace) -> tuple[float, float, float]:
        b = t.candidate_breakdown
        assert b is not None  # consider() filters None
        return (b.completion_rate, b.slot_accuracy,
                _smoothness(b.mean_turns_per_checkpoint))

    @classmethod
    def _dominates(cls, a: RoundTrace, b: RoundTrace) -> bool:
        va, vb = cls._vector(a), cls._vector(b)
        return all(x >= y for x, y in zip(va, vb)) and va != vb

    def consider(self, t: RoundTrace) -> None:
        """Add `t` unless dominated; evict members it dominates."""
        if t.candidate_breakdown is None:
            return
        if any(self._dominates(m, t) for m in self.members):
            return
        self.members = [m for m in self.members if not self._dominates(t, m)]
        self.members.append(t)
```

**Step 4: Run it — verify it passes** — all PASS.

**Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git commit -am "feat(playbook): round trace and informational Pareto frontier"
```

---

### Task 6: `optimize()` — the paired-round loop

Round 0 evaluates the input once (baseline + reflection seed). Each round proposes
edits, then evaluates incumbent AND candidate fresh; accepts iff the candidate's
same-round objective strictly beats the incumbent's. Output: the final incumbent,
emitted in the source format. All LLMs/HTTP injected — fully offline in tests.

**Files:**
- Modify: `src/superdialog/playbook/optimize.py`
- Modify: `tests/playbook/test_optimize.py`

**Step 1: Write the failing test**

Append to `tests/playbook/test_optimize.py`:
```python
import textwrap

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.editable import SimpleDoc
from superdialog.playbook.eval_bridge import PersonaSpec
from superdialog.playbook.models import Playbook
from superdialog.playbook.optimize import OptimizeReport, optimize
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_eval_bridge import ScriptedUser
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE = {"slots": {}, "advance": None, "note": None}
_ADVANCE = {"slots": {"city": "Pune", "date": "2026-06-12"},
            "advance": "booking.confirm", "note": None}
_HOLD_OK = (200, {"data": {"hold_id": "h1"}})

_PERSONAS = [PersonaSpec(
    name="closer", traits="direct", goal="book in Pune",
    ground_truth_slots={"city": "Pune", "date": "2026-06-12"})]


def _improving_agent_factory(playbook: Playbook) -> PlaybookAgent:
    """Director completes the booking only after the 'warmly' mutation."""
    improved = "warmly" in playbook.checkpoint("booking.collect").guidance
    return PlaybookAgent(
        playbook=playbook,
        talker_llm=StreamLLM(["Which", " city?"]),
        director_llm=CannedLLM(_ADVANCE if improved else _IDLE),
        http=FakeHttp([_HOLD_OK] * 4),
    )


async def test_optimize_improves_and_emits_final_incumbent() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    llm = CannedEditsLLM([_edit_json(new_text="Collect warmly.")])
    report = await optimize(
        doc, personas=_PERSONAS, candidate_llm=llm,
        user_llm=ScriptedUser(["Pune on 2026-06-12 please", "ok"]),
        agent_factory=_improving_agent_factory, rounds=2, n=1)
    assert isinstance(report, OptimizeReport)
    assert "Collect warmly." in report.final_yaml
    assert report.final_breakdown.objective > report.initial_breakdown.objective
    accepted = [t for t in report.trace if t.accepted]
    assert accepted and accepted[0].edits[0].address == _GUIDANCE
    assert accepted[0].candidate_breakdown is not None


async def test_no_valid_candidate_keeps_the_input() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    report = await optimize(
        doc, personas=_PERSONAS, candidate_llm=CannedEditsLLM(["not json"]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory, rounds=1, n=1)
    assert report.final_yaml == doc.emit()
    assert report.trace[0].accepted is False
    assert report.trace[0].detail == "no valid candidate"


async def test_noop_edit_is_never_accepted_and_round_cap_holds() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    noop = _edit_json(new_text="Collect naturally.")  # identical prose
    report = await optimize(
        doc, personas=_PERSONAS, candidate_llm=CannedEditsLLM([noop]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory, rounds=3, n=1, patience=99)
    assert len(report.trace) == 3
    assert not any(t.accepted for t in report.trace)


async def test_patience_stops_early() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    noop = _edit_json(new_text="Collect naturally.")
    report = await optimize(
        doc, personas=_PERSONAS, candidate_llm=CannedEditsLLM([noop]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory, rounds=5, n=1, patience=2)
    assert len(report.trace) == 2  # stopped after `patience` stale rounds


_SIMPLE_TWO_STEP = textwrap.dedent("""
    name: "Mini"
    goal: "Say hello and close."
    persona:
      identity: "You are a tiny demo agent."
    opening: "Greet the caller."
    playbook:
      - id: hello
        purpose: "Open the call."
        say: "Greet and ask how to help."
        done_when: "Caller responded."
      - id: done
        purpose: "Close."
        say: "Wrap up."
        done_when: "Closed."
""")


def _simple_improving_factory(playbook: Playbook) -> PlaybookAgent:
    improved = "warmly" in playbook.checkpoint("main.hello").guidance
    verdict = ({"slots": {}, "advance": "main.done", "note": None}
               if improved else _IDLE)
    return PlaybookAgent(
        playbook=playbook,
        talker_llm=StreamLLM(["Hi", " there"]),
        director_llm=CannedLLM(verdict),
        http=FakeHttp([]),
    )


async def test_simple_doc_optimizes_and_stays_simple() -> None:
    doc = SimpleDoc.from_text(_SIMPLE_TWO_STEP)
    edit = json.dumps([{"address": "steps.hello.say",
                        "new_text": "Greet warmly and ask how to help."}])
    report = await optimize(
        doc, personas=[PersonaSpec(name="p", traits="brief", goal="say hi")],
        candidate_llm=CannedEditsLLM([edit]),
        user_llm=ScriptedUser(["hello", "bye"]),
        agent_factory=_simple_improving_factory, rounds=1, n=1)
    assert any(t.accepted for t in report.trace)
    out = _yaml.safe_load(report.final_yaml)
    assert "playbook" in out and "journeys" not in out  # still simple format
    assert "warmly" in out["playbook"][0]["say"]
```

Also add `import yaml as _yaml` to the test module imports if not already present
from Task 3's additions.

**Step 2: Run it — verify it fails** — ImportError on `optimize`/`OptimizeReport`.

**Step 3: Implement**

Append to `src/superdialog/playbook/optimize.py`:
```python
from typing import Callable

from .agent import PlaybookAgent
from .eval_bridge import PersonaSpec, SpeaksUser, run_eval
from .models import Playbook

AgentFactory = Callable[[Playbook], PlaybookAgent]


class OptimizeReport(BaseModel):
    """The optimize run's result: final artifact plus the full metric trace."""

    final_yaml: str
    initial_breakdown: ObjectiveBreakdown
    final_breakdown: ObjectiveBreakdown
    trace: list[RoundTrace]
    frontier: list[RoundTrace]


async def optimize(
    doc: EditableDoc,
    *,
    personas: list[PersonaSpec],
    candidate_llm: CompletesLLM,
    user_llm: SpeaksUser,
    agent_factory: AgentFactory,
    rounds: int = 3,
    n: int = 1,
    patience: int = 2,
    reflect_attempts: int = 3,
) -> OptimizeReport:
    """Paired-round reflective optimization. Returns the final incumbent.

    Acceptance compares only same-round scores: each round evaluates the
    incumbent AND the candidate fresh, so both face the same sampling noise.
    The Pareto frontier is reported but never picks the output.
    """

    async def _eval(d: EditableDoc) -> EvalReport:
        playbook = d.compile()
        return await run_eval(
            lambda: agent_factory(playbook), personas, user_llm, n
        )

    incumbent = doc
    last_report = await _eval(incumbent)
    initial_b = score_report(last_report)
    final_b = initial_b
    frontier = ParetoFrontier()
    trace: list[RoundTrace] = []
    stale = 0
    for round_no in range(1, rounds + 1):
        proposal = await propose_edits(
            incumbent, last_report, candidate_llm, max_attempts=reflect_attempts
        )
        if proposal is None:
            trace.append(RoundTrace(
                round_no=round_no, accepted=False,
                incumbent_breakdown=final_b, detail="no valid candidate",
            ))
            stale += 1
        else:
            candidate, edits = proposal
            inc_report = await _eval(incumbent)
            cand_report = await _eval(candidate)
            inc_b = score_report(inc_report)
            cand_b = score_report(cand_report)
            accepted = cand_b.objective > inc_b.objective
            t = RoundTrace(
                round_no=round_no, accepted=accepted,
                incumbent_breakdown=inc_b, candidate_breakdown=cand_b,
                edits=edits,
            )
            trace.append(t)
            frontier.consider(t)
            if accepted:
                incumbent, last_report, final_b = candidate, cand_report, cand_b
                stale = 0
            else:
                last_report, final_b = inc_report, inc_b
                stale += 1
        if stale >= patience:
            break
    return OptimizeReport(
        final_yaml=incumbent.emit(),
        initial_breakdown=initial_b,
        final_breakdown=final_b,
        trace=trace,
        frontier=frontier.members,
    )
```

**Step 4: Run it — verify it passes**

Run: `uv run pytest tests/playbook/test_optimize.py -v` — all PASS.
Also run the neighbors: `uv run pytest tests/playbook -v` — no regressions.

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git commit -am "feat(playbook): optimize() paired-round loop with convergence"
```

---

## Phase 4 — Personas

### Task 7: `personas.py` — load, save, generate, derive

Persona resolution (design §4): explicit `--personas` path wins; else a cache file
`<playbook stem>.personas.yaml` beside the playbook; else generate 4 personas along
fixed diversity axes and write the cache. Generation failure falls back (in the CLI)
to one derived persona.

**Files:**
- Create: `src/superdialog/playbook/personas.py`
- Create: `tests/playbook/test_personas.py`

**Step 1: Write the failing test**

`tests/playbook/test_personas.py`:
```python
"""Tests for persona suite load/save/generate/derive."""

import json

import pytest

from superdialog.playbook.eval_bridge import PersonaSpec
from superdialog.playbook.models import Playbook
from superdialog.playbook.personas import (
    derive_default_persona,
    generate_personas,
    load_personas,
    persona_cache_path,
    save_personas,
)
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_optimize import CannedEditsLLM


def _personas_json(count: int = 4, drop_slot: bool = False) -> str:
    slots = {"city": "Pune", "date": "2026-06-12"}
    if drop_slot:
        slots = {"city": "Pune"}
    return json.dumps([
        {"name": f"p{i}", "traits": "direct", "goal": "book a slot",
         "ground_truth_slots": slots}
        for i in range(count)
    ])


def test_save_load_round_trip(tmp_path) -> None:
    personas = [PersonaSpec(name="a", traits="t", goal="g",
                            ground_truth_slots={"city": "Pune"})]
    path = tmp_path / "x.personas.yaml"
    save_personas(personas, str(path))
    loaded = load_personas(str(path))
    assert loaded == personas


def test_load_rejects_non_list(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: not-a-list\n")
    with pytest.raises(ValueError):
        load_personas(str(path))


def test_cache_path_sits_beside_the_playbook(tmp_path) -> None:
    p = tmp_path / "booking.yaml"
    assert persona_cache_path(str(p)) == str(tmp_path / "booking.personas.yaml")


def test_derive_default_uses_initial_checkpoint_goal() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    persona = derive_default_persona(pb)
    assert "Have city and date" in persona.goal
    assert persona.ground_truth_slots == {}


async def test_generate_returns_validated_suite() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    llm = CannedEditsLLM([_personas_json()])
    personas = await generate_personas(pb, llm)
    assert len(personas) == 4
    assert all(p.ground_truth_slots.keys() >= {"city", "date"} for p in personas)
    # the prompt enumerated the slot schema and the diversity axes
    prompt = " ".join(m["content"] for m in llm.calls[0])
    assert "city" in prompt and "date" in prompt
    assert "tangent" in prompt


async def test_generate_retries_then_raises_on_missing_required_slots() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    llm = CannedEditsLLM([_personas_json(drop_slot=True)])
    with pytest.raises(ValueError):
        await generate_personas(pb, llm, max_attempts=2)
    assert len(llm.calls) == 2
```

**Step 2: Run it — verify it fails** — ModuleNotFoundError.

**Step 3: Implement**

`src/superdialog/playbook/personas.py`:
```python
"""Persona suites for the optimizer: load, save, generate, derive."""

from __future__ import annotations

import json
import os

import yaml

from .director import CompletesLLM
from .eval_bridge import PersonaSpec
from .models import Playbook

_AXES = (
    "cooperative and forthcoming",
    "terse and impatient",
    "tangent-prone (wanders off-topic, must be steered back)",
    "error-making (gives one wrong slot value, then corrects it when asked)",
)

_GEN_SYSTEM = """\
You create test personas for evaluating a conversational agent. Given the
playbook summary and its slot schema, return ONLY a JSON array of exactly
{count} personas, one per diversity axis:
{axes}

Each persona: {{"name": str, "traits": str, "goal": str,
"ground_truth_slots": {{...}}}}. ground_truth_slots MUST contain a concrete,
plausible value for EVERY required slot listed. No commentary, no fences.
"""


def persona_cache_path(playbook_path: str) -> str:
    """The conventional persona-suite cache path beside the playbook."""
    root, _ = os.path.splitext(playbook_path)
    return f"{root}.personas.yaml"


def load_personas(path: str) -> list[PersonaSpec]:
    """Load a YAML/JSON list of PersonaSpec dicts; ValueError when malformed."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    data = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of personas")
    return [PersonaSpec.model_validate(item) for item in data]


def save_personas(personas: list[PersonaSpec], path: str) -> None:
    """Write a persona suite as reviewable YAML."""
    dumped = yaml.safe_dump(
        [p.model_dump() for p in personas], sort_keys=False, allow_unicode=True
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumped)


def derive_default_persona(playbook: Playbook) -> PersonaSpec:
    """Single fallback persona derived from the initial checkpoint's goal."""
    cp = playbook.checkpoint(playbook.initial_checkpoint_id)
    goal = cp.goal or "complete the conversation"
    return PersonaSpec(
        name="default",
        traits="cooperative, concise",
        goal=f"Work with the agent so it can: {goal}",
    )


def _required_slots(playbook: Playbook) -> dict[str, str]:
    """Map of required slot key -> type across all checkpoints."""
    out: dict[str, str] = {}
    for journey in playbook.journeys.values():
        for cp in journey.checkpoints:
            for key, spec in cp.slots.items():
                if spec.required:
                    out[key] = spec.type
    return out


def _summary(playbook: Playbook) -> str:
    lines: list[str] = []
    for jname, journey in playbook.journeys.items():
        for cp in journey.checkpoints:
            lines.append(f"- {jname}.{cp.id}: goal={cp.goal!r}")
    return "\n".join(lines)


async def generate_personas(
    playbook: Playbook,
    llm: CompletesLLM,
    *,
    count: int = 4,
    max_attempts: int = 3,
) -> list[PersonaSpec]:
    """Generate a diverse persona suite; ValueError after max_attempts."""
    required = _required_slots(playbook)
    system = _GEN_SYSTEM.format(
        count=count, axes="\n".join(f"- {a}" for a in _AXES)
    )
    user = (
        f"CHECKPOINTS:\n{_summary(playbook)}\n\n"
        f"REQUIRED SLOTS (every persona needs concrete values for all):\n"
        + "\n".join(f"- {k} ({t})" for k, t in required.items())
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = "no attempts made"
    for _ in range(max_attempts):
        raw = await llm.complete(messages)
        try:
            data = json.loads(raw.strip().strip("`"))
            if not isinstance(data, list):
                raise ValueError("expected a JSON array of personas")
            personas = [PersonaSpec.model_validate(item) for item in data]
            missing = [
                p.name
                for p in personas
                if not set(required) <= set(p.ground_truth_slots)
            ]
            if missing:
                raise ValueError(f"personas missing required slots: {missing}")
        except ValueError as exc:
            last_error = str(exc)
            continue
        return personas
    raise ValueError(f"persona generation failed: {last_error}")
```

**Step 4: Run it — verify it passes**

Run: `uv run pytest tests/playbook/test_personas.py -v` — all PASS.

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add src/superdialog/playbook/personas.py tests/playbook/test_personas.py
git commit -am "feat(playbook): persona suites - load, save, generate, derive"
```

---

## Phase 5 — CLI, exports, docs

### Task 8: `superdialog optimize` subcommand + package exports

Mirror the `chat` wiring: a thin `_cmd_optimize` (pre-flight validation, clean
errors) delegating to `_run_optimize` (provider construction, persona resolution,
the async loop) which tests patch — exactly as `test_chat.py` patches
`_run_playbook_repl`. Exports land here (not in the docs task) so the real command
works at this commit.

**Files:**
- Modify: `src/superdialog/cli/main.py`
- Modify: `src/superdialog/playbook/__init__.py`
- Create: `tests/cli/test_optimize.py`

**Step 1: Write the failing test**

`tests/cli/test_optimize.py`:
```python
"""CLI tests for the optimize subcommand (heavy lifting patched out)."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

from superdialog.cli.main import main
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_simple import SIMPLE

_cli = importlib.import_module("superdialog.cli.main")


def _write(tmp_path: Path, text: str = MINIMAL_YAML) -> Path:
    p = tmp_path / "play.yaml"
    p.write_text(text)
    return p


def test_optimize_writes_out_and_prints_trace(tmp_path, capsys) -> None:
    src = _write(tmp_path)
    out = tmp_path / "improved.yaml"
    improved = MINIMAL_YAML.replace("Collect naturally.", "Collect warmly.")
    lines = ["round 1: incumbent 0.40 vs candidate 0.70 - accepted (1 edit)"]
    with patch.object(_cli, "_run_optimize", return_value=(improved, lines)) as m:
        rc = main(["optimize", "--playbook", str(src), "--rounds", "1",
                   "--out", str(out)])
    assert rc == 0
    m.assert_called_once()
    assert out.read_text() == improved
    printed = capsys.readouterr().out
    assert "round 1" in printed and "accepted" in printed
    assert str(out) in printed


def test_optimize_default_out_is_improved_basename(tmp_path) -> None:
    src = _write(tmp_path)
    with patch.object(_cli, "_run_optimize", return_value=("y: 1\n", [])):
        rc = main(["optimize", "--playbook", str(src)])
    assert rc == 0
    assert (tmp_path / "improved.play.yaml").read_text() == "y: 1\n"


def test_optimize_missing_playbook_returns_1(capsys) -> None:
    rc = main(["optimize", "--playbook", "/nope.yaml"])
    assert rc == 1
    assert "/nope.yaml" in capsys.readouterr().err


def test_optimize_invalid_playbook_exits_clean(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("journeys: not-a-dict\n")
    with patch.object(_cli, "_run_optimize") as m:
        rc = main(["optimize", "--playbook", str(bad)])
    assert rc == 1
    m.assert_not_called()
    assert "Invalid playbook" in capsys.readouterr().err


def test_optimize_accepts_simple_format(tmp_path) -> None:
    src = tmp_path / "simple.yaml"
    src.write_text(SIMPLE)
    with patch.object(_cli, "_run_optimize", return_value=("y: 1\n", [])) as m:
        rc = main(["optimize", "--playbook", str(src)])
    assert rc == 0
    m.assert_called_once()


def test_package_exports() -> None:
    import superdialog.playbook as pb

    for name in ("optimize", "OptimizeReport", "ObjectiveBreakdown",
                 "RoundTrace", "FullDoc", "SimpleDoc", "MutationError",
                 "Edit", "make_editable", "load_personas",
                 "generate_personas"):
        assert hasattr(pb, name), name
        assert name in pb.__all__, name
```

**Step 2: Run it — verify it fails**

Run: `uv run pytest tests/cli/test_optimize.py -v`
Expected: FAIL — argparse error: `invalid choice: 'optimize'`.

**Step 3: Implement**

3a. In `src/superdialog/playbook/__init__.py` add (keep `__all__` sorted for ruff):
```python
from .editable import Edit, FullDoc, MutationError, SimpleDoc, make_editable
from .optimize import (
    ObjectiveBreakdown,
    OptimizeReport,
    RoundTrace,
    optimize,
)
from .personas import generate_personas, load_personas
```
and the corresponding names in `__all__`.

3b. In `src/superdialog/cli/main.py` add (follow the file's lazy-import style):
```python
def _run_optimize(
    playbook_path: str,
    *,
    rounds: int,
    n: int,
    personas_path: str | None,
    llm: str,
    candidate_llm: str | None,
    user_llm: str | None,
) -> tuple[str, list[str]]:
    """Run the optimize loop against real providers; return (yaml, trace lines)."""
    import asyncio
    from pathlib import Path as _Path

    from ..llm.resolver import resolve_llm
    from ..playbook import (
        PlaybookAgent,
        httpx_http,
        make_editable,
        optimize,
        provider_adapters,
    )
    from ..playbook.personas import (
        derive_default_persona,
        generate_personas,
        load_personas,
        persona_cache_path,
        save_personas,
    )

    doc = make_editable(_Path(playbook_path).read_text(encoding="utf-8"))
    director, talker = provider_adapters(resolve_llm(llm))
    cand = (provider_adapters(resolve_llm(candidate_llm))[0]
            if candidate_llm else director)
    user = (provider_adapters(resolve_llm(user_llm))[0]
            if user_llm else director)

    def agent_factory(pb):  # type: ignore[no-untyped-def]
        return PlaybookAgent(
            playbook=pb, talker_llm=talker, director_llm=director,
            http=httpx_http,
        )

    notes: list[str] = []

    async def _go():  # type: ignore[no-untyped-def]
        playbook = doc.compile()
        cache = persona_cache_path(playbook_path)
        if personas_path:
            personas = load_personas(personas_path)
        elif os.path.exists(cache):
            personas = load_personas(cache)
            notes.append(f"personas: loaded cache {cache}")
        else:
            try:
                personas = await generate_personas(playbook, cand)
                save_personas(personas, cache)
                notes.append(f"personas: generated suite -> {cache} (review it)")
            except ValueError as exc:
                personas = [derive_default_persona(playbook)]
                notes.append(f"personas: generation failed ({exc}); "
                             "using one derived persona")
        return await optimize(
            doc, personas=personas, candidate_llm=cand, user_llm=user,
            agent_factory=agent_factory, rounds=rounds, n=n,
        )

    report = asyncio.run(_go())
    lines = list(notes)
    for t in report.trace:
        if t.candidate_breakdown is None:
            lines.append(f"round {t.round_no}: {t.detail}")
        else:
            verdict = (f"accepted ({len(t.edits)} edit"
                       f"{'s' if len(t.edits) != 1 else ''})"
                       if t.accepted else "rejected")
            lines.append(
                f"round {t.round_no}: incumbent "
                f"{t.incumbent_breakdown.objective:.2f} vs candidate "
                f"{t.candidate_breakdown.objective:.2f} - {verdict}"
            )
    lines.append(
        f"objective: {report.initial_breakdown.objective:.2f} -> "
        f"{report.final_breakdown.objective:.2f}"
    )
    return report.final_yaml, lines


def _cmd_optimize(args: argparse.Namespace) -> int:
    """Validate inputs, run the loop, write the improved playbook."""
    path = args.playbook
    if not os.path.exists(path):
        print(f"Playbook not found: {path}", file=sys.stderr)
        return 1
    try:  # pre-flight either format; surface schema errors as one line
        if _looks_like_simple_playbook(path):
            from ..playbook.simple import load_simple

            load_simple(path)
        else:
            from ..playbook import Playbook

            Playbook.load(path)
    except Exception as exc:
        print(f"Invalid playbook {path}: {exc}", file=sys.stderr)
        return 1
    out = args.out or os.path.join(
        os.path.dirname(path) or ".", f"improved.{os.path.basename(path)}"
    )
    final_yaml, lines = _run_optimize(
        path, rounds=args.rounds, n=args.n, personas_path=args.personas,
        llm=args.llm, candidate_llm=args.candidate_llm, user_llm=args.user_llm,
    )
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(final_yaml)
    for line in lines:
        print(line)
    print(f"Wrote {out}")
    return 0
```
(`import os` is already at the top of main.py — verify; add if missing.)

3c. Register the parser in `_build_parser` (after the `flow` block):
```python
    opt = sub.add_parser(
        "optimize", help="Reflectively improve a playbook's prose via self-play"
    )
    opt.add_argument("--playbook", required=True,
                     help="Path to a playbook (full or simple format)")
    opt.add_argument("--rounds", type=int, default=3)
    opt.add_argument("--n", type=int, default=1,
                     help="Eval sessions per persona per side")
    opt.add_argument("--personas", default=None,
                     help="Path to a PersonaSpec list (YAML/JSON)")
    opt.add_argument("--llm", default="openai/gpt-4o-mini")
    opt.add_argument("--candidate-llm", default=None,
                     help="Override the reflecting LLM (default: --llm)")
    opt.add_argument("--user-llm", default=None,
                     help="Override the caller-simulator LLM (default: --llm)")
    opt.add_argument("--out", default=None,
                     help="Output path (default: improved.<name>, same format)")
    opt.set_defaults(fn=_cmd_optimize)
```

**Step 4: Run it — verify it passes**

Run: `uv run pytest tests/cli/test_optimize.py tests/cli/test_chat.py -v` — all PASS.

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add -A src tests
git commit -m "feat(cli): superdialog optimize subcommand with persona resolution"
```

---

### Task 9: Docs

**Files:**
- Modify: `docs/04-playbook-guide.md`

**Step 1: Update the guide**

- Add an **optimize** subsection under §6 documenting: the paired-round loop, the
  prose whitelist (both formats), simple-format round-trip, persona suite
  generation/cache, the CLI invocation
  (`superdialog optimize --playbook X.yaml [--rounds N] [--n K] [--personas p.yaml]
  [--llm M] [--candidate-llm M] [--user-llm M] [--out path]`), and a minimal Python
  example (`make_editable` → `optimize(...)` → `report.final_yaml`). Note the cost
  model: each round ≈ 2 evals × personas × n × ~2 LLM calls/turn + 1 reflect call —
  the most expensive command in the tool.
- In the Roadmap (**§9** — renumbered by the simple-format work; there is no §7):
  move `superdialog optimize` from "Clearly future" to shipped, noting prose-only
  scope and that structure mutation (checkpoint split/merge, schema tightening)
  remains future.

**Step 2: Verify the full suite**

```bash
uv run pytest tests/playbook tests/cli -v
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
```

**Step 3: Commit**

```bash
git add docs/04-playbook-guide.md
git commit -m "docs(playbook): document the optimize command and update roadmap"
```

---

## Honest scope (read before shipping)

- **v1 optimizes PROSE only**, enforced by construction: only whitelisted addresses
  are applicable. Frozen: `expr` whens, dispatch intents, interrupt whens, silence
  prompts, simple-format facts/objections/boundaries/fallbacks, all structure.
- **Acceptance is noise-resistant, not noise-free.** Paired same-round evals remove
  between-round drift; within-round sampling noise remains. Raise `--n` when it
  matters.
- **Reflection quality tracks the candidate LLM.** Worst case the loop returns the
  input unchanged — the accept gate never regresses the artifact.
- **Output fidelity:** key order and authored defaults survive (dict-level edits);
  YAML comments do not (PyYAML). ruamel.yaml round-trip is noted future polish.
- **Cost:** each round ≈ 2 evals × |personas| × n × ~2 LLM calls/turn + 1 reflect
  call; `run_eval` is strictly sequential, and a non-advancing playbook burns full
  `max_turns` per session.

## Explicitly deferred (NOT in this plan)

Structure-stage mutation; frontier picker UI / GEPA-style frontier parent sampling;
production-log feedback ingestion; CI threshold gates; ResponseCache reuse;
latency/hard-gate-wait scoring (no timing in `SessionMetrics`); LLM-judged smoothness
(`SessionAuditor` is machine-substrate-only); replay regression guard (paired evals
gate regressions; `replay` has a spurious-diff caveat on pipeline-failure logs).
