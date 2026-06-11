# `superdialog optimize` — Reflective Prompt Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the deferred `superdialog optimize` command — a GEPA-style reflective
prompt optimizer for the Playbook engine. It runs persona self-play against a playbook,
scores the sessions into a scalar objective, asks a candidate LLM to mutate the playbook's
**prose** (guidance / `advance_when.when` / persona / `never_say`), validates and re-runs
each candidate, and keeps a Pareto frontier of non-dominated improvements. The output is a
better, git-diffable playbook YAML plus a per-round metric trace.

**Architecture:** A pure-ish loop `optimize(...)` on top of the **shipped** eval substrate.
Each round: **RUN** (`run_eval` — persona self-play → `EvalReport` of `SessionMetrics` +
event logs) → **SCORE** (aggregate metrics into a scalar objective + per-dimension
breakdown) → **REFLECT** (feed the worst sessions' event logs + current playbook YAML to a
candidate `CompletesLLM`; it returns a mutated playbook as YAML; validate via
`Playbook.from_yaml`, reject/retry on invalid or structural change) → **VALIDATE** (re-run
`run_eval` on the candidate; accept if the objective improves) → keep a **Pareto frontier**
(smoothness vs completion vs slot-accuracy), not a single winner; stop after `rounds` or on
convergence. v1 mutates **prose only**; structure mutation is the deferred "structure stage".

**Tech Stack:** Python ≥3.10, pydantic v2, the `superdialog.playbook` package
(`eval_bridge`, `models`, `replay`, `director.CompletesLLM`, `providers`), pytest
(`asyncio_mode = "auto"`), uv, ruff, pyrefly.

**Design doc:** `docs/plans/2026-06-10-checkpoint-compound-architecture-design.md` §4 (the
optimize loop) — read it before starting. This plan realizes §4's run→score→reflect→validate
loop with the prose-only v1 scope decided there ("Structure stage — once prompt-level metrics
are stable… the optimizer may propose checkpoint splits/merges/reorders").

**Shipped substrate this builds on (read first; cite real APIs):**
- `src/superdialog/playbook/eval_bridge.py` — `PersonaSpec`, `SessionMetrics`
  (`completed`, `outcome`, `turns`, `turns_per_checkpoint`, `slot_accuracy`, `slot_diffs`,
  `repair_count`, `degraded_count`, `event_log_jsonl`), `EvalReport`
  (`.completion_rate`, `.mean_slot_accuracy`), `SpeaksUser` protocol, `run_session`,
  `run_eval(playbook_factory, personas, user_llm, n)`. **This is the run + score substrate.**
- `src/superdialog/playbook/models.py` — `Playbook` is pydantic; `from_yaml` /
  `model_validate` / `model_dump`; per-`Checkpoint` optimizable prose fields are
  `guidance`, `advance_when[].when`, and playbook-level `persona`, plus per-checkpoint
  `never_say`. Reference validation raises `ValueError` on dangling refs. The artifact
  round-trips to git-diffable YAML.
- `src/superdialog/playbook/agent.py` — `PlaybookAgent(playbook, talker_llm, director_llm, http)`.
- `src/superdialog/playbook/director.py` — `CompletesLLM` protocol
  (`async complete(messages, **kwargs) -> str`); this is the candidate-LLM shape.
- `src/superdialog/playbook/providers.py` — `provider_adapters(provider) -> (Director, Talker)`.
- `src/superdialog/playbook/replay.py` — `replay` / `ReplayReport` (decision diffing): the
  optional A/B regression primitive (Task 6, optional gate).
- `src/superdialog/cli/main.py` — `eval`-style subcommand wiring lives here; `chat` already
  routes to the playbook REPL via `_run_playbook_repl`. Add `optimize` the same way.
- `src/superdialog/llm/resolver.py` — `resolve_llm(model)` builds an `LLMProvider`.
- Test fakes already in the suite: `tests/playbook/test_director.py::CannedLLM`
  (returns a fixed `json.dumps(payload)`), `tests/playbook/test_talker.py::StreamLLM`,
  `tests/playbook/test_toolexec.py::FakeHttp`, `tests/playbook/test_eval_bridge.py::ScriptedUser`.

**Conventions for every task:**
- Branch: `feat/playbook-engine` (continue the existing branch; do not commit to `main`).
- Run `uv run pytest <test file> -v` after each test/impl step.
- Run `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check` before each
  commit; fix what they flag (line length 88; type hints required; explicit None checks).
- `pyproject.toml` sets `asyncio_mode = "auto"` — plain `async def` test functions run with no
  markers. Use the **anyio** library for any concurrency inside source/tests; do **not** add
  `@pytest.mark.anyio` or an `anyio_backend` fixture.
- **No network in tests.** Every test uses scripted fakes: a `CannedLLM`-style candidate that
  returns a known mutated YAML, a `ScriptedUser` persona `user_llm`, `FakeHttp`.
- New source module: `src/superdialog/playbook/optimize.py`. It may import only from within
  `superdialog.playbook` and stdlib/pydantic/yaml — never from `superdialog.machine`.
- If a `scripts/run_tests.sh` exists in the repo, register the new test module there; if it
  does not (it currently does not), skip that step.
- Commit after every green test, conventional style: `feat(playbook): …`.

---

## Phase 1 — Optimizer core (scoring is pure and LLM-free)

### Task 1: Objective scoring — `score_report` (pure, no LLM, no network)

The objective and its per-dimension breakdown are pure functions over a shipped
`EvalReport`. Land them first so every later task asserts against a real number.

**Files:**
- Create: `src/superdialog/playbook/optimize.py`
- Create: `tests/playbook/test_optimize.py`

**Step 1: Write failing test**

`tests/playbook/test_optimize.py`:
```python
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
    assert b.completion_rate == 0.5            # 1 of 2 completed
    assert b.slot_accuracy == 1.0              # both 1.0
    # smoothness proxy = mean turns_per_checkpoint (lower is smoother)
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
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/playbook/test_optimize.py -v`
Expected: FAIL — `ModuleNotFoundError: superdialog.playbook.optimize`.

**Step 3: Implement (this task only)**

Create `src/superdialog/playbook/optimize.py` with the scoring layer:
- `class ObjectiveBreakdown(BaseModel)`: fields `objective: float`, `completion_rate: float`,
  `slot_accuracy: float`, `mean_turns_per_checkpoint: float`, `repair_rate: float`.
- `def score_report(report: EvalReport) -> ObjectiveBreakdown:` — pure, no LLM:
  - `completion_rate = report.completion_rate`; `slot_accuracy = report.mean_slot_accuracy`.
  - `mean_turns_per_checkpoint`: mean over all sessions of
    `mean(session.turns_per_checkpoint.values())` (0.0 when a session has no checkpoints; 0.0
    for an empty report). This is the **smoothness proxy** named in the task spec.
  - `repair_rate`: total `repair_count` / total `turns` across sessions (0.0 if no turns).
  - `objective`: a weighted sum normalized to `[0, 1]`. Define module constants
    `W_COMPLETION`, `W_SLOT`, `W_SMOOTHNESS`, `W_REPAIR` (UPPER_SNAKE_CASE) summing to 1.0.
    Map smoothness to `[0,1]` with `smoothness = 1 / (1 + max(0, mean_turns_per_cp - 1))`
    (1 turn/checkpoint → 1.0; more turns → less). `repair` term is `1 - min(1, repair_rate)`.
    `objective = W_COMPLETION*completion_rate + W_SLOT*slot_accuracy +
    W_SMOOTHNESS*smoothness + W_REPAIR*(1 - min(1, repair_rate))`. Empty report → 0.0.
  - Add docstrings; type hints on everything; keep functions small.

**Step 4: Run to verify pass** — 4 PASS.

**Step 5: Format, typecheck, commit**
```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add src/superdialog/playbook/optimize.py tests/playbook/test_optimize.py
git commit -m "feat(playbook): objective scoring for the optimize loop"
```

---

### Task 2: Mutation safety — structural skeleton + prose-only guard

The reflect step may only edit prose. Enforce it by extracting a **structural skeleton**
from a `Playbook` and rejecting any candidate whose skeleton differs from the original.
This is a pure function — land and test it before any LLM is involved.

**Files:**
- Edit: `src/superdialog/playbook/optimize.py`
- Edit: `tests/playbook/test_optimize.py`

**Step 1: Write failing test**

Append to `tests/playbook/test_optimize.py`:
```python
import pytest

from superdialog.playbook.models import Playbook
from superdialog.playbook.optimize import (
    MutationError, structural_skeleton, validate_prose_only_mutation,
)
from tests.playbook.test_models import MINIMAL_YAML


def test_skeleton_ignores_prose_fields() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    edited = MINIMAL_YAML.replace("Collect naturally.", "Gently collect the details.")
    edited = edited.replace("details complete", "the caller has given everything")
    edited = edited.replace(
        'persona: "You are a booking assistant."', 'persona: "You are warm and helpful."')
    pb2 = Playbook.from_yaml(edited)
    assert structural_skeleton(pb) == structural_skeleton(pb2)  # prose differs, shape same


def test_validate_accepts_prose_only_change() -> None:
    original = Playbook.from_yaml(MINIMAL_YAML)
    candidate = Playbook.from_yaml(
        MINIMAL_YAML.replace("Collect naturally.", "Collect warmly."))
    validate_prose_only_mutation(original, candidate)  # does not raise


def test_validate_rejects_new_checkpoint_id() -> None:
    original = Playbook.from_yaml(MINIMAL_YAML)
    bad = MINIMAL_YAML.replace("- id: close", "- id: renamed_close")
    bad = bad.replace("to: booking.close}", "to: booking.renamed_close}")
    candidate = Playbook.from_yaml(bad)
    with pytest.raises(MutationError):
        validate_prose_only_mutation(original, candidate)


def test_validate_rejects_slot_or_gate_change() -> None:
    original = Playbook.from_yaml(MINIMAL_YAML)
    gate = Playbook.from_yaml(MINIMAL_YAML.replace("gate: hard", "gate: soft"))
    with pytest.raises(MutationError):
        validate_prose_only_mutation(original, gate)
    slots = Playbook.from_yaml(
        MINIMAL_YAML.replace("city: {type: str, required: true, invalidates: [course_id]}",
                             "city: {type: str, required: false}"))
    with pytest.raises(MutationError):
        validate_prose_only_mutation(original, slots)


def test_validate_rejects_changed_advance_target() -> None:
    original = Playbook.from_yaml(MINIMAL_YAML)
    # re-pointing a rule's `to` is structure, not prose
    bad = MINIMAL_YAML.replace("to: booking.confirm,", "to: booking.close,")
    candidate = Playbook.from_yaml(bad)
    with pytest.raises(MutationError):
        validate_prose_only_mutation(original, candidate)
```

**Step 2: Run to verify failure** — FAIL (names missing).

**Step 3: Implement**

In `optimize.py`:
- `class MutationError(ValueError): ...`
- `def structural_skeleton(pb: Playbook) -> dict:` — return a canonical dict capturing
  **everything except prose**. Build it from `pb.model_dump()`, then strip the editable prose
  fields so they don't affect equality:
  - playbook-level: drop `persona`.
  - per checkpoint (each journey, in order): keep `id`, `gate`, `auto`, `terminal`,
    `outcome`, `pipeline`, `on_enter`, `on_failure`, `turn_budget`, the **slots** map (full
    `SlotSpec` dump — type/required/values/authoritative/invalidates is structure), and for
    each `advance_when` rule keep `judge`, `to`, `requires`, `set` but **drop** `when`. Drop
    `guidance`, `say_verbatim`, `never_say` (prose). (`never_say` and `say_verbatim` are
    editable prose in v1.)
  - keep `dispatch`, `tools`, `pipelines`, `handlers`, `interrupts` (drop interrupt `when`
    prose but keep `id`/`judge`/`to`/`resume`), `policies`, `middleware`, `env`, `views`,
    `initial` unchanged — all structure.
  - Return a plain nested dict with deterministic ordering (sort dict keys) so two skeletons
    compare by value.
- `def validate_prose_only_mutation(original: Playbook, candidate: Playbook) -> None:` —
  raise `MutationError` with a precise message when
  `structural_skeleton(original) != structural_skeleton(candidate)`. Keep the message short
  and human-readable (e.g. first differing top-level key / checkpoint id).

**Step 4: Run to verify pass** — all PASS.

**Step 5: Commit**
```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git commit -am "feat(playbook): prose-only mutation guard via structural skeleton"
```

---

### Task 3: The reflect step — candidate proposal + load/validate/retry

Wrap the candidate `CompletesLLM` call: build a reflect prompt from the failing traces + the
current playbook YAML, parse the returned YAML, and validate it (loads as a `Playbook` AND
passes the prose-only guard). On any failure, retry up to `max_attempts`; if all attempts
fail, return `None` (the round falls back to the incumbent).

**Files:**
- Edit: `src/superdialog/playbook/optimize.py`
- Edit: `tests/playbook/test_optimize.py`

**Step 1: Write failing test**

Append to `tests/playbook/test_optimize.py`:
```python
from superdialog.playbook.eval_bridge import EvalReport, SessionMetrics
from superdialog.playbook.optimize import propose_mutation


class CannedYamlLLM:
    """Candidate LLM: returns scripted YAML strings, recording prompts seen."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


def _report(**kw) -> EvalReport:
    base = dict(
        persona="p", completed=False, outcome=None, turns=6,
        turns_per_checkpoint={"booking.collect": 6}, slot_accuracy=0.0,
        slot_diffs={"city": ("Pune", None)}, repair_count=2, degraded_count=0,
        event_log_jsonl='{"type":"utterance","version":1,"role":"user","text":"uh"}',
    )
    base.update(kw)
    return EvalReport(sessions=[SessionMetrics(**base)])


async def test_propose_returns_validated_prose_mutation() -> None:
    original = Playbook.from_yaml(MINIMAL_YAML)
    good = MINIMAL_YAML.replace("Collect naturally.", "Ask for the city first, warmly.")
    llm = CannedYamlLLM([good])
    candidate = await propose_mutation(original, _report(), llm, max_attempts=3)
    assert candidate is not None
    assert candidate.checkpoint("booking.collect").guidance == "Ask for the city first, warmly."
    # the failing trace and current YAML were shown to the candidate
    prompt = " ".join(m["content"] for m in llm.calls[0])
    assert "Collect naturally." in prompt and "city" in prompt


async def test_invalid_yaml_retries_then_falls_back() -> None:
    original = Playbook.from_yaml(MINIMAL_YAML)
    llm = CannedYamlLLM(["this is not: yaml: : :", "journeys: not-a-dict"])
    candidate = await propose_mutation(original, _report(), llm, max_attempts=2)
    assert candidate is None
    assert len(llm.calls) == 2  # retried


async def test_structural_mutation_rejected_and_falls_back() -> None:
    original = Playbook.from_yaml(MINIMAL_YAML)
    structural = MINIMAL_YAML.replace("- id: close", "- id: closed_out").replace(
        "to: booking.close}", "to: booking.closed_out}")
    llm = CannedYamlLLM([structural])
    candidate = await propose_mutation(original, _report(), llm, max_attempts=1)
    assert candidate is None  # structural change rejected by the guard
```

**Step 2: Run to verify failure** — FAIL.

**Step 3: Implement**

In `optimize.py`:
- A module constant `_REFLECT_RULES` (string): the rules the candidate must obey — "edit ONLY
  prose: per-checkpoint `guidance`, `advance_when[].when`, `never_say`, `say_verbatim`, and the
  top-level `persona`. Do NOT add/remove/rename checkpoints, journeys, slots, tools, pipelines,
  interrupts, or change `gate`/`judge`/`to`/`requires`. Return the COMPLETE playbook as YAML,
  nothing else." Treat the candidate output as untrusted; never execute it.
- `def _worst_sessions(report: EvalReport, k: int = 3) -> list[SessionMetrics]:` — sort
  sessions by `(completed, slot_accuracy, -repair_count)` ascending and take the worst `k`.
- `def _reflect_prompt(playbook: Playbook, report: EvalReport, k: int = 3) -> list[dict[str, str]]:`
  — a system message with `_REFLECT_RULES`; a user message containing the current
  `yaml.safe_dump(playbook.model_dump(exclude_defaults=True))` and, for each worst session, its
  `slot_diffs`, `repair_count`, `turns_per_checkpoint`, and a trimmed `event_log_jsonl`
  (cap length so a runaway log can't blow the context).
- `async def propose_mutation(playbook, report, candidate_llm: CompletesLLM, *, max_attempts=3) -> Playbook | None:`
  - Loop up to `max_attempts`: `raw = await candidate_llm.complete(_reflect_prompt(...))`;
    strip Markdown fences if present; `try: cand = Playbook.from_yaml(raw)` — on
    `yaml.YAMLError`/`ValidationError`/`ValueError` continue to next attempt; then
    `try: validate_prose_only_mutation(playbook, cand)` — on `MutationError` continue.
    On success `return cand`.
  - All attempts exhausted → `return None`. Import `CompletesLLM` from `.director`.

**Step 4: Run to verify pass** — all PASS.

**Step 5: Commit** — `git commit -am "feat(playbook): reflect step — candidate proposal with validate/retry"`

---

### Task 4: Pareto frontier — keep non-dominated candidates

The loop keeps a Pareto frontier across three dimensions (smoothness, completion,
slot-accuracy), not a single winner. Land the frontier data structure as a pure helper.

**Files:**
- Edit: `src/superdialog/playbook/optimize.py`
- Edit: `tests/playbook/test_optimize.py`

**Step 1: Write failing test**

Append to `tests/playbook/test_optimize.py`:
```python
from superdialog.playbook.optimize import ParetoFrontier, RoundTrace


def _trace(round_no, completion, slot, smoothness, objective) -> RoundTrace:
    return RoundTrace(
        round=round_no, accepted=True,
        breakdown=ObjectiveBreakdown(
            objective=objective, completion_rate=completion, slot_accuracy=slot,
            mean_turns_per_checkpoint=1.0 / smoothness, repair_rate=0.0),
        playbook_yaml="persona: x\njourneys: {}\n",
    )


def test_frontier_keeps_non_dominated() -> None:
    f = ParetoFrontier()
    f.consider(_trace(1, completion=0.9, slot=0.5, smoothness=0.5, objective=0.6))
    f.consider(_trace(2, completion=0.5, slot=0.9, smoothness=0.5, objective=0.6))  # trades off
    f.consider(_trace(3, completion=0.4, slot=0.4, smoothness=0.4, objective=0.4))  # dominated
    rounds = sorted(t.round for t in f.members)
    assert rounds == [1, 2]  # #3 dominated by #1 and #2; both #1 and #2 kept


def test_frontier_drops_a_newly_dominated_member() -> None:
    f = ParetoFrontier()
    f.consider(_trace(1, completion=0.6, slot=0.6, smoothness=0.6, objective=0.6))
    f.consider(_trace(2, completion=0.9, slot=0.9, smoothness=0.9, objective=0.9))  # dominates #1
    assert [t.round for t in f.members] == [2]


def test_best_is_max_objective() -> None:
    f = ParetoFrontier()
    f.consider(_trace(1, 0.9, 0.5, 0.5, objective=0.6))
    f.consider(_trace(2, 0.5, 0.9, 0.9, objective=0.8))
    assert f.best().round == 2
```

**Step 2: Run to verify failure** — FAIL.

**Step 3: Implement**

In `optimize.py`:
- `class RoundTrace(BaseModel)`: `round: int`, `accepted: bool`, `breakdown: ObjectiveBreakdown`,
  `playbook_yaml: str`, optional `detail: str = ""` (e.g. "fallback: no valid candidate").
- `class ParetoFrontier(BaseModel)`: `members: list[RoundTrace] = Field(default_factory=list)`.
  - `def _vector(self, t: RoundTrace) -> tuple[float, float, float]:` — `(completion_rate,
    slot_accuracy, smoothness)` where `smoothness = 1/(1+max(0, mean_turns_per_checkpoint-1))`
    (reuse the same mapping as `score_report`; factor it into a module helper `_smoothness`).
  - `def _dominates(self, a, b) -> bool:` — `a` dominates `b` iff `a` is `>=` on every
    dimension and `>` on at least one.
  - `def consider(self, t: RoundTrace) -> None:` — drop any existing member dominated by `t`;
    add `t` unless an existing member dominates it.
  - `def best(self) -> RoundTrace:` — the member with max `breakdown.objective` (raises if
    empty).

**Step 4: Run to verify pass** — all PASS.

**Step 5: Commit** — `git commit -am "feat(playbook): Pareto frontier over completion/slot/smoothness"`

---

### Task 5: `optimize` — the full loop (round cap, convergence, accept-on-improve)

Compose Tasks 1–4 into the driver. Each round runs the eval, scores it, proposes a mutation,
re-evaluates the candidate, accepts on objective improvement, records a `RoundTrace`, and
updates the frontier. Stop after `rounds` or on convergence (no acceptance for `patience`
rounds). All LLMs/HTTP are injected — fully offline in tests.

**Files:**
- Edit: `src/superdialog/playbook/optimize.py`
- Edit: `tests/playbook/test_optimize.py`

**Step 1: Write failing test**

Append to `tests/playbook/test_optimize.py`:
```python
from superdialog.playbook.agent import PlaybookAgent
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


def _improving_agent_factory(playbook):
    """After mutation the Director starts completing; before it idles."""
    improved = "warmly" in playbook.checkpoint("booking.collect").guidance
    return PlaybookAgent(
        playbook=playbook,
        talker_llm=StreamLLM(["Which", " city?"]),
        director_llm=CannedLLM(_ADVANCE if improved else _IDLE),
        http=FakeHttp([_HOLD_OK] if improved else []),
    )


async def test_optimize_improves_and_returns_best() -> None:
    base = Playbook.from_yaml(MINIMAL_YAML)
    good = MINIMAL_YAML.replace("Collect naturally.", "Collect warmly.")
    report = await optimize(
        base, personas=_PERSONAS,
        candidate_llm=CannedYamlLLM([good]),
        user_llm=ScriptedUser(["Pune on 2026-06-12 please", "ok"]),
        agent_factory=_improving_agent_factory, rounds=2, n=1)
    assert isinstance(report, OptimizeReport)
    assert "warmly" in report.best_playbook.checkpoint("booking.collect").guidance
    assert report.best_breakdown.objective > report.initial_breakdown.objective
    assert len(report.trace) >= 1
    assert any(t.accepted for t in report.trace)


async def test_invalid_candidate_keeps_incumbent() -> None:
    base = Playbook.from_yaml(MINIMAL_YAML)
    report = await optimize(
        base, personas=_PERSONAS,
        candidate_llm=CannedYamlLLM(["not: valid: yaml: :"]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory, rounds=1, n=1)
    assert report.best_playbook.model_dump() == base.model_dump()  # unchanged
    assert report.trace[0].accepted is False


async def test_round_cap_respected() -> None:
    base = Playbook.from_yaml(MINIMAL_YAML)
    # a no-op mutation never improves the objective -> never accepted
    noop = MINIMAL_YAML  # identical prose
    report = await optimize(
        base, personas=_PERSONAS,
        candidate_llm=CannedYamlLLM([noop, noop, noop]),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory, rounds=3, n=1, patience=99)
    assert len(report.trace) == 3
    assert all(not t.accepted for t in report.trace)


async def test_convergence_stops_early() -> None:
    base = Playbook.from_yaml(MINIMAL_YAML)
    report = await optimize(
        base, personas=_PERSONAS,
        candidate_llm=CannedYamlLLM([MINIMAL_YAML] * 5),
        user_llm=ScriptedUser(["x"]),
        agent_factory=_improving_agent_factory, rounds=5, n=1, patience=1)
    assert len(report.trace) < 5  # stopped after `patience` non-improving rounds
```

**Step 2: Run to verify failure** — FAIL.

**Step 3: Implement**

In `optimize.py`:
- `AgentFactory = Callable[[Playbook], PlaybookAgent]` — production builds one with
  `provider_adapters`; tests inject a fake. (The eval substrate's `run_eval` wants a
  zero-arg `() -> PlaybookAgent`; wrap with `lambda: agent_factory(playbook)`.)
- `class OptimizeReport(BaseModel)`: `best_playbook: Playbook`, `best_breakdown: ObjectiveBreakdown`,
  `initial_breakdown: ObjectiveBreakdown`, `trace: list[RoundTrace]`,
  `frontier: list[RoundTrace]`. Add `def best_yaml(self) -> str:` →
  `yaml.safe_dump(self.best_playbook.model_dump(exclude_defaults=True), sort_keys=False)`.
- ```python
  async def optimize(
      playbook: Playbook,
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
  ```
  Logic:
  1. `async def _eval(pb): return await run_eval(lambda: agent_factory(pb), personas, user_llm, n)`.
  2. Initial: `incumbent = playbook`; `incumbent_b = score_report(await _eval(incumbent))`;
     `initial_b = incumbent_b`. Seed the frontier with the incumbent round-0 trace.
  3. For `round_no in range(1, rounds + 1)`:
     a. `cand = await propose_mutation(incumbent, last_report, candidate_llm, max_attempts=reflect_attempts)`.
        (Keep `last_report` = the most recent incumbent `EvalReport`.)
     b. If `cand is None`: append a non-accepted fallback `RoundTrace`; `stale += 1`.
     c. Else re-evaluate: `cand_b = score_report(await _eval(cand))`. Append a `RoundTrace`
        (`accepted = cand_b.objective > incumbent_b.objective`), feed it to the frontier.
        If accepted: `incumbent, incumbent_b, last_report = cand, cand_b, <cand report>`;
        `stale = 0`. Else `stale += 1`.
     d. If `stale >= patience`: break (convergence).
  4. `best = frontier.best()` (frontier always contains at least the round-0 incumbent, so
     `best_playbook` is never worse than the input). Build `OptimizeReport` from
     `incumbent`-or-`best` (pick `frontier.best()` for `best_playbook`/`best_breakdown`),
     `initial_b`, the trace list, and `frontier.members`.
- Imports: `from .eval_bridge import EvalReport, PersonaSpec, SpeaksUser, run_eval`,
  `from .agent import PlaybookAgent`, `from .director import CompletesLLM`,
  `from .models import Playbook`, `from typing import Callable`.

**Step 4: Run to verify pass** — all PASS.

**Step 5: Commit** — `git commit -am "feat(playbook): optimize() reflective loop with frontier and convergence"`

---

## Phase 2 — Optional A/B regression gate

### Task 6 (optional): Replay-based regression check on accepted candidates

A defensive gate: before accepting a candidate, optionally replay the incumbent's recorded
logs against it (`replay`) and reject if it would destabilize already-correct decisions.
Wire it as an **opt-in** flag so the core loop stays simple. This task is optional — skip if
time-boxed; the core loop is complete without it.

**Files:**
- Edit: `src/superdialog/playbook/optimize.py`
- Edit: `tests/playbook/test_optimize.py`

**Step 1: Write failing test**
```python
from superdialog.playbook.events import EventLog


async def test_regression_guard_rejects_destabilizing_candidate() -> None:
    # Build a recorded log whose Director decisions the candidate LLM would diverge from;
    # with guard_regressions=True the candidate is not accepted even if its objective rises.
    ...  # assert the trace records detail="rejected: replay unstable"
```
(Construct the log from `EventLog`/`UtteranceEvent`/`AdvanceEvent` as in
`tests/playbook/test_replay.py`; use a `CannedLLM` whose verdict diverges.)

**Step 2–4:** Add a `guard_regressions: bool = False` param to `optimize`. When set and a
candidate would otherwise be accepted, run `replay(log, cand, director_llm)` over each worst
session's `EventLog.from_jsonl(session.event_log_jsonl)` and reject (mark not accepted,
`detail="rejected: replay unstable"`) if any `ReplayReport.stable` is False. Note the shipped
caveat in `replay.py`: pipeline-failure logs can report spurious diffs, so the guard is
opt-in only.

**Step 5: Commit** — `git commit -am "feat(playbook): optional replay regression guard in optimize"`

---

## Phase 3 — CLI, exports, docs

### Task 7: `superdialog optimize` subcommand

Mirror the `chat`/`flow generate` wiring in `src/superdialog/cli/main.py`: a thin
`_cmd_optimize` that loads the playbook, builds provider-backed LLMs, runs `optimize`, writes
the improved YAML, and prints the per-round metric trace. The heavy lifting (`optimize`,
provider construction) is factored into a `_run_optimize(...)` helper patched by tests exactly
as `test_chat.py` patches `_run_playbook_repl`.

**Files:**
- Edit: `src/superdialog/cli/main.py`
- Edit: `tests/cli/test_chat.py` (or a new `tests/cli/test_optimize.py`; prefer the latter to
  keep files focused)

**Step 1: Write failing test**

`tests/cli/test_optimize.py`:
```python
from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

from superdialog.cli.main import main

_cli = importlib.import_module("superdialog.cli.main")

_PLAYBOOK = """\
persona: "You are a tiny demo agent."
journeys:
  demo:
    checkpoints:
      - id: collect
        goal: "Have a name"
        guidance: "Ask for the caller's name."
        advance_when:
          - {when: "name given", judge: llm, to: demo.done, requires: []}
      - id: done
        terminal: true
        outcome: finished
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "play.yaml"
    p.write_text(_PLAYBOOK)
    return p


def test_optimize_writes_out_and_prints_trace(tmp_path, capsys) -> None:
    src = _write(tmp_path)
    out = tmp_path / "improved.yaml"
    improved_yaml = _PLAYBOOK.replace("Ask for the caller's name.", "Warmly ask their name.")

    def fake_run(playbook_path, rounds, personas_path, llm, out_path):
        # mimic _run_optimize: returns (best_yaml, printable_trace_lines)
        return improved_yaml, ["round 1: objective 0.40 -> 0.70 (accepted)"]

    with patch.object(_cli, "_run_optimize", side_effect=fake_run) as m:
        rc = main(["optimize", "--playbook", str(src), "--rounds", "1",
                   "--out", str(out)])
    assert rc == 0
    m.assert_called_once()
    assert out.read_text() == improved_yaml          # improved playbook written
    printed = capsys.readouterr().out
    assert "round 1" in printed and "accepted" in printed


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
```

**Step 2: Run to verify failure** — FAIL (`optimize` subcommand unknown).

**Step 3: Implement** in `src/superdialog/cli/main.py`:
- `_run_optimize(playbook_path, rounds, personas_path, llm, out_path) -> tuple[str, list[str]]`:
  - `from ..llm.resolver import resolve_llm`; `from ..playbook import (Playbook,
    PlaybookAgent, httpx_http, provider_adapters, optimize)` and `PersonaSpec`.
  - Build `director, talker = provider_adapters(resolve_llm(llm))`; `candidate_llm = director`
    (the candidate uses the Director adapter — a `CompletesLLM`); `user_llm = director`
    (persona self-play uses the same provider; `SpeaksUser` is satisfied by the Director
    adapter's `complete`).
  - `agent_factory = lambda pb: PlaybookAgent(pb, talker_llm=talker, director_llm=director,
    http=httpx_http)`.
  - Load personas from `personas_path` (a JSON/YAML list of `PersonaSpec`) when given, else a
    small built-in default persona derived from the playbook's first journey goal.
  - `report = asyncio.run(optimize(Playbook.load(playbook_path), personas=personas,
    candidate_llm=candidate_llm, user_llm=user_llm, agent_factory=agent_factory,
    rounds=rounds))`.
  - Return `(report.best_yaml(), [<one line per RoundTrace: round, objective delta,
    accepted>])`.
- `_cmd_optimize(args)`: validate the `--playbook` exists (missing → print to stderr, return 1)
  and pre-flight `Playbook.load` (invalid → "Invalid playbook …", return 1) exactly like
  `_chat_playbook`. Then call `_run_optimize(...)`, write the returned YAML to `--out` (default
  `improved.<basename>`), print the trace lines to stdout, return 0.
- Register the parser in `_build_parser`:
  ```python
  opt = sub.add_parser("optimize", help="Reflectively improve a playbook's prose")
  opt.add_argument("--playbook", required=True, help="Path to the playbook YAML/JSON")
  opt.add_argument("--rounds", type=int, default=3)
  opt.add_argument("--personas", default=None, help="Path to a PersonaSpec list (JSON/YAML)")
  opt.add_argument("--llm", default="openai/gpt-4o-mini")
  opt.add_argument("--out", default=None, help="Output path (default: improved.<name>)")
  opt.set_defaults(fn=_cmd_optimize)
  ```

**Step 4: Run to verify pass** — `uv run pytest tests/cli/test_optimize.py -v` — all PASS.

**Step 5: Commit** — `git commit -am "feat(cli): superdialog optimize subcommand"`

---

### Task 8: Exports + docs

**Files:**
- Edit: `src/superdialog/playbook/__init__.py`
- Edit: `docs/04-playbook-guide.md`

**Step 1: Write failing test**

Append to `tests/playbook/test_optimize.py`:
```python
def test_optimize_exported_from_package() -> None:
    import superdialog.playbook as pb
    assert hasattr(pb, "optimize") and hasattr(pb, "OptimizeReport")
    assert "optimize" in pb.__all__ and "OptimizeReport" in pb.__all__
```

**Step 2: Run to verify failure** — FAIL.

**Step 3: Implement**
- In `src/superdialog/playbook/__init__.py`: add
  `from .optimize import ObjectiveBreakdown, OptimizeReport, RoundTrace, optimize` and add
  `"ObjectiveBreakdown"`, `"OptimizeReport"`, `"RoundTrace"`, `"optimize"` to `__all__`
  (keep the list sorted to satisfy ruff).
- In `docs/04-playbook-guide.md`:
  - Add an **optimize** subsection to **§6** documenting the loop, the prose-only scope, the
    `superdialog optimize --playbook X.yaml [--rounds N] [--personas path] [--llm model]
    [--out improved.yaml]` invocation, and a minimal Python example
    (`report = await optimize(pb, personas=[...], candidate_llm=..., user_llm=...,
    agent_factory=...)` then `report.best_yaml()`). Note the cost model (each round ≈
    N personas × turns × 2 LLM calls + 1 reflect call) and that large/ultracode budgets suit it.
  - In **§7 Roadmap**: move `superdialog optimize` from "Clearly future" to a shipped note —
    "Shipped: `superdialog optimize` runs a reflective prose optimizer (guidance /
    advance_when / persona / never_say); **structure mutation** (split/merge checkpoints,
    schema tightening) remains future (the §4 structure stage)."

**Step 4: Run to verify pass** — PASS.

**Step 5: Format, typecheck, full suite, commit**
```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
uv run pytest tests/playbook tests/cli -v
git commit -am "feat(playbook): export optimize/OptimizeReport and document the command"
```

---

## Honest scope (read before shipping)

- **v1 optimizes PROSE only** — per-checkpoint `guidance`, `advance_when[].when`, `never_say`,
  `say_verbatim`, and the top-level `persona`. The prose-only guard (Task 2) is the hard
  boundary; any structural change is rejected.
- **Reflection quality depends on the candidate LLM.** The loop is correct and offline-testable
  regardless, but real gains track the reflecting model's strength. The accept-on-improve gate
  means a weak candidate never regresses the artifact — worst case it returns the input.
- **Cost.** One round ≈ (incumbent eval + candidate eval) × N personas × turns × 2 LLM calls
  (Talker + Director per turn) **+ 1 reflect call**. This is the most expensive command in the
  tool; large/ultracode budgets suit it. `n=1` and a 1–3 persona set keep dev runs cheap.

---

## Explicitly deferred (NOT in this plan)

- **Structure-stage mutation** — split/merge/reorder checkpoints, slot-schema tightening,
  pipeline/tool edits. v1's guard rejects all of it; this is the §4 "structure stage", gated on
  prose-level metrics being stable and trusted.
- **Multi-objective auto-selection UI** — the loop returns a Pareto frontier; choosing among
  non-dominated artifacts (a picker / interactive review of the frontier) is out of scope. v1
  writes `frontier.best()` (max objective) and exposes the full frontier in `OptimizeReport`.
- **Production-log feedback ingestion** — folding opt-in recorded production event logs into the
  optimization corpus as regression cases (design doc §4 closing paragraph).
- **CI threshold gates** — wiring `objective`/`completion_rate` floors into CI as regression
  gates (design doc §5 "Simulation in CI"). The scoring is reusable; the gate is not built here.
- **ResponseCache reuse** — the design doc's "ResponseCache keeps iterations cheap" optimization;
  v1 re-runs evals each round without caching.
