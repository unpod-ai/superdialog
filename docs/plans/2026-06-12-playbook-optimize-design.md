# `superdialog optimize` — Validated Design

Status: validated 2026-06-12 (brainstorm stress-test of
`2026-06-12-playbook-optimize-command.md`). This design supersedes that plan's
Tasks 2–3 and amends its loop, CLI, and output semantics. The plan doc should
be revised to match before execution (deltas listed in §8).

## 1. Goal and audiences

Reflectively improve a playbook's prose by running persona self-play, scoring
the sessions, asking a candidate LLM for targeted prose edits, and keeping only
edits that win a paired evaluation. Output: an improved playbook **in the
source format**, plus a per-round metric trace.

Designed local-first (author iterates, reviews the git diff, commits), without
painting CI runs or external adopters into a corner: personas are a reviewable
committed artifact, LLM roles are separable flags, and acceptance is
noise-resistant by construction.

## 2. The loop (paired rounds)

```
Round 0:  eval(incumbent) → baseline breakdown; seeds reflection input
Round r:  REFLECT   worst-k sessions' evidence + source YAML
                    → candidate LLM returns JSON list of targeted edits
          APPLY     edits → new doc → compile → validated candidate Playbook
                    (failure: retry up to reflect_attempts, else no-op round)
          PAIR-EVAL run_eval(incumbent) AND run_eval(candidate), fresh, same
                    personas, same n — both face this round's noise
          ACCEPT    iff candidate.objective > incumbent.objective
                    (same-round scores only; cross-round never compared)
          accepted → incumbent ← candidate, stale ← 0; else stale += 1
Stop:     after `rounds` (default 3) or stale ≥ patience (default 2)
Output:   final incumbent, emitted in the source format
```

The Pareto frontier (completion / slot-accuracy / smoothness) is maintained and
reported — `OptimizeReport.frontier` and trace lines show which rounds traded
off what — but it is **informational only**. The written artifact is the last
accepted incumbent: the only playbook whose every step was a same-round paired
win.

Cost: each round ≈ 2 evals × |personas| × n sessions × ~2 LLM calls/turn,
plus 1 reflect call. Round 0 adds one eval. Note `run_eval` is strictly
sequential and a non-advancing playbook burns the full `max_turns` per session.

## 3. Mutation pipeline: `EditableDoc` and targeted edits

The reflector never returns YAML. It returns a JSON list of
`{address, new_text}` edits against an enumerated whitelist, applied by code:

```python
class EditableDoc(Protocol):
    def fields(self) -> list[FieldRef]      # address + current text — the whitelist
    def apply(self, edits) -> "EditableDoc" # new doc; MutationError on bad address
    def compile(self) -> Playbook           # identity / simple_to_playbook
    def emit(self) -> str                   # YAML in the source format
```

Prose-only is enforced **by construction** — no structural-skeleton diff, no
edge-case rulings, no whole-file re-emission. Edits are applied to the parsed
source dict (not the pydantic model), so the author's key order and
explicitly-authored defaults survive; diffs touch only edited values. PyYAML
still drops comments; ruamel.yaml round-trip is optional later polish, not v1.

**FullDoc whitelist** (full-format YAML): top-level `persona`; per checkpoint
`guidance`, `goal`, `never_say` (entries editable or addable — constraints only
tighten), `say_verbatim` **only where already present** (adding one would
bypass the Talker LLM — behavior, not prose), `slots.<name>.description`, and
`advance_when[i].when` **only where `judge == "llm"`**. Frozen: `expr` whens
(evaluated code), dispatch intents, interrupt whens, silence prompts, and all
structure.

**SimpleDoc whitelist** (simple format): per step `say`, `done_when`,
`purpose`; top-level `opening`, `closing`, `persona.identity`,
`persona.voice_style`. `facts`, `objections`, `boundaries`, and
`fallback_actions` are **not** whitelisted; each round recompiles via
`simple_to_playbook`, so the folded persona's reference facts and hard
boundaries are preserved by construction. Output stays simple-format — the
author's surface is never destroyed.

Validation after apply, before any eval spend:
1. Recompile through `Playbook.model_validate` / `simple_to_playbook` —
   catching both `ValidationError` and `yaml.YAMLError` (distinct hierarchies).
2. Jinja `parse()` syntax check on edited `guidance` / `say_verbatim` — broken
   templates pass model validation and only fail at eval time.

Reflect-prompt rules additionally forbid altering factual claims, prices, or
boundary statements inside a full-format `persona` (eval is the backstop), and
require true/false booleans (the custom YAML loader treats yes/no/on/off as
strings).

## 4. Personas

New `load_personas(path) -> list[PersonaSpec]` (YAML/JSON list of `PersonaSpec`
fields). Resolution order:

1. `--personas path` → load it; never generate.
2. Cache exists (`<playbook stem>.personas.yaml` beside the playbook) → load.
3. Otherwise **generate once**: one LLM call sees the compiled playbook's
   journeys, checkpoint goals, and slot schema and returns 4 personas along
   fixed diversity axes — cooperative, terse/impatient, tangent-prone,
   error-making (wrong slot value, then corrects) — each with concrete
   `ground_truth_slots` covering the slot schema (that is what
   `slot_accuracy` scores against). Validate, write the cache, print its
   location with a review/edit nudge.

Generation failure falls back to a single persona derived from the **initial
checkpoint's** `goal` (Journey has no goal field), with a printed warning. The
suite is a reviewable, committable artifact — what CI needs later.

## 5. Scoring and acceptance

`score_report(report) -> ObjectiveBreakdown`, pure and LLM-free, as in the
original plan, with one amendment: `mean_turns_per_checkpoint` is computed over
**completed sessions only** (no completed sessions → smoothness term 0), so
fail-fast sessions cannot game the smoothness mean — incomplete sessions are
penalized by the completion term alone.

```
smoothness = 1 / (1 + max(0, mean_turns_per_cp − 1))
objective  = 0.4·completion_rate + 0.3·slot_accuracy
           + 0.2·smoothness + 0.1·(1 − min(1, repair_rate))
```

Weights are hardcoded module constants in v1. Acceptance is strict
`candidate > incumbent` on same-round paired scores — no epsilon (pairing
already removes the between-round noise an epsilon would compensate for).

`RoundTrace` (field `round_no`, not `round`) records both same-round
breakdowns, the accepted flag, the applied edit list (it *is* the diff), and a
`detail` string for fallback rounds. The replay regression guard (plan Task 6)
is cut: paired evals gate regressions, and `replay` has a known spurious-diff
caveat on pipeline-failure logs.

## 6. CLI

```
superdialog optimize --playbook X.yaml [--rounds 3] [--n 1] [--personas p.yaml]
                     [--llm openai/gpt-4o-mini] [--candidate-llm M] [--user-llm M]
                     [--out path]
```

- Format auto-detect mirrors `chat` (top-level `playbook` list → SimpleDoc).
  `--out` defaults to `improved.<basename>` in the **same format as input**.
- One `--llm` plays all roles by default; `--candidate-llm` / `--user-llm`
  override the reflector and caller-simulator separately (the single-model
  default lets the reflector optimize for what its own twin judges — the flags
  make before/after comparisons clean when it matters). Wiring:
  `resolve_llm(uri)` → `provider_adapters` → director adapter (a
  `CompletesLLM`, which also satisfies `SpeaksUser`); agent factory closes over
  `(talker, director, httpx_http)`.
- Prints per-round trace lines (both objectives, accepted/why-not), a
  baseline→final summary, frontier trade-off notes, and the persona cache path.
- Errors mirror `chat`: missing file → stderr, rc 1; invalid playbook →
  "Invalid playbook …", rc 1. The heavy lifting lives in a `_run_optimize`
  helper that tests patch.

## 7. Testing (fully offline)

Reuse verified fakes (`ScriptedUser`, `StreamLLM`, `FakeHttp`, `CannedLLM`);
add `CannedEditsLLM` returning scripted edit-JSON. Cover: `EditableDoc` both
formats (whitelist enforcement, expr-when rejection, say_verbatim-add
rejection, emit fidelity — only edited lines differ); `score_report` edges
(empty report, completed-only smoothness); paired-loop accept/patience/
fallback; persona loader + generation fallback; CLI via patched
`_run_optimize`. `asyncio_mode = "auto"`; anyio for concurrency; no network.

## 8. Plan deltas (revise the implementation plan to match)

- **Replace Tasks 2–3** (structural skeleton + whole-YAML reflect) with
  `EditableDoc` + targeted edits. This also retires the two broken plan tests
  found in review (incomplete checkpoint renames left dangling
  interrupt/silence refs, so `from_yaml` raised before the guard ran).
- `run_eval`'s first parameter is an **agent** factory
  (`Callable[[], PlaybookAgent]`), not a playbook factory.
- Export `optimize` from `superdialog.playbook` in the CLI task, not the docs
  task (the plan's ordering leaves the real command broken at the Task 7
  commit).
- Docs guide Roadmap is **§9** (simple-format work renumbered it); there is no
  `eval` subcommand precedent in `cli/main.py` — `optimize` is the first CLI
  surface over the eval substrate.
- Plan Task 5's test snippet is missing the `PersonaSpec` import; fix when
  revising.

## 9. Ratified divergences from the architecture design doc (§4)

Deliberate, not drift: latency / hard-gate wait times unscored (no timing in
`SessionMetrics`); LLM-judged smoothness unported (`SessionAuditor` is
machine-substrate-only) — the turns-per-checkpoint proxy stands in; the
frontier is a record-keeper, not a GEPA-style mutation-parent sampler; the
Director's verdict prompt is module code, not an optimizable artifact field.
Still deferred: structure-stage mutation, ResponseCache, production-log
ingestion, CI threshold gates, frontier picker UI.
