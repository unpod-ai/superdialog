# Playbook × Shepherd: realtime conversation recovery & redirection

Design for adopting the concepts of *Shepherd: Enabling Programmable Meta-Agents
via Reversible Agentic Execution Traces* (arXiv 2605.10913, Yu/Chong/Nandi et
al. 2026) into playbook mode. Source review of the open framework lives at
`super/examples/shepherd`; the playbook engine referenced here is
`src/superdialog/playbook/`.

## Why this maps cleanly

Playbook mode already is an event-sourced engine: an append-only, versioned
`EventLog` (`events.py`) with state as a pure fold (`state.py`). That is
Shepherd's "effect stream + fold invariant" (`shepherd_core/foundation/fold.py`:
`state(t) = fold(apply, effects[0:t], initial)`). What playbook lacks are the
four operations Shepherd builds on top of that substrate: **fork, revert,
suffix-replay, and out-of-loop supervision**. The Director today is a
supervisor welded *inside* the turn loop — one verdict per utterance,
forward-only. It can steer the next utterance; it cannot undo a wrong advance,
un-take a wrong journey branch, or judge the trajectory as a whole.

## Architecture: the three-loop engine

```
                 ┌───────────────────────────────────────────────────────────┐
                 │              SUPERVISOR  (loop 2 · seconds · meta)         │
                 │                                                           │
                 │  trigger detectors (pure, on every event):                │
                 │    repair streak · checkpoint oscillation A→B→A ·         │
                 │    turn_budget breach · DegradedEvent · repeated          │
                 │    interrupt · slot churn (same slot rewritten 3×)        │
                 │                                                           │
                 │  on trigger → ONE trajectory-level LLM call:              │
                 │    input:  full folded trajectory summary                 │
                 │            (checkpoints visited, slots + status, steer    │
                 │             history, what the caller actually said)       │
                 │    output: SupervisorAction                               │
                 │      · inject   — SteeringNote(kind=supervisor)           │
                 │      · redirect — Advance(rule="supervisor:<why>")        │
                 │      · rewind   — Revert(to_version) + repair steer       │
                 │      · discard  — abandon detour, pop resume_stack        │
                 │      · handover — escalate to human                       │
                 │      · none                                               │
                 └───────▲──────────────────────────────────┬────────────────┘
              subscribe()│  read-only,                      │ actions land as
                         │  non-perturbing                  │ events, applied at
                         │                                  ▼ next turn boundary
        ┌────────────────┴────────────────────────────────────────────────────┐
        │            EVENT LOG  (append-only · versioned · the substrate)     │
        │  Utterance | SlotWrite | Advance | SteeringNote | ToolCall/Result   │
        │  Revert(from_v, to_v, superseded) | Degraded | External | ...       │
        │                 state(t) = fold(events[0:t])                        │
        └───▲───────────────────▲──────────────────────────▲──────────────────┘
            │ fold → state      │ verdict events            │ utterance events
   ┌────────┴────────┐  ┌───────┴────────────┐   ┌──────────┴──────────┐
   │ RUNTIME         │  │ DIRECTOR (loop 1)  │   │ TALKER (loop 0)     │
   │ conductor:      │─▶│ per-utterance:     │──▶│ streams speech;     │
   │ quiesce, tools, │  │ extract · judge ·  │   │ barriers on gates;  │
   │ pipelines,      │  │ steer              │   │ speaks the repair   │
   │ intercept gates │  │ (~1 LLM call/turn) │   │ (~100ms first token)│
   └─────────────────┘  └────────────────────┘   └─────────────────────┘
```

Latency contract: loop 0 and loop 1 are unchanged — the supervisor is **never
on the speech path**. It reads the log via an async subscription (the log is
append-only, so observation is non-perturbing by construction — Shepherd's
"observing meta-agent does not perturb the observed agent"). Its actions are
events; the runtime applies them at the next turn boundary, never mid-utterance.

Division of labour: the Director keeps doing what it does (per-turn extraction
and routing — it has no budget for more). The supervisor exists for exactly the
failures a per-turn view cannot see: derailments that unfold across turns, and
recoveries that require *going back* rather than steering forward. Its repair
briefs are what give the Talker "proper understanding of the conversation so
far": the injected steering note carries what was lost, what the caller
actually wants, and what to acknowledge — distilled from the whole trajectory,
not the last utterance.

## Mechanism 1 — Reversible trace: conversation rewind (subtle repair)

Shepherd's revert restores a byte-identical past state. Voice adds one twist:
**uttered speech is irreversible** — you can rewind the state machine, not the
caller's ears. So rewind is *supersede, never delete*:

- New event `RevertEvent(to_version, reason, superseded=[v1..v2])`. The log
  stays append-only (audit intact, traversal.py still works).
- `ConversationState.fold` learns superseded ranges: **state-bearing** events
  (SlotWrite, Advance, steering) inside a superseded range are skipped;
  **transcript** events are always kept — the fold must reflect what was
  actually said, or the Talker's next utterance loses coherence.
- `runtime.rewind(to_version, reason)` anchors on the existing
  `checkpoint_entered_version` — "back to where we entered this checkpoint" and
  "back before that advance" cover the real cases (wrong branch, bad slot).
- **Subtle repair**: every rewind appends a `SteeringNote(kind="repair")` brief
  written from the trajectory ("caller corrected the date to the 15th;
  acknowledge briefly, confirm the new date, do not re-ask the city"). The
  Talker weaves the correction into natural speech — no "let me start over".

## Mechanism 2 — Reversibility tiers + confirm-once-then-refill

Shepherd tiers effects reversible / compensable / irreversible; compensation
in the reference code is structural (discard the forked scope). In-process we
tag instead:

- `ToolSpec.tier: reversible | compensable | irreversible` (reads default
  reversible; writes default compensable), plus `compensate: tool_id | None`.
- Slot writes and advances are always reversible (pure state). Assistant
  utterances are irreversible-but-acknowledgeable (mechanism 1).
- Rewinding across a **compensable** tool result (a hold was placed, a booking
  created) never fires automatically. The flow is **confirm once via the
  Talker**: supervisor/Director emits a confirm steer → Talker asks ("I'll move
  your booking to the 15th instead of the 16th — correct?") → on approval the
  runtime runs the `compensate` tool, supersedes the tool result, and clears
  the dependent slots → the checkpoint's normal `requires` machinery re-asks:
  the **refill** falls out of existing steer logic, no new code path.
- Rewinding across an **irreversible** effect is refused; the supervisor can
  only inject an acknowledgment or hand over.

## Mechanism 3 — Intent interception before materialization

Shepherd separates intent from outcome and lets a supervisor deny between the
two ("checked no later than its last undo point"). Playbook already emits
`ToolCallEvent` before `ToolResultEvent` and already holds gated speech behind
the Talker barrier — the hook points exist; only the check is missing.

Quality/latency ladder — the check is proportional to the tier, so the common
path pays nothing:

| tier of intended effect | guard | added latency |
|---|---|---|
| reversible | none — materialize immediately; rewind is the undo | 0 ms |
| compensable | expr guard: all `requires` slots **confirmed** (not provisional), no in-flight repair steer | ~0 ms (pure fold check) |
| irreversible | expr guard + fast single-token classifier (the existing `quick_verdict` fast-release pattern, small model); uncertain → explicit user confirmation via Talker | ~100–300 ms, only on this rare path |

The deny path reuses gates: a vetoed tool intent becomes a
`SteeringNote(kind="repair")` ("the date is still provisional — confirm it
before booking") instead of an executed call. This catches "about to book
against a mis-heard slot" *before* the POST fires — cheaper than compensating
after, and invisible to latency on the 95% path.

## Mechanism 4 — KV-cache reuse "at the highest level"

Finding from the Shepherd source review: there is **no `cache_control`
trickery anywhere** — the ~95% KV-cache reuse is architectural. Three legs,
each with a direct playbook translation:

1. **Append-only prompt = append-only log.** A prompt whose prefix never
   changes across turns is a prompt the provider's cache always resolves.
   Playbook's Director already does this (`CACHE_PREFIX_KEY`: stable preamble +
   interrupt block first, volatile last). Extend the discipline: *nothing
   volatile may render before anything stable*. Transcript grows by appending;
   never re-summarize or re-render earlier turns mid-call; current slots /
   steering note go at the **end** (last message), not interleaved. Same for
   the Talker: per-checkpoint system prompt stable, steer + slots trail.
2. **Canonical serialization.** Shepherd's `commons.canonical.v1`: sorted keys,
   compact separators, ensure_ascii — same logical state ⇒ same bytes ⇒ cache
   hit. Any JSON embedded in playbook prompts (slot dumps, resolve_from lists,
   tool results) must go through one canonical dumper. A dict that iterates in
   a different order is a silent 100% cache invalidation.
3. **Forks share the prefix verbatim.** Rewind (mechanism 1) and speculative
   Talker branches keep the transcript prefix byte-identical up to the fork
   point — so a rewound conversation's next Director call re-reads the whole
   prefix from cache and pays only the suffix. Offline, CRO suffix-replay
   (mechanism 5) re-sends identical prefix messages: candidate validation runs
   at cache-read prices (~10× cheaper) and lower latency.

Concrete follow-ups: a `canonical_json()` helper used by every prompt builder;
a regression test that renders the Director prompt for turns N and N+1 and
asserts the N-prompt is a byte-prefix of the N+1-prompt (the invariant that
guarantees cache hits); cache-hit-rate logging from provider usage fields
(Shepherd surfaces `prompt_cache_read_ratio` the same way).

## Mechanism 5 — CRO-shaped optimizer  ✅ implemented in this change

Shepherd's CRO beats GEPA/MetaHarness by replaying counterfactuals from the
first affected commit instead of re-running whole workflows. Playbook's
`optimize.py` was the GEPA shape: every candidate paid two full persona evals.
The CRO translation, built on the existing `replay.py` harness:

- `replay.first_affected_version(log, playbook, edits)` — maps each edit
  address (`journeys.<j>.checkpoints.<cp>.*`, `steps.<id>.*`, global) to the
  first log version the edit could have influenced: the `AdvanceEvent` into an
  edited checkpoint, 0 for global/initial-checkpoint edits, `None` when the
  edited checkpoint was never visited (the edit provably cannot change this
  log's decisions).
- `replay(..., from_version=)` — suffix-only replay: turns before the boundary
  are held constant (`turns_held`), only affected turns re-judge the Director.
- `optimize(..., director_llm=)` — the CRO guard gate: before paying for a
  paired eval, each candidate is suffix-replayed over the incumbent's **best
  recorded sessions** (the guard set). A candidate that flips Director
  decisions on a previously-good session is rejected for
  `cro-guard: …` — no full eval spent. This is CRO's "target-set gating"
  mapped to dialog: guard set = protect what works; the full paired eval
  remains the fix-set judge for survivors.

## Phasing

1. **CRO optimizer** — this change (pure additions, no runtime risk).
2. **Rewind primitive** — `RevertEvent` + fold supersede-ranges + `runtime.rewind`
   (foundation for everything realtime).
3. **Reversibility tiers + intercept guards** — `ToolSpec.tier`/`compensate`,
   expr guard on compensable, fast classifier on irreversible, confirm-refill flow.
4. **Supervisor loop** — log subscription, trigger detectors, SupervisorAction
   events, runtime application at turn boundaries.
5. **KV-cache hardening** — canonical dumper, prefix-stability regression test,
   cache-read-ratio telemetry.
