# Playbook Continuity v2 — Design

**Date:** 2026-08-07
**Status:** Validated in brainstorm; ready for implementation planning
**Evidence base:** Three production traversals (westgate ×2, golfai ×1, all 2026-08-07) replayed against the engine at `src/superdialog/playbook/`; full audit in the session preceding this design.

## Problem

Live calls lose conversational continuity: the bot re-asks answered questions, changes topic mid-thread, writes contradictory facts, and loses its place after detours. The audit traced every failure to one structural imbalance plus a set of engine defects:

- **~90% of checkpoint transitions ride on a single director-LLM JSON verdict** (westgate1: 16/18 LLM-judged; westgate2: 11/11; golfai: 3/3). The playbook's advance rules are natural-language prose the LLM interprets, not rails the engine enforces.
- The director is the **weakest model in the stack** (llama-3.3-70b / gpt-oss-20b), yet it holds sole authority over slot writes, transitions, and interrupts. Direct A/B in the data: haiku-4.5 director completed its call; both llama-directed calls derailed and never completed.
- Four engine defects destroy state even when the LLM is right: self-interrupt resume corruption (E1), interrupts permanently dying after one visit (E2), junk slot values confirmed as truth (E4), and user turns silently dropped by the host under its turn lock (E7).
- The Supervisor — built precisely for these failures (oscillation, slot churn, repeated interrupts) — exists, is tested, and **has never run in production**: `guidelines.supervisor` defaults false, no playbook sets it, and the lite_v2 host never passes `supervisor_llm`. All three traversals show its trigger patterns firing with no intervention.

## Decisions (from brainstorm)

| Question | Decision |
|---|---|
| Scope | Architectural rebalance of transition authority |
| Rollout posture | New default semantics + `legacy_continuity: true` escape hatch |
| Advance mechanism | Director still proposes (light path); engine classifies every advance as corroborated / uncorroborated |
| Strictness | Soft-gate: uncorroborated advances pass **with** a steer and an audit event; nothing blocks |
| Supervisor | Default-on under v2; trigger-gated (not per-turn), reuses the director LLM |
| Cost ceiling | 2 LLM calls per turn on the happy path; +8-16% on fully-conveyor calls (hard ceiling ~+24%, see §4), zero on healthy calls; sticky triggers watermarked so identical stale trajectories are never re-reviewed |

Non-goals (YAGNI): compiled intent enums, spoken-goal ledger, new per-turn LLM calls, playbook format changes, playbook linter, model-floor policy. The last two are follow-on tracks.

## Design

### 1. Shape and principles

Target authority distribution: LLM proposes (~60%), engine validates and steers (~25%), supervisor corrects trajectory (~15%). The engine becomes the **accountability layer**, not a straitjacket — no advance is blocked; every advance is classified, and prose-only advances become visible, steerable, and supervisable.

Two-tier change policy:

1. **Unconditional bug fixes** — apply even under `legacy_continuity` (defects, not semantics):
   - **E1** self-interrupt resume-stack corruption (`director.py` interrupt handling)
   - **E2** interrupt suppression scoped to `state.completed` (whole-call) instead of the active detour
   - **E3** silent no-op when the verdict names an unknown advance target
   - **E9** barrier filler bypassing `never_say` and polluting the transcript
   - **E5** `mark_interrupted` never called by any host (barge-in truncation dead)
   - **E7** user turns dropped when the lite_v2 turn lock is held
   - **E8** `language` never passed into `turn()` (sticky-language directive and language-aware filler are dead code)
2. **Flagged semantics** — new default; `legacy_continuity: true` opts out:
   - soft-gated (classified) advances
   - junk-slot rejection
   - slot-churn dampener
   - supervisor default-on with the new triggers

### 2. Director and the evidence model

**Verdict shape unchanged.** The director returns `{slots, advance, note, interrupt}` in one call. No new fields, no second call, prompts nearly identical.

**The engine computes corroboration; it never asks the LLM.** After the verdict, the engine classifies the proposed advance:

- **Corroborated** — any of:
  (a) the matched rule has `requires` and all are met by real slot values;
  (b) an expr rule fired;
  (c) at least one slot declared on the current checkpoint was written this turn with a real value.
  Corroborated advances pass silently, exactly as today.
- **Uncorroborated** — none of the above (prose-only rules such as westgate's "the caller has heard the mention and responded in any way"). The advance still lands, but the engine:
  - appends a steering note to the **target** checkpoint: *"entering `<goal>` without confirmed input — address the caller's last utterance first, then pursue this goal"*;
  - marks the advance in the log (`corroborated: false` on `AdvanceEvent`).

**Junk-write guard.** `_coerce_slot` rejects `""`, `"None"`, `"null"`, `"n/a"`, `"not specified"`, `"unknown"` (case-insensitive) for `str` and enum slots; the value is treated as not-extracted, same as a failed cast today. This kills the observed `configuration='None'`, `investment_or_self_use='None'`, and `city=''` confirmed writes at the source.

**Churn dampener.** Identical-value rewrites of a confirmed slot are dropped (no event, no version churn — westgate2 wrote `staying` four times with the same value). A third *distinct* value for one slot raises the existing `slot_churn` supervisor trigger.

### 3. Interrupt and resume repairs

These fix the westgate2 failure sequence (steps 10–12: `global_price_lookup_guardrail` fired while already inside `answer_pricing_question`, pushed the checkpoint onto its own resume stack, "resumed" to itself, and stranded `qualify_location` forever).

- **E1 — self-interrupt guard.** In `Director.evaluate`, an interrupt whose target equals the current checkpoint is downgraded to a steering note ("the caller is still on `<topic>` — continue handling it here"). No `AdvanceEvent`, no stack push, no self-loop.
- **E2 — detour-scoped suppression.** `already_handled` changes from `spec.to in state.completed` to `spec.to in state.resume_stack or spec.to == state.checkpoint_id`. Interrupts stay alive for the whole call; only re-entry into an *active* detour is suppressed.
- **Resume integrity.**
  - A resume advance validates its target exists; a stack entry older than 6 advances is dropped with a `DegradedEvent` instead of teleporting the caller backward minutes later.
  - The resume-priority branch in `PlaybookRuntime.on_user_text` keeps applying the director's slot writes (unchanged), but now honors a **corroborated** advance over the forced return. Uncorroborated advances still defer to resume.
- **E3 — loud unknown targets.** A verdict advance with no matching rule logs `DegradedEvent(component="director", detail="unknown_advance_target:<x>")`.

### 4. Supervisor default-on and host truthfulness

**Supervisor.** `guidelines.supervisor` defaults to `true` under v2 (legacy keeps `false`). It reuses the director LLM as already wired in `PlaybookAgent.__init__`. Cost stays trigger-gated: pure-Python detection on every completed turn, one LLM call only when a trigger fires past the 2-turn cooldown (compensation-pending reviews bypass the cooldown), off the speech path. Sticky triggers are additionally **watermarked** (see deviations §6): a firing requires NEW qualifying evidence since the last review, so identical stale trajectories are never re-reviewed. Measured against westgate1 (25 turns, 50 LLM calls, ~16 directed advances): an all-conveyor call fires at most once per new uncorroborated advance past cooldown — ~8 firings for westgate1's advance clustering ≈ +8-16% on fully-conveyor calls (hard cooldown-bound ceiling ~12 firings ≈ +24% if fresh conveyor advances land in every 2-turn window for the whole call); in practice streak evidence between reviews is consumed in batches, so observed cost stays well under. A healthy call pays zero. Two new triggers in `detect_triggers`:

- `uncorroborated_streak` — ≥2 consecutive uncorroborated advances;
- `junk_rejected:<key>` — the same slot rejected ≥2 times.

Merging the supervisor into the director was considered and rejected: the trajectory prompt (~1–2k tokens of advance history and checkpoint catalogue) would be paid every turn on the latency-critical barrier path, and it would ask the already-overloaded director model to do two jobs in one verdict.

**Host truthfulness (lite_v2).** The engine cannot keep continuity over a transcript that lies about what was said and heard:

- **E7 — dropped turns.** When `_pb_turn_lock` is held, queue the skipped utterance and deliver it on release instead of discarding it. The adapter's latest-user-text extraction back-fills any user messages missing from the event log, in order.
- **E5 — barge-in.** The interruption callback calls `pb_agent.mark_interrupted(heard_prefix)` with the transcript-synced prefix LiveKit already computes. The director stops reasoning over words the caller never heard.
- **E8 — language.** The host passes bridge-detected language into `turn(text, language=...)`, activating the sticky-language directive and the language-aware barrier filler (currently dead code; the English filler spoken mid-Hindi call in westgate1 is the symptom).
- **E9 — filler.** `Talker.speak` runs the filler line through the same `never_say` excision as streamed tokens, and tags filler utterances in the log so traversals and director prompts can discount them.

### 5. Observability, rollout, testing

**Observability:**

- Traversals record every user turn, not only transitions; dwell turns appear as steps with `from == to`. This closes the invisible-turn gaps (westgate1 turns 10–12 and 21–24) and fixes the misleading bot/user ordering.
- `AdvanceEvent` gains `corroborated: bool`; traversals and Langfuse show how much of a call ran on prose.
- `_LLMTimer` records talker TTFT separately from full-stream duration (westgate talker full-stream p95 was 19–42 s; TTFT is currently unmeasurable). Traversal `latency` gains `ttft_ms`.
- Traversal `playbook_file` comes from `playbook.source_path`, not the host's static agent name (both westgate and golf sessions currently claim `unpod-prod-general-agent-v3`).

**Rollout:**

1. Ship the unconditional bug fixes (E1/E2/E3/E9 engine; E5/E7/E8 host). No flag.
2. Ship v2 semantics behind the flag with default **legacy** for one release; enable v2 on one QA agent; compare traversals.
3. Flip the default to v2. Westgate and golf get `legacy_continuity: true` only if regressions surface.
4. Release note for downstream test suites: scripted-director fixtures on v2 playbooks must pin `guidelines: {supervisor: false}` — under v2 the supervisor defaults on and reuses the director LLM, so a scripted verdict sequence would otherwise be consumed by supervisor reviews.

**Testing** (registered as a new `playbook-continuity` module in `scripts/run_tests.sh`):

- **Traversal regression fixtures:** replay the three production event sequences and assert — westgate2's self-interrupt yields a steer, not an advance, and `qualify_location` resumes; golfai's backward bounce is marked uncorroborated; `'None'` and `''` writes are rejected.
- **Unit:** corroboration classifier (all three clauses plus negatives), junk-guard table, detour-scoped suppression, resume-stack expiry, filler excision through `never_say`.
- **Supervisor:** trigger tests for `uncorroborated_streak` and `junk_rejected`.

## Implementation deviations (reconciled)

Deltas between this design (and the implementation plan's snippets) and what shipped on `feat/playbook-continuity-v2`:

1. **Corroboration clause (c) ordering.** `slot_written_this_turn` is computed BEFORE `rule.set` writes are appended — a rule's own `set:` stamp never vouches for its advance; clause (c) evidence is caller-derived verdict extraction only. The implementation plan's Task 8 snippet carried the flaw (it counted events already appended, including set writes); fixed in d30edb7.
2. **Recorded decision: old evidence still corroborates clause (a).** `requires` met by OLD state corroborate an advance — clause (a) has no recency requirement; recency lives entirely in clause (c).
3. **Resume-stack expiry is fold-pure and SILENT.** No `DegradedEvent` — the fold cannot append events; the non-resume is visible in traversals as the absent return. Caveat: `max_hops` (8) exceeds the age budget (6), so one exhausted quiescence chain can expire an entry within a single turn.
4. **Nested-resume fold fix.** On a resume pop, `entered_via_resume = bool(remaining stack)` — nested detours unwind level by level. Accepted bounded ceiling: a Task-9-stranded entry can be resumed-to if a later detour pops on top of it within the 6-advance window; clear-on-fall-through was rejected (ambiguous for nested chains); the expiry reaper bounds the effect.
5. **Junk signal shape.** Implemented as `DegradedEvent(detail="junk_rejected:<entity-namespaced key>")` rather than a new event type; the generic `degraded` supervisor trigger excludes it so a single benign rejection never spends a supervisor call.
6. **Sticky supervisor triggers are WATERMARKED** (post-plan addition). Sticky triggers (`uncorroborated_streak`, `junk_rejected`, `slot_churn`, `repeated_interrupt`) only fire when their newest qualifying event is newer than the supervisor's last-reviewed version. Added after cost analysis showed un-watermarked all-conveyor calls would breach the +8% budget ~3x (an established streak re-fires every cooldown window: ~12 of 50 baseline calls ≈ +24%). Updated cost figures in §4.
7. **Deferred / known gaps:**
   - `say_verbatim` still bypasses `never_say` (the verbatim yield skips the stream excision filter).
   - `compiler.py`'s flows_json path has no `legacy_continuity` source — flow-compiled playbooks cannot opt out at the flow level.
   - The churn dampener removes the accidental repair-note delivery for long-confirmed re-asks (that accidental delivery was the false-positive source).
   - `SupervisorDecision` accepts garbage that type-checks as `action="none"` silently — harden with a required `action` or `extra="forbid"` in a follow-up.
   - The `slot_churn` trigger label is not entity-namespaced while `junk_rejected` is — align in a follow-up.
   - `expr_gated_advance_target` refinement: NOT implemented (YAGNI — the verdict prompt only advertises llm-rule targets, so the mislabel window is negligible).
   - Release-note item: scripted-director fixtures on v2 playbooks must pin `guidelines: {supervisor: false}` (see Rollout step 4).

## Appendix — audit findings referenced above

| ID | Layer | Finding |
|---|---|---|
| E1 | engine | Self-interrupt corrupts resume stack; caller's place lost (westgate2 steps 10–12) |
| E2 | engine | `already_handled` uses append-only `state.completed`; interrupts die after one visit |
| E3 | engine | Unknown verdict advance target is a silent no-op |
| E4 | engine | `_coerce_slot` accepts `"None"`/`""`; soft slots confirmed on first write; junk renders into the talker prompt every turn |
| E5 | engine/host | `mark_interrupted` has zero callers; barge-ins log unheard speech as heard |
| E6 | engine | Suppressed-but-spoken stale talker chunks; log/audio order divergence (accepted limitation, out of scope) |
| E7 | host | lite_v2 skips turns under `_pb_turn_lock`; utterances never reach the event log |
| E8 | host | `language` never passed to `turn()`; language plumbing dead |
| E9 | engine | Barrier filler bypasses `never_say`, gets logged, and ignores call language |
| E10 | engine | Repair detection keyed on `"?"` in the reply (out of scope for v2) |
| E11 | engine | Director sees only the last 12 transcript messages (out of scope for v2) |
| E12 | config | Supervisor implemented but never active anywhere |
| P1–P6 | playbook | Conveyor advance rules; catch-all backward edge (golf); decline≠WhatsApp interrupt; empty goals on interrupt targets; 22 KB persona vs 4 k talker budget; every checkpoint hard-gated |
| L1–L3 | deployment | Director model too weak for its authority; talker drift; YAML/resolved model mismatch |
| O1–O2 | observability | Transition-only traversals; per-turn latency misalignment |

Playbook authoring fixes (P1–P6) and the model-floor policy (L1) are deliberate follow-on tracks; this design makes their failures visible and survivable rather than fixing the authoring itself.
