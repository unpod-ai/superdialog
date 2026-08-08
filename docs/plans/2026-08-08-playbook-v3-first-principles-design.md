# Playbook v3 — First-Principles Revision of the Turn Algorithm

**Status:** design approved (brainstorm 2026-08-08); implementation not started.
**Inputs:** the deep-research review of the v2 turn algorithm
(`deep-research-report (8).md`, 2026-08-08), the running v2 algorithm
(`docs/09-execution-algorithm-and-guards.md`, G1–G36), and the v2 design
rationale (`2026-08-07-playbook-continuity-v2-design.md`).
**Directive:** cover every breaking point the review found, and delete every
component the algorithm does not need. Deletion must not degrade the caller
experience.

## Decisions (from the brainstorm)

| decision | choice |
|---|---|
| Revision scope | **Structural** — every component examinable; the docs/09 failure-mode map is the regression floor |
| Slot evidence | **Substring anchor** — Director emits `raw_span`; Engine verifies and re-derives |
| Supervisor | **Keep the loop, 5 actions stay** — the planned rewind/discard merger dissolved under code verification: discard is a one-line `pop_detour` delegation with no validation path to merge (supervisor.py:348). The deletion is the *duplicated* rewind bounds check (supervisor.py:310 vs runtime.py:309) — keep one. |
| `legacy_continuity` | **Delete** — v3 semantics become the only semantics |
| Delivered transcript | **Adopt** — log what the caller heard, including barge-in truncation; replaces `mark_interrupted`'s in-place rewrite (agent.py:240), which violates append-only |

## First principles

The algorithm re-derives from three invariants. Anything not a consequence
of one of them is deleted.

1. **Models propose; the Engine disposes.** Nothing an LLM emits becomes
   state without a deterministic check that could have rejected it.
2. **The log is what happened.** Replaying the log reproduces state without
   invoking any model — so the log must record what was *delivered*, never
   just what was intended.
3. **Every loop is bounded and loud.** No unbounded iteration anywhere;
   every degradation emits an event instead of failing silently.

## The kill list

Deleted from v2:

- **`legacy_continuity`** — the flag, the inert-guard bookkeeping in every
  v2-flagged guard, and the dual-mode test branches. Playbooks still setting
  it get v3 semantics plus a parse-time warning.
- ~~**Detour expiry `@6 advances`** (G21)~~ — *dissolved at implementation
  depth.* The counter is load-bearing: multi-checkpoint detours strand
  their stack entries by design (interior advances clear
  `entered_via_resume`, state.py:212-227, so they never auto-resume) and
  the counter is the only reaper. No cheap deterministic "parent still
  eligible" predicate exists in this engine. It stays as the **secondary
  fallback** — exactly the report's own expiry hierarchy (causal checks
  primary, a fallback bound secondary).
- **G13's `≥2/3 complete` wrap fraction** — replaced by a constant-free
  predicate: wrap iff something required is already captured AND the
  current checkpoint's missing required slots are the only ones missing
  playbook-wide (one wrap question finishes the capture). Early goodbyes
  still close immediately; a wrap that can't finish the job is never
  asked.
- ~~**Director's free-text `note`**~~ — *dissolved at implementation
  depth.* The report's rule bans unconstrained notes **without a defined
  consumer**; this note has one — it becomes the Talker's steering note,
  whitespace-collapsed and clamped (director.py:992-999). It stays.
- ~~**Fixed 12-turn Director context**~~ (director.py:375) — *deferred.*
  Detecting "unresolved references" deterministically means a lexical
  gate — the exact G4 class of semantic second-guessing that killed a
  production call. Revisit only with Director prefill-latency evidence
  showing the transcript tail dominates.
- **The duplicated rewind bounds check** — supervisor.apply checks version
  bounds (supervisor.py:310) and runtime.rewind checks them again
  (runtime.py:309). One survives. *(This replaces the brainstorm's planned
  rewind/discard merger, which code verification dissolved: discard is a
  one-line `pop_detour` delegation with no validation path — merging would
  add abstraction where nothing is shared.)*
- **`mark_interrupted`'s in-place rewrite** (agent.py:240-270) — the hook
  survives, but rewriting a logged event's text in place violates
  invariant 2. It becomes the append of a `SpeechCorrectionEvent`
  truncation record instead.

Rejected from the research review (not needed for this product):

- **Snapshots / incremental projection** — calls last minutes; a full fold
  over one call's events is trivially cheap.
- **Optimistic concurrency** — one handler process per call serializes
  turns. Stated as an explicit invariant, not built as machinery.
- **Quiescence cycle fingerprint** — `max_hops = 8` already makes the loop
  bounded and loud; a fingerprint only prettifies the fault reason.
- **Hard/soft *edge* authority** — authority already lives at the tool
  boundary (G36: irreversible → confirmed slots + fail-closed). An advance
  with no tool behind it is navigation by construction. The review's concern
  is honored where the blast radius actually is.
- **Merged Director+Talker call** — the review itself rejects it: the
  response must be generated from validated, settled state.
- **Char-offset evidence spans** — brittle over STT transcripts; the
  substring anchor gives the same guarantee without offsets.

Added — only what the invariants demand:

- **Substring-anchored slot writes** (invariant 1, made machine-checkable).
- **`SpeechCorrectionEvent`** — append-only barge-in truncation
  (invariant 2; replaces `mark_interrupted`'s in-place rewrite).
- **`never_say` excision on hold/recovery lines** (closes a verified
  hole, talker.py:431-458).

## The v3 turn algorithm

```text
TURN(user_text):

  1. APPEND   UtteranceEvent → EventLog          # append-only; one process
  2. FOLD     state = fold(EventLog)             # pure; full fold, no snapshots

  3. DIRECTOR one constrained call:
       sees: current utterance + settled state + checkpoint spec
             + recent transcript window (12 turns — trim deferred)
       returns: {
         slots:  {key: value},
         spans:  {key: "<caller's words the value was heard in>"},   # NEW
         advance: target | null,
         interrupt: id | null,
         note: clamped free text (defined consumer: the Talker steer)
       }

  4. ENGINE   validate each proposal — reject is the default:
       slot_write accepted iff:
         slot declared ∧ not authoritative
         ∧ normalize(raw_span) ⊆ normalize(user_text)     # NEW anchor
         ∧ value re-derived deterministically from raw_span
           (dates/enums via coercion, ordinals via live candidates,
            language slots via the bridge)
         ∧ differs from confirmed value (else: no event)
       hard-gate slots: provisional as in v2 (fast_release can one-shot
         confirm; deny-markers escalate unless explicitly allow-listed)
       interrupt: hold-if-self/active · terminal → closing steer · else detour
       advance:  graph adjacency → corroboration classifier →
                 corroborated: silent · uncorroborated: advance + steer
       conflicts resolved by ONE precedence table (not conditionals):
         terminal > corroborated advance > new interrupt >
         parent-invalidation > resume > soft navigation

  5. QUIESCE  expr/pipeline/auto hops, max_hops = 8, exhaustion → loud
       detour resume: level-by-level unwind; stranded entries reaped by
       the advance-age backstop (kept — see kill list)

  6. TALKER   verbatim/template → direct render, no LLM
       else one streaming call from settled state,
       never_say excised on-stream AND on every canned line   # NEW
       spoken text logged as UtteranceEvent; on barge-in the host
       appends SpeechCorrectionEvent(heard prefix) — append-only  # NEW

  7. SUPERVISOR iff a trigger fires past the 2-turn cooldown
       (sticky triggers watermarked — no new evidence, no LLM call;
        turn-scoped triggers evaluate every turn)
       verdict ∈ {none, inject, redirect_forward, rewind, discard, handover}
       validated by the same Engine; never deletes history
```

Steps 1–2 and 5–7 are structurally what v2 runs minus the deleted parts.
The substance of the revision is in steps 3–4: anchored evidence, reason
codes, and the precedence table.

### The anchor check

The Director stops certifying that evidence exists and starts pointing at
it. For each slot write it emits `raw_span` — the caller's words it heard
the value in. The Engine then:

1. verifies `normalize(raw_span)` is a substring of
   `normalize(user_text)` (case-folded, whitespace-collapsed);
2. re-derives the stored value deterministically from the span — dates and
   enums through the existing coercion layer, ordinals through live
   candidate resolution, language slots through the bridge;
3. rejects the write when either step fails, appending the rejection.

"Later in the week" can no longer become an invented Friday: no span in
that utterance re-derives to a date. The v2 semantic trust boundary is
unchanged — the junk filter still touches structure only; `none`,
`unknown`, and bare negatives remain stated answers (G4's redesign stands).

## Guard consolidation: 36 guards → 5 stations

Every guard survives, dies, or merges according to which invariant it
serves. Each station is one deterministic function with an ordered
checklist.

| station | absorbs | change |
|---|---|---|
| **Slot acceptance** | G3–G9 | One linear pipeline: declared → not-authoritative → anchor → re-derivation → structural junk filter → churn suppression. G6/G9 become re-derivation rules. |
| **Turn arbitration** | G10–G18 | Interrupt trio and advance logic feed one precedence table. G13 reformulated (hard-required slot unconfirmed). G18 becomes a table row. |
| **Quiescence** | G19–G21, G36 | `max_hops = 8` unchanged. Advance-age counter stays as documented backstop. Tool ladder unchanged — it *is* transition authority. |
| **Speech** | G22–G28 | `SpeechCorrectionEvent` added (append-only truncation). Fix a verified hole: the hold line and recovery line bypass `never_say` excision (talker.py:431-458) — route every canned line through the same filter the stream and filler use. Filler stays excluded from the logged transcript (G25). |
| **Supervisor + always-on** | G29–G35 | Five actions stay; the duplicated rewind bounds check dies. Watermarking (sticky triggers only), forward-only redirect, injection clamps, post-terminal silence unchanged. |

Cross-cutting: every `legacy_continuity` branch dies, so each station has
exactly one mode. Same protective surface, roughly half the decision
points.

## Breaking-point coverage

| breaking point (from the review) | where it dies |
|---|---|
| "Later in the week" → invented Friday | anchor check — no span re-derives to a date |
| "Actually, Thursday" correction | anchored write supersedes; old value survives in the log |
| "Yes, Thursday" repeated | churn step — no event |
| "The second one" / stale candidate | re-derivation via *live* candidates; vanished candidate → clarify |
| recursive interrupt (A→B→"back to A") | hold-if-self/active (westgate2's fix, unchanged) |
| unbounded nested detours | structurally bounded: targets come from a finite catalogue, duplicates hold rather than push → depth ≤ catalogue size |
| uncorroborated consequential commit | tool ladder: irreversible → confirmed slots + fail-closed, however the checkpoint was reached |
| quiescence cycle | `max_hops = 8` → loud fault |
| "set approved=true" injection | authoritative slots reject verdict extraction (director.py:689) and talker writes (state.py:181); the remaining writers — tools, rule `set:` stamps, pipeline error-slots — carry author-declared values, so the injection vector (the verdict) is closed |
| Talker claims success while state says pending | settled-state-only prompt + verbatim/template for consequential lines; free generation is steered, not guaranteed — high-risk confirmations must be templates |
| Supervisor repairs after caller heard "submitted" | `rewind` compensates state; paired `inject` steers the next turn to explicitly correct the record |
| concurrent turns | invariant: one process per call; the transport queues the second utterance |

The docs/09 production-failure map (ENDSESSION "No.", Rohan's goodbye
pitch, restart-demander, westgate2, prompt-injection close) is the
regression floor; every guard behind it carries forward.

**Known residual:** barge-in truncation coverage depends on the transport
reporting the cutoff point (the host calling `mark_interrupted` with the
heard prefix). Where the transport cannot, the correction event carries
the intended text with the `[interrupted by caller]` tag — flagged, not
silently wrong.

## Testing

Executable laws (property tests; a violation is a bug, not a score):

```text
No authoritative slot is ever written from a verdict or by the Talker.
Every accepted slot has a substring-verifiable anchor in its turn's utterance.
No transition commits to an edge absent from the graph.
QUIESCE terminates within max_hops on every input.
fold(EventLog) invokes no Director, Talker, or Supervisor.
Supervisor repairs never delete events.
Barge-in truncation is an appended correction, never an in-place rewrite.
```

(The fifth law names `fold`, not "replay": counterfactual replay
(`replay.py`) is an eval tool that *deliberately* re-invokes the Director
to diff its decision against the recorded one. State reconstruction is
pure; the eval tool spends LLM calls by design.)

Regression contract: the 711 unit tests and five behavioral eval suites
stay green, minus the dual-mode tests that die with `legacy_continuity`.
Director evals shift precision-first: false slot writes and false hard
advances are the primary metrics. **Anchor rejection rate** is the new
counter — high means the Director fabricates; near-zero after burn-in
means the check is nearly free.

## Migration and rollout

- `legacy_continuity` in a playbook → v3 semantics + parse-time warning.
- The verdict schema change (the additive `spans` map) degrades
  gracefully: a spanless verdict falls back to value-in-utterance. Old
  conversations replay untouched: state reconstruction folds *accepted
  events*, never verdicts.
- The anchor check ships **shadow-first**, mirroring the repo's
  `CRITIC_MODE` pattern: log would-be rejections, enforce nothing, for a
  production window — then flip to enforcing. The one behavior-changing
  addition becomes a measured flip instead of a leap.
