# Algorithm Optimization Roadmap

**Scope:** research-grade optimization techniques for the playbook execution
algorithm documented in `09-execution-algorithm-and-guards.md`. Each entry
states the technique, why it wins, its cost, and what to measure.

**Framing.** The engine is a **shielded hierarchical semi-MDP**: a fast
speech policy (Talker), a per-step structured-inference controller
(Director), a deterministic shield (guards G1–G36), and an episodic
meta-controller (Supervisor), over an event-sourced belief state (the fold).
Every technique below exploits that structure. The strategic asset: the
event-sourced fold + shield is a **simulator** — calls can be replayed,
counterfactualed, and policies trained offline with safety guaranteed by
construction.

---

## A. Decode-time constraints replace post-hoc guards *(highest leverage)*

Guards G3/G4/G17 let the model waste probability mass on invalid outputs,
then reject them. **Grammar-constrained decoding** (JSON schema → token
masking; vLLM/outlines, or gateway `response_format` with schema) makes
invalid outputs unrepresentable:

- `advance` constrained to the current checkpoint's legal targets ∪ {null}
  — the unknown-target class (G17) vanishes at the decoder.
- Slot keys constrained to the declared allowlist; enum slots to their
  declared values — G3/G5 rejection becomes a prior constraint.
- Information-theoretically strictly better: the model's distribution is
  renormalized over *valid* actions only. Weak models gain the most — and
  the Director is deliberately the cheapest model in the stack.

**Cost:** near zero (decoder feature). **Measure:** unknown-target and
structural-junk audit-event rates (expect ~0); verdict parse-failure rate.

## B. Cascaded inference with uncertainty routing

Most verdicts are trivial (`STAY` + one slot write). Run a **cascade**: a
small/local model (or the distilled head below) answers first; escalate to
the big model only on uncertainty. Training-free uncertainty signals:
constrained-decode logprob margin on the `advance` span, or k=2
self-consistency disagreement.

**Distillation dataset already exists:** every traversal logs
(prompt-state, verdict) and eval judges label whether the turn went well.
Distill the easy 80% into a ~1B structured decoder; keep the big model for
escalations. The haiku-vs-llama production A/B showed the ceiling is model
quality — the cascade buys cost/latency without lowering that ceiling.

**Measure:** director p50/p95 (the barrier path), cost per call, escalation
rate, quality delta on the behavioral suites.

## C. Speculative dialog execution

The Talker already speaks speculatively with rollback via stale-speech
suppression (G26) — speculative decoding at the dialog level. Formalize it:

- A tiny transition predictor (a bigram table over `(checkpoint, advance)`
  from the traversal corpus — production flows are near-deterministic
  conveyors) predicts the post-verdict state; the Talker streams from the
  *predicted* state.
- Prediction hit → zero-added-latency turn; miss → existing suppression path.
- Speculatively prefetch **idempotent** `on_enter` GET tools for the top-1
  predicted target.

**Measure:** speculation hit-rate per playbook; TTFT at hard gates (already
instrumented); suppression rate (rollback cost).

## D. Calibrated evidence instead of binary corroboration

`corroborated ∈ {True, False, None}` is a 1-bit shadow of
**P(advance correct | features)**. Fit a logistic/isotonic calibrator on
logged features — requires-met count, slot recency, uncorroborated-streak
length, checkpoint depth, director model, interrupt density — against
eval-judge outcomes. Then steer strength, barrier engagement, and supervisor
thresholds become functions of calibrated risk, and the magic constants
(cooldown 2, expiry 6, junk ≥2) are set where the calibration curve says
regret crosses cost.

**Cost:** small; the event logs + judge scores are already the labeled
dataset. **Measure:** calibration curve (Brier), supervisor firings per
call, false-steer rate.

## E. Close the learning loop: offline RL under the shield

The guard layer is a **shield automaton** in the safe-RL sense — it makes
*training* the Director safe:

- **DPO/KTO on verdict pairs**: same state, verdicts leading to judge-good
  vs judge-bad continuations (A/B runs generate these naturally).
- **Counterfactual replay as off-policy evaluation**: the event-sourced
  fold + `replay.py` re-runs a logged call under a different director
  policy deterministically at the tool seams — a true offline A/B with zero
  live cost. Exploit this before any live experiment.

**Measure:** offline policy value vs. logged policy; then paired live suites.

## F. Retrieval-gated interrupt set

Every interrupt rides in every verdict prompt with absolute priority — token
cost and a false-fire attractor for weak models (westgate2 guardrail
over-firing). Gate by bi-encoder similarity between the utterance and each
interrupt's `when` description; include top-k plus the goodbye class
(always). Shrinking the per-step action space is the same class of win as
(A) by other means.

**Measure:** interrupt false-fire rate, verdict prompt tokens, detour rate.

## G. Value-of-information confirmation policy

Hard-gate confirmation (G8) is rule-based: always escalate. Frame each
re-confirmation as a **VoI decision**: ask iff
`P(wrong) × cost(wrong) > cost(extra turn)`. With (D)'s calibrated
confidence, phone/OTP-class slots keep near-certain confirmation while
low-stakes slots stop burning turns — directly attacks the turn-count and
latency the judges penalize.

**Measure:** turns-to-completion, re-confirmation count, wrong-slot rate.

## H. Statistically honest evals *(prerequisite for B–G)*

Observed composite swings of ±0.25 at n=1 exceed most expected effect
sizes. Fix the estimator first:

- reps ≥ 3 per case; variance-aware gates with CIs;
- **paired-comparison scoring** (same case, transcript A vs B,
  Bradley-Terry) instead of absolute judge ratings — judges rank far more
  reliably than they rate.

**Measure:** judge-score variance per case; minimum detectable effect at
current rep count.

---

## Ranked plan (leverage ÷ effort)

| # | technique | rationale |
|---|---|---|
| 1 | A. Constrained decoding | free correctness for the weakest component; deletes two guard classes at the decoder |
| 2 | H. Paired evals + reps | everything else is unmeasurable without it |
| 3 | C. Speculative execution + prefetch | flows are near-deterministic; latency is the felt product |
| 4 | D. Calibrated corroboration | five magic constants → one learned curve |
| 5 | B → E. Cascade, then DPO under the shield | cost first, then raise the ceiling |

Techniques compose: A shrinks the action space, C hides the remaining
latency, D prices the residual risk, E learns the policy, H proves it.
