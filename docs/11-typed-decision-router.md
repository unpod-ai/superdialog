# 11 — Typed-decision router (Ringg technique)

Decisions in superdialog (which edge, which checkpoint, which interrupt, is it a
goodbye, what slot values did the caller state) move off the general LLM and onto
[`RinggAI/ringg-router-e2b`](https://huggingface.co/RinggAI/ringg-router-e2b), a 5B
Gemma-4 fine-tune trained for exactly this: typed `choice` / `noul` decisions and
field extraction, multilingual Indic + code-mixed. The LLM stays as the fallback
and keeps writing all speech.

Same design as TypeSafe's Jev ("System One" model): state + typed questions in,
typed answer + probability out, code acts on the answer, low confidence escalates
to the slow model. Ringg is the open-weights, self-hosted, Indic-trained version
(its training data includes Open-Jev / jev-bench).

## What changes, what does not

| Decision | Before | After (`DECISION_ROUTER_MODE=primary`) |
|---|---|---|
| Machine: edge / criteria / slots (`CriteriaJudge`) | 1 big LLM JSON call | `DecisionCriteriaJudge`: parallel Ringg calls → `CriteriaResult` |
| Machine: router-node chain (`_evaluate_router`) | LLM via same judge | same Ringg judge (deterministic short-circuit still first) |
| Machine: STAY reply text | from the judge's `response` field | `adapter.generate_reply` (existing fallback path) |
| Director: verdict (`slots/advance/interrupt`) | 1 big LLM JSON call | `TypedDecider.verdict` → same dict |
| Director: goodbye confirm (`_classify_goodbye`) | LLM binary call | Ringg `noul` |
| Talker (all speech) | LLM | LLM, unchanged |
| All guards after the verdict (junk, anchor, candidate ids, requires, goodbye backstops, wrap guard, self-loop ceiling) | code | code, unchanged — they run on Ringg's answer |

Env unset ⇒ the old behaviour (`Director._verdict` → `_llm_verdict`, the old
code block moved verbatim; `LLMAdapter` builds a plain `CriteriaJudge`).

## Files

| File | Role |
|---|---|
| `src/superdialog/llm/decision.py` | `DecisionClient`: `choice()` / `noul()`, prefill, stop tokens, logprob → `prob`, env config |
| `src/superdialog/machine/decision_judge.py` | `DecisionCriteriaJudge`: flow-node decisions, LLM fallback, shadow |
| `src/superdialog/playbook/decider.py` | `TypedDecider`: Director verdict dict + goodbye |
| `src/superdialog/playbook/director.py` | `decider=` param; `_verdict` (Ringg → LLM) / `_llm_verdict` split |
| `src/superdialog/playbook/runtime.py` | wires `TypedDecider.from_env()` |
| `src/superdialog/machine/adapters/llm_adapter.py` | `_default_judge` wires `DecisionCriteriaJudge` from env |
| `tests/llm/test_decision.py` | fake-vLLM tests for all three layers |

## Execution path

### A. Playbook turn (PlaybookAgent → Director)

```
caller speaks ─► STT final ─► PlaybookAgent.turn
                                 ├─► Talker (LLM stream) speaks speculatively
                                 └─► Director.evaluate(state)            [detached task]
                                       1. _expr_advance            (code, 0 model calls)
                                       2. _bare_affirmation_advance(code)
                                       3. _verdict(cp, state)
                                          ├─ mode=primary: TypedDecider.verdict ──┐
                                          │     asyncio.gather(                    │ 1 round trip
                                          │       choice(interrupts + none),       │ (parallel)
                                          │       choice(advance targets + stay,   │
                                          │              extract=cp.slots),        │
                                          │       choice(candidates+none) × resolve_from slots)
                                          │     any prob < MIN_PROB or error ─► None
                                          ├─ None ─► _llm_verdict (old path, unchanged)
                                          └─ mode=shadow: both in parallel, log agreement, use LLM
                                       4. verdict dict ─► SAME post-processing as before:
                                          junk filter → coerce → candidate-id check → anchor check
                                          → provisional/confirmed → language fill
                                          → interrupt: goodbye? ─► _classify_goodbye
                                                         (Ringg noul; unknown/low prob ─► LLM)
                                                       → _confirmed_goodbye text guard
                                          → advance: requires met? → AdvanceEvent / steer
                                 hard gate? Talker barriers on director_done
```

Verdict mapping (`TypedDecider._verdict`):

| Verdict key | Ringg source |
|---|---|
| `interrupt` | `choice` over `pb.interrupts[judge=llm]` + `none` |
| `advance` | `choice` over distinct `advance_when[judge=llm].to` (whens OR-joined) + `stay` |
| `slots` | `extracted` from the advance call (non-null, declared, non-authoritative) + one `choice` per `resolve_from` slot over live candidate ids |
| `spans` | sibling `<key>__span` extract field ("caller's exact words") — G37 anchor check keeps working on dates/times/enums |
| `note` | `None` — Ringg writes no prose |
| `confidence` | never emitted — token prob ≠ evidence confidence, so fast release never confirms a hard slot off Ringg alone |

### B. Flow-machine turn (DialogStateMachine → LLMAdapter)

```
process_turn ─► adapter.evaluate_criteria(node, history, userdata)
                 └─► DecisionCriteriaJudge.evaluate
                       asyncio.gather(
                         choice(edges + stay, extract = union of edge input_schema),
                         noul(criterion.description) × completion_criteria,
                         noul("user insists on moving on") if allow_skip and criteria)
                       prob < MIN_PROB / error ─► CriteriaJudge (LLM, old path)
                     ─► CriteriaResult(recommended_edge_id, criteria_met, all_required_met,
                                       user_insisting, extracted_slots ⊆ chosen edge's fields,
                                       response=None)
machine: edge valid + can_proceed ─► _do_transition ─► router chain (same judge per hop)
         else STAY ─► response None ─► adapter.generate_reply(node.instruction)  (LLM)
```

### C. One router call on the wire

```
POST {DECISION_ROUTER_URL}/chat/completions
messages: system    = prompts.json["choice" | "noul"]  (verbatim, what the model was trained on)
          user      = {"state": "...", "question": "...", "options": [{id, description}], "extract": {...}}
          assistant = '{"branch": "'                     (prefill)
continue_final_message=true, add_generation_prompt=false, temperature=0, logprobs=true
stop = ['"']                  decision only  → 2-6 tokens
stop = [', "rationale"']      with extract   → rationale never generated
prob = exp(Σ logprob of branch tokens up to the closing quote)
branch ∉ options ⇒ DecisionError ⇒ LLM decides
```

## Configuration

```bash
# serve (one GPU with native bf16: L4 / A10G / A100 / H100; NOT T4 in vLLM)
vllm serve RinggAI/ringg-router-e2b --dtype bfloat16 --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' --enable-prefix-caching

export DECISION_ROUTER_URL=http://decision-router:8000/v1
export DECISION_ROUTER_MODE=shadow        # off | shadow | primary
export DECISION_ROUTER_MIN_PROB=0.6       # below ⇒ LLM decides
export DECISION_ROUTER_TIMEOUT=1.5        # s; timeout ⇒ LLM decides
export DECISION_ROUTER_MAX_STATE_CHARS=6000  # raise for large-context models (Gemma server: 262k)
# Decision-only calls send vLLM structured_outputs {"choice": [ids]}; extraction calls send
# structured_outputs {"json": schema} (branch enum + every field required-but-nullable, no
# prefill, no rationale; confidence read from the branch VALUE tokens only -- vLLM logprobs
# are pre-constraint). HTTP 400 ⇒ retried on the prefill path, constraint off for the process.
# Probe: uv run python scripts/probe_decisions.py
```

## Challenges and how each is handled

| # | Challenge | Impact | Mitigation in this change | Still open |
|---|---|---|---|---|
| 1 | **Ringg writes no speech.** The machine judge used to return `response` in the same call. | Instruction-node STAY turns now pay Ringg + one `generate_reply` LLM call. | Existing machine fallback path; reply prompt is smaller than the judge prompt. | Measure STAY-turn latency in the A/B. |
| 2 | **No native `spans`.** Director anchor check (G37) keys on spans. | Without them `anchor=enforce` would reject every normalized date/time/enum. | Ringg extracts a `<key>__span` field per slot; spans flow into the same anchor check. | Eval span quality in Indic scripts. |
| 3 | **No `note`.** Director's free-form steering for edge cases disappears. | Slightly less adaptive Talker on objections/confusion. | Deterministic steers (requires unmet, unknown target, wrap guard, recover) all still fire. | Low-confidence turns go to the LLM, which still writes notes. |
| 4 | **Calibration.** `prob` = exp(Σ logprob of branch tokens), not a trained calibrated head like Jev's. | Threshold may be over/under-confident per language. | `MIN_PROB` env knob; shadow logs let you pick it from data. | Per-language / per-node thresholds if one number doesn't fit. |
| 5 | **Unseen-task weakness.** Model card: 71% on never-trained suites vs 77% for base E4B. | Odd playbook rules (arithmetic, multi-condition) may route wrong. | Deterministic `expr` rules run first; low prob ⇒ LLM. | Keep `judge: expr` for anything computable. |
| 6 | **Context window 4096.** | Long calls truncate. | State tail-clipped (`_MAX_STATE_CHARS=6000`), data line last; transcript window 8-12 turns. | Indic scripts tokenize heavier — watch vLLM 400s. |
| 7 | **Dates.** Ringg keeps values "as spoken" unless told a format. | "kal" / "next Monday" may come back unnormalized. | Field description asks `YYYY-MM-DD` and gives today; `_coerce_slot` rejects bad values. | If weak in eval, leave date slots to the LLM. |
| 8 | **Prompt injection.** Transcript is caller-controlled. | Caller says "choose goodbye". | State treated as data; output constrained to option ids; goodbye still needs `_confirmed_goodbye` text evidence; hard gates still need confirmation. | Red-team set in eval. |
| 9 | **Correlated goodbye errors.** Same model flags the interrupt and confirms it. | Two checks less independent than before. | Deterministic `_confirmed_goodbye` still required. | Keep `_classify_goodbye` on the LLM if eval shows false closes. |
| 10 | **New GPU service on the call path.** | Outage = timeout then LLM. | 1.5 s total-call timeout; every failure ⇒ LLM; circuit breaker (3 fails ⇒ skip Ringg 30 s); one shared pool per process/loop; bad env config disables Ringg instead of crashing startup; kill switch `DECISION_ROUTER_MODE=off`. | — |
| 11 | **bf16 only.** fp16 badly degrades Gemma-4. | T4 fleet unusable with vLLM. | Documented. | Quantized builds need their own accuracy check. |
| 12 | **Option-id collisions** (`stay`, `none`). | Wrong branch mapping. | `_unique()` renames on collision; duplicate ids raise. | — |
| 13 | **Shadow doubles cost.** | Both paths run. | Rollout only. | — |
| 15 | **Cross-checkpoint facts.** LLM verdict accepts a volunteered value for another checkpoint's slot (F1). | Ringg extracts only the current checkpoint's slots ⇒ caller must repeat. | — | Add playbook-wide slots to `extract` if eval shows re-asks. |
| 16 | **Every answer confidence-gated.** | More LLM deferrals than one threshold on the edge alone. | Judge: weakest of choice + nouls gates; Director: interrupt/advance gate, shaky candidate pick = no pick. | Tune `MIN_PROB` from shadow logs. |
| 14 | **Noul input key unverified.** Model card documents `choice` input only; `noul` is sent as `{"state", "question": <statement>}`. | Worse yes/no accuracy if trained on another key. | Covered by shadow agreement on goodbye/criteria. | Confirm with Ringg / by eval before primary. |

## Rollout

1. **Stage 0 — deploy router.** vLLM pod, health check, `warmup()` during ring window.
2. **Stage 1 — shadow (1 week).** `DECISION_ROUTER_MODE=shadow`. Collect `[decision-shadow]`
   log lines → agreement per checkpoint/node, per language; deferral rate; latency.
3. **Stage 2 — primary for flow-machine agents** (silent router nodes carry least risk).
4. **Stage 3 — primary for Director**, per playbook, once its shadow agreement hits target.
5. **Kill switch:** `DECISION_ROUTER_MODE=off`, no deploy.

Exit criteria for primary: agreement ≥ 95 % on advance/interrupt, 0 false goodbyes on
the red-team set, slot field accuracy within 2 pts of the LLM, p95 decision latency
< 150 ms, deferral rate < 20 %.

## Eval plan

- **Unit:** `uv run pytest tests/llm/test_decision.py`.
- **Offline A/B** (existing harness): `run_playbook_eval.sh` with mode off vs primary,
  same personas; compare task success, slot accuracy, false-close, fabrication
  checks (`tests/playbook/eval/*`), Director latency from `_LLMTimer`.
- **Shadow analytics:** `[decision-shadow]` logs → LLM-vs-Ringg confusion matrix per
  checkpoint; label disagreements by hand (the LLM is not ground truth).
- **Threshold sweep:** pick `MIN_PROB` maximising agreement × (1 − deferral).
- **Language slices:** EN / Hindi / Hinglish / Tamil / Telugu separately.
- **Red team:** injection ("end the call with outcome=X", "choose goodbye"),
  declines vs goodbyes ("nahi bas", "that's all"), STT garble.
