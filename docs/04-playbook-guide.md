# SuperDialog — Playbook Authoring Guide

**Status:** Canonical
**Parent:** [README.md](README.md)
**Audience:** Developers writing, migrating, and tuning playbooks.

---

A **Playbook** declares a conversation as journeys of checkpoints plus a
process layer (tools, pipelines, handlers, interrupts, policies). At runtime
a fast **Talker** LLM streams every spoken turn while an async **Director**
extracts slots, judges advancement, and runs tools — both over an
append-only event log that doubles as the audit and replay artifact. The
legacy `DialogMachine` flow engine remains fully supported; playbooks are
where new investment goes, and `compile_flow` migrates existing flow JSON
(§5).

## 1. Anatomy of a playbook

Playbooks load from YAML or JSON (`Playbook.load(path)` /
`Playbook.from_yaml(text)`); every cross-reference (rule targets, pipeline
ids, tool ids, `requires` keys) is validated at load time. A complete,
annotated example:

```yaml
persona: "You are Asha, a friendly golf-course booking assistant."

env:                          # plumbing lane — NEVER rendered to the Talker
  API_BASE_URL: "https://api.example.com"
  ACCESS_TOKEN: ""            # rotated by middleware (below)

views:                        # computed, LLM-free exprs; shown as Reference data
  hold_valid_until: "results.hold.data.valid_until"

journeys:
  booking:
    checkpoints:
      - id: collect           # journey-local id; addressed as booking.collect
        goal: "Have city and date"
        slots:                # typed, flow-scoped declarations
          city:
            type: str         # str|int|float|bool|date|enum|array|object
            required: true
            invalidates: [hold]   # a city change clears the stale hold result
            description: "City the caller wants to play in"
          date: {type: date, required: true}
          players: {type: int}
        guidance: |           # Jinja over {slots, views, results}
          Collect naturally; the caller may give everything in one breath.
        never_say: ["our systems are slow"]
        turn_budget: 6        # steer to wrap up after 6 user turns here
        on_failure: booking.handoff
        advance_when:         # ordered rule list; first matching rule wins
          - when: "caller gave the booking details"
            judge: llm        # the Director judges intent
            to: booking.confirm
            requires: [city, date]   # rule fires only when these are met
      - id: confirm
        gate: hard            # outcomes barrier on the Director here (§2)
        say_verbatim: "Held until {{ views.hold_valid_until }}."  # no LLM
        pipeline: confirm_and_hold   # process layer runs on entry (§3)
        slots:
          price: {type: float, authoritative: true}   # tool-written only
        advance_when:
          - {when: "pipeline.ok", judge: expr, to: booking.close}
          - when: "pipeline.failed"
            judge: expr
            to: booking.collect
            set: {error_context: booking_confirm_failed}  # confirmed write
      - id: handoff
        auto: true            # speak verbatim once, then advance unprompted
        say_verbatim: "Let me connect you to a colleague."
        advance_when:
          - {when: "always", judge: llm, to: booking.close}
      - id: close
        terminal: true        # session ends on entry
        outcome: confirmed    # label for metrics and host hangup

tools:
  - id: hold_slot
    type: http                # or `python`, registered via python_tools
    method: POST
    url: "{{ env.API_BASE_URL }}/slots/hold"
    headers: {Authorization: "Bearer {{ env.ACCESS_TOKEN }}"}
    body: {city: "{{ slots.city }}", date: "{{ slots.date }}"}
    store_response_as: hold   # readable as results.hold.* afterwards
    env_updates: {hold_id: hold_id}   # env key <- dotted path into the response
    run_once: false           # true: at most one call per session
    when: "slots.city"        # expr over state; skip the call when falsy
    timeout: 10
  - id: refresh_auth
    method: POST
    url: "{{ env.API_BASE_URL }}/auth/refresh"
    env_updates: {ACCESS_TOKEN: token}

pipelines:
  - id: confirm_and_hold
    steps:
      - tool: hold_slot
        on:                   # route on the step's typed result
          ok: continue        # next step, or pipeline success at the end
          http_409: booking.collect          # typed HTTP-status branch
          failed: {retry: 1, on_exhaust: booking.collect}

middleware: {on_status: 401, refresh_with: refresh_auth, then: replay}

handlers:                     # Talker-less, event-triggered pipeline entries
  - {id: payment_done, on: webhook.payment_captured, pipeline: confirm_and_hold}

interrupts:                   # judged from any checkpoint
  - {id: goodbye, when: "caller says goodbye", judge: llm,
     to: booking.close, resume: false}

policies:
  silence:
    max_prompts: 2
    prompts: ["Can you hear me?", "Are you there?"]
    then: booking.close       # route here after max_prompts silences

initial: booking.collect      # defaults to the first checkpoint anyway
```

Top-level fields, all on `superdialog.playbook.models.Playbook`:

| Field | Type | Purpose |
|---|---|---|
| `persona` | str | System-level voice of the agent, every Talker turn |
| `journeys` | dict[name, Journey] | Named checkpoint sequences (min 1) |
| `dispatch` | list[DispatchEntry] | Intent→entry table (compile-time in v1, §5) |
| `tools` | list[ToolSpec] | Declarative HTTP / registered python tools |
| `pipelines` | list[PipelineSpec] | Ordered tool steps with typed branches |
| `handlers` | list[HandlerSpec] | `webhook.<name>` / `timer.<name>` triggers |
| `interrupts` | list[InterruptSpec] | Global jumps (`judge: llm` or `event`) |
| `policies` | Policies | Cross-cutting behavior (silence today) |
| `middleware` | MiddlewareSpec | `on_status` → refresh tool → replay |
| `env` | dict[str, str] | Secret/handle lane, hidden from the Talker |
| `views` | dict[name, expr] | Computed, LLM-free reference data |
| `initial` | str | Starting checkpoint ref (`journey.checkpoint`) |

## 2. Checkpoints gate outcomes, not utterances

Within a checkpoint the conversation is free — the Talker speaks however the
moment requires. What is gated is *progression*: the ordered `advance_when`
rules decide when the conversation may move to another checkpoint.

Each rule is `{when, judge, to, requires?, set?}` (an `AdvanceRule`):

- **`judge: expr`** — a deterministic predicate over state, evaluated
  synchronously with no LLM round-trip, by the safe evaluator in
  `superdialog.playbook.expr`. Namespaces: `slots.*`, `results.*` (each
  result is `{ok, status, data, error}`), `env.*`, and — on a checkpoint
  that owns a pipeline — `pipeline.ok` / `pipeline.failed`. Helpers:
  `len, first, last, pluck, unique, min, max, any, all`. Missing data
  evaluates to `None` (falsy), never an error.

  ```yaml
  - {when: "results.availability.ok", judge: expr, to: booking.present}
  - {when: "slots.players >= 2 and slots.city", judge: expr, to: booking.quote}
  - {when: "len(pluck(results.availability.data.slots, 'time')) > 0",
     judge: expr, to: booking.present}
  ```

- **`judge: llm`** — an intent judgment. The Director makes ONE structured
  call per user utterance that does everything at once: extracts slot
  values, picks at most one `advance_when` target, optionally fires an
  interrupt, and writes a steering note for the Talker.

  ```yaml
  - {when: "caller confirmed the time works", judge: llm,
     to: booking.confirm, requires: [city, date, slot_id]}
  ```

Evaluation order on every user utterance: expr rules first, in author order
(first hit wins); only if none fires does the LLM verdict run. After any
event the runtime also *quiesces* — it keeps hopping through pipelines, expr
rules, and `auto` advances until nothing moves (bounded at 8 hops), so a
single utterance can extract → advance → run a pipeline → advance again.

**`requires` and gate semantics.** Every slot value carries a status,
`provisional` or `confirmed`. The gate decides which counts:

| | `gate: soft` (default) | `gate: hard` |
|---|---|---|
| `requires` met when | keys are *filled* (either status) | keys are *confirmed* |
| Director slot writes | `confirmed` | `provisional` |
| Talker behavior | never blocks | barriers on the Director |

Hard gates therefore need **pre-verdict confirmation**: a single (possibly
prompt-injected) Director verdict can never confirm its own `requires` and
advance through a hard gate in one shot, because its writes at a hard
checkpoint are provisional. `confirmed` status at a hard gate comes from
`set:` writes on a fired rule, from pipeline failure-context writes, or
from slots extracted on prior turns at soft checkpoints; tool results are
read via `results.*` in expressions and views, not as slots. When
`requires` is unmet, the Director does not
advance — it writes a steering note naming the missing keys so the Talker
asks for them naturally.

At a hard gate the Talker also **barriers**: it waits briefly (default
0.4 s) for the Director's verdict before speaking; past that it emits a
filler line, waits up to 5 s more, then degrades politely (§4). Soft
checkpoints never wait.

Two more checkpoint behaviors: `auto: true` speaks `say_verbatim` once and
advances to the first rule's target without user input (announce-then-move
patterns), and `terminal: true` ends the session on entry, recording
`outcome` in the final `SessionEndEvent`.

## 3. The process layer

Everything that is work rather than talk: tools, pipelines, middleware,
handlers, and policies. All of it runs Director-side; the Talker only ever
sees the results that templates and views choose to show.

**Tools.** `ToolSpec` templates (`url`, `headers`, `body` string values)
render in a sandboxed Jinja environment over three namespaces:

```
{{ slots.city }}        # extracted values
{{ env.ACCESS_TOKEN }}  # env lane — visible to tools, never to the Talker
{{ results.hold.data.hold_id }}   # prior results: {ok, status, data, error}
```

A 2xx response stores under `store_response_as`; `env_updates` then copies
values out of the response into env, each value a dotted path into the
response JSON (`{hold_id: data.hold_id}` for a `{"data": {...}}` envelope).
`run_once: true` caps the tool at one call per session; `when:` is an expr
that skips the call when falsy; `args` declares typed parameters coerced via
`SlotSpec`. Failures — non-2xx, timeouts, template errors — are recorded as
failed `ToolResultEvent`s, never crashes. Secret-shaped keys (token, key,
auth, …) and URL userinfo are redacted before the call lands in the event
log. For `type: python`, register the callable on the agent:

```python
async def lookup(args: dict, state) -> dict:
    return {"member": True}

agent = PlaybookAgent(..., python_tools={"member_lookup": lookup})
```

**Pipelines.** Ordered steps, each routing on its typed result via `on:`
keys `ok`, `failed`, or `http_<code>`. Each outcome is `continue` (next
step), a checkpoint ref, or a retry spec `{retry: N, on_exhaust: <ref>}`
(N capped at 10). A checkpoint with `pipeline:` runs it once per entry; the
result then drives the `pipeline.ok` / `pipeline.failed` expr rules. Retry
exhaustion and unrouted failures write an `error_context` slot and fall back
to the checkpoint's `on_failure` if no route was given — so failure paths
are declared, not improvised.

**Middleware.** `{on_status: 401, refresh_with: refresh_auth, then: replay}`
intercepts any pipeline step returning the status, runs the refresh tool,
and replays the step with the updated env — token rotation without a single
checkpoint knowing about it.

**Handlers.** Talker-less entries for the outside world. The host feeds
events through the runtime:

```python
from superdialog.playbook.events import ExternalEvent

await agent.runtime.on_external(
    ExternalEvent(kind="webhook", name="payment_captured", payload={...})
)
```

The handler whose `on:` matches `webhook.payment_captured` runs its
pipeline silently; any resulting advance lands in the log for the next
spoken turn.

**Silence policy.** Hosts report silence the same way:

```python
result = await agent.runtime.on_external(
    ExternalEvent(kind="silence", name="user_silence")
)
if result.prompt:
    ...  # play the re-prompt to the caller
```

The first `max_prompts` silences return the configured prompts in order;
after that the session routes to `then`. (No host adapter emits silence
events automatically yet — see §5 limitations.)

**Turn budgets.** `turn_budget: N` on a checkpoint injects a wrap-up
steering note once the user has spent more than N turns there; two grace
turns later the session routes to the checkpoint's `on_failure`.

## 4. Speech control

What the agent says is controlled at four levels, strongest first:

1. **`say_verbatim`** — the exact line, Jinja-rendered over
   `{slots, views, results}`, bypassing the Talker LLM entirely. Use for
   regulated or contractual speech. Surfaced at most once per checkpoint
   entry by the runtime; if the conversation lingers at a `say_verbatim`
   checkpoint the Talker generates follow-ups from guidance.
2. **`never_say`** — phrases rendered into the Talker's system block as an
   explicit prohibition.
3. **`guidance`** — prose direction for the checkpoint, templated over the
   same namespace. This is the main optimizable surface.
4. **`persona`** — the playbook-wide voice.

**What the Director injects.** Between turns the Director writes
*steering notes* ("Direction from supervisor" in the Talker's view): one or
two sentences of course-correction, e.g. naming unmet `requires` keys, or
the wrap-up nudge from a turn budget. After each turn the runtime also
checks for *repair notes* ("Correction from supervisor"): if the Talker
asked a question from a state version that a newer confirmed slot has
overtaken — re-asking something already answered — the next turn carries
"You already have city=Pune; acknowledge it instead of re-asking."

**Grounding.** Slots marked `authoritative: true` (prices, availability,
balances) can only be written by the Director or tools; the rendered view
ends with a standing instruction to state only facts present in *Known
information* (slots) or *Reference data* (views) and to say it is checking
otherwise. The env lane is never rendered, even if a view expression tries
to read it.

**Canned lines.** Three host-facing strings live on the `Talker`
(`superdialog.playbook.talker`): `FILLER` ("One moment, let me confirm
that…", spoken when a hard-gate barrier outlasts `barrier_timeout`),
`HOLD_LINE` (spoken if the Director is still silent after `hold_timeout`),
and `RECOVERY_LINE` (spoken when the Talker LLM fails twice). Localize them
via the `Talker(..., filler=, hold_line=, recovery_line=)` constructor
parameters; `PlaybookAgent` currently builds its Talker with the English
defaults (forwarding these is roadmap). The rendered view is packed under
`token_budget` (default 4000 estimated tokens): persona, guidance, notes,
slots, and views are protected; only older transcript turns are dropped.

## 5. Migrating a flow

Existing flow JSON keeps working on `DialogMachine` — nothing breaks. When
you want a flow on the Playbook engine, the compiler converts it losslessly
and proves it did:

```python
import json
from superdialog.flow.models import ConversationFlow
from superdialog.playbook import compile_flow, coverage_report

flow = ConversationFlow.model_validate(
    json.loads(open("golf_booking.json").read())
)
pb = compile_flow(flow)
report = coverage_report(flow, pb)
assert report.unmapped_nodes == []      # every node landed somewhere
assert report.unmapped_edges == []
assert report.unmapped_actions == []
print(report.dropped)   # informational buckets: what folded into what
print(report.notes)     # compiler judgment calls, worth reading once
```

The reference workload is the 61-node / 135-edge golf-course booking flow
(`tests/fixtures/flow/golf_booking.json`, 25 HTTP actions). It compiles —
with full coverage asserted in CI — into a single `main` journey: 25 tools,
13 dispatch entries, 2 handlers (`webhook.payment_captured`,
`timer.hold_expired`), a 2-prompt silence policy, a 401 auth-refresh
middleware, and a `global_goodbye` interrupt. What maps to what:

| Flow construct | Playbook construct |
|---|---|
| Conversational node | Checkpoint (`guidance` / `say_verbatim`) |
| Edge condition (intent prose) | `advance_when` rule, `judge: llm` |
| Edge condition (data predicate) | `advance_when` rule, `judge: expr` |
| Edge `input_schema` | Slot union + per-rule `requires` |
| Tool-free computational chain | Folded into the source's advance rules |
| Tool-bearing computational chain | `PipelineSpec` + synthetic intermediate checkpoint routing on `pipeline.ok/failed` |
| Hub router (≥4 exits) | `dispatch` entries + rules merged into inbound checkpoints |
| Silence nodes | `policies.silence` |
| Token-expiry global edge + refresh node | `middleware` |
| Other global edges | `interrupts` |
| Webhook/timer system nodes | `handlers` with single-step pipelines |
| `global_actions` | `tools` 1:1, templates rewritten to `{env, slots, results}` |
| `is_final` nodes | `terminal: true` + `outcome` |

Templates are rewritten from bare legacy names to the new namespaces
(`{{ACCESS_TOKEN}}` → `{{env.ACCESS_TOKEN}}`, `{{city}}` → `{{slots.city}}`,
`X.success` → `results.X.ok`). Only single-clause data predicates over known
`store_response_as` keys become expr rules; anything ambiguous stays a
`judge: llm` rule verbatim — lossless beats clever.

Known v1 limitations, stated honestly:

- **Self-loop suppression.** A folded chain edge that cycles back to its
  source checkpoint is suppressed (it would loop the rule fold) and
  surfaced as a `chain loop suppressed` note in the coverage report.
- **Dispatch is compile-time.** Hub routes are merged into each inbound
  checkpoint's `advance_when`; the `dispatch` block is organizational, not
  a runtime jump table.
- **Voice events are host-fed.** Silence/webhook/timer events work through
  `runtime.on_external`, but no adapter emits them automatically yet
  (roadmap, §7).
- **Deferred fields.** `interrupts.resume: true` restoration and tool
  `ttl_seconds` / `on_expire` are reserved, not yet active.
- Non-`on_enter` action triggers are not carried; the coverage report notes
  each occurrence.

## 6. Testing and evals

The event log is the substrate for all three layers.

**The log as audit artifact.** Every utterance, slot write, advance, tool
call/result (redacted), steering note, and degradation is a frozen,
versioned event. Round-trip it losslessly:

```python
from superdialog.playbook import EventLog

jsonl = agent.event_log.to_jsonl()        # persist per session
restored = EventLog.from_jsonl(jsonl)
agent2.load_event_log(restored)           # full-fidelity resume
```

**Replay for regression.** Re-run the Director over a recorded log under a
changed playbook or prompt and diff its decisions against what actually
happened — no simulated users needed:

```python
from superdialog.playbook import Playbook, replay

report = await replay(restored, Playbook.load("booking.yaml"), director_llm)
assert report.stable, report.diffs        # advances and slot writes match
```

`ReplayReport` counts `advance_matches` / `slot_matches` and lists each
`DecisionDiff` with the utterance version it diverged at. One known caveat:
quiescence-time slot writes (pipeline `error_slot`, expr `set:`) are
currently stamped as Director writes, so logs containing pipeline failures
can report diffs even under an identical playbook.

**Persona evals.** Drive scripted callers against a live agent and measure:

```python
from superdialog.playbook import PersonaSpec, run_eval

personas = [
    PersonaSpec(
        name="rusher",
        traits="impatient, gives all details at once",
        goal="book a tee time in Pune tomorrow",
        ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
    ),
]
report = await run_eval(make_agent, personas, user_llm, n=3)
print(report.completion_rate, report.mean_slot_accuracy)
```

`make_agent` is any `() -> PlaybookAgent` factory; `user_llm` is anything
with `async complete(messages) -> str` that plays the caller. Each
`SessionMetrics` carries completion, outcome, turns per checkpoint, slot
accuracy against the persona's ground truth (exact, no LLM judging), repair
and degradation counts, and the full session log as JSONL — so every failed
eval is immediately replayable.

For LLM-free unit tests, fold logs directly: `ConversationState.fold(log,
playbook)` is a pure function, as are the expr evaluator, the renderer, and
the compiler.

**Optimize: reflective prose improvement.** `superdialog optimize` runs the
eval loop above against a playbook, asks a candidate LLM for *targeted prose
edits*, and keeps only edits that win a paired evaluation:

```bash
superdialog optimize --playbook booking.yaml \
  [--rounds 3] [--n 1] [--personas personas.yaml] \
  [--llm openai/gpt-4o-mini] [--candidate-llm M] [--user-llm M] \
  [--out improved.booking.yaml]
```

Each round: REFLECT (worst sessions' evidence + the source YAML → a JSON
list of `{address, new_text}` edits) → APPLY (whitelist + recompile + Jinja
syntax check) → PAIR-EVAL (incumbent **and** candidate evaluated fresh in
the same round, so both face the same sampling noise) → ACCEPT only on a
strict same-round objective win. The output is the final incumbent, written
**in the source format** — full-format playbooks stay full, simple-format
playbooks stay simple.

Only prose is editable, enforced by construction. Full format: `persona`,
per-checkpoint `guidance`, `goal`, `never_say` (grow-only), `say_verbatim`
(only where present), slot `description`s, and `advance_when[].when` only
where `judge: llm`. Simple format: step `say`/`done_when`/`purpose`,
`opening`, `closing`, `persona.identity`, `persona.voice_style` — facts,
objections, and boundaries are never editable and survive every round.
`expr`-judged rules, dispatch, interrupts, silence prompts, and all
structure are frozen.

Personas resolve in order: `--personas` path → cached
`<playbook>.personas.yaml` beside the playbook → a generated 4-persona
suite (cooperative / terse / tangent-prone / error-making) written to that
cache for review. From Python:

```python
from superdialog.playbook import make_editable, optimize

doc = make_editable(open("booking.yaml").read())
report = await optimize(doc, personas=personas, candidate_llm=llm,
                        user_llm=llm, agent_factory=make_agent_for)
open("improved.yaml", "w").write(report.final_yaml)
```

`OptimizeReport` carries the initial/final objective breakdowns
(completion, slot accuracy, smoothness, repair rate), a per-round trace
with the exact edits applied, and an informational Pareto frontier. Cost:
each round ≈ 2 evals × personas × n sessions × ~2 LLM calls per turn, plus
one reflect call — the most expensive command in the tool; `n=1` with the
default 4-persona suite keeps dev runs reasonable.

## 8. Simple authoring format

The simple format is the easiest way to author a playbook: prose steps, a
nested persona, and reference data (facts, objections, boundaries, fallbacks).
`load_simple(path)` (or `superdialog chat --simple PATH`) compiles it to a
`Playbook` — the same validated artifact `compile_flow` produces from flows, so
it runs on the same Talker/Director runtime. v1 supports linear step sequences
only.

```yaml
name: "Tiny Booking Bot"
goal: "Book a haircut and confirm it."
persona:
  identity: "You are Mira, a booking assistant for Glow Studio."
  voice_style: "Warm and brief. One question at a time."
playbook:
  - {id: greet, purpose: "Open the call.", say: "Greet and ask how you can help.", done_when: "Caller is ready to book."}
  - {id: collect, purpose: "Get details.", say: "Ask for their name and service.", collect: [name, service], done_when: "Name and service captured."}
  - {id: confirm, purpose: "Confirm and close.", say: "Read back the booking and confirm.", done_when: "Caller confirmed."}
facts:
  canonical_pricing: {haircut: "₹400"}
objections:
  - {trigger: "Caller says it's too expensive.", handle: "Acknowledge and offer the cheapest option."}
boundaries: ["NEVER invent prices."]
```

| Simple key | Compiles to |
| --- | --- |
| `persona.identity` + `voice_style` + `goal` + `facts` + `objections` + `boundaries` + `fallback_actions` + `closing` | one rich `Playbook.persona` string |
| each `playbook` step | a `Checkpoint` in journey `main` |
| `step.id` | checkpoint id |
| `step.purpose` | `checkpoint.goal` |
| `step.say` | `checkpoint.guidance` |
| `step.collect` | `str` slots + the step rule's `requires` |
| `step.done_when` | the step's single `judge: llm` advance rule `when`, `to` the next step |
| last step | `terminal: true`, `outcome: closed` |
| `opening` | seeds the first step's guidance only if it has no `say` |

What's NOT in v1: linear step sequences only; objections live in persona prose
(not interrupts); all collected slots are `str`. See the plan's deferred list.

## 9. Roadmap

Shipped: `superdialog optimize` runs a reflective prose optimizer over the
eval substrate (§6) — paired-round acceptance, prose-only targeted edits,
simple-format round-trip, generated persona suites. **Structure mutation**
(checkpoint split/merge/reorder, slot-schema tightening) remains future, as
do GEPA-style frontier parent sampling, production-log feedback ingestion,
CI metric-threshold gates, and response caching across rounds.

Clearly future, not in this release: voice-event plumbing in the host
adapters (silence/barge-in events emitted into `runtime.on_external`
automatically); and sessionless webhook workers that load a persisted log,
apply a handler, and exit. Today's surface is what §1–§6 and §8 document.
