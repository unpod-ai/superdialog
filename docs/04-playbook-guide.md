# SuperDialog - Playbook Guide

**Status:** Canonical
**Parent:** [README.md](README.md)
**Audience:** Developers writing, migrating, and tuning playbooks.

---

This guide is in two deliberately separate parts:

- **Part 1 - Authoring: the playbook formats.** What you *write*: the
  simple format (start here), the full format (when you need precision),
  and how one maps onto the other.
- **Part 2 - Technical design: how the engine runs it.** What *happens*:
  the Talker/Director runtime, gating semantics, the process layer, speech
  control, and the testing/optimization substrate.

If you're stuck writing YAML, your answer is in Part 1. If your playbook
loads but the conversation behaves unexpectedly - doesn't advance, speaks
a filler line, re-asks a question - your answer is in Part 2. The two
parts meet at exactly one object: the validated `Playbook` artifact that
every format compiles into and the engine executes. There is **one
engine**; formats differ only in what they can express, never in how they
run (paired evals measured a dead quality tie and identical latency).

---

# Part 1 - Authoring: the playbook formats

## 1. Two formats, one engine

```
simple YAML ──(auto-detected)──▶ simple_to_playbook ─▶ Playbook ─▶ one runtime
full YAML ─────────────────────▶ Playbook.load ──────▶ (the IR)    (Talker+Director)
legacy flow JSON ──────────────▶ compile_flow ───────▶
```

`Playbook.load(path)` / `from_yaml` / `from_json` auto-detect the simple
format (a top-level `playbook:` list) and lower it at load time - callers
and the CLI never branch on format; `load_simple` and
`superdialog chat --simple PATH` remain as explicit routes.

**Which format when:**

| You need… | Use |
| --- | --- |
| A linear conversation: greet → qualify → close, with early exits | **Simple** - less YAML, no hand-wired transitions |
| Tools/pipelines, hard gates, typed slots, multiple journeys or outcomes | **Full** |
| To keep authoring simple AND `superdialog optimize` output in your format | **Simple** (optimize round-trips it) |
| An existing flow JSON on the new engine | `compile_flow` (§4) |

## 2. The simple format

Prose steps, a structured persona, and reference data as real YAML.

```yaml
goal: "Book a haircut and confirm it."
persona:
  name: Mira
  language: ["en", "hi"]
  voice_style: "Warm and brief. One question at a time."
  identity: "You are Mira, a booking assistant for Glow Studio."
opening: "Greet the caller warmly."
closing: "Thank them and say goodbye."
playbook:
  - id: greet
    purpose: "Open the call."
    say: "Greet the caller and ask how you can help."
    done_when: "Caller is ready to book."
  - id: collect
    purpose: "Get the booking details."
    say: "Ask for their name and preferred service."
    collect: [name, service]
    done_when: "Name and service are captured."
  - id: confirm
    purpose: "Confirm and close."
    say: "Read back the booking and confirm."
    done_when: "Caller has confirmed."
facts:
  canonical_pricing: {haircut: "₹400"}
boundaries: ["NEVER invent prices."]
interrupts:
  - {when: "Caller says goodbye or asks to end the call.", to: main.confirm}
```

### Section reference

**`name`** (string, optional) - a human-readable title. Metadata only;
not folded into the compiled artifact.

**`goal`** (string) - the call's mission statement. Folds into the persona
as `Overall goal: …`. Write it as the success definition: what makes this
call a win, including acceptable fallbacks.

**`persona`** (mapping) - compiles into one rich `Playbook.persona` string
the Talker sees every turn:

- `identity` - who the agent is; the verbatim first paragraph and the
  highest-leverage prose in the file.
- `name` - folded as `Your name is <name>.` only when the identity prose
  doesn't already mention it.
- `language` - a name (`English`), an ISO 639-1 code (`hi`), or a list of
  either (`["en", "hi"]`): first entry is the default, the rest fold as
  `Also speaks: …`. The code map covers 59 common languages; unmapped
  values pass through as written. Quote the Norwegian code (`"no"`) -
  unquoted YAML parses it as a boolean.
- `voice_style` - folded as `Voice & manner: …`: tone, pacing, sentence
  length, language-switching rules.

**`opening`** (string, optional) - seed guidance for the **first step
only, and only when that step has no `say`**. Prefer putting the opening
line in the first step's `say` and omitting this.

**`closing`** (string, optional) - folds into the persona under
`## Closing line`. An *instruction*, not an auto-spoken line - pair it
with a final step whose `say` tells the agent to deliver it.

**`playbook`** (list of steps) - each step becomes a `Checkpoint` in a
single journey named `main`, chained **linearly in list order**: step N's
`done_when` advances to step N+1; the last step is `terminal: true,
outcome: closed`. Reordering the list re-wires the chain - there are no
hand-written `to:` targets to maintain. Per step:

- `id` - the checkpoint id; addressable as `main.<id>` in logs, metrics,
  and replay.
- `purpose` - compiles to `Checkpoint.goal`. Director-facing context: what
  this step is *for*. One sentence.
- `say` - compiles to `Checkpoint.guidance`, the Talker's playbook for the
  step. May contain Jinja over `{slots, views, results}`. This is the
  prose `superdialog optimize` mutates most.
- `collect` - slot keys to capture; compiled to untyped (`str`) slots
  **plus** the advance rule's `requires`. ⚠️ All collected keys gate
  advancement together: a 3-slot step needs all three extracted before the
  conversation moves - measured in evals as the single biggest source of
  stalls. Prefer 1–2 slots per step.
- `done_when` - compiles to a single `judge: llm` advance rule the
  Director judges each turn. Write an observable condition ("Caller has
  confirmed a day and time"), not an intention.

**`facts`** (mapping, optional) - folds under `## Reference facts (never
invent beyond these)`. The agent's grounding data: pricing, amenities,
policies. It lives in the persona (not `env`) deliberately - the `env`
lane is never rendered to the Talker, so facts must ride the persona to
stay visible during speech. Keep it canonical; anything here is recited.

**`objections`** (list of `{trigger, handle}`, optional) - folds as
`## Objection handling` bullets. Prose-level steering, not control flow:
handled *within* the current step; they cannot re-route the journey.

**`boundaries`** (list of strings, optional) - folds as `## Hard
boundaries`. Compliance-critical "NEVER…" rules. Prose-enforced; the full
format's `never_say` is the stronger mechanism.

**`fallback_actions`** (mapping `{name: instruction}`, optional) - folds
as `## Fallback actions`: what to do when the happy path fails (callback,
message, reschedule, do-not-call). Pair with an `interrupts:` entry that
routes there, or the instructions have no path to fire on.

**`interrupts`** (list of `{id?, when, to}`, optional) - global jumps,
judged from any step: when the Director sees `when` matching, the
conversation re-routes to the `to` step (`main.<id>` ref, validated at
load). Compiles to engine interrupts with `judge: llm`, `resume: false`;
ids default to `interrupt_<n>`. **Use at least a goodbye interrupt** - in
a 56-session assessment, linear playbooks with no early exit never
completed a single call (a satisfied or busy caller loops until the turn
cap), while the same playbook with goodbye/busy interrupts completed 8/8.

### What the simple format cannot express

| Engine feature | Why it matters |
| --- | --- |
| Multiple terminals / outcomes | One `closed` outcome can't distinguish booked vs callback vs DNC in metrics. |
| `gate: hard`, pipelines, tools | Transactional steps (holds, payments) with barriered speech (Part 2 §6). |
| `judge: expr` rules | Machine-evaluated transitions - zero LLM cost, zero latency. |
| Typed/required slots, `never_say`, `say_verbatim`, silence policy, multi-journey, dispatch | Precision controls. |

When you need any of these, move to the full format. The escape hatch is
one-way: compile your simple file (`Playbook.load(...)` then
`yaml.safe_dump(pb.model_dump(exclude_defaults=True))`) and continue
authoring the result; there is no decompiler back.

## 3. The full format

Everything the engine can do, stated explicitly. A complete, annotated
example:

```yaml
persona: "You are Asha, a friendly golf-course booking assistant."

env:                          # plumbing lane - NEVER rendered to the Talker
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
        gate: hard            # outcomes barrier on the Director here (Part 2 §6)
        say_verbatim: "Held until {{ views.hold_valid_until }}."  # no LLM
        pipeline: confirm_and_hold   # process layer runs on entry (Part 2 §7)
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
  hold_timeout: 4.0           # post-filler wait for a slow Director (Part 2 §8)

initial: booking.collect      # defaults to the first checkpoint anyway
```

Top-level fields, all on `superdialog.playbook.models.Playbook`:

| Field | Type | Purpose |
|---|---|---|
| `persona` | str | System-level voice of the agent, every Talker turn |
| `journeys` | dict[name, Journey] | Named checkpoint sequences (min 1) |
| `dispatch` | list[DispatchEntry] | Intent→entry table (compile-time in v1, §4) |
| `tools` | list[ToolSpec] | Declarative HTTP / registered python tools |
| `pipelines` | list[PipelineSpec] | Ordered tool steps with typed branches |
| `handlers` | list[HandlerSpec] | `webhook.<name>` / `timer.<name>` triggers |
| `interrupts` | list[InterruptSpec] | Global jumps (`judge: llm` or `event`) |
| `policies` | Policies | Silence handling; `hold_timeout` (default 4.0 s) |
| `middleware` | MiddlewareSpec | `on_status` → refresh tool → replay |
| `env` | dict[str, str] | Secret/handle lane, hidden from the Talker |
| `views` | dict[name, expr] | Computed, LLM-free reference data |
| `initial` | str | Starting checkpoint ref (`journey.checkpoint`) |

Every cross-reference (rule targets, pipeline ids, tool ids, `requires`
keys) is validated at load time.

**How simple maps onto full** - useful when graduating a file:

| Simple key | Compiles to |
| --- | --- |
| `persona.*` + `goal` + `facts` + `objections` + `boundaries` + `fallback_actions` + `closing` | one rich `persona` string |
| each `playbook` step | a `Checkpoint` in journey `main` |
| `step.purpose` / `step.say` | `goal` / `guidance` |
| `step.collect` | `str` slots + the step rule's `requires` |
| `step.done_when` | a `judge: llm` rule, `to` the next step |
| last step | `terminal: true`, `outcome: closed` |
| `interrupts[{when, to}]` | `InterruptSpec` (`judge: llm`, `resume: false`) |
| `opening` | first step's guidance, only if it has no `say` |

## 4. Migrating a legacy flow

Flow JSON needs no migration step to *run*: the unified loader detects it
(`nodes` + `initial_node`) and compiles it onto the Playbook engine
automatically - `superdialog chat --flow legacy.json` just works, and
`--mode flow` opts back into the original `DialogMachine` runtime when you
want the graph engine itself. For a permanent conversion, the compiler is
explicit and proves its coverage:

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
(`tests/fixtures/flow/golf_booking.json`, 25 HTTP actions). It compiles -
with full coverage asserted in CI - into a single `main` journey: 25 tools,
13 dispatch entries, 2 handlers, a 2-prompt silence policy, a 401
auth-refresh middleware, and a `global_goodbye` interrupt. What maps to
what:

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
`judge: llm` rule verbatim - lossless beats clever.

Known v1 limitations, stated honestly:

- **Self-loop suppression.** A folded chain edge that cycles back to its
  source checkpoint is suppressed (it would loop the rule fold) and
  surfaced as a `chain loop suppressed` note in the coverage report.
- **Dispatch is compile-time.** Hub routes are merged into each inbound
  checkpoint's `advance_when`; the `dispatch` block is organizational, not
  a runtime jump table.
- **Voice events are host-fed.** Silence/webhook/timer events work through
  `runtime.on_external`, but no adapter emits them automatically yet
  (roadmap, §10).
- **Deferred fields.** `interrupts.resume: true` restoration and tool
  `ttl_seconds` / `on_expire` are reserved, not yet active.
- Non-`on_enter` action triggers are not carried; the coverage report notes
  each occurrence.

## 5. Tooling, whichever format you author in

- `superdialog chat --playbook X.yaml` - REPL; auto-detects both formats.
- `superdialog optimize --playbook X.yaml` - reflective prose optimizer
  (Part 2 §9); **emits improved YAML in your source format**. For simple
  files it edits `say`/`done_when`/`purpose`/`opening`/`closing`/
  `persona.identity`/`persona.voice_style`; facts, objections, boundaries,
  and interrupt conditions are never touched.
- Persona evals and replay (Part 2 §9) operate on the compiled artifact;
  metrics key on `journey.<id>` checkpoint ids (`main.<step id>` for
  simple-origin files).

---

# Part 2 - Technical design: how the engine runs it

A **Playbook** declares a conversation as journeys of checkpoints plus a
process layer (tools, pipelines, handlers, interrupts, policies). At
runtime a fast **Talker** LLM streams every spoken turn while an async
**Director** extracts slots, judges advancement, and runs tools - both
over an append-only event log that doubles as the audit and replay
artifact. Within each turn they run **concurrently**: the Talker speaks
from a pre-decision snapshot while the Director settles in parallel, so
per-turn latency is max(Talker, Director), not the sum. The one
synchronization point is a hard gate (§6).

## 6. Checkpoints gate outcomes, not utterances

Within a checkpoint the conversation is free - the Talker speaks however the
moment requires. What is gated is *progression*: the ordered `advance_when`
rules decide when the conversation may move to another checkpoint.

Each rule is `{when, judge, to, requires?, set?}` (an `AdvanceRule`):

- **`judge: expr`** - a deterministic predicate over state, evaluated
  synchronously with no LLM round-trip, by the safe evaluator in
  `superdialog.playbook.expr`. Namespaces: `slots.*`, `results.*` (each
  result is `{ok, status, data, error}`), `env.*`, and - on a checkpoint
  that owns a pipeline - `pipeline.ok` / `pipeline.failed`. Helpers:
  `len, first, last, pluck, unique, min, max, any, all`. Missing data
  evaluates to `None` (falsy), never an error.

  ```yaml
  - {when: "results.availability.ok", judge: expr, to: booking.present}
  - {when: "slots.players >= 2 and slots.city", judge: expr, to: booking.quote}
  - {when: "len(pluck(results.availability.data.slots, 'time')) > 0",
     judge: expr, to: booking.present}
  ```

- **`judge: llm`** - an intent judgment. The Director makes ONE structured
  call per user utterance that does everything at once: extracts slot
  values, picks at most one `advance_when` target, optionally fires an
  interrupt, and writes a steering note for the Talker.

  ```yaml
  - {when: "caller confirmed the time works", judge: llm,
     to: booking.confirm, requires: [city, date, slot_id]}
  ```

Evaluation order on every user utterance: expr rules first, in author order
(first hit wins); only if none fires does the LLM verdict run. After any
event the runtime also *quiesces* - it keeps hopping through pipelines, expr
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
advance - it writes a steering note naming the missing keys so the Talker
asks for them naturally.

At a hard gate the Talker also **barriers**: it waits briefly (default
0.4 s) for the Director's verdict before speaking; past that it emits a
filler line, then waits up to the playbook's `policies.hold_timeout`
(default 4 s, must be > 0) before degrading politely (§8). Soft
checkpoints never wait - Talker and Director run fully concurrent there.

Two more checkpoint behaviors: `auto: true` speaks `say_verbatim` once and
advances to the first rule's target without user input (announce-then-move
patterns), and `terminal: true` ends the session on entry, recording
`outcome` in the final `SessionEndEvent`.

## 7. The process layer

Everything that is work rather than talk: tools, pipelines, middleware,
handlers, and policies. All of it runs Director-side; the Talker only ever
sees the results that templates and views choose to show.

**Tools.** `ToolSpec` templates (`url`, `headers`, `body` string values)
render in a sandboxed Jinja environment over three namespaces:

```
{{ slots.city }}        # extracted values
{{ env.ACCESS_TOKEN }}  # env lane - visible to tools, never to the Talker
{{ results.hold.data.hold_id }}   # prior results: {ok, status, data, error}
```

A 2xx response stores under `store_response_as`; `env_updates` then copies
values out of the response into env, each value a dotted path into the
response JSON (`{hold_id: data.hold_id}` for a `{"data": {...}}` envelope).
`run_once: true` caps the tool at one call per session; `when:` is an expr
that skips the call when falsy; `args` declares typed parameters coerced via
`SlotSpec`. Failures - non-2xx, timeouts, template errors - are recorded as
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
to the checkpoint's `on_failure` if no route was given - so failure paths
are declared, not improvised.

**Middleware.** `{on_status: 401, refresh_with: refresh_auth, then: replay}`
intercepts any pipeline step returning the status, runs the refresh tool,
and replays the step with the updated env - token rotation without a single
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
events automatically yet - see §10 limitations.)

**Turn budgets.** `turn_budget: N` on a checkpoint injects a wrap-up
steering note once the user has spent more than N turns there; two grace
turns later the session routes to the checkpoint's `on_failure`.

## 8. Speech control

What the agent says is controlled at four levels, strongest first:

1. **`say_verbatim`** - the exact line, Jinja-rendered over
   `{slots, views, results}`, bypassing the Talker LLM entirely. Use for
   regulated or contractual speech. Surfaced at most once per checkpoint
   entry by the runtime; if the conversation lingers at a `say_verbatim`
   checkpoint the Talker generates follow-ups from guidance.
2. **`never_say`** - phrases rendered into the Talker's system block as an
   explicit prohibition.
3. **`guidance`** - prose direction for the checkpoint, templated over the
   same namespace. This is the main optimizable surface.
4. **`persona`** - the playbook-wide voice.

**What the Director injects.** Between turns the Director writes
*steering notes* ("Direction from supervisor" in the Talker's view): one or
two sentences of course-correction, e.g. naming unmet `requires` keys, or
the wrap-up nudge from a turn budget. After each turn the runtime also
checks for *repair notes* ("Correction from supervisor"): if the Talker
asked a question from a state version that a newer confirmed slot has
overtaken - re-asking something already answered - the next turn carries
"You already have city=Pune; acknowledge it instead of re-asking."

**Grounding.** Slots marked `authoritative: true` (prices, availability,
balances) can only be written by the Director or tools; the rendered view
ends with a standing instruction to state only facts present in *Known
information* (slots) or *Reference data* (views) and to say it is checking
otherwise. The env lane is never rendered, even if a view expression tries
to read it.

**Canned lines and timeouts.** Three host-facing strings live on the
`Talker` (`superdialog.playbook.talker`): `FILLER` ("One moment, let me
confirm that…", spoken when a hard-gate barrier outlasts `barrier_timeout`,
default 0.4 s), `HOLD_LINE` (spoken if the Director is still silent after
the hold window - the playbook's `policies.hold_timeout`, default 4 s, or
an explicit `PlaybookAgent(hold_timeout=…)` override), and `RECOVERY_LINE`
(spoken when the Talker LLM fails twice). Localize them via the
`Talker(..., filler=, hold_line=, recovery_line=)` constructor parameters;
`PlaybookAgent` currently builds its Talker with the English defaults
(forwarding these is roadmap). The rendered view is packed under
`token_budget` (default 4000 estimated tokens): persona, guidance, notes,
slots, and views are protected; only older transcript turns are dropped.

## 9. Testing, evals, and optimization

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
happened - no simulated users needed:

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
and degradation counts, and the full session log as JSONL - so every failed
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
**in the source format** - full-format playbooks stay full, simple-format
playbooks stay simple.

Only prose is editable, enforced by construction. Full format: `persona`,
per-checkpoint `guidance`, `goal`, `never_say` (grow-only), `say_verbatim`
(only where present), slot `description`s, and `advance_when[].when` only
where `judge: llm`. Simple format: step `say`/`done_when`/`purpose`,
`opening`, `closing`, `persona.identity`, `persona.voice_style` - facts,
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
one reflect call - the most expensive command in the tool; `n=1` with the
default 4-persona suite keeps dev runs reasonable.

## 10. Roadmap

Shipped: `superdialog optimize` (reflective prose optimizer - paired-round
acceptance, prose-only targeted edits, simple-format round-trip, generated
persona suites); simple-format `interrupts`; the unified loader;
configurable `policies.hold_timeout`. **Structure mutation** in optimize
(checkpoint split/merge/reorder, slot-schema tightening) remains future, as
do GEPA-style frontier parent sampling, production-log feedback ingestion,
CI metric-threshold gates, and response caching across rounds.

Clearly future, not in this release: voice-event plumbing in the host
adapters (silence/barge-in events emitted into `runtime.on_external`
automatically); simple-format sugar for multiple terminal outcomes; and
sessionless webhook workers that load a persisted log, apply a handler,
and exit. Today's surface is what Parts 1–2 document.
