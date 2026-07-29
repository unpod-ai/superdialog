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
  `Also speaks: …`. The code map (`playbook/simple.py::_LANG_NAMES`)
  covers the common ISO 639-1 codes; unmapped values pass through as
  written. Quote the Norwegian code (`"no"`) -
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
hand-written `to:` targets to maintain. When a step must leave the chain
(a fallback that should close the call, a pitch that branches on the
answer), `then:`, `terminal:`, and `branches:` route it explicitly.
Per step:

- `id` - the checkpoint id; addressable as `main.<id>` in logs, metrics,
  and replay.
- `purpose` - compiles to `Checkpoint.goal`. Director-facing context: what
  this step is *for*. One sentence.
- `say` - compiles to `Checkpoint.guidance`, the Talker's playbook for the
  step. May contain Jinja over `{slots, views, results}`. This is the
  prose `superdialog optimize` mutates most.
- `collect` - slot keys to capture; compiled to untyped (`str`) slots.
  Whether they **gate advancement** depends on the step's shape: a focused
  capture step (**≤2 slots**) requires all of them filled before the
  Director may advance (plus a free deterministic expr rule fires the
  moment they are all filled - compiled **only at the default
  `entity: caller`**, because the expr `slots.*` namespace is
  caller-scoped, so a step with any other `entity:` advances on the LLM
  judge alone: `playbook/simple.py::simple_to_playbook`); a **branchy step
  collecting >2 per-path alternatives requires none** - demanding all of a
  14-slot category qualifier's slots deadlocks the step, since no caller
  fills every branch. Override the heuristic with `require:`.
- `require` (optional) - explicit subset of `collect` that gates the
  advance, when the auto heuristic guesses wrong (e.g. one mandatory key
  on an otherwise-branchy step: `require: [inquiry_category]`).
- `done_when` - compiles to a single `judge: llm` advance rule the
  Director judges each turn. Write an observable condition ("Caller has
  confirmed a day and time"), not an intention.
- `then` (optional) - step id to advance to when `done_when` holds,
  instead of the next list element. This is how fallback steps close the
  call rather than falling into whatever happens to follow them:

  ```yaml
  - id: advisor_callback
    say: "Capture a callback time, then close politely."
    collect: [callback_time]
    done_when: "Callback time captured."
    then: deliver_closing          # not the next step in the list
  ```

  Unknown targets and self-targets are rejected at load.
- `terminal` (optional, default false) + `outcome` (optional, default
  `closed`) - marks the step as a call ending: compiles to a terminal
  checkpoint with no advance rules, recording `outcome` on `SessionEnd`.
  Previously only the *last* list element could end the call, which made
  fallback endings (callback scheduled, DNC respected) structurally
  unclosable and metrically indistinguishable. `outcome` on a
  non-terminal step is rejected - it would be silently ignored at
  runtime.
- `branches` (optional) - multi-way routing, judged by the Director in
  author order and ahead of the `done_when` default. Each entry
  `{when, to, requires?}` compiles to one `judge: llm` advance rule:

  ```yaml
  - id: pitch_site_visit
    say: "Offer the site visit; handle the objection on hesitation."
    done_when: "Customer clearly accepts or is open to the visit."
    branches:
      - when: "Customer firmly declines after the objection was handled."
        to: advisor_callback
  ```

  A branch may not target the step's default next step (use `done_when`
  for that path - the branch would shadow its slot gate), may not target
  itself, and terminal steps cannot have branches.
- `then_say` (optional) - a line to deliver *while advancing out of* this
  step, rendered (Jinja over slots) and injected as a one-shot steer.
  Use it for post-capture speech: guidance written after the capture in
  `say` is unreachable, because the deterministic expr rule advances the
  instant the collected slots fill. Spoken **only on a Director rule
  advance**: `playbook/runtime.py::PlaybookRuntime._emit_exit_say`
  allowlists rule ids prefixed `llm:` / `expr:`, so interrupts, policy
  advances (including the turn-budget backstop below), `resume` returns,
  supervisor redirects, and `auto` exits all skip it - detours and
  failures never get the happy-path pitch. Rejected on terminal steps
  (they never advance out).
- `turn_budget` (optional, default 4) - user turns on this step before the
  runtime steers "wrap this step up". The steer is not the end of it: two
  grace turns later (`runtime.py::_TURN_BUDGET_GRACE`)
  `playbook/runtime.py::PlaybookRuntime._apply_turn_budget`
  **force-advances**, so a call can never wedge on an advance gate the
  Director cannot satisfy. Simple playbooks compile no `on_failure`, so
  the forced target is the **journey-order successor**
  (`Playbook.next_checkpoint_id`), logged as a `DegradedEvent`
  (`turn_budget_forced:<checkpoint>`) plus an advance with rule
  `policy:turn_budget_forced`. Two consequences to author around: a step
  routed with `then:` can be pushed to its *list neighbour*, which is not
  where its happy path goes; and only the journey's last step is exempt,
  having no successor. Graduate to the full format and declare
  `on_failure` when the list neighbour is the wrong place to land.
- `kb` (optional) - whether this step's Talker prompt carries the (large)
  `facts.knowledge_base`. Unset = legacy heuristic (the step's `say`
  mentions `knowledge_base`). Set `kb: false` on steps that merely
  *reference* the KB to keep them lean - off-step KB questions route
  through a KB-answer step via a `global_kb_query`-style interrupt.
- `gate` (optional, `hard` | `soft`; default `hard`) - the Talker's sync
  barrier on this step. Hard makes the Talker wait for the Director's
  verdict so it speaks from post-advance state; `soft` speaks immediately,
  saving the Director's settle time on pure-talk steps that capture
  nothing. Collected slots are compiled with slot-level `gate: soft`
  regardless, so a filled slot satisfies `requires` without a separate
  confirmation round-trip (`playbook/simple.py::_step_to_checkpoint`).
- `entity` (optional, default `caller`) - whose details this step collects
  when one call covers more than one person. Must match
  `[a-z_][a-z0-9_]*`; requires top-level `multi_entity: true` to change
  storage behavior (§3).

**Voice and runtime knobs** (optional, top-level) - the simple format
spells at top level what the full format puts in its `guidelines:` block
(§3); `playbook/simple.py::simple_to_playbook` copies them across:

| Key | Default | Effect |
| --- | --- | --- |
| `channel` | `voice` | `text` suppresses the baseline speaking-style block |
| `tone` | `professional` | Tone injected with the voice block |
| `call_type` | unset | `sales` / `support` / `booking` domain pattern block |
| `timezone` | `UTC` | IANA tz for the per-turn date anchor |
| `memory_enabled` | `false` | Guard beside a prior-call summary, when one is present |
| `followup_enabled` | `false` | Follow-ups & callbacks block |
| `multi_entity` | `false` | Scope slot storage/lookup per step `entity` |
| `supervisor` | `false` | Enable the Loop-2 Supervisor (Part 2 §7) |

`persona.language` supplies `guidelines.language`. Everything else is
full-format only: the top-level `llm:`, `pronunciations:`, and `policies:`
blocks, per-slot typing, `strict`, and tool tiers.

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
format's `never_say` is the stronger mechanism: authored `never_say`
phrases are deterministically excised from the Talker's token stream
before TTS, whatever the LLM emits.

**`fallback_actions`** (mapping `{name: instruction}`, optional) - folds
as `## Fallback actions`: what to do when the happy path fails (callback,
message, reschedule, do-not-call). Pair with an `interrupts:` entry that
routes there, or the instructions have no path to fire on.

**`interrupts`** (list of `{id?, when, to, resume?}`, optional) - global
jumps, judged from any step: when the Director sees `when` matching, the
conversation re-routes to the `to` step (`main.<id>` ref, validated at
load). Compiles to engine interrupts with `judge: llm`; ids default to
`interrupt_<n>`. `resume: true` (default false) makes the jump a *detour*:
the step it left is pushed on a resume stack and the conversation returns
there once the detour ends without firing another interrupt
(`playbook/runtime.py::PlaybookRuntime.on_user_text`).
**Use at least a goodbye interrupt** - in
a 56-session assessment, linear playbooks with no early exit never
completed a single call (a satisfied or busy caller loops until the turn
cap), while the same playbook with goodbye/busy interrupts completed 8/8.

### Unknown keys fail loudly

Keys the format does not recognize - a typo'd `done_wehn`, an invented
top-level `language_lock:` section - **raise at load** with the dotted
path of every offender, instead of being silently dropped by pydantic
(configuration theater: the author believes it is configured; the runtime
never sees it). `simple_to_playbook(doc, strict=False)` /
`load_simple(path, strict=False)` downgrade the error to a warning for
live loaders that must not kill a call over a stale authored file.

### What the simple format cannot express

| Engine feature | Why it matters |
| --- | --- |
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
            type: str         # str|int|float|bool|date|time|enum|array|object
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
| `multi_entity` | bool | Scope slot storage/lookup per checkpoint `entity` (default false) |
| `guidelines` | GuidelineConfig | Voice/channel knobs and the Supervisor opt-in (below) |
| `llm` | LLMConfig \| None | The model this playbook declares (below); `None` keeps the host-supplied models |
| `pronunciations` | list[PronunciationSpec] | Authored pronunciation rules, exported to the host voice runtime (below) |
| `journeys` | dict[name, Journey] | Named checkpoint sequences (min 1) |
| `dispatch` | list[DispatchEntry] | Intent→entry table (compile-time in v1, §4) |
| `tools` | list[ToolSpec] | Declarative HTTP / registered python tools |
| `pipelines` | list[PipelineSpec] | Ordered tool steps with typed branches |
| `handlers` | list[HandlerSpec] | `webhook.<name>` / `timer.<name>` triggers |
| `interrupts` | list[InterruptSpec] | Global jumps (`judge: llm` or `event`) |
| `policies` | Policies | Silence handling; `hold_timeout` (default 4.0 s); authored `filler` / `hold_line` barrier lines (Part 2 §8) |
| `middleware` | MiddlewareSpec | `on_status` → refresh tool → replay |
| `env` | dict[str, str] | Secret/handle lane, hidden from the Talker |
| `views` | dict[name, expr] | Computed, LLM-free reference data |
| `knowledge_base` | str | Global KB text (Jinja-renderable) injected into the Talker prompt of KB steps (`uses_kb` / simple `kb`) |
| `initial` | str | Starting checkpoint ref (`journey.checkpoint`) |
| `source_path` | str | Set by `Playbook.load()`; empty when built from text |

Every cross-reference (rule targets, pipeline ids, tool ids, `requires`
keys) is validated at load time. Per-field types and defaults for
`Checkpoint`, `SlotSpec`, and `ToolSpec` live in
[02-api-reference.md](02-api-reference.md#the-artifact-model); this guide
covers what changes how you *author*.

### `guidelines:` — voice and channel configuration

`guidelines:` is an optional top-level block that controls how the runtime
injects speaking-style instructions. All fields have defaults, so existing
playbooks that omit the block are unaffected.

| Field | Type | Default | Description |
|---|---|---|---|
| `channel` | `voice` \| `text` | `voice` | Delivery channel. `voice` injects the baseline TTS/speaking-style block; `text` suppresses it. |
| `tone` | `professional` \| `casual` | `professional` | Speaking tone injected as part of the voice block. |
| `language` | str or list[str] | `en` | ISO 639-1 code(s) or language names. A non-English value adds the Language & Accent and Hinglish-examples blocks. |
| `call_type` | `sales` \| `support` \| `booking` | — | Adds a domain-specific pattern block (`## Pre-Sales Flows`, `## Customer Support Flows`, or `## Appointment Booking Flows`) to the voice guidelines. Omit the key to disable it - `call_type: none` parses as the string `"none"` and fails validation. |
| `timezone` | IANA tz string | `UTC` | The timezone used for the per-call date/time anchor injected into every Talker turn. |
| `memory_enabled` | bool | `false` | When `true`, injects a "Using Past Context" guard beside the conversation summary when one is present. |
| `followup_enabled` | bool | `false` | When `true`, injects the Follow-ups & Callbacks block. |
| `gender` | `male` \| `female` \| `neutral` | `neutral` | Pins the agent's own gendered verb/adjective forms (करूँगी vs करूँगा) instead of letting the model guess from the persona name. Emitted only for a non-English `language`; `neutral` emits nothing. |
| `supervisor` | bool | `false` | Opt into the Loop-2 Supervisor - a trajectory-level reviewer that runs off the speech path (Part 2 §7). |

**Voice channel behavior.** For `channel: voice` (the default), a baseline
voice/TTS guideline block — covering one-thought-per-turn speaking style,
conversational leadership, tone, and (when applicable) language/accent and
domain patterns — is appended after the persona in every Talker system
prompt. The block is session-constant and lands in the stable cache prefix
alongside the persona and the date anchor. For `channel: text` the block is
suppressed entirely.

**Date/time anchor and date slots.** A `## CURRENT DATE & TIME` line (derived
from `timezone`) is injected into every Talker turn so the agent resolves
relative references ("tomorrow", "next Monday") to exact absolute dates.
Slots declared with `type: date` default to `gate: hard` — the slot must be
confirmed before it satisfies a rule's `requires` check. Set `gate: soft` on
the slot to opt out of this default.

**Handover checkpoints.** A checkpoint with `handover: true` causes the
handover summary instruction to appear in that turn's system prompt: the agent
is instructed to hand over a 1–2 sentence neutral summary (caller name, reason
for calling, request) when transferring to a human.

**Deprecated: `guidelines.director_model` / `guidelines.talker_model`.**
Both still parse, but no host ever read them back. Setting either without a
top-level `llm:` block *emits* a `DeprecationWarning`
(`playbook/models.py::Playbook._warn_deprecated_llm_fields` calls
`warnings.warn`) - the playbook still loads. Declare models in `llm:`
instead (§ below).

### `llm:` — the model the playbook declares

Optional. `None` (the default) preserves today's behavior byte-for-byte: the
host's session model drives both roles, unvalidated.

```yaml
llm:
  provider: openai
  model: gpt-4.1-mini
  director:                     # optional split; unset ⇒ Director shares the talker model
    provider: openai
    model: gpt-4.1-nano
```

`Playbook.llm_uri()` and `.director_llm_uri()` read it back as
`provider/model` URIs. `await pb.resolve_llm_providers(override=...)` does the
whole job in one call - priority (an explicit `override` wins, warning when it
shadows a declared `llm:`; neither present raises), a live availability check
per model (a confirmed-invalid model raises, an unverifiable one warns), and
resolution to provider instances.

### Fields beyond the annotated example

Four shipped authoring surfaces the example above does not show. Types and
defaults: [02-api-reference.md](02-api-reference.md#the-artifact-model).

**`strict: true` on a checkpoint** - never paraphrase. `say_verbatim` alone
still lets the Talker generate follow-ups if the conversation lingers on the
checkpoint; `strict` removes that escape hatch, so the step speaks its
rendered `say_verbatim` and nothing else. A `strict` checkpoint with no
`say_verbatim` authored speaks the Talker's recovery line rather than
improvising (`playbook/talker.py::Talker.speak`). Use it for regulated
disclosures.

**`resolve_from` on a slot** - for the one value a caller never utters
verbatim: an opaque id. The Director is handed the live candidate list from a
prior tool result and renders `"<name>" -> <id>` pairs into its verdict
prompt, so it maps the spoken name (tolerating STT drift) to the canonical id
itself - no fuzzy matching and no domain alias table in the engine:

```yaml
slots:
  course_id:
    type: str
    resolve_from:
      result: courses        # a tool's store_response_as key
      list_field: items      # path under results.courses.data
      name_field: name       # the human-spoken field
      id_field: id           # the canonical id to write into the slot
```

The block renders only when the named result is present and its list is
non-empty; the Director is instructed to omit the slot when no candidate
clearly matches, never to invent an id.

**`multi_entity: true` + checkpoint `entity`** - one call that collects for
more than one person (caller and partner, policyholder and nominee). Off (the
default) slots are stored bare and everything behaves exactly as before. On,
each checkpoint's slots are stored under its `entity` prefix; the Talker's
"Known information" block groups them by entity, `{{ slots.dob }}` resolves to
the *active* entity's value overlaying the caller's, and the Director's prompt
is prefixed with "You are collecting details for: `<entity>`" so it never asks
"whose date of birth?" (`playbook/render.py::_active_slots`,
`playbook/render.py::_known_info`).

**`pronunciations:`** - authored TTS pronunciation rules, respelling-first
(`respelling` is the operative field; `ipa` is advisory):

```yaml
pronunciations:
  - {word: "Cartesia", respelling: "kar-TEE-zha"}
  - {word: "Sure", language: "hi", respelling: "श्योर", context: force}
```

> SuperDialog validates and carries these; it never applies them. It emits
> text, not audio. `PronunciationSpec.to_rule()` maps each entry onto the
> field names a host voice runtime's pronunciation manager expects
> (`respelling` → `replacement`), which is the whole integration contract.

**How simple maps onto full** - useful when graduating a file:

| Simple key | Compiles to |
| --- | --- |
| `persona.*` + `goal` + `facts` + `objections` + `boundaries` + `fallback_actions` + `closing` | one rich `persona` string |
| each `playbook` step | a `Checkpoint` in journey `main` |
| `step.purpose` / `step.say` | `goal` / `guidance` |
| `step.collect` | `str` slots + the step rule's `requires` |
| `step.done_when` | a `judge: llm` rule, `to` the next step (or `step.then`) |
| `step.branches` | `judge: llm` rules ahead of the `done_when` rule, in author order |
| `step.then_say` | `Checkpoint.exit_say` - a one-shot steer on the advance out |
| `step.kb` | `Checkpoint.uses_kb` |
| `step.gate` | `Checkpoint.gate` (default `hard`); collected slots always compile with slot-level `gate: soft` |
| `step.entity` | `Checkpoint.entity` |
| last step / `step.terminal` | `terminal: true`, `outcome:` from the step (default `closed`) |
| `interrupts[{when, to}]` | `InterruptSpec` (`judge: llm`, `resume:` from the entry) |
| `opening` | first step's guidance, only if it has no `say` |
| `channel` / `tone` / `call_type` / `timezone` / `memory_enabled` / `followup_enabled` / `supervisor` + `persona.language` | the `guidelines:` block |
| `multi_entity` | `Playbook.multi_entity` |
| `facts.knowledge_base` | `Playbook.knowledge_base` (YAML-dumped) |

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
- **Deferred fields.** Tool `ttl_seconds` / `on_expire` are reserved model
  fields with no runtime consumer. (`interrupts.resume: true` restoration
  has since shipped - see §2.)
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

| | `gate: soft` | `gate: hard` (default) |
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

The gate is per checkpoint **and** per slot: `SlotSpec.gate` overrides the
checkpoint's for one key (`None` inherits it), and a `type: date` slot
defaults to `gate: hard` on its own. A turn barriers when the checkpoint is
hard *or* any of its slots is (`playbook/talker.py::Talker._is_gated`).

At a hard gate the Talker also **barriers**: it waits `barrier_timeout` for
the Director's verdict before speaking; past that it emits a filler line,
then waits up to the playbook's `policies.hold_timeout` (default 4 s, must
be > 0) before degrading politely (§8). Soft checkpoints never wait -
Talker and Director run fully concurrent there.

> `barrier_timeout` has two different defaults depending on the entry
> point: **0.4 s** via `Talker` and the `DialogMachine` facade, **4.0 s**
> via `PlaybookAgent` directly - a 10x longer silence before the filler.
> Pass `barrier_timeout=0.4` explicitly when constructing a `PlaybookAgent`
> yourself. See [01-architecture.md](01-architecture.md) §3.4.

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
turns later (`runtime.py::_TURN_BUDGET_GRACE`) the session routes to the
checkpoint's `on_failure`. That route is a **hard backstop, not an
opt-in**: when no `on_failure` is declared - the case for every
simple-format step -
`playbook/runtime.py::PlaybookRuntime._apply_turn_budget` force-advances
to the journey-order successor (`Playbook.next_checkpoint_id`) instead,
recording a `DegradedEvent` (`turn_budget_forced:<checkpoint>`) and an
advance with rule `policy:turn_budget_forced`. The design intent is that a
call can never livelock on a gate the Director cannot satisfy; the price
is that the forced target is list order, not the step's own `then:`
routing. Only the journey's last checkpoint is exempt, having no
successor. Declare `on_failure` wherever the successor is the wrong place
to land.

### Reversibility: tool tiers, rewind, compensation

Some tool calls change the world. `ToolSpec.tier` classifies how badly:

| `tier` | Meaning | On rewind across it |
| --- | --- | --- |
| `reversible` | Pure state (a read) | Undone structurally |
| `compensable` | Real side effect that `compensate: <tool id>` can undo | Needs caller confirmation, then the compensation tool runs |
| `irreversible` | Cannot be undone | Rewind is refused |

`tier` is optional and **unannotated playbooks behave exactly as before**:
`ToolSpec.effective_tier` guesses conservatively at rewind time (safe HTTP
methods → reversible, everything else → compensable-without-`compensate`,
so a rewind refuses) and never activates the interception guard. Load-time
validation: `compensate` requires `tier: compensable`, may not name the
tool itself, and must reference a declared tool.

```yaml
tools:
  - {id: hold_slot, method: POST, url: "...", tier: compensable,
     compensate: release_hold}
  - {id: release_hold, method: POST, url: "..."}
  - {id: charge_card, method: POST, url: "...", tier: irreversible}
```

**Rewind** (`playbook/runtime.py::PlaybookRuntime.rewind`) restores
conversation *state* to a past log version. Speech is irreversible, so
nothing is deleted: a `RevertEvent` marks the version range superseded and
the fold skips those state effects while every utterance stays in the
transcript. It returns a `RewindOutcome` with status `done`,
`needs_confirmation` (compensable effects in range - the pending tool ids
come back with it), or `refused` (an irreversible effect, or a compensable
one with no `compensate` tool). Compensation tools run *before* the revert
so their templates still see the results about to be reverted (a hold id,
say). A `repair_note` lands as a repair steer so the Talker acknowledges
the correction instead of resetting the conversation.

**Interception.** Explicitly tiered tools are additionally guarded *before*
they fire (`playbook/runtime.py::PlaybookRuntime._intercept`), with no
check at all for `reversible` or untiered ones. Neither tier may fire while
a repair is in flight. Beyond that, a `compensable` tool needs the
checkpoint's required slots to meet the same per-slot gate the Director
uses to advance; an `irreversible` one needs every required slot
*confirmed*, plus - when the host passes
`PlaybookAgent(intercept_llm=...)` - a single fast classifier call that
fails **closed**.

### Loop 2 — the Supervisor (opt-in)

The Director judges one utterance at a time and structurally cannot see
failures that unfold *across* turns. The Supervisor
(`playbook/supervisor.py::Supervisor`) is a trajectory-level reviewer that
runs after each completed turn, off the speech path. Enable it with
`guidelines: {supervisor: true}` (top-level `supervisor: true` in the
simple format) - it then reuses the Director's model - or hand a dedicated
model to `PlaybookAgent(supervisor_llm=...)`, which wins over the
playbook's opt-in. Unset, nothing runs and behavior is unchanged.

Cost is proportional to trouble: `detect_triggers` is a pure function over
the folded state and costs nothing, and an LLM verdict is spent only when
one fires - and then only off a 2-turn cooldown, which a pending
compensation confirmation bypasses. The triggers:

| Trigger | Fires when |
| --- | --- |
| `repair_streak` | ≥2 repair notes since the checkpoint was entered |
| `oscillation` | A full A,B,A,B round trip in advance targets |
| `turn_budget` | User turns exceed the checkpoint's `turn_budget` |
| `degraded` | A `DegradedEvent` since the checkpoint was entered |
| `slot_churn:<key>` | ≥3 distinct Director-written values for one slot |
| `repeated_interrupt:<id>` | The same interrupt fired ≥2 times |
| `compensation_pending` | A rewind is awaiting the caller's confirmation |

The verdict maps to five verbs, all landing as ordinary events at the turn
boundary: **inject** a repair brief, **redirect** to another checkpoint,
**rewind** wrong state, **discard** the current detour, or **handover** to
a human. Two guardrails are structural, not prompt-hope: a redirect to a
terminal checkpoint is blocked (ending a call is the Director's job), and a
rewind that fires compensation only honors the verdict's `confirmed` flag
when a confirmation was actually surfaced to the caller on the previous
turn. A third is a floor rather than a ceiling: a `turn_budget` trigger may
not resolve to a no-op, so a passive verdict is replaced with a forward
steer toward the checkpoint's goal before the runtime's hard backstop has
to force-advance. A malformed verdict records a `DegradedEvent` and is
dropped - it never disturbs the call.

## 8. Speech control

What the agent says is controlled at four levels, strongest first:

1. **`say_verbatim`** - the exact line, Jinja-rendered over
   `{slots, views, results}`, bypassing the Talker LLM entirely. Use for
   regulated or contractual speech. Two paths surface it: the runtime
   speaks it as pass-through at most once per checkpoint entry
   (`playbook/runtime.py::PlaybookRuntime._speak_verbatim`, deduped against
   what was already said since that entry), and the Talker yields it
   instead of streaming whenever it speaks from a `say_verbatim`
   checkpoint - so a caller who lingers there hears the line again, never
   an improvisation. `strict: true` extends the same discipline to a
   checkpoint with *no* `say_verbatim`: it speaks the recovery line rather
   than generating one (§3).
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
`Talker` (`playbook/talker.py`): `FILLER` ("One moment, let me confirm
that…", spoken when a hard-gate barrier outlasts `barrier_timeout`),
`HOLD_LINE` (spoken if the Director is still silent after the hold window -
the playbook's `policies.hold_timeout`, default 4 s, or an explicit
`PlaybookAgent(hold_timeout=…)` override), and `RECOVERY_LINE` (spoken when
the Talker LLM fails twice).

The first two are **authorable in the playbook**, which is how a
non-English persona localizes them without a host wiring Python overrides:

```yaml
policies:
  hold_timeout: 4.0
  filler: "एक सेकंड, मैं देख लेती हूँ…"
  hold_line: "थोड़ा समय लग रहा है — लाइन पर बने रहिए।"
```

Resolution order, per `playbook/agent.py::PlaybookAgent.__init__`: an
explicit `PlaybookAgent(filler=…, hold_line=…)` argument wins, else the
playbook's `policies.filler` / `policies.hold_line`, else the Talker's
built-in English defaults. Because the resolution happens inside
`PlaybookAgent`, authored lines also reach playbooks run through the
`DialogMachine` facade, which has no filler arguments of its own.

The two **`PlaybookAgent` arguments** accept a `SpokenLine`
(`playbook/talker.py::SpokenLine`) - a plain string, or a
`Callable[[ConversationState], str]` called with the live state at speak
time (a language-aware filler). A provider that raises degrades to the
built-in default; a filler must never kill the turn. The **authored YAML
fields are plain strings only**: `playbook/models.py::Policies` declares
`filler: str | None` and `hold_line: str | None`, so anything but a string
under `policies:` fails validation at load. `recovery_line` stays a
`Talker` constructor parameter only.

The rendered view is packed under `token_budget` (default 4000 estimated
tokens): persona, guidance, notes, slots, and views are protected; only
older transcript turns are dropped.

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

## 10. Status and roadmap

**Shipped.** `superdialog optimize` (reflective prose optimizer - paired-round
acceptance, prose-only targeted edits, simple-format round-trip, generated
persona suites); simple-format `interrupts` including `resume: true`
detours; simple-format routing (`then:`, `terminal:`/`outcome:`,
`branches:`, `then_say:`) with strict unknown-key validation; the unified
loader; configurable `policies.hold_timeout`; and the unversioned wave this
guide documents - `guidelines:` voice configuration, the global
`knowledge_base`, authored `policies.filler` / `policies.hold_line`, the
top-level `llm:` block, `multi_entity` + checkpoint `entity`,
`pronunciations:`, `strict`, `resolve_from`, tool tiers with rewind and
compensation, and the Loop-2 Supervisor. Today's surface is what Parts 1-2
document.

> **Status: roadmap — not built.** Optimize **structure mutation**
> (checkpoint split/merge/reorder, slot-schema tightening), GEPA-style
> frontier parent sampling, production-log feedback ingestion, CI
> metric-threshold gates, and response caching across rounds; voice-event
> plumbing in the host adapters (silence/barge-in events emitted into
> `runtime.on_external` automatically); and sessionless webhook workers
> that load a persisted log, apply a handler, and exit.
