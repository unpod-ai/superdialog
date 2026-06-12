# The Simple Authoring Format — Section-by-Section Reference

The simple format is the friendliest way to author a playbook: prose steps,
a structured persona, and reference data as real YAML instead of folded
strings. It is **purely an authoring surface** — `simple_to_playbook`
lowers it at load time into the same validated `Playbook` artifact that
full-format YAML and `compile_flow` produce, and the engine
(Director/Talker/runtime) never knows which format you wrote. There is no
performance difference between the formats: paired evals measured a dead
tie in quality, and the lowering is a one-time, millisecond-scale compile.

Loading is automatic everywhere: `Playbook.load` / `from_yaml` / `from_json`
detect a top-level `playbook:` list and lower it; `superdialog chat` and
`superdialog optimize` accept simple files directly, and `optimize` writes
its improved output back **in simple format**.

## Minimal example

```yaml
goal: "Book a haircut and confirm it."
persona:
  name: Mira
  language: English
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
```

## Section reference

### `name` (string, optional)

A human-readable title for the file. **Metadata only** — it is not folded
into the compiled artifact. Use it for repository hygiene.

### `goal` (string)

The call's mission statement. Folds into the persona as `Overall goal: …`,
so both the Talker (speech) and the conversation framing carry it every
turn. Write it as the success definition: what makes this call a win,
including the acceptable fallbacks.

### `persona` (mapping)

Compiles into one rich `Playbook.persona` string the Talker sees on every
turn. Four fields:

- **`identity`** — who the agent is, verbatim first paragraph of the
  persona. The highest-leverage prose in the file.
- **`name`** — the agent's name. Folded as `Your name is <name>.` **only
  when the identity prose doesn't already mention it** (no duplication).
- **`language`** — a language name (`English`), an ISO 639-1 code (`hi`),
  or a list of either (`["en", "hi"]`). The first entry is the default,
  the rest are also spoken; codes map to readable names. Folds as
  `Default conversation language: English. Also speaks: Hindi.` The code
  map covers the [Soniox translation
  set](https://soniox.com/docs/translation/supported-languages) (59
  languages); unmapped values pass through as written. In YAML, quote the
  Norwegian code (`"no"`) — unquoted it parses as a boolean.
- **`voice_style`** — folded as `Voice & manner: …`. Tone, pacing,
  sentence length, language-switching rules.

### `opening` (string, optional)

Seed guidance for the **first step only, and only when that step has no
`say` of its own**. If your first step has a `say`, `opening` is unused —
prefer putting the opening line directly in the first step's `say` and
omitting this field.

### `closing` (string, optional)

Folds into the persona under `## Closing line`. It is an *instruction*,
not an auto-spoken line — pair it with a final step whose `say` tells the
agent to deliver it (see `deliver_closing` in the realestate example).

### `playbook` (list of steps) — the journey

Each step becomes a `Checkpoint` in a single journey named `main`,
chained **linearly in list order**: step N's `done_when` advances to step
N+1; the last step is `terminal: true, outcome: closed`. Reordering the
list re-wires the chain automatically — there are no hand-written `to:`
targets to maintain.

Per-step fields:

- **`id`** — the checkpoint id; addressable as `main.<id>` in logs,
  metrics (`turns_per_checkpoint`), and replay.
- **`purpose`** — compiles to `Checkpoint.goal`. Director-facing context:
  *what this step is for*. Keep it one sentence.
- **`say`** — compiles to `Checkpoint.guidance`, the Talker's playbook for
  the step: what to say and how. May contain Jinja over
  `{slots, views, results}`. This is the prose `superdialog optimize`
  mutates most.
- **`collect`** — slot keys to capture, compiled to untyped (`str`),
  optional slots **plus** the advance rule's `requires` list. ⚠️ All
  collected keys gate advancement together: a 3-slot step needs all three
  extracted before the conversation moves on — measured in evals as the
  single biggest source of stalls. Prefer 1–2 slots per step; split big
  collections across steps.
- **`done_when`** — compiles to a single `judge: llm` advance rule: the
  Director judges this prose against the conversation each turn. Write it
  as an observable condition ("Caller has confirmed a day and time"), not
  an intention.

### `facts` (mapping, optional)

Folded into the persona under `## Reference facts (never invent beyond
these)` as YAML. This is the agent's grounding data — pricing tables,
amenities, policies. It lives in the persona (not `env`) deliberately:
the `env` lane is never rendered to the Talker, so facts must ride the
persona to stay visible during speech. Anything here is recited verbatim
risk — keep it canonical and current.

### `objections` (list of `{trigger, handle}`, optional)

Folded as `## Objection handling` bullets (`If <trigger> -> <handle>`).
These are prose-level steering, not control flow: the agent handles the
objection *within the current step*. They cannot re-route the journey.

### `boundaries` (list of strings, optional)

Folded as `## Hard boundaries`. Compliance-critical "NEVER…" rules.
Prose-enforced only — the engine's `never_say` (full format) is the
stronger mechanism. Note: `superdialog optimize` never edits facts,
objections, or boundaries; they are frozen by construction.

### `fallback_actions` (mapping of `{name: instruction}`, optional)

Folded as `## Fallback actions`. Describes *what to do* when the happy
path fails (callback, message, reschedule, do-not-call) — but provides no
*path* for it: the linear chain has no branch to a fallback step, which
is the format's main structural limit (below).

## What the simple format cannot express

The lowering targets a deliberately small subset of the engine. Not
available, by design:

| Engine feature | Why it matters |
| --- | --- |
| `interrupts` (goodbye / busy re-routing) | Without an early exit, a satisfied or busy caller loops to the turn cap — in our 56-session assessment, **no simple-origin session ever completed**, while a structure-enriched variant completed 8/8 (ΔCO +0.42). |
| Multiple terminals / outcomes | One `closed` outcome can't distinguish booked vs callback vs DNC. |
| `gate: hard`, pipelines, tools | Transactional steps (holds, bookings) with barriered speech. |
| `judge: expr` rules | Machine-evaluated transitions — zero LLM cost. |
| Typed/required slots, `never_say`, `say_verbatim`, silence policy, multi-journey, dispatch | Precision controls. |

**When you need any of these, switch to the full format** (or wait for
the planned `on_goodbye:`/`on_busy:` sugar, which will compile to
interrupts). The escape hatch is one-way today: compile your simple file
(`Playbook.load(...)` + `yaml.safe_dump(pb.model_dump(exclude_defaults=True))`)
and continue authoring the result; there is no decompiler back.

## Tooling that understands the format natively

- `superdialog chat --simple X.yaml` (or auto-detect via `--flow`/`--playbook`).
- `superdialog optimize --playbook X.simple.yaml` — edits **simple-format
  fields** (`say`, `done_when`, `purpose`, `opening`, `closing`,
  `persona.identity`, `persona.voice_style`) and emits improved simple
  YAML; facts/objections/boundaries are never touched.
- Persona evals (`run_eval`) and replay operate on the compiled artifact —
  metrics key on `main.<step id>` checkpoint ids.
