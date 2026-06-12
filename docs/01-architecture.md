# SuperDialog — Architecture

**Status:** Canonical
**Parent:** [README.md](README.md)
**Purpose:** Internal design of the framework. Two engines, one contract:
the checkpoint-compound Playbook runtime (the default — the unified loader
runs playbooks, simple-format files, and compiled flow JSON on it) and the
legacy graph-railed DialogMachine (explicit opt-in, `--mode flow`).

---

## 1. System overview

One Python package. No services, no daemons. Everything in-process.

SuperDialog now ships **two conversation engines** behind the same `Agent`
protocol (`superdialog.agent.Agent`: `turn` / `assist` / `chat_ctx` /
`load_chat_ctx`). Hosts, sessions, and adapters do not know which engine they
are driving.

```
                       Host platforms
        LiveKit · PipeCat · FastAPI · WebSocket · CLI
                            │
                  superdialog.adapters            (thin shims)
                            │
                      SessionWorker               (sessions, stores, locks)
                            │
       Agent protocol — turn / assist / chat_ctx / load_chat_ctx
              ┌─────────────┴─────────────┐
       DialogMachine                PlaybookAgent
       (Engine A)                   (Engine B)
              │                            │
     DialogStateMachine            PlaybookRuntime ── EventLog
     flow graph · TransitionGate   Talker ∥ Director · pipelines
     CriteriaJudge · FlowContext   expr rules · policies · tools
```

- **Engine A — DialogMachine.** The stable, graph-railed state machine. A
  flow graph decides every transition; the LLM speaks within the rails.
  Fully supported; existing flows keep working unchanged.
- **Engine B — Playbook.** The checkpoint compound runtime for fluid
  conversations. Checkpoints gate *outcomes*, not utterances; a fast Talker
  streams every spoken turn while an async Director extracts, judges, and
  steers over an event-sourced log. This is where new investment goes.
- **The bridge.** `compile_flow` converts existing flow JSON into Playbooks
  losslessly (§4), so migration is a compile step, not a rewrite.

Package layout:

```
superdialog/
  ├─ flow/                # Flow graph: nodes, edges, serialization
  ├─ machine/             # DialogStateMachine engine (Engine A internals)
  ├─ dialog_machine.py    # Public DialogMachine facade
  ├─ playbook/            # Playbook engine (Engine B) — models, events,
  │                       #   runtime, talker, director, compiler, replay
  ├─ agent.py             # Agent Protocol + TurnResult
  ├─ agents/              # LLMAgent, LangChainAgent (non-DM brains;
  │                       #   LangChainAgent imports from
  │                       #   superdialog.agents.langchain_agent,
  │                       #   `langchain` extra required)
  ├─ session/             # Session, SessionHandle, SessionWorker, stores
  ├─ chat_context.py      # ChatContext, ChatMessage (LiveKit-aligned)
  ├─ llm/                 # Model URI resolver and provider adapters
  ├─ tools/               # Python / HTTP / MCP tool wrappers
  ├─ cli/                 # `superdialog chat / flow lint / draw / generate`
  └─ adapters/            # LiveKit, PipeCat, FastAPI, WebSocket
```

Shared substrate — `SessionWorker` (one agent per session, pluggable
`SessionStore`), the model URI resolver (`openai/gpt-5.1`,
`anthropic/claude-opus-4-7`, `custom/...`), tools, adapters, and the CLI —
is documented in [02-api-reference.md](02-api-reference.md) and
[03-embedding-guides.md](03-embedding-guides.md). Note one seam difference:
`DialogMachine` takes a model URI; `PlaybookAgent` takes two small LLM
protocols (`StreamsLLM` for the Talker, `CompletesLLM` for the Director),
so any provider — or a scripted fake in tests — plugs in directly.

## 2. Engine A — DialogMachine (graph-railed)

The mature engine. A `ConversationFlow` is a directed graph: nodes (states,
each with an instruction or static text), edges (transitions with
natural-language conditions), `global_edges`, and `actions` (declarative HTTP
calls). The public `Flow` facade loads/saves it as version-controllable JSON;
`create_dialog_flow(prompt=..., llm=...)` bootstraps one from a prompt (the
LLM is used at construction time only, never at runtime).

```python
dialog_machine = DialogMachine(flow=flow, llm="anthropic/claude-opus-4-7")
reply = await dialog_machine.turn("hello")           # complete Turn
stream = await dialog_machine.turn("hello", stream=True)  # StreamChunk iter
```

Internals (`superdialog.machine`):

| Component | Role |
|---|---|
| `DialogStateMachine` | Runtime core; two execution models: criteria-based `process_turn(user_input)` (CriteriaJudge picks the edge) and tool-call-based `apply_transition(edge_id)` (the host LLM's tool callback names the edge) |
| Adapters (`TextAdapter`, `ToolCallAdapter`, `LLMAdapter`) | How the LLM is consulted per turn — compose a reply, or pick an edge via a tool call |
| `CriteriaJudge` | LLM-based node-completion evaluator for the criteria path |
| `TransitionGate` | Validates every transition, in order: edge valid → node content spoken → self-loop limit → completion criteria (slots) → user spoke → CriteriaJudge verdict |
| `FlowContext` | Mutable state bag that travels with the machine: `ConversationData` (history, variables) + `MachineState` (current node, visit counts, intent stack) + legacy `node_slots` |

Flexibility on this engine is rail-shaped: `__stay_on_node__`, global edges
with an intent stack, fallback edges, skip/auto-proceed handling. Each turn
costs a route decision plus a speak call. That is exactly the failure mode
the Playbook engine was designed to remove — users do not follow graphs.

`DialogMachine` remains fully supported: `turn` / `reset` / `set_llm` /
`switch_flow` / `assist`, FlowSets, streaming, sessions, and all four host
adapters. Full signatures and worked examples:
[02-api-reference.md](02-api-reference.md).

## 3. Engine B — Playbook (checkpoint compound)

Design rationale: [the checkpoint compound architecture design
doc](plans/2026-06-10-checkpoint-compound-architecture-design.md).
Source: `src/superdialog/playbook/`. Public surface: `superdialog.playbook`
(`Playbook`, `PlaybookAgent`, `EventLog`, `ConversationState`,
`compile_flow`, `coverage_report`, `replay`, `run_eval`, …).

### 3.1 The artifact

A **Playbook** (`playbook/models.py`) is the authored, git-diffable artifact,
loaded from YAML or JSON via `Playbook.load(path)` with full cross-reference
validation (unknown checkpoint/pipeline/tool ids, undeclared `requires` keys,
duplicate ids, and the reserved `pipeline` result key all fail at load time).

Two layers:

- **Conversation layer** — `journeys` of `Checkpoint`s plus a `persona` and
  a reusable `dispatch` table. A checkpoint is a call-center-script unit:
  `goal`, typed `slots` (`SlotSpec`), `guidance` prose (Jinja over
  `{slots, views, results}`), an ordered `advance_when` rule list
  (`AdvanceRule`: `when` / `judge: llm|expr` / `to` / `requires` / `set`),
  `gate: soft|hard`, optional `say_verbatim`, `never_say`, `auto`,
  `on_failure`, `terminal` + `outcome`, `turn_budget`.
- **Process layer** — everything that is *not* conversation: `tools`
  (`ToolSpec`: templated HTTP or registered python, `store_response_as`,
  `run_once`, `when:`, `env_updates`, timeout), `pipelines` (`PipelineSpec`:
  ordered steps with typed `on: {ok | failed | http_<code>}` branches and
  capped `RetrySpec`), `handlers` (`HandlerSpec`: webhook/timer-triggered
  pipelines), `interrupts` (`InterruptSpec`), `policies` (silence), optional
  auth `middleware` (`on_status: 401 → refresh_with → replay`), an `env`
  lane, and computed `views` (LLM-free expressions).

One worked checkpoint (an excerpt; `to:` targets live elsewhere in the file):

```yaml
journeys:
  booking:
    checkpoints:
      - id: collect_details
        goal: "Have city, course preference, date, and party size"
        slots:
          city:    {type: str, required: true,
                    invalidates: [availability_result]}
          date:    {type: date, required: true}
          players: {type: int, required: true}
        guidance: |
          Collect naturally; the caller may give everything in one
          breath or nothing. Known cities: {{ views.registered_cities }}.
        advance_when:
          - {when: "details complete and caller picked a course",
             judge: llm, to: booking.availability,
             requires: [city, date, players]}
          - {when: "caller asks what courses exist",
             judge: llm, to: course_info.list_city, requires: [city]}
        gate: soft
        turn_budget: 6
```

How each field behaves at runtime: within the checkpoint the conversation is
free — the Talker speaks from `guidance`, and the caller may answer in any
order. The Director extracts `city`/`date`/`players` into the typed slots
(an `invalidates:` write clears stale downstream data on change-of-mind).
The rules are ordered and multi-way: the Director judges the `llm` rules and
may only fire one whose `requires` are met. `gate: soft` means provisional
slot values suffice and the Talker never blocks; past `turn_budget` the
runtime injects a wrap-up steering note, then routes to `on_failure` after a
grace window.

### 3.2 The event-sourced log

Every mutation is an event; state is a fold; the log is the audit artifact.

`playbook/events.py` defines twelve frozen, versioned pydantic events:
`utterance`, `slot_write`, `advance`, `steering_note`, `tool_call`,
`tool_result`, `env_write`, `scratchpad`, `summary`, `external`
(silence/webhook/timer), `degraded`, `session_end`. `EventLog` is
append-only with contiguous versions stamped from 1; it serializes to JSONL
(`to_jsonl` / `from_jsonl`) and is the single persistence payload.

`ConversationState.fold(log, playbook)` (`playbook/state.py`) is a pure
function from log to snapshot: transcript, slots, env, tool results,
steering note, summary, checkpoint position, silence/turn counters, ended +
outcome. Fold semantics encode the lane rules: slot values carry
`provisional | confirmed` status and never downgrade; `authoritative` slots
ignore Talker writes; `invalidates` is applied non-transitively and skipped
when a write re-asserts the same value.

Because the log is the artifact, replay is free: `replay(log, playbook,
director_llm)` (`playbook/replay.py`) re-runs the Director over each recorded
user utterance and diffs its decisions against what was recorded
(`ReplayReport`) — regression evidence for prompt or model changes.
`eval_bridge.py` (`PersonaSpec`, `run_session`, `run_eval`) drives persona
self-play sessions and scores checkpoint completion, slot accuracy, and
turns-per-checkpoint from the same logs.

### 3.3 The compound runtime — one turn

`PlaybookAgent` (`playbook/agent.py`) implements the `Agent` protocol, so
`SessionWorker` and every host adapter run it unchanged — and streaming is
real (tokens leave as the Talker produces them). Internally it composes
`PlaybookRuntime` (event log owner + quiescence conductor,
`playbook/runtime.py`), `Talker` (`playbook/talker.py`), and `Director`
(`playbook/director.py`).

A user turn, in order:

1. **User text arrives** via `await agent.turn(text, stream=True)`. The
   agent snapshots the current state (version *N*) for the Talker.
2. **Director starts concurrently** in a cancellation-shielded task:
   `runtime.on_user_text(text)` appends the `UtteranceEvent`, then
   `Director.evaluate(state)` makes **one structured LLM call** that does
   three jobs — extract slot values into the checkpoint's typed schema,
   judge the `llm` advance rules and interrupts, and write a 1–3 sentence
   **steering note** for the Talker's next context ("user already gave the
   date; nudge toward time selection").
3. **Talker streams concurrently** from snapshot *N*:
   `render_view(pb, state, token_budget)` packs persona → guidance →
   steering note → slots → computed views → summary → recent transcript,
   and one streaming call sends tokens straight to the host (TTS). At a
   hard gate it barriers first (§3.4).
4. **Quiescence.** After the Director's decision is applied, the runtime
   hops (bounded by `max_hops=8`) until nothing moves: the entered
   checkpoint's **pipeline** runs (`PipelineRunner.run`, with typed
   branches, capped retries, and 401-refresh-replay middleware), then
   **`judge: expr` rules** are evaluated synchronously in the fold — no LLM
   round-trip, which is what makes compiled router chains instant — then
   **`auto` checkpoints** speak their verbatim line and advance, and a
   **terminal** checkpoint appends `SessionEndEvent` with its `outcome`.
   `say_verbatim` lines crossed during quiescence surface as pass-through
   speech after the Talker's stream.
5. **Join and repair.** The Talker's speech is logged exactly once as an
   `UtteranceEvent` stamped `spoke_from_version=N`; `runtime.check_repairs()`
   compares that stamp against later confirmed slot writes and, if the
   Talker re-asked for something already answered, appends a **repair**
   steering note — the Talker self-corrects next turn instead of silently
   accumulating drift.
6. **Done chunk** carries `Turn` metadata: `checkpoint`, state `version`,
   `ended`, and `outcome` when terminal.

```
 user text ──► PlaybookAgent.turn
     │ snapshot state (version N)
     ├────────────► Director task (shielded; survives barge-in)
     │                append UtteranceEvent
     │                Director.evaluate: slots · advance · steering note
     │                quiesce: pipeline → expr rules → auto hops → terminal
     ▼
 Talker.speak(state@N)
     │ soft gate: stream immediately
     │ hard gate: barrier ≤0.4s ─miss─► filler ─► wait ≤5s ─miss─► hold line
     │ tokens ────────────────────────────────────────────► host / TTS
     ▼
 join (Director done) ─► log speech (spoke_from_version=N)
     ─► check_repairs ─► pass-through verbatim ─► done {checkpoint, outcome}
```

Barge-in is safe by construction: aborting the stream cancels *speech*, not
the state machine. The Director runs to completion in a shielded scope,
partial Talker speech is logged exactly once, and `check_repairs` still runs.

External events use the same log: hosts call
`runtime.on_external(ExternalEvent(...))` for silence (silence policy:
re-prompt up to `max_prompts`, then route), webhooks, and timers (matched
`HandlerSpec` pipelines run without the Talker).

### 3.4 Gates, barrier, degradation

**Soft gates never block.** Provisional slot values satisfy `requires`; the
Talker streams immediately; correctness converges via the Director.

**Hard gates** (payments, identity, anything irreversible) buy correctness
at the moments it matters:

- `requires` must be **confirmed**, not provisional
  (`ConversationState.confirmed`).
- The Talker **barriers**: `Talker.speak(state, director_done=...)` waits up
  to `barrier_timeout` (0.4s) for the quiescent post-verdict state, emits
  the natural filler (`FILLER`) if exceeded, waits up to `hold_timeout`
  (5s), and emits `HOLD_LINE` if the Director never lands — politely
  degraded, never hung.
- `say_verbatim` bypasses the Talker LLM entirely (template → speech) for
  regulated lines; `never_say` lists are injected as renderer constraints.

**Degradation ladder** — the session never dies with a model:

| Failure | Behavior |
|---|---|
| Director LLM error / bad JSON | `DegradedEvent(component="director")` appended; Talker continues solo; LLM-free policies (turn budget, silence) still apply; slots settle later |
| Talker stream failure | One instant retry, then the canned `RECOVERY_LINE` |
| Hard-gate barrier miss | Filler, then hold line (above) |
| Quiescence hop exhaustion | `DegradedEvent(detail="quiesce_hop_exhaustion")` — a runaway hop loop is audited, never spun |
| Tool/pipeline failure | A failed `ToolResultEvent` plus a typed `error_context` slot; declarative `retry` / `on_exhaust` / `on_failure` routing |

Every rung is an *event in the log* — degraded mode is auditable, not silent.

## 4. The compiler — flows become playbooks

`compile_flow(flow: ConversationFlow) -> Playbook` (`playbook/compiler.py`)
converts a legacy graph into a single-journey playbook, lossless by
construction. `FlowIndex` first classifies every node by degree and shape:

| Class | Test | Becomes |
|---|---|---|
| conversational | speaks/listens | a `Checkpoint` in journey `"main"` |
| computational | router or `auto_proceed` | folded into rules, or pipelines |
| system | indegree 0, not initial | webhook/timer `handlers` |

The mapping, validated against the 61-node golf flow
(`tests/fixtures/flow/golf_booking.json`):

- **Edge conditions** compile per `compile_edge_condition`: anchored data
  predicates over known result keys become `judge: expr` rules
  (`X.success == true` → `results.X.ok`, `X.status == 409` →
  `results.X.status == 409`); everything not confidently translatable stays
  `judge: llm` with the prose verbatim — lossless beats clever.
- **Tool-bearing computational chains** linearize into a `PipelineSpec` plus
  a synthetic intermediate checkpoint that runs it on entry and routes on
  `pipeline.ok` / `pipeline.failed`; status/failure branch edges become step
  `on:` routes. Tool-free routers fold into their sources' advance rules.
- **Hub routers** (≥4 exits) become dispatch entries merged into every
  inbound checkpoint.
- **Silence nodes** become `policies.silence` (prompts kept in chain order);
  the token-expiry global edge + refresh node become `middleware`; other
  global edges become `interrupts`; `is_final` nodes become `terminal` +
  `outcome`.
- **`global_actions`** map 1:1 to `tools`, with Jinja templates rewritten
  into the `{env, slots, results}` namespace; edge `input_schema`s union
  into optional slot declarations plus per-rule `requires`
  (`union_slot_schemas`).

`coverage_report(flow, pb) -> CoverageReport` is the lossless proof: it
re-derives compile provenance and lists every node, edge, and action that did
not map anywhere (`unmapped_*` — any entry is a compiler bug) alongside
informational `dropped` buckets (constructs absorbed into policies,
middleware, pipelines, or dispatch). Run it in CI next to the compiled
artifact.

```python
from superdialog.playbook import compile_flow, coverage_report

pb = compile_flow(flow)
report = coverage_report(flow, pb)
assert not report.unmapped_nodes and not report.unmapped_edges
```

Positioning: flows keep working on Engine A; the compiler makes Engine B the
zero-rewrite destination for them.

## 5. Security model

The playbook artifact is data — possibly optimizer-generated, possibly
third-party — and the transcript is untrusted user speech. Defenses, by
layer:

- **Sandboxed Jinja.** All template rendering (`render.py`, `toolexec.py`)
  uses `jinja2.sandbox.SandboxedEnvironment`: attribute-walking SSTI
  payloads are blocked, not executed. Template errors degrade — raw text on
  the speaking path, a failed `ToolResultEvent` on the tool path — never a
  crash mid-call.
- **AST-whitelisted expressions.** `expr.evaluate` parses `judge: expr`
  rules and computed views against a strict node whitelist: no
  comprehensions, lambdas, dunders, imports, or non-whitelisted calls;
  builtins are stripped; namespaces are guarded wrappers (`slots`,
  `results`, `env`); expressions are length-capped; missing data evaluates
  to `None` (falsy), never an exception.
- **Hard gates require pre-verdict confirmation.** Director
  verdict-extracted slots are written `provisional` at hard gates, so a
  single (possibly prompt-injected) verdict can never confirm its own
  `requires` and advance through a hard gate in one shot — `confirmed`
  comes from tools, expr `set:` writes, or prior soft-checkpoint
  extraction. The Director's prompt additionally pins the transcript as
  untrusted input. `authoritative` slots are tool/Director-only, and the
  rendered view instructs the Talker never to assert facts absent from it.
- **Secret redaction in event recording.** `ToolExecutor` records redacted
  tool calls: secret-shaped keys (token, api-key, password, bearer, otp, …)
  are masked recursively in bodies, and URLs are stripped of userinfo and
  masked query secrets before the `ToolCallEvent` lands — the real request
  still goes to the wire untouched. The **env lane is never rendered** to
  the Talker: the renderer shadows `env` in view expressions, so
  `ACCESS_TOKEN`-class values cannot leak into speech or the packed prompt.

## 6. Roadmap (future, not shipped)

Clearly labeled non-features today: a `superdialog optimize` command closing
the run → score → reflect loop over event logs; a playbook CLI mode; host
adapter plumbing that feeds LiveKit silence/barge-in signals in as
`ExternalEvent`s (the Agent-protocol text path already works with the
existing adapters); sessionless webhook workers for `handlers`;
`resume: true` interrupt restoration; tool TTL scheduling. None of these are
promised for a specific version.

## 7. What lives outside this library

Audio processing, STT/TTS, telephony/SIP/RTP, media servers and Rooms,
numbers, voice profiles, billing. All of those are Voice Infra's problem.
SuperDialog ends at text in, text out — on both engines.
