# SuperDialog - Architecture

**Status:** Canonical
**Parent:** [README.md](README.md)
**Purpose:** Internal design of the framework. Two engines, one contract:
the checkpoint-compound Playbook runtime (the default - the unified loader
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
       Agent protocol - turn / assist / chat_ctx / load_chat_ctx
              ┌─────────────┴─────────────┐
       DialogMachine                PlaybookAgent
       (Engine A)                   (Engine B)
              │                            │
     DialogStateMachine            PlaybookRuntime ── EventLog
     flow graph · TransitionGate   Talker ∥ Director · pipelines
     CriteriaJudge · FlowContext   expr rules · policies · tools
```

- **Engine A - DialogMachine.** The legacy, graph-railed state machine. A
  flow graph decides every transition; the LLM speaks within the rails.
  Fully supported; existing flows keep working unchanged.
- **Engine B - Playbook.** The default engine: the checkpoint compound runtime for fluid
  conversations. Checkpoints gate *outcomes*, not utterances; a fast Talker
  streams every spoken turn while an async Director extracts, judges, and
  steers over an event-sourced log. This is where new investment goes.
- **The frontends.** Two compilers lower authoring surfaces onto Engine B:
  `simple_to_playbook` (`playbook/simple.py`) compiles the simple authoring
  format - prose steps, structured persona, facts/objections/boundaries -
  and `compile_flow` converts existing flow JSON losslessly (§4), so
  migration is a compile step, not a rewrite. The unified loader invokes
  both automatically.

Package layout:

```
superdialog/
  ├─ flow/                # Flow graph: nodes, edges, serialization
  ├─ machine/             # DialogStateMachine engine (Engine A internals)
  ├─ dialog_machine.py    # Public DialogMachine facade
  ├─ playbook/            # Playbook engine (Engine B) - models, events,
  │                       #   runtime, talker, director, compiler, replay
  ├─ agent.py             # Agent Protocol + TurnResult
  ├─ agents/              # LLMAgent, LangChainAgent (non-DM brains;
  │                       #   LangChainAgent imports from
  │                       #   superdialog.agents.langchain_agent,
  │                       #   `langchain` extra required)
  ├─ session/             # Session, SessionHandle, SessionWorker, stores
  ├─ chat_context.py      # ChatContext, ChatMessage (LiveKit-aligned)
  ├─ llm/                 # Model URI resolver, backends, resilience,
  │                       #   prompt caching (§5.3-5.5)
  ├─ observability/       # Observer protocol, TracingProvider, Langfuse
  │                       #   sink (§5.1)
  ├─ eval/                # Eval runner, metrics registry, dataset gen,
  │                       #   OpenAI-compatible server (`eval serve`)
  ├─ benchmark/           # `superdialog benchmark` - deterministic +
  │                       #   RAGAS playbook scoring
  ├─ traversal/           # Session traversal recordings (build/save
  │                       #   full dialog histories to JSON)
  ├─ tools/               # Python / HTTP / MCP tool wrappers
  ├─ cli/                 # `superdialog generate / chat / optimize / playbook / flow / eval`
  └─ adapters/            # LiveKit, PipeCat, FastAPI, WebSocket
```

Shared substrate - `SessionWorker` (one agent per session, pluggable
`SessionStore`), the model URI resolver (`openai/gpt-5.1`,
`anthropic/claude-opus-4-7`, `custom/...`), tools, adapters, and the CLI -
is documented in [02-api-reference.md](02-api-reference.md) and
[03-embedding-guides.md](03-embedding-guides.md). Two engines, one entry
point: `DialogMachine(source, llm, *, engine=...)` is the recommended way in
and drives either engine - the Playbook engine by default, the legacy graph
runtime with `engine="flow"`. The lower-level `PlaybookAgent` takes two small
LLM protocols (`StreamsLLM` for the Talker, `CompletesLLM` for the Director),
so any provider - or a scripted fake in tests - plugs in directly.

> **Naming disambiguation** - three collisions to keep straight when reading
> code or older prose:
>
> 1. **`DialogMachine` names two things.** Today it is the unified *facade*
>    (`dialog_machine.py::DialogMachine`) that drives either engine and
>    defaults to Playbook. In older prose it named the legacy graph engine
>    itself. When this doc says "the legacy DialogMachine engine" it means
>    Engine A's internals (`superdialog.machine`); everywhere else,
>    `DialogMachine` means the facade.
> 2. **The engine axis has three spellings.** CLI: `superdialog chat
>    --mode {playbook,flow}`. Constructor API: `DialogMachine(...,
>    engine="auto"|"playbook"|"flow")`. Internally the resolved value is
>    `Literal["graph", "playbook"]`
>    (`dialog_machine.py::_select_engine`) - so `--mode flow`,
>    `engine="flow"`, and internal `"graph"` all mean Engine A.
> 3. **"Supervisor" names two things.** The Talker's rendered prompt says
>    "Direction from supervisor" (`playbook/render.py`) - that "supervisor"
>    is the **Director**'s steering note. The separate
>    `playbook/supervisor.py::Supervisor` is an opt-in Loop-2 trajectory
>    reviewer (enabled via `guidelines.supervisor` or
>    `PlaybookAgent(supervisor_llm=)`) that fires on
>    `detect_triggers` derailment patterns and can inject, redirect,
>    rewind, or hand over (`SupervisorDecision`). Different component,
>    unfortunate shared name.

## 2. Engine A - DialogMachine (legacy, graph-railed)

The legacy engine, opted into via `superdialog chat --mode flow`. A `ConversationFlow` is a directed graph: nodes (states,
each with an instruction or static text), edges (transitions with
natural-language conditions), `global_edges`, and `actions` (declarative HTTP
calls). The public `Flow` facade loads/saves it as version-controllable JSON;
`create_dialog_flow(prompt=..., llm=...)` bootstraps one from a prompt (the
LLM is used at construction time only, never at runtime). For new agents,
prefer the default creation path - `superdialog generate` /
`generate_simple_playbook` - which produces a playbook directly.

```python
# engine="flow" selects this legacy graph runtime; the default is Playbook.
dialog_machine = DialogMachine(flow, llm="anthropic/claude-opus-4-7", engine="flow")
reply = await dialog_machine.turn("hello")           # complete Turn
stream = await dialog_machine.turn("hello", stream=True)  # StreamChunk iter
```

Internals (`superdialog.machine`):

| Component | Role |
|---|---|
| `DialogStateMachine` | Runtime core; two execution models: criteria-based `process_turn(user_input)` (CriteriaJudge picks the edge) and tool-call-based `apply_transition(edge_id)` (the host LLM's tool callback names the edge) |
| Adapters (`TextAdapter`, `ToolCallAdapter`, `LLMAdapter`) | How the LLM is consulted per turn - compose a reply, or pick an edge via a tool call |
| `CriteriaJudge` | LLM-based node-completion evaluator for the criteria path |
| `TransitionGate` | Validates every transition, in order: edge valid → node content spoken → self-loop limit → completion criteria (slots) → user spoke → CriteriaJudge verdict |
| `FlowContext` | Mutable state bag that travels with the machine: `ConversationData` (history, variables) + `MachineState` (current node, visit counts, intent stack) + legacy `node_slots` |

Flexibility on this engine is rail-shaped: `__stay_on_node__`, global edges
with an intent stack, fallback edges, skip/auto-proceed handling. Each turn
costs a route decision plus a speak call. That is exactly the failure mode
the Playbook engine was designed to remove - users do not follow graphs.

`DialogMachine` remains fully supported: `turn` / `reset` / `set_llm` /
`switch_flow` / `assist`, FlowSets, streaming, sessions, and all four host
adapters. Full signatures and worked examples:
[02-api-reference.md](02-api-reference.md).

## 3. Engine B - Playbook (default, checkpoint compound)

Design rationale: the decision record in
[decisions.md §6](decisions.md#6-decision-records).
Source: `src/superdialog/playbook/`. Public surface: `superdialog.playbook`
(`Playbook`, `PlaybookAgent`, `EventLog`, `ConversationState`,
`compile_flow`, `coverage_report`, `replay`, `run_eval`, …).

### 3.1 The artifact

A **Playbook** (`playbook/models.py`) is the authored, git-diffable artifact,
loaded from YAML or JSON via `Playbook.load(path)` - which auto-detects all
three formats (full playbooks, the simple authoring format, and legacy flow
JSON compiled through `compile_flow`), so callers never route by format -
with full cross-reference
validation (unknown checkpoint/pipeline/tool ids, undeclared `requires` keys,
duplicate ids, and the reserved `pipeline` result key all fail at load time).

Two layers:

- **Conversation layer** - `journeys` of `Checkpoint`s plus a `persona` and
  a reusable `dispatch` table. A checkpoint is a call-center-script unit:
  `goal`, typed `slots` (`SlotSpec`), `guidance` prose (Jinja over
  `{slots, views, results}`), an ordered `advance_when` rule list
  (`AdvanceRule`: `when` / `judge: llm|expr` / `to` / `requires` / `set`),
  `gate: soft|hard`, optional `say_verbatim`, `never_say`, `auto`,
  `on_failure`, `terminal` + `outcome`, `turn_budget`.
- **Process layer** - everything that is *not* conversation: `tools`
  (`ToolSpec`: templated HTTP or registered python, `store_response_as`,
  `run_once`, `when:`, `env_updates`, timeout), `pipelines` (`PipelineSpec`:
  ordered steps with typed `on: {ok | failed | http_<code>}` branches and
  capped `RetrySpec`), `handlers` (`HandlerSpec`: webhook/timer-triggered
  pipelines), `interrupts` (`InterruptSpec`), `policies` (silence, `hold_timeout`), optional
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
free - the Talker speaks from `guidance`, and the caller may answer in any
order. The Director extracts `city`/`date`/`players` into the typed slots
(an `invalidates:` write clears stale downstream data on change-of-mind).
The rules are ordered and multi-way: the Director judges the `llm` rules and
may only fire one whose `requires` are met. `gate: soft` means provisional
slot values suffice and the Talker never blocks; past `turn_budget` the
runtime injects a wrap-up steering note, then routes to `on_failure` after a
grace window.

### 3.2 The event-sourced log

Every mutation is an event; state is a fold; the log is the audit artifact.

`playbook/events.py` defines fourteen frozen, versioned pydantic events:
`utterance`, `slot_write`, `advance`, `steering_note`, `tool_call`,
`tool_result`, `env_write`, `session_start` (the per-call date/time anchor),
`scratchpad`, `summary`, `external` (silence/webhook/timer), `degraded`,
`session_end`, `revert` (rewind: supersede a version range of earlier state
effects). `EventLog` is
append-only with contiguous versions stamped from 1; it serializes to JSONL
(`to_jsonl` / `from_jsonl`) and is the single persistence payload.

`ConversationState.fold(log, playbook)` (`playbook/state.py`) is a pure
function from log to snapshot: transcript, slots, env, tool results,
steering note, summary, checkpoint position, silence/turn counters, ended +
outcome. Fold semantics encode the lane rules: slot values carry
`provisional | confirmed` status and never downgrade; `authoritative` slots
ignore Talker writes; `invalidates` is applied non-transitively and skipped
when a write re-asserts the same value. A `revert` makes the fold skip the
state effects of events in its superseded version range while utterances stay
in the transcript (the caller heard them) and `session_start` is never
superseded — the log stays append-only, so the audit trail is complete.

Because the log is the artifact, replay is free: `replay(log, playbook,
director_llm)` (`playbook/replay.py`) re-runs the Director over each recorded
user utterance and diffs its decisions against what was recorded
(`ReplayReport`) - regression evidence for prompt or model changes.
`eval_bridge.py` (`PersonaSpec`, `run_session`, `run_eval`) drives persona
self-play sessions and scores checkpoint completion, slot accuracy, and
turns-per-checkpoint from the same logs. `superdialog optimize`
(`playbook/optimize.py`) closes the loop: paired persona evals (generated
suites by default) score reflective, prose-only edits and emit improved
YAML in the source format.

### 3.3 The compound runtime - one turn

`PlaybookAgent` (`playbook/agent.py`) implements the `Agent` protocol, so
`SessionWorker` and every host adapter run it unchanged - and streaming is
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
   three jobs - extract slot values into the checkpoint's typed schema,
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
   **`judge: expr` rules** are evaluated synchronously in the fold - no LLM
   round-trip, which is what makes compiled router chains instant - then
   **`auto` checkpoints** speak their verbatim line and advance, and a
   **terminal** checkpoint appends `SessionEndEvent` with its `outcome`.
   `say_verbatim` lines crossed during quiescence surface as pass-through
   speech after the Talker's stream.
5. **Join and repair.** The Talker's speech is logged exactly once as an
   `UtteranceEvent` stamped `spoke_from_version=N`; `runtime.check_repairs()`
   compares that stamp against later confirmed slot writes and, if the
   Talker re-asked for something already answered, appends a **repair**
   steering note - the Talker self-corrects next turn instead of silently
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
     │ hard gate: barrier ≤0.4s ─miss─► filler ─► wait ≤4s ─miss─► hold line
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
  (default 4.0s, per playbook via `policies.hold_timeout`), and emits
  `HOLD_LINE` if the Director never lands - politely
  degraded, never hung.
- `say_verbatim` bypasses the Talker LLM entirely (template → speech) for
  regulated lines; `never_say` lists are injected as renderer constraints.

**Degradation ladder** - the session never dies with a model:

| Failure | Behavior |
|---|---|
| Director LLM error / bad JSON | `DegradedEvent(component="director")` appended; Talker continues solo; LLM-free policies (turn budget, silence) still apply; slots settle later |
| Talker stream failure | One instant retry, then the canned `RECOVERY_LINE` |
| Hard-gate barrier miss | Filler, then hold line (above) |
| Quiescence hop exhaustion | `DegradedEvent(detail="quiesce_hop_exhaustion")` - a runaway hop loop is audited, never spun |
| Tool/pipeline failure | A failed `ToolResultEvent` plus a typed `error_context` slot; declarative `retry` / `on_exhaust` / `on_failure` routing |

Every rung is an *event in the log* - degraded mode is auditable, not silent.

## 4. The compiler - flows become playbooks

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
  `judge: llm` with the prose verbatim - lossless beats clever.
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
not map anywhere (`unmapped_*` - any entry is a compiler bug) alongside
informational `dropped` buckets (constructs absorbed into policies,
middleware, pipelines, or dispatch). Run it in CI next to the compiled
artifact.

```python
from superdialog.playbook import compile_flow, coverage_report

pb = compile_flow(flow)
report = coverage_report(flow, pb)
assert not report.unmapped_nodes and not report.unmapped_edges
```

Positioning: the unified loader makes Engine B the zero-rewrite default
destination for flow JSON - `superdialog chat` compiles and runs it
automatically; Engine A remains available via `--mode flow`.

## 5. Observability and LLM routing

Two cross-cutting layers sit under both engines: an observability seam that
traces sessions, LLM generations, and tool calls to a pluggable sink, and the
LLM resolution stack that turns a model URI into a resilient, optionally
prompt-cached provider. Both downstream platforms (supervoice's playbook
pool, unpod-sdk's adapter) wire into these seams, so their contracts are
load-bearing.

```
 DialogMachine.set_observer(observer, trace_id)
        │                                  DialogMachine.register_llm_callback(fn)
        ▼                                  (per-LLM-call usage → billing ledger)
 TracingProvider ── wraps ──┐
        │                   ▼
 Observer sink       resolve_llm(uri)
 (Langfuse / Null)          │
                     ResilientProvider          .inner = raw backend
                     timeout · retry · hedge
                            │
                     mark_cache_prefix          (prompt caching, opt-in)
                            │
                     backend: any-llm / LiteLLM / OpenAI SDK / LiveKit gateway
```

### 5.1 The Observer seam (`observability/observer.py`)

`Observer` is a `runtime_checkable` Protocol - a backend-agnostic sink for
session and LLM observability:

| Method | Fires on | Returns |
|---|---|---|
| `on_session_start(session_id, metadata)` | session open | `trace_id` |
| `on_generation_start(trace_id, name, input_messages)` | LLM call begins | `observation_id` |
| `on_generation_end(observation_id, output, tool_calls, metadata)` | LLM call ends | - |
| `on_tool_call(trace_id, name, args, result)` | tool execution | - |
| `on_flow_node(trace_id, node_id, slots, *, prev_node=)` | node/checkpoint transition | - |
| `on_voice_turn(trace_id, metrics)` | voice-turn metrics from a host | - |
| `on_session_end(trace_id, output)` | session close | - |

Naming quirk: `on_flow_node` is named for graph nodes but receives
*checkpoint ids* on the playbook path - one callback serves both engines.

Three implementations ship:

- **`NullObserver`** - the no-op default; zero external dependencies. Also
  carries `on_error` and `flush` (best-effort extras the Langfuse sink
  implements beyond the Protocol).
- **`LangfuseObserver`** - wraps an *injected* Langfuse client; every method
  is best-effort and never raises (failures log at debug and the call
  continues). Exported alias: `SuperdialogObserver`.
- **`TracingProvider`** - not a sink but a provider wrapper: it wraps any
  `LLMProvider` and records `complete`/`stream` calls as generations against
  a `trace_id`. Optional `role=` ("talker"/"director") prefixes generation
  names so multiple LLM roles in one turn stay distinguishable; `model_uri=`
  tags the end metadata.

**`build_observer(public_key=, secret_key=, host=, *, enable_tracing=)`**
constructs the right sink. Enable priority: the `enable_tracing` kwarg, then
`SUPERDIALOG_TRACING` (`1/true/on` enable, `0/false/off` disable), then
auto-detect from keys. Keys resolve from the kwargs or
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`; host from
`LANGFUSE_BASE_URL`. Anything missing or failing → `NullObserver`; tracing
can never take a session down.

Slot payloads are PII-masked before leaving the process:
`observability/observer.py::redact_slots` masks values whose key (or any
`_`/`-` token of it) is in the redaction set - a built-in PII list plus
`SUPERDIALOG_REDACT_FIELDS` (CSV).

**Wiring:** hosts call
`DialogMachine.set_observer(observer, trace_id)` - both engines. Graph:
the active provider is wrapped in `TracingProvider` immediately. Playbook:
the cached backend is nullified so the next
`dialog_machine.py::DialogMachine._ensure_backend` rebuilds it with traced
Talker and Director providers (roles tagged separately).

### 5.2 The billing hook - `register_llm_callback`

`DialogMachine.register_llm_callback(fn)` registers a per-LLM-call usage
callback for both engines - this is the hook unpod-sdk wires so superdialog
token usage (including cache read/write tokens) reaches the platform usage
ledger. Playbook: the callback is attached as `on_llm_complete` on the
Director and Talker provider adapters - immediately if built, else at
`_ensure_backend`, so registration order does not matter on this engine.
Graph: it lands on the toolcall/llm adapter's `_on_llm_complete` when the
adapter is already built; a callback registered before that relies on the
SDK re-attaching before streaming.

### 5.3 LLM resolution (`llm/resolver.py`)

`resolve_llm(uri)` is the single entry: it resolves the URI to a backend
provider via `resolve_backend(uri)` and wraps it in `ResilientProvider`
(§5.4), so every engine path - flow adapters, playbook Director/Talker, edge
evaluation - inherits timeout/retry/hedge without per-call-site code. The
raw backend stays reachable as `.inner`. `resolve_backend` alone (no
wrapping) exists for hedge legs.

Backend selection is layered on top of the model URI:

1. An explicit backend scheme prefix wins.
2. Else `SUPERDIALOG_LLM_BACKEND` (`anyllm` | `litellm` | `openai`).
3. Else the default (`anyllm`), which silently falls back to LiteLLM when
   the optional `any-llm-sdk` package is not installed.

| URI form | Resolves to |
|---|---|
| `openai/gpt-4.1-mini` | default backend (any-llm, LiteLLM fallback) |
| `anyllm/<provider>/<model>` | `AnyLlmProvider` (forced) |
| `litellm/<provider>/<model>` | `LitellmProvider` (forced) |
| `oai/<model>` | `OpenAIProvider` - naked OpenAI SDK (forced) |
| `livekit/<model>` | `OpenAIProvider` against the LiveKit inference gateway (minted JWT; ignores the ambient backend selector) |
| `custom/<name>/<model>` | LiteLLM against a registered base_url + key (`llm/registry.py::get_custom`) |
| `custom/lk-inference/<model>` | as above, but self-registers the LiveKit gateway from env (`llm/livekit_gateway.py::register_livekit_inference`) when unregistered |
| `vllm/<model>@<host>` | LiteLLM `hosted_vllm/<model>` with `api_base=<host>` |
| `ollama/<model>@<host>` | LiteLLM `ollama/<model>` with `api_base=<host>` |

`custom/` and `@host` forms always route through LiteLLM regardless of the
selected backend - they depend on LiteLLM features. SDK-pool backends
(any-llm / OpenAI) are cached per `(SUPERDIALOG_LLM_BACKEND, uri)` so
sessions share one warm client pool; LiteLLM-backed results are built fresh
(LiteLLM keeps its own global cache, and `custom/` credentials live in a
mutable registry).

When `SUPERDIALOG_LLM_FALLBACK_MODELS` is set (comma-separated URIs),
`resolve_llm` returns a `FallbackProvider` chain (`llm/fallback.py`): leg 0
is the normal resilient wrap, fallback legs are cheap wraps (no retry, no
hedge). Unset ⇒ identical to the plain resilient wrap.

### 5.4 Resilience (`llm/resilience.py`)

`ResilientProvider` wraps any `LLMProvider` with a per-request timeout,
bounded retry-with-backoff, and an optional cross-provider hedge (race the
primary against a delayed alternate; first success wins). The happy path is
transparent - one attempt returning before the timeout behaves exactly like
the bare backend. Retries fire only on timeouts and transient failures
(`_is_retryable`: transient HTTP statuses, type-name/message markers);
auth and bad-request errors fail fast. Streams retry only while nothing has
been emitted yet - after the first token a stall is surfaced, never a silent
restart of a partially-spoken turn. Exhaustion raises
`LLMResilienceError`. The wrapped backend is always available as `.inner`,
and unknown attributes (e.g. `.model`) delegate to it.

`ResilienceConfig.from_env()` reads the knobs (defaults are a safety net;
voice deployments tune the timeout down and/or enable hedging):

| Env var | Default | Meaning |
|---|---|---|
| `SUPERDIALOG_LLM_TIMEOUT_S` | `60.0` | per-request timeout (`0`/`none`/`off` disables) |
| `SUPERDIALOG_LLM_MAX_RETRIES` | `2` | extra attempts after the first |
| `SUPERDIALOG_LLM_BACKOFF_BASE_S` | `0.5` | exponential backoff base |
| `SUPERDIALOG_LLM_BACKOFF_MAX_S` | `8.0` | backoff cap |
| `SUPERDIALOG_LLM_HEDGE` | off | enable the hedge leg |
| `SUPERDIALOG_LLM_HEDGE_MODEL` | - | hedge model URI (resolved via `resolve_backend`) |
| `SUPERDIALOG_LLM_HEDGE_DELAY_S` | `2.0` | delay before the hedge fires |

### 5.5 Prompt caching (`llm/prompt_cache.py`)

Opt-in provider-side prompt caching. Prompt assemblers annotate the leading
system message with a private `_cache_prefix` key holding the stable
persona/preamble substring; `mark_cache_prefix` runs *once* at the
`ResilientProvider` seam - the last step before the backend call, so markers
apply before any retry/hedge - and either:

- strips the private key and returns byte-identical legacy messages
  (caching off, an automatic-cache provider such as OpenAI/Deepseek/xAI, or
  any error), or
- splits the system content at the stable boundary and tags the stable
  block (and the last tool) with `cache_control` for explicit-cache
  providers (Anthropic, Bedrock, Vertex, Gemini).

`PromptCacheConfig.from_env()`: `SUPERDIALOG_PROMPT_CACHING`
(`1/true/yes/on` enables; disabled by default) and
`SUPERDIALOG_PROMPT_CACHE_TTL` (unset → provider default). Automatic-cache
providers benefit even without markers - the assembler guarantees a stable
prefix, which is all their server-side cache needs.

## 6. Security model

The playbook artifact is data - possibly optimizer-generated, possibly
third-party - and the transcript is untrusted user speech. Defenses, by
layer:

- **Sandboxed Jinja.** All template rendering (`render.py`, `toolexec.py`)
  uses `jinja2.sandbox.SandboxedEnvironment`: attribute-walking SSTI
  payloads are blocked, not executed. Template errors degrade - raw text on
  the speaking path, a failed `ToolResultEvent` on the tool path - never a
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
  `requires` and advance through a hard gate in one shot - `confirmed`
  comes from tools, expr `set:` writes, or prior soft-checkpoint
  extraction. The Director's prompt additionally pins the transcript as
  untrusted input. `authoritative` slots are tool/Director-only, and the
  rendered view instructs the Talker never to assert facts absent from it.
- **Secret redaction in event recording.** `ToolExecutor` records redacted
  tool calls: secret-shaped keys (token, api-key, password, bearer, otp, …)
  are masked recursively in bodies, and URLs are stripped of userinfo and
  masked query secrets before the `ToolCallEvent` lands - the real request
  still goes to the wire untouched. The **env lane is never rendered** to
  the Talker: the renderer shadows `env` in view expressions, so
  `ACCESS_TOKEN`-class values cannot leak into speech or the packed prompt.

## 7. Roadmap (future, not shipped)

Clearly labeled non-features today: host
adapter plumbing that feeds LiveKit silence/barge-in signals in as
`ExternalEvent`s (the Agent-protocol text path already works with the
existing adapters); sessionless webhook workers for `handlers`;
`resume: true` interrupt restoration; tool TTL scheduling. None of these are
promised for a specific version.

## 8. What lives outside this library

Audio processing, STT/TTS, telephony/SIP/RTP, media servers and Rooms,
numbers, voice profiles, billing. All of those are Voice Infra's problem.
SuperDialog ends at text in, text out - on both engines.
