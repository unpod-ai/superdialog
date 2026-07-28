
# SuperDialog - API Reference

**Status:** Draft
**Parent:** [README.md](README.md)
**Audience:** Developers writing code against the library.

---

SuperDialog ships **two engines, one entry point**. `DialogMachine` is the
recommended way in; it drives either engine behind the same `Agent` protocol:

- **Playbook engine** - the **default** runtime: checkpoint-compound
  journeys for fluid conversations. Full playbooks, the simple authoring
  format, and legacy flow JSON all auto-detect and run here via the unified
  loader - callers never route by format. See
  [Playbook engine](#playbook-engine).
- **Graph engine** - the supported **legacy** graph-railed state machine.
  Select it with `DialogMachine(..., engine="flow")` (or `superdialog chat
  --mode flow`); flows keep working unchanged.

  

---

## Construction

### `async generate_simple_playbook(prompt, llm, *, max_attempts=3) -> str`

**The default creation path**: bootstrap a simple-format playbook from a
natural-language description. The output is validated (parsed and compiled
through `simple_to_playbook`) before being returned, so a successful return
is always loadable. CLI equivalent: `superdialog generate "<prompt>"
--output playbook.yaml`.

```python
from superdialog.playbook import generate_simple_playbook

yaml_text = await generate_simple_playbook(
    "An agent that books demo calls and captures a day and time.",
    director,                       # any CompletesLLM
)
open("playbook.yaml", "w").write(yaml_text)
```

### `async create_dialog_flow(prompt, llm, **kwargs) -> Flow` (legacy)

Bootstrap a legacy flow graph from a prompt using a one-shot LLM call.
Prefer `generate_simple_playbook` for new agents; generated flows still run
on the Playbook engine by default via the unified loader.

```python
import asyncio
from superdialog import create_dialog_flow

flow = await create_dialog_flow(            # inside async code
    prompt="Confirm appointment. Ask if Friday 4pm works; offer 5pm if not.",
    llm="openai/gpt-5.1",
)
# or, from a sync entry point:
# flow = asyncio.run(create_dialog_flow(prompt=..., llm=...))
```

The `llm` parameter is used **only at construction**. The runtime model is set on `DialogMachine`.

### `Flow.save(path)` / `Flow.load(path)`

Serialize / deserialize. JSON. Version-controllable.

### `FlowSet(flows: dict[str, Flow])`

Container for multiple small flows. Switch between them at runtime.

```python
flowset = FlowSet({
    "main": main_flow,
    "escalation": escalation_flow,
    "billing": billing_flow,
})
```

---

## DialogMachine (unified entry point)

`DialogMachine` is the recommended way in: one class, one model URI. It runs
the Playbook engine by default; pass `engine="flow"` for the legacy graph
runtime. The lower-level `PlaybookAgent` (below) stays available for explicit
Talker/Director LLMs or a custom HTTP executor.

### Construction

```python
DialogMachine(
    source: Flow | FlowSet | Playbook | str | dict,
    llm: str | None = None,             # model URI (Talker, and Director unless split)
    tools: list[Tool] | None = None,    # any Tool; both engines
    memory: ContextStore | None = None, # graph-only; default in-memory
    config: dict | None = None,         # graph-only: max_tokens, temperature, etc.
    traversal_dir: str | Path | None = None,  # graph engine: auto-save traversal JSON
    adapter: str = "toolcall",          # graph-only adapter selection
    *,
    flow: Flow | FlowSet | None = None, # back-compat keyword alias for source
    engine: str = "auto",               # "auto" | "playbook" | "flow"
    director_llm: str | None = None,    # Playbook: strong-Director override
    barrier_timeout: float = 0.4,       # Playbook: hard-gate Director wait
    hold_timeout: float | None = None,  # Playbook: None -> policies.hold_timeout
    settle_before_speak: bool = False,  # Playbook: offline-eval knob (see PlaybookAgent)
)
```

Engine selection (`engine="auto"`, the default): a `Flow`/`FlowSet` object
keeps the legacy graph engine (back-compat); a `Playbook` object, a path
string, or a parsed dict runs the Playbook engine. Override with
`engine="flow"` (force the graph runtime; `ValueError` on a `Playbook`) or
`engine="playbook"` (force the Playbook engine, compiling a flow if needed).
A missing `llm` in Playbook mode raises a clear `ValueError`.

In Playbook mode, `llm` is the Talker and the Director unless `director_llm`
splits them; `tools=` bridges each `Tool` through its own `execute()`. Graph-only
methods (`switch_flow`, `seed`, `load_flow_state`) raise `NotImplementedError`
in Playbook mode, and `flow_state` returns `None`.

Set `traversal_dir` (graph engine) to a directory path and the machine will write a timestamped JSON file recording every node visited, every turn, and slot values collected - automatically when `is_complete` becomes `True`. Useful for debugging flows, building eval datasets, and auditing production conversations.

Traversal recording is **not** graph-only: the Playbook engine has its own
recorder. Pass `PlaybookAgent(traversal_dir=...)` (plus optional
`traversal_source` / `traversal_model` labels) and the agent saves a
checkpoint-level traversal on session end, built by
`playbook/traversal.py::build_playbook_traversal` and saved by
`playbook/traversal.py::save_playbook_traversal` (both exported from
`superdialog.playbook`).

### `async turn(text, context=None, stream=False) -> Turn | AsyncIterator[StreamChunk]`

The primary method. One method, one parameter for streaming mode. **Always
async** - there is no synchronous wrapper. Drive it from `asyncio.run(...)`
or any async runtime.

```python
# Non-streaming
reply = await dialog_machine.turn("hello")
print(reply.text)

# Streaming - `turn(stream=True)` returns a coroutine that resolves to an
# async iterator, so the iterator must be awaited out of the coroutine first.
stream = await dialog_machine.turn("hello", stream=True)
async for chunk in stream:
    print(chunk.text, end="")
```

`Turn` has:
- `text: str`
- `tool_calls: list[ToolCall]`
- `metadata: dict` (latency, tokens, model used)

> **Streaming policy (v0.2):** the v0.2 implementation resolves the turn
> in one shot, then surfaces the response as whitespace-delimited chunks.
> True provider-level streaming inference is planned for v0.4. The chunk
> shape (`StreamChunk(text, done, turn)`) is stable. The Playbook engine's
> `PlaybookAgent` already streams provider tokens live - see
> [Playbook engine](#playbook-engine).

### `reset()`

Clear conversation memory, restart from the flow's initial node. Useful between independent conversations on the same `DialogMachine` instance.

### `set_llm(uri: str)`

Hot-swap the model. Applies to next turn (in-flight streaming continues on the old model).

```python
dialog_machine.set_llm("anthropic/claude-haiku-4-5")
```

### `switch_flow(name: str)`

If the machine was constructed with a `FlowSet`, switch to a named flow. State is reset by default; pass `preserve_memory=True` to keep history.

```python
dialog_machine.switch_flow("escalation")
```

### `assist(text: str)`

Push a system-level instruction that takes effect next turn. Used for mid-call context injection.

```python
dialog_machine.assist("Customer is upset. Be especially empathetic.")
```

> `inject_system(...)` is preserved as a deprecated alias and emits a
> `DeprecationWarning` on call. Slated for removal in v0.4.

---

## Sessions (v0.2)

Sessions add a lifecycle and persistence layer on top of any `Agent`-protocol
brain - `PlaybookAgent` (the default engine) and the legacy `DialogMachine`
alike. Use them when you need to **resume a
conversation across process boundaries** (async HTTP handlers, multi-worker
deployments, long-lived chat across days).

### The Agent Protocol

```python
class Agent(Protocol):
    async def turn(text: str, *, stream: bool = False) -> TurnResult | AsyncIterator[StreamChunk]
    def assist(text: str) -> None
    @property def chat_ctx -> ChatContext
    def load_chat_ctx(ctx: ChatContext) -> None
```

`DialogMachine`, `LLMAgent`, `LangChainAgent`, and the Playbook engine's
`PlaybookAgent` all implement this Protocol.

### `Session` and `SessionWorker`

```python
# agent_factory may return any Agent-protocol brain - a PlaybookAgent
# factory works identically; shown here with the legacy DialogMachine.
from superdialog import DialogMachine, SessionWorker, InMemorySessionStore

flow = Flow.load("kyc.json")
tools = [PythonTool.of(lookup_customer)]

# One Worker per process; one Agent (and one Session) per active conversation.
worker = SessionWorker(
    agent_factory=lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1", tools=tools),
    store=InMemorySessionStore(),
)

async with worker.acquire("user-42") as h:
    result = await h.turn("hello")
    h.assist("Customer sounds upset; be empathetic.")
```

- **`SessionWorker(*, agent_factory=None, agent_factory_ctx=None, store=None,
  lock_backend=None, max_sessions=1000)`** - process-level multiplexer. All
  arguments are keyword-only. Provide **exactly one** of `agent_factory`
  (zero-arg, one identical brain per session) or `agent_factory_ctx`
  (`Callable[[SessionInit], Agent]` - a *different* brain per session;
  `ValueError` if both or neither are given). Defaults:
  `InMemorySessionStore`, `AsyncioLockBackend`.
- **`worker.acquire(session_id, *, init=None)`** - async context manager.
  Loads or creates the session, locks it for the duration of the block,
  persists state on exit. `init` is a `SessionInit` consumed only when the
  session is first created (bind-at-creation); a cached session is returned
  untouched.
- **`SessionInit(session_id, metadata={})`** - per-session construction
  context handed to `agent_factory_ctx`. `metadata` carries arbitrary keys
  (e.g. `playbook_id`) so one worker process can serve many different
  playbooks - this is the seam supervoice's playbook pool builds on
  (`session/session.py::SessionInit`).
- **`worker.close_session(session_id, *, persist=True)`** - explicit
  eviction. `persist=False` deletes the store record so a terminal session
  cannot be resumed.
- **`SessionHandle`** - yielded inside the with-block. `.turn(text, *, stream)`,
  `.assist(text)`, `.session`, `.agent`.
- **`Session`** - the durable data (`id`, `chat_ctx`, `flow_state`,
  `metadata`). Not normally constructed directly.

```python
# Context-aware factory: one pool process, many playbooks.
worker = SessionWorker(
    agent_factory_ctx=lambda init: build_agent(init.metadata["playbook_id"]),
    store=InMemorySessionStore(),
)
async with worker.acquire(
    "call-7", init=SessionInit(session_id="call-7", metadata={"playbook_id": "hotel"})
) as h:
    await h.turn("hi")
```

### `ChatContext` and `FlowState`

`ChatContext` is LiveKit-aligned message history:

```python
@dataclass class ChatMessage: role: Literal["system","user","assistant","tool"]; content: str
@dataclass class ChatContext: items: list[ChatMessage]
```

`FlowState` is DM-specific runtime state (current node, slots, etc.) - used
only when the session's brain is a `DialogMachine`. Sessions bound to
non-DM brains have `flow_state=None`.

### `SessionStore` and `LockBackend`

Pluggable backends:

| Protocol | Ships in superdialog |
|---|---|
| `SessionStore` | `InMemorySessionStore`, `NullSessionStore` |
| `LockBackend` | `AsyncioLockBackend` |

`InMemorySessionStore` persists for the process lifetime; `NullSessionStore`
drops every write - use it for voice (one DM per call) where persistence
is unwanted.

superdialog ships the **protocol**; production backends live downstream. A
working Redis-backed store ships in supervoice
(`playbook_pool/session_store.py::RedisSessionStore`) - implement
`SessionStore` (`session/store.py::SessionStore`) the same way for your own
backend.

### Other agents (`LLMAgent`, `LangChainAgent`)

```python
from superdialog import LLMAgent, SessionWorker

worker = SessionWorker(
    agent_factory=lambda: LLMAgent(llm="openai/gpt-5.1", system_prompt="Be helpful."),
    store=InMemorySessionStore(),
)
```

`LLMAgent` is a raw chat brain - no flow, no slots. Useful when you want
sessions/persistence/concurrency but no state-machine opinion.

`LangChainAgent` (requires `pip install superdialog[langchain]`) wraps an
async LangChain runnable.

### `assist(text)` - pushing system messages

Both `DialogMachine.assist(...)` and `SessionHandle.assist(...)` are the
canonical way to push a system-level instruction mid-conversation.
`DialogMachine.inject_system` remains as a deprecated alias (slated for
removal in v0.4) and emits a `DeprecationWarning` on call.

---

## Tools

### `PythonTool.of(fn, name=None, description=None)`

The convenience constructor. Infers `id`, `name`, `description`, and
`input_schema` from the function's signature and docstring.

```python
def lookup_customer(customer_id: str) -> dict:
    """Look up customer record by ID."""
    return crm.get(customer_id)

tool = PythonTool.of(lookup_customer)
```

The bare `PythonTool(id=..., name=..., description=..., fn=...)` constructor
is also available when you need to override identity or schema explicitly.

### `HttpTool(id, name, description, url, method="POST", auth=None, input_schema=None)`

```python
tool = HttpTool(
    id="lookup",
    name="lookup",
    description="Look up a customer by partial Aadhaar",
    url="https://api.kerali.io/customer/lookup",
    auth={"type": "bearer", "token": os.environ["KERALI_KEY"]},
)
```

`auth` accepts a dict in v0.2:
- `{"type": "bearer", "token": "..."}` - Bearer token in `Authorization`.

> **Status: roadmap — not built.** Additional auth shapes (`basic`,
> `api_key`, callable). Today only `bearer` is honored
> (`tools/http_tool.py::HttpTool`).

### `MCPTool(id, name, description, server, input_schema=None)`

```python
tool = MCPTool(id="search", name="search", description="...", server="https://mcp.kerali.io")
```

> **Status (v0.2):** the MCPTool wrapper forwards `execute(args)` to
> `session.call_tool(self.id, args)` against the configured server.
> Auto-discovery and namespacing of *all* tools on an MCP server (so one
> `MCPTool` registration exposes every tool the server publishes) is
> planned for a follow-up.

---

## LLM provider registration

### `register_llm_provider(name, base_url, api_key, api_style="openai")`

Process-global. Once registered, the URI `custom/<name>/<model>` works in `set_llm()`, `DialogMachine(llm=...)`, and `create_dialog_flow(llm=...)`.

`api_key` accepts a plain string **or a zero-arg callable returning a
string** (`llm/registry.py::ApiKey = str | Callable[[], str]`). A callable
is invoked on every LLM call, so short-lived credentials stay fresh - the
LiveKit inference gateway uses this to mint a fresh JWT per call.

```python
register_llm_provider(
    name="kerali-internal",
    base_url="https://llm.kerali.io/v1",
    api_key=os.environ["KERALI_KEY"],          # or: lambda: mint_fresh_token()
    api_style="openai",
)
dialog_machine = DialogMachine(flow=flow, llm="custom/kerali-internal/llama-3-70b-tuned")
```

---

## Eval

The eval surface is the shipped `superdialog.eval` package (runner,
endpoints, metrics registry, report writer, and an OpenAI-compatible eval
server), driven by the `superdialog eval` CLI group. There is no `Eval`
class - the entry points are the CLI and the persona bridge below.

- **Concepts and workflow:** [05-eval-guide.md](05-eval-guide.md)
- **Running evals (CLI walkthrough):** [07-running-evals.md](07-running-evals.md)
- **In-code persona evals:** see
  [Eval bridge](#eval-bridge-personaspec--run_session--run_eval) - the
  Playbook engine's `PersonaSpec` / `run_session` / `run_eval` surface.

CLI surface (see [CLI](#cli) for the full table): `superdialog eval
{flow, gen-dataset, run, bench, serve, suite}` plus the top-level
`superdialog benchmark`.

---

## Adapters

The actual module paths shipped in v0.2:

| Import | Purpose |
|---|---|
| `superdialog.adapters.livekit.DialogMachineLLM` | LiveKit `Agent(llm=...)` plugin (livekit-plugins-langchain-style) |
| `superdialog.adapters.pipecat.make_processor` | Factory that builds a PipeCat `FrameProcessor` |
| `superdialog.adapters.fastapi.FastAPIRouter` | Mountable FastAPI router exposing `/turn`, `/stream`, `/reset` |
| `superdialog.adapters.websocket.WebSocketRunner` | Standalone WSS server (Unpod Voice Infra) |

See `docs/03-embedding-guides.md` for working snippets per host.

---

## CLI

| Command | Purpose | Status |
|---|---|---|
| `superdialog generate "<prompt>" --output playbook.yaml` | Bootstrap a playbook from a prompt (default creation path) | shipped |
| `superdialog chat [--flow <path>]` | Interactive REPL (default path `./playbook.yaml`, then `./flow.json`; runs on the Playbook engine; `--mode flow` for legacy DialogMachine; `--adapter` applies only in flow mode; default `--llm openai/gpt-4.1-mini`) | shipped |
| `superdialog flow lint <flow.json>` | Validate graph (legacy) | shipped |
| `superdialog flow draw <flow.json>` | Render Mermaid diagram (legacy) | shipped |
| `superdialog flow generate "<prompt>" --llm openai/gpt-5.1` | Bootstrap flow.json from a prompt (legacy) | shipped |
| `superdialog optimize --playbook <pb>` | Reflective prose optimizer: paired persona evals, targeted prose edits, improved YAML in the source format | shipped |
| `superdialog playbook compile <flow.json>` | Compile legacy flow JSON to Playbook YAML | shipped |
| `superdialog playbook chat --playbook <pb.yaml>` | REPL against an existing playbook YAML | shipped |
| `superdialog playbook run <flow.json>` | Compile a flow and immediately start a Playbook REPL | shipped |
| `superdialog eval flow --flow <flow.json> [--traversal <session.json>]` | Audit a recorded session or run a synthetic eval (legacy flow harness) | shipped |
| `superdialog eval gen-dataset --playbook <pb.yaml>` | Build `<playbook>.evalcases.yaml` from personas (auto-generated unless `--personas`) | shipped |
| `superdialog eval run --playbook <pb> --dataset <cases> --out <dir>` | A/B evaluate a dataset (`--modes vanilla,playbook`), write a report | shipped |
| `superdialog eval bench --playbook <pb> --models <uris>` | One shot: gen dataset if missing + A/B every model (one report dir each) | shipped |
| `superdialog eval serve --playbook <pb> [--port 8000]` | Serve the playbook as an OpenAI-compatible endpoint | shipped |
| `superdialog eval suite --config <registry.yaml>` | Run eval suites as behavioral gates (`--tier smoke\|full`) | shipped |
| `superdialog benchmark [--data <name\|.jsonl>] [--flow <pb>]` | RAGAS + deterministic benchmark over a dataset, raw LLM vs SuperDialog | shipped |

`eval` is a **required subcommand group** - bare `superdialog eval --flow ...`
(the pre-0.2.18 spelling) no longer parses; use `superdialog eval flow`.
Full eval workflow: [07-running-evals.md](07-running-evals.md).

---

## Worked example - end to end (legacy DialogMachine)

A KYC bot built once, deployed four ways. Same `DialogMachine` object passes through every host.

```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine, PythonTool

# ── 1. Bootstrap a flow from a prompt (one-shot LLM call at construction) ─
async def build_flow():
    flow = await create_dialog_flow(
        prompt="Verify customer KYC. Ask for Aadhaar last 4. Confirm DOB.",
        llm="openai/gpt-5.1",
    )
    flow.save("kyc.json")                     # version-control it
    return flow

flow = asyncio.run(build_flow())

# ── 2. Register a tool (Python callable; HTTP or MCP equally valid) ──────
def lookup_customer(aadhaar_last_4: str) -> dict:
    """Lookup customer by partial Aadhaar."""
    return crm.lookup_by_partial_aadhaar(aadhaar_last_4)

# ── 3. Build the runtime machine ─────────────────────────────────────────
dialog_machine = DialogMachine(
    flow=flow,
    llm="anthropic/claude-haiku-4-5",         # runtime model, cost lever
    tools=[PythonTool.of(lookup_customer)],
)

# ── 4a. Test as a CLI chatbot - no infrastructure needed ─────────────────
async def repl():
    while True:
        user = input("> ")
        if user.strip() in {"quit", "exit"}: break
        reply = await dialog_machine.turn(user)
        print(reply.text)

asyncio.run(repl())

# ── 4b. Or use the bundled CLI ───────────────────────────────────────────
#       $ superdialog chat --flow kyc.json --mode flow

# ── 5. Drop into LiveKit (LLM-plugin pattern; same dialog_machine) ───────
from livekit.agents import Agent, AgentSession
from superdialog.adapters.livekit import DialogMachineLLM

async def lk_entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(dialog_machine))
    await AgentSession().start(agent=agent, room=ctx.room)

# ── 6. Or drop into PipeCat ──────────────────────────────────────────────
from superdialog.adapters.pipecat import make_processor
pipecat_node = make_processor(dialog_machine)

# ── 7. Or expose to Unpod Voice Infra via WSS runner ─────────────────────
from superdialog.adapters.websocket import WebSocketRunner
WebSocketRunner(
    agent=dialog_machine,
    agent_id="kerali-kyc-bot",                # bind this name in Unpod portal
).serve(port=8080)
```

| Step | Host | LoC added |
|---|---|---|
| 1-3 | (none - just construct the machine) | ~10 |
| 4 | CLI chatbot | ~3 |
| 5 | LiveKit agent | ~6 |
| 6 | PipeCat pipeline | ~2 |
| 7 | Unpod Voice Infra (WSS runner) | ~5 |

One `DialogMachine` instance, four hosts, one product surface.

For the **full Unpod Voice Infra journey** - portal config (voice profile, number, agent binding) alongside the SDK code - see the Unpod voice platform docs.

---

## Playbook engine

The Playbook engine (`superdialog.playbook`) runs declarative
checkpoint-compound journeys instead of node-railed graphs. Two LLM roles
share one append-only event log: a fast **Talker** streams every spoken turn
while an async **Director** extracts slots, judges advancement, and runs
tools. Checkpoints gate *outcomes*, not utterances - the Talker speaks
freely; the Director decides when a step's goal is actually met.

Authoring guide: [docs/04-playbook-guide.md](04-playbook-guide.md) - Part 1
the playbook formats (full and simple), Part 2 the technical design.
Concepts and rationale:
[decisions.md §6](decisions.md#6-decision-records);
module overview: `src/superdialog/playbook/README.md`.

Public exports (`from superdialog.playbook import ...`): `Playbook`,
`PlaybookAgent`, `EventLog`, `ConversationState`, `CompletesLLM`,
`StreamsLLM`, `HttpFn`, `PythonToolFn`, `httpx_http`, `compile_flow`,
`coverage_report`, `replay`, `ReplayReport`, `PersonaSpec`, `SessionMetrics`,
`EvalReport`, `run_session`, `run_eval`, `run_multi_model`,
`MultiModelReport`, `ModelScore`, `score_report`, `SessionAuditor`,
`AuditReport`, `CorpusGenerator`, `CorpusSpec`, `EdgeScenario`, `EvalCache`,
`ScriptedUser`, `SpeaksUser`, `TimingLLM`, `cached_speaker`,
`derive_default_persona`, `persona_cache_path`, `save_personas`,
`generate_simple_playbook`, `load_simple`, `simple_to_playbook`,
`is_simple_playbook`, `optimize`, `OptimizeReport`, `RoundTrace`,
`ObjectiveBreakdown`, `generate_personas`, `load_personas`,
`ProviderTalker`, `ProviderDirector`, `provider_adapters`, `make_editable`,
`Edit`, `SimpleDoc`, `FullDoc`, `MutationError`,
`build_playbook_traversal`, `save_playbook_traversal`, `load_traversal`,
`traversal_to_persona`, `traversal_to_scripted_user`.

### `Playbook`

The authored artifact. A pydantic model: construct directly, or load:

```python
from superdialog.playbook import Playbook

pb = Playbook.from_yaml(text)    # YAML 1.2-style booleans: on/off/yes/no
pb = Playbook.from_json(text)    #   stay strings; only true/false are bool
pb = Playbook.load(path)         # picks YAML for .yaml/.yml, else JSON
```

All three loaders auto-detect the **simple authoring format** (a top-level
`playbook:` list, lowered via `simple_to_playbook`) **and legacy flow
JSON** (`nodes` + `initial_node`, compiled via `compile_flow`) - callers
never route by format; the Playbook engine is the default for every
artifact. `load_simple` remains as an explicit alias.

Top-level fields:

| Field | Type / default | Meaning |
|---|---|---|
| `persona` | `str = ""` | System persona rendered into every Talker view |
| `multi_entity` | `bool = False` | Opt-in: scope slot storage/lookups per checkpoint `entity` |
| `guidelines` | `GuidelineConfig` | Channel/tone/language/timezone knobs, `memory_enabled`, `supervisor` opt-in (deprecated: `director_model` / `talker_model` - use `llm:`) |
| `llm` | `LLMConfig \| None` | The model this playbook declares (`provider` + `model`, optional `director` split); `None` keeps host-supplied models. See `Playbook.resolve_llm_providers` |
| `pronunciations` | `list[PronunciationSpec] = []` | Authored pronunciation rules for the voice runtime (respelling-first; IPA advisory) |
| `journeys` | `dict[str, Journey]` (≥ 1) | Named checkpoint lists; refs are `"journey.checkpoint"` |
| `dispatch` | `list[DispatchEntry] = []` | Intent → checkpoint routes (compile-time organization in v1) |
| `tools` | `list[ToolSpec] = []` | HTTP / python tools |
| `pipelines` | `list[PipelineSpec] = []` | Ordered tool steps with typed result branches |
| `handlers` | `list[HandlerSpec] = []` | Webhook / timer → pipeline bindings |
| `interrupts` | `list[InterruptSpec] = []` | Global "drop everything" routes |
| `policies` | `Policies` | `silence: SilencePolicy \| None`; `hold_timeout: float = 4.0` - Talker post-filler wait; `filler` / `hold_line: str \| None` - authored barrier lines (see `PlaybookAgent`) |
| `middleware` | `MiddlewareSpec \| None` | Auth-refresh-and-replay for pipeline steps |
| `env` | `dict[str, str] = {}` | Seed env lane (never rendered to the Talker) |
| `views` | `dict[str, str] = {}` | Computed views: name → expr, rendered as reference data |
| `knowledge_base` | `str = ""` | Global KB text (Jinja-renderable); injected into a step's Talker prompt when `Checkpoint.uses_kb` allows (explicit value wins; `None` = heuristic: guidance mentions `knowledge_base`) |
| `initial` | `str \| None` | Start ref; defaults to `initial_checkpoint_id` |
| `source_path` | `str = ""` | Set by `Playbook.load()`; empty when built from text |

Lookups:

- `pb.checkpoint(ref: str) -> Checkpoint` - `ref` is `"journey.id"`; raises
  `KeyError` for an unknown journey or checkpoint.
- `pb.initial_checkpoint_id -> str` - `initial` if set, else
  `"<first journey>.<its first checkpoint>"`.
- `pb.checkpoint_ids() -> set[str]`, `pb.tool(id)`, `pb.pipeline(id)`,
  `pb.slot_spec(key)` (first declaration wins).

**Validation** runs on construction (so on `from_yaml`/`from_json`/`load`
too) and raises `ValueError` - surfaced as `pydantic.ValidationError`, a
`ValueError` subclass - for:

- duplicate checkpoint ids within a journey; duplicate tool or pipeline ids
- a journey name containing `"."`
- `store_response_as: "pipeline"` on any tool (reserved result key - it
  gates the `pipeline.ok` / `pipeline.failed` expr namespace)
- unknown checkpoint refs anywhere: `advance_when[].to`, `on_failure`,
  `dispatch[].to`, `interrupts[].to`, `policies.silence.then`, `initial`,
  pipeline step routes and `RetrySpec.on_exhaust`
- unknown pipeline refs (`checkpoint.pipeline`, `handlers[].pipeline`) and
  unknown tool refs (`on_enter`, pipeline steps, `middleware.refresh_with`)
- an `advance_when[].requires` key not declared in any checkpoint's `slots`
  and not written by that rule's own `set` (a typo'd key at a hard gate
  would deadlock the checkpoint)

### `PlaybookAgent`

The engine behind the public `Agent` protocol - drop it into
`SessionWorker` and every host adapter unchanged.

```python
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

agent = PlaybookAgent(
    playbook=Playbook.load("booking.yaml"),
    talker_llm=talker,          # StreamsLLM
    director_llm=director,      # CompletesLLM
    http=httpx_http,            # HttpFn
    python_tools=None,          # dict[str, PythonToolFn] | None
    token_budget=4000,          # Talker view budget (estimated tokens)
    barrier_timeout=4.0,        # hard gate: wait this long for the Director
    hold_timeout=None,          # then filler + this much more before degrading;
                                # None -> the playbook's policies.hold_timeout (4.0)
    traversal_dir=None,         # str | Path | None: auto-save traversal on session end
    traversal_source="",        # display name in the traversal (default: source file)
    traversal_model="",         # model label (default: director_llm.model_id)
    settle_before_speak=False,  # offline-eval: Talker waits for the Director EVERY
                                # turn, not just hard gates (off in live voice)
    supervisor_llm=None,        # CompletesLLM | None: Loop-2 reviewer; explicit arg
                                # wins, else guidelines.supervisor reuses the Director
    intercept_llm=None,         # CompletesLLM | None: fast classifier guarding
                                # irreversible tools (fails closed); None = expr guard
    filler=None,                # SpokenLine | None: barrier filler line override
    hold_line=None,             # SpokenLine | None: hold line override
    allow_private_hosts=False,  # permit HTTP tools to hit private/loopback hosts
)
```

`SpokenLine = str | Callable[[ConversationState], str]`
(`playbook/talker.py::SpokenLine`) - pass a callable for language-aware
lines rendered against the live state. Explicit `filler` / `hold_line`
arguments win over the playbook's authored `policies.filler` /
`policies.hold_line`; `None` everywhere keeps the Talker's built-in
defaults.

- **`async turn(text, *, stream=False, language=None)`** - Agent protocol.
  Non-streaming returns `TurnResult(text, metadata)`; `metadata` carries
  `checkpoint`, `version`, `ended` (and `outcome` once ended). With
  `stream=True` the returned iterator yields **live provider tokens**
  (`StreamChunk(text=...)`) while the Director settles concurrently, then
  any pass-through `say_verbatim` lines, then the `done=True` chunk.
  `language` is the bridge-detected language of this turn, stamped on the
  user utterance so the reply adheres to the caller (a `PlaybookAgent`
  extension - the base `Agent` protocol's `turn` has no `language`).
- **`stream_turn(text, language=None)`** - returns the async iterator
  directly, no `await` needed: `async for chunk in agent.stream_turn(t)`.
- **`mark_interrupted(heard_text=None)`** - host barge-in repair: truncates
  the last logged assistant utterance to what the caller actually heard and
  tags it `[interrupted by caller]`; `None` keeps the text, just tags it.
- **`apply_memory(summary)`** - seed prior-call context for a returning
  caller (rendered under "Earlier in this conversation" when
  `guidelines.memory_enabled` is set); takes effect next turn.
- **Barge-in safety** - aborting the stream (host `aclose()` or
  cancellation) interrupts *speech*, not the state machine: the Director
  runs to completion in a shielded scope, partial Talker speech is logged
  exactly once, and `check_repairs` still runs.
- **Hard-gate barrier** - at a `gate: hard` checkpoint the Talker waits up
  to `barrier_timeout` for the quiescent state; on timeout it speaks a
  filler line, waits up to `hold_timeout`, then degrades to a hold line
  rather than hang. The lines (`FILLER`, `HOLD_LINE`, `RECOVERY_LINE`) live
  in `superdialog.playbook.talker`; the Talker retries a failed stream once
  before speaking `RECOVERY_LINE`.
- **`assist(text)`** - appends a system `UtteranceEvent`; takes effect next
  turn.
- **`chat_ctx` / `load_chat_ctx(ctx)`** - brain-agnostic transcript view;
  loading seeds a *fresh* log from the context's utterances (tool messages
  are skipped). Lossy by design - for full fidelity use the event log:
- **`event_log` / `load_event_log(log)`** - the runtime's append-only
  `EventLog` (single source of truth) and its lossless wholesale restore.
- **`runtime`** - public `PlaybookRuntime`. Hosts may call
  `agent.runtime.start()` to seed the session eagerly, feed external
  events, or inspect state; a turn on a never-started runtime starts it
  automatically.

### `PlaybookRuntime`

The central conductor: owns the log, applies policies, runs to quiescence.

```python
PlaybookRuntime(
    playbook: Playbook,
    director_llm: CompletesLLM,
    http: HttpFn,
    python_tools: dict[str, PythonToolFn] | None = None,
    max_hops: int = 8,
    intercept_llm: CompletesLLM | None = None,   # irreversible-tool classifier
    allow_private_hosts: bool = False,           # HTTP tools may hit private hosts
)
```

| Member | Behavior |
|---|---|
| `log` | The `EventLog`; `rt.log.append(...)` is a supported public pattern |
| `state` | Cached `ConversationState` fold, refreshed when the log grows; treat as read-only |
| `async start() -> list[str]` | Seed the date anchor + `env`, enter the initial checkpoint, quiesce; returns pass-through speech |
| `async on_user_text(text, *, record=True, language=None) -> list[str]` | Append the utterance, one Director verdict, policies, quiesce. `record=False` when the caller already appended (the streaming agent does); `language` stamps the utterance |
| `async on_external(event) -> ExternalResult` | Record an `ExternalEvent`; silence policy or webhook/timer handler |
| `async check_repairs()` | Emit a repair steering note when the Talker re-asked an answered slot |
| `async rewind(to_version, reason, *, by="runtime", confirmed=False, repair_note=None) -> RewindOutcome` | Rewind conversation *state* to `to_version` (below) |
| `async redirect(to, reason) -> list[str]` | Supervisor-grade redirect: advance to `to`, quiesce; returns pass-through speech |
| `async pop_detour() -> list[str]` | Abandon the current detour: return to the step under it on the resume stack, if any |
| `load_log(log)` | Replace the event log wholesale (invalidates the state cache) |

**Rewind and tool tiers.** Speech is irreversible, so `rewind` deletes
nothing: a `RevertEvent` marks the range `(to_version, now]` superseded and
the fold skips those state effects while every utterance stays in the
transcript. Side effects in the range are guarded by each tool's
`effective_tier` (`ToolSpec.effective_tier`): *reversible* tools are undone
structurally; *compensable* tools require `confirmed=True` and run their
`compensate` tool **before** the revert (so its templates still see the
to-be-reverted results); *irreversible* tools (or compensable ones with no
`compensate`) refuse the rewind. The result is a `RewindOutcome` pydantic
model: `status: "done" | "needs_confirmation" | "refused"`, `detail`, and
`pending` (the blocking tool ids). `repair_note` becomes a repair steer so
the Talker acknowledges the correction naturally.

**Interception (`intercept_llm` seam).** Before an explicitly-tiered tool
materializes, the runtime runs a guard ladder with cost proportional to
risk (`runtime.py::PlaybookRuntime._intercept`): reversible/unannotated -
no check; compensable - a pure expr guard (required slots meet the same
per-slot gate the Director uses, no repair in flight); irreversible - every
required slot must be *confirmed*, plus, when an `intercept_llm` is
configured, one fast classifier call that **fails closed**. Unannotated
playbooks never activate the guard.

**Quiescence guarantee:** when `start()` or `on_user_text()` returns, the
runtime is quiescent - every pipeline, expr rule, auto-advance, and policy
hop has resolved (bounded by `max_hops`; exhaustion is recorded as a
`DegradedEvent`, never an exception). `PlaybookAgent`'s hard-gate barrier
relies on this as API. The returned `list[str]` is pass-through speech
(`say_verbatim` lines traversed during quiescence) - already logged as
assistant utterances; the host must play them. A Director LLM failure
appends `DegradedEvent` and the Talker continues solo; LLM-free policies
still apply.

External events (hosts deliver these themselves on the text path):

```python
from superdialog.playbook.events import ExternalEvent

res = await agent.runtime.on_external(ExternalEvent(kind="silence", name="vad"))
if res.prompt:                      # silence-policy line for the host to play
    speak(res.prompt)
await agent.runtime.on_external(
    ExternalEvent(kind="webhook", name="payment_captured", payload={...})
)
```

`kind` is `"silence" | "webhook" | "timer"`; webhook/timer events match a
`HandlerSpec` whose `on` equals `"<kind>.<name>"`. Handler advances stay
silent (no pass-through speech is fabricated for an absent listener).

### The artifact model

#### `Checkpoint`

| Field | Type / default | Meaning |
|---|---|---|
| `id` | `str` (required) | Unique within its journey; referenced as `"journey.id"` |
| `goal` | `str = ""` | What "done" means; shown to Talker and Director |
| `entity` | `str = "caller"` | Which person this step's slots describe (storage-key prefix under `multi_entity`); must match `[a-z_][a-z0-9_]*` |
| `slots` | `dict[str, SlotSpec] = {}` | Typed slots to extract while here |
| `guidance` | `str = ""` | Talker prose; Jinja over `{slots, views, results}` |
| `say_verbatim` | `str \| None` | Exact line (same Jinja namespace); bypasses the Talker LLM |
| `never_say` | `list[str] = []` | Rendered into the Talker view as hard prohibitions |
| `exit_say` | `str = ""` | Spoken on the turn that *leaves* this checkpoint via a Director rule: rendered and injected as a one-shot steer (capture-then-pitch steps). Simple format spells it `then_say` |
| `advance_when` | `list[AdvanceRule] = []` | Outcome gates (below) |
| `gate` | `"soft" \| "hard" = "hard"` | Default hard: Talker barriers on the Director, `requires` need *confirmed* slots. Set `soft` for free collection (provisional slots suffice, never blocks) |
| `auto` | `bool = False` | Speak verbatim once, then advance via the first rule without user input |
| `strict` | `bool = False` | Speak `say_verbatim` word-for-word; never paraphrase via the LLM |
| `handover` | `bool = False` | Inject the handover-summary instruction at this step |
| `pipeline` | `str \| None` | Pipeline run once per entry; routes on `pipeline.ok` / `pipeline.failed` |
| `on_enter` | `list[str] = []` | Tool ids executed on entry; failures are data, not exceptions |
| `on_failure` | `str \| None` | Route on pipeline failure or turn-budget exhaustion |
| `terminal` | `bool = False` | Entering ends the session (`SessionEndEvent`) |
| `outcome` | `str \| None` | Outcome label recorded when the terminal checkpoint ends the session |
| `turn_budget` | `int \| None` | User turns before a wrap-up steering note; 2 grace turns later, route to `on_failure` |
| `uses_kb` | `bool \| None` | Inject `Playbook.knowledge_base` into this step's Talker prompt. `None` = heuristic (guidance mentions `knowledge_base`); set explicitly to keep the KB off hot steps. Simple format spells it `kb` |

#### `SlotSpec`

| Field | Type / default | Meaning |
|---|---|---|
| `type` | `"str"` (also `int float bool date time enum array object`) | Verdict values are cast; bad casts / enum misses are dropped, never stored |
| `required` | `False` | Surfaces in the Talker's "Still needed" list and the Director prompt |
| `values` | `list[str] \| None` | Enum members; non-members are rejected |
| `resolve_from` | `ResolveFrom \| None` | Candidate source for id-like slots the user never speaks verbatim: the Director gets the live list from `tool_results` and resolves the spoken value to the matching id. `ResolveFrom(result, list_field="items", name_field="name", id_field="id")` |
| `authoritative` | `False` | Only tools, pipelines, and expr `set:` may write it; Talker writes and Director verdict extraction are ignored |
| `invalidates` | `[]` | Slots/results cleared when this value *changes* (non-transitive - list the closure) |
| `description` | `""` | Shown to the Director's extractor |
| `gate` | `"soft" \| "hard" \| None` | Per-slot confirmation gate (risk class). `None` inherits the checkpoint's `gate`; `"hard"` marks an intrinsically risky slot that must be Director-confirmed; `"soft"` lets it advance on a provisional fill even inside a hard checkpoint. `date` slots default to hard |

#### `AdvanceRule`

| Field | Type / default | Meaning |
|---|---|---|
| `when` | `str` (required) | Prose condition (`judge: llm`) or expr text (`judge: expr`) |
| `judge` | `"llm" \| "expr" = "llm"` | Who decides (below) |
| `to` | `str` (required) | Target checkpoint ref `"journey.id"` (validated) |
| `requires` | `list[str] = []` | Slot keys that must be *filled* (soft gate) / *confirmed* (hard gate) |
| `set` | `dict[str, Any] = {}` | Slot writes (confirmed, by director) applied on advance |

Judge semantics:

- **`expr`** - evaluated LLM-free at every quiescence hop and before any
  LLM verdict; first matching expr rule in author order wins. Inside a
  pipeline-owned checkpoint the namespace also exposes `pipeline.ok` /
  `pipeline.failed`.
- **`llm`** - the Director makes ONE structured call per user utterance
  (extract slots + pick an advance target + steer). The first `llm` rule
  whose `to` matches the verdict's target wins, in author order. When
  `requires` is unmet the advance is withheld and a steering note names
  the missing keys instead.
- Verdict-extracted slots are **provisional at hard gates** - a single
  (possibly prompt-injected) verdict can never confirm its own `requires`
  and pass a hard gate in one shot. Confirmation at hard gates comes from
  tools, expr `set:` writes, or prior soft-checkpoint extraction.

#### `ToolSpec`

| Field | Type / default | Meaning |
|---|---|---|
| `id` | `str` (required) | Unique; python tools are looked up by it in `python_tools` |
| `type` | `"http" \| "python" = "http"` | |
| `method` / `url` / `headers` / `body` | `"GET"` / `""` / `{}` / `{}` | Request parts; string values are sandboxed Jinja over `{slots, env, results}` |
| `store_response_as` | `str \| None` | Result key for `results.<key>`; `"pipeline"` is reserved (ValueError) |
| `env_updates` | `dict[str, str] = {}` | env key → dot-path into response `data`; applied on 2xx |
| `slot_updates` | `dict[str, str] = {}` | slot key → dot-path into response `data`; a confirmed write (tools may confirm what a verdict cannot) |
| `run_once` | `False` | Skip when the tool already ran this session |
| `when` | `str \| None` | Expr over state; skip when falsy (or invalid) |
| `timeout` | `30.0` | Seconds, passed to the `HttpFn` |
| `ttl_seconds` / `on_expire` | `None` | Reserved - TTL scheduling is deferred |
| `args` | `dict[str, SlotSpec] = {}` | Typed call args, coerced before execution |
| `tier` | `"reversible" \| "compensable" \| "irreversible" \| None` | Reversibility tier governing rewind and interception. `None` (default) infers conservatively at rewind time and never activates the interception guard |
| `compensate` | `str \| None` | Tool id that undoes an ok result on rewind (required for a compensable tool to be rewindable) |

`ToolSpec.effective_tier` (property) is the tier rewind safety actually
uses: the explicit `tier` if set, else a conservative guess - safe HTTP
methods → `"reversible"`, anything else (writes, python tools) →
`"compensable"`. See [PlaybookRuntime](#playbookruntime) for rewind and
interception semantics.

Tool failures (HTTP error status, exceptions, template errors) are recorded
as a failed `ToolResultEvent` - data, never a crash. The logged
`ToolCallEvent` redacts secret-like URL params and body keys; the real
request is sent unredacted. A body of exactly `{"_template": "..."}` (the
compiler's escape hatch for Jinja-in-JSON bodies) is rendered, then
JSON-parsed into the actual request body.

#### `PipelineSpec`, `PipelineStep`, `RetrySpec`

```yaml
pipelines:
  - id: confirm_and_hold
    steps:
      - tool: hold_slot
        on:
          ok: continue                    # default for ok
          http_409: booking.offer_other   # typed status branch
          failed: {retry: 2, on_exhaust: booking.collect}
```

- `PipelineSpec(id, steps)`; `PipelineStep(tool, on={})`.
- `on` keys: `ok`, `failed`, `http_<code>` (exact status first, then the
  `ok`/`failed` fallback). Values: `"continue"`, a checkpoint ref, or a
  `RetrySpec`.
- `RetrySpec(retry, on_exhaust)` - `retry` is capped at **10** (`0 ≤ n ≤ 10`;
  an unbounded retry would become an HTTP hot loop inside a live call).
  Exhaustion marks the run failed, routes to `on_exhaust`, and writes the
  `error_context` slot (`"<pipeline>:<tool>"`).
- A failing step with no matching branch stops the pipeline, sets
  `error_context`, and lets the checkpoint's `on_failure` route.

#### `MiddlewareSpec`

`MiddlewareSpec(on_status=401, refresh_with=<tool id>, then="replay")` - one
per playbook. Inside pipelines, a step result with `on_status` triggers the
`refresh_with` tool (typically writing a fresh token via `env_updates`) and
replays the step once. `"replay"` is the only `then` value.

#### `HandlerSpec`, `InterruptSpec`, `SilencePolicy`

- `HandlerSpec(id, on, pipeline)` - `on` is `"webhook.<name>"` or
  `"timer.<name>"`; fired via `runtime.on_external(...)`.
- `InterruptSpec(id, when, judge="llm", to, resume=False)` - global routes;
  `judge` is `"llm" | "event"`; `judge: llm` interrupts ride the Director
  verdict. `resume=True` restoration is deferred (validation accepts it;
  the runtime does not restore yet).
- `SilencePolicy(max_prompts=2, prompts=[], then="")` - the *n*-th silence
  since checkpoint entry speaks `prompts[n-1]` (returned as
  `ExternalResult.prompt`); past `max_prompts` the session routes to `then`.

### Protocols: `CompletesLLM`, `StreamsLLM`, `HttpFn`, `PythonToolFn`

```python
class CompletesLLM(Protocol):            # Director seam
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...

class StreamsLLM(Protocol):              # Talker seam
    def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]: ...

HttpFn = Callable[..., Awaitable[tuple[int, Any]]]   # keyword-called:
# (method=, url=, headers=, body=, timeout=) -> (status_code, parsed_json)

class PythonToolFn(Protocol):            # registered via python_tools={id: fn}
    async def __call__(self, args: dict[str, Any], state: ConversationState) -> Any: ...
```

Provider adapter pattern - the Director wants plain text back, the Talker
wants raw tokens (an `async def` generator satisfies `StreamsLLM`):

```python
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

class Director:                                   # any provider -> CompletesLLM
    def __init__(self, client): self._client = client
    async def complete(self, messages, **kwargs):
        return (await self._client.chat(messages)).text

class Talker:                                     # any provider -> StreamsLLM
    def __init__(self, client): self._client = client
    async def stream(self, messages, **kwargs):
        async for chunk in self._client.stream_chat(messages):
            yield chunk.text

agent = PlaybookAgent(
    playbook=Playbook.load("booking.yaml"),
    talker_llm=Talker(client),
    director_llm=Director(client),
    http=httpx_http,        # production HttpFn backed by httpx
)
```

`httpx_http` sends `body` as JSON and returns `(status_code,
response_json)` - non-JSON responses come back as `{"text": ...}`.

### The expr language

Used by `judge: expr` rules, `ToolSpec.when`, and `Playbook.views`. Safe and
LLM-free: a restricted Python expression evaluated against state.

```python
slots.city == "Pune"                    # slot value; None when unset
results.hold.ok                         # tool result: ok / status / data / error
results.search.status == 404
env.BOOKING_API                         # env lane
pipeline.ok                             # pipeline-owned checkpoints only
len(results.search.data.slots) > 0
first(pluck(results.search.data.slots, "time"))
```

Namespaces per context:

| Context | `slots` | `results` | `env` | `pipeline` |
|---|---|---|---|---|
| `advance_when` (`judge: expr`) | yes | yes | yes | yes (`.ok` / `.failed`) |
| `ToolSpec.when` | yes | yes | yes | no |
| `Playbook.views` | yes | yes | shadowed to `None` | no |

Helpers (the only callables): `len` (None → 0), `first`, `last` (None on
empty), `pluck(items, key)`, `unique`, `min`, `max`, `any`, `all` - note
`all([])` is `None`, not `True`: missing data never fires a predicate.

Allowed syntax: literals, lists/tuples, names, attribute and subscript
access, comparisons (`== != < <= > >= in not in is is not`), `and/or/not`,
unary minus, helper calls. **Forbidden** (raises `ExprError`, an
authoring-time `ValueError`): arithmetic operators, comprehensions,
lambdas, dict literals, f-strings, any `_`-prefixed name or attribute,
non-whitelisted calls, unknown root names, expressions over 4096 chars.
Missing values and runtime errors evaluate to `None` - falsy, not fatal.

Jinja templates are a separate, sandboxed surface: `guidance` /
`say_verbatim` render over `{slots, views, results}` (env is never
Talker-visible); tool `url`/`headers`/`body` render over
`{slots, env, results}`. Template errors degrade (raw text in speech, a
failed result for tools) - they never crash a live call.

### `compile_flow` / `coverage_report` - migrating flows

```python
from superdialog import Flow                    # Flow is ConversationFlow
from superdialog.playbook import compile_flow, coverage_report

flow = Flow.load("golf_booking.json")
pb = compile_flow(flow)                # single-journey "main" Playbook
report = coverage_report(flow, pb)     # CoverageReport
assert not report.unmapped_nodes
assert not report.unmapped_edges
assert not report.unmapped_actions
```

`compile_flow(flow: ConversationFlow) -> Playbook` is lossless by
construction - every legacy construct lands somewhere:

| Legacy construct | Becomes |
|---|---|
| Conversational nodes | Checkpoints in journey `"main"` |
| Tool-free computational nodes | Folded into their sources' advance rules |
| Tool-bearing computational chains | A `PipelineSpec` + synthetic intermediate checkpoint routing on `pipeline.ok` / `pipeline.failed` |
| Hub routers (≥ 4-exit computational) | `dispatch` entries + rules merged into inbound checkpoints |
| Silence nodes | `policies.silence` (prompts in chain order) |
| Token-expiry global edge + refresh action | `middleware` |
| Other global edges | `interrupts` |
| Webhook/timer system nodes | `handlers` with single-step pipelines |
| `global_actions` | `tools`, 1:1, templates rewritten to `{env, slots, results}` |

Deterministic edge conditions (`X.success == true`, `X.status == 404`)
compile to `judge: expr`; everything else stays `judge: llm` with the prose
passed through verbatim. `coverage_report(flow, pb) -> CoverageReport`
audits the mapping: `unmapped_nodes` / `unmapped_edges` /
`unmapped_actions` (any entry is a compiler bug), `orphans`, `dropped`
(informational buckets), `notes`. The compiler is validated against the
61-node golf-booking flow at `tests/fixtures/flow/golf_booking.json`.

`FlowIndex` (`superdialog.playbook.compiler`) exposes the underlying
degree/classification index when you need to inspect a flow yourself.

### `EventLog` and `ConversationState`

The event log is the single source of truth; state is a pure fold over it.

```python
from superdialog.playbook import ConversationState, EventLog

text = agent.event_log.to_jsonl()                 # persist (JSONL, one event/line)
agent.load_event_log(EventLog.from_jsonl(text))   # lossless restore
state = ConversationState.fold(agent.event_log, playbook)
```

- `EventLog(events=None)` - versions must be contiguous from 1, else
  `ValueError`. `append(event)` stamps the next version and returns the
  stamped copy (appending an already-stamped event raises `ValueError`);
  `version` property; `replay()` iterates; `to_jsonl()` / `from_jsonl()`.
  Events are frozen pydantic models discriminated on `type`:
  `utterance`, `slot_write`, `advance`, `steering_note`, `tool_call`,
  `tool_result`, `env_write`, `scratchpad`, `summary`, `external`,
  `degraded`, `session_end` (import from `superdialog.playbook.events`).
- `ConversationState.fold(log, playbook=None)` - derived snapshot:
  `checkpoint_id`, `slots` (value + provisional/confirmed status +
  provenance), `transcript`, `env`, `tool_results`, `steering_note`,
  `silence_count`, `user_turns_in_checkpoint`, `ended`, `outcome`, etc.
  Confirmed slot values are never downgraded by provisional writes; a
  changed slot value clears its `invalidates` dependents. Helpers:
  `slot_value(key)`, `confirmed(keys)`, `filled(keys)`.

### `replay` / `ReplayReport` - regression over recorded logs

```python
from superdialog.playbook import replay

report = await replay(log, playbook, director_llm)   # pure: never mutates log
report.stable            # True when every replayed decision matched
report.turns, report.advance_matches, report.slot_matches
for d in report.diffs:   # DecisionDiff(at_version, kind, recorded, replayed)
    ...                  # kind: advance | slot | missing_advance | extra_advance
```

Each recorded user utterance is re-evaluated by the Director under a
(possibly edited) playbook, and the decision is diffed against what the log
shows the Director actually did - the inner evaluation primitive for prompt
and playbook changes.

### Eval bridge: `PersonaSpec` / `run_session` / `run_eval`

Persona-scripted sessions measured from their event logs:

```python
from superdialog.playbook import PersonaSpec, run_eval, run_session

personas = [
    PersonaSpec(
        name="impatient",
        traits="impatient, gives all details at once",
        goal="book a tee time in Pune tomorrow",
        max_turns=12,
        opening="Hello",
        ground_truth_slots={"city": "Pune", "players": 2},
    ),
]
metrics = await run_session(agent, personas[0], user_llm)   # one session
report = await run_eval(                                    # fresh agent per run
    playbook_factory=lambda: make_agent(),
    personas=personas,
    user_llm=user_llm,           # async complete(messages) -> str (the caller)
    n=1,
)
report.completion_rate, report.mean_slot_accuracy
```

`SessionMetrics` per session: `completed`, `outcome`, `turns`,
`turns_per_checkpoint`, `slot_accuracy` + `slot_diffs` (against
`ground_truth_slots`), `repair_count`, `degraded_count`, and the full
`event_log_jsonl` for replay/audit. `EvalReport` aggregates sessions with
`completion_rate` and `mean_slot_accuracy`.

### Voice events (host-fed today)

`superdialog optimize` (reflective prose optimization over paired persona
evals, with generated persona suites) and the playbook CLI
(`superdialog playbook {compile,chat,run}`) are shipped. Voice-event plumbing (silence/barge-in `ExternalEvent`s wired
by the LiveKit adapter) remains the one unshipped piece. Today the Agent-protocol text path
works with the existing adapters, and hosts deliver external events through
`runtime.on_external` themselves.
