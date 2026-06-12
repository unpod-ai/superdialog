# Unified `DialogMachine` Entry Point — Validated Design

Status: validated 2026-06-12 (brainstorm). Makes `DialogMachine` the single
entry-level API that drives either engine — the Playbook engine by default,
the legacy graph runtime on request — while `PlaybookAgent` remains the
advanced lower-level seam.

## 1. Goal

The framework's syntax has moved to the Playbook engine, but the developer's
first touchpoint should stay the familiar `DialogMachine` class with one model
URI. `DialogMachine` becomes a thin facade that selects and drives one of two
backends; passing either engine through one entry point is the priority.

Reconciles with the just-shipped "Playbook is the default engine" positioning:
the *class* keeps its name, the *engine* it runs defaults to Playbook.

## 2. Shape: a dispatcher over two backends

`DialogMachine` holds one backend, chosen at construction:

- **Playbook backend** — wraps a `PlaybookAgent` (existing class, unchanged).
- **Graph backend** — the current `DialogStateMachine` lazy-build path, unchanged.

Both already satisfy the `Agent` protocol (`turn` / `assist` / `chat_ctx` /
`load_chat_ctx` / `start`), so `DialogMachine`'s public methods become straight
delegations. No engine logic moves; we add a selection layer in `__init__` and
forward the protocol methods. Engine *selection* is synchronous and
network-free in `__init__`; the heavy backend builds lazily on first
`turn`/`start`, preserving today's construction discipline.

```python
def __init__(
    self,
    source: Flow | FlowSet | Playbook | str | dict,   # was: flow: Flow | FlowSet
    llm: str | None = None,
    *,
    engine: Literal["auto", "playbook", "flow"] = "auto",
    director_llm: str | None = None,          # Playbook: strong-judge override
    tools: list[Tool] | None = None,          # any Tool, both engines
    memory: ContextStore | None = None,       # graph-only
    config: dict[str, Any] | None = None,     # graph-only
    traversal_dir: str | Path | None = None,  # graph-only
    adapter: str = "toolcall",                # graph-only
) -> None:
```

The `source` positional stays compatible: `DialogMachine(flow_obj, "uri")`
binds `flow_obj` to `source` exactly as before.

## 3. Engine selection

A pure helper `_select_engine(source, engine) -> Literal["graph", "playbook"]`.

With `engine="auto"` (default):

| `source` | resolves to | rationale |
|---|---|---|
| `Flow` / `FlowSet` object | **graph** | back-compat: existing callers unchanged |
| `Playbook` object | **playbook** | already the playbook artifact |
| `str` path | **playbook** | new path calls get the default engine; `Playbook.load` compiles flow/simple/full transparently |
| `dict` (parsed doc) | **playbook** | same as path |

Explicit override:
- `engine="flow"` — force the graph runtime. Valid for a `Flow` object or a
  flow-JSON path/dict; `ValueError` if given a playbook artifact (no graph
  runtime exists for it).
- `engine="playbook"` — force the Playbook engine, including on a `Flow`
  object or flow JSON (compiled via `compile_flow`). The escape hatch to move
  an existing flow onto the new engine without rewriting the call.

Deliberate default: `DialogMachine("flow.json")` → Playbook engine (compiled);
`engine="flow"` keeps the graph runtime on a path-based flow. The string-path
branch reuses the shipped `Playbook.load` — no detection is re-implemented.

## 4. Playbook-mode wiring

```python
playbook = source if isinstance(source, Playbook) else Playbook.load(source)
director, talker = provider_adapters(resolve_llm(llm))      # llm = Talker + default Director
if director_llm:
    director, _ = provider_adapters(resolve_llm(director_llm))  # strong-judge override
self._backend = PlaybookAgent(
    playbook=playbook,
    talker_llm=talker,
    director_llm=director,
    http=httpx_http,                                # internal default, not exposed
    python_tools={t.id: _as_python_tool_fn(t) for t in (tools or [])},
)
```

- **`llm`** is the Talker, and the Director too unless `director_llm` is set.
  Required in Playbook mode; a missing `llm` raises a clear `ValueError`
  ("DialogMachine needs an llm= for the Playbook engine"), never a silent None.
- **`director_llm`** is the only model override — the cheap-Talker /
  strong-Director latency split. `talker_llm` is not exposed (set `llm`).
- **`http`** is always `httpx_http`. It is the executor for the playbook's
  declared HTTP tools/pipelines — an injection seam for production vs tests,
  not an entry-level concern. A custom executor ⇒ drop to `PlaybookAgent`.
- **`adapter`** (toolcall/llm) is graph-only; ignored in Playbook mode.

## 5. Tools: pass any `Tool`, same as before

Every `Tool` subclass shares `id` + `name` + `async execute(args) -> ToolResult`.
The Playbook engine's `PythonToolFn` is `async (args, state) -> data-dict`. One
uniform bridge covers all three types — no special-casing, no rejection:

```python
def _as_python_tool_fn(tool: Tool) -> PythonToolFn:
    async def fn(args: dict, state) -> dict:
        return (await tool.execute(args)).data    # ToolResult.data
    return fn
```

Because the adapter calls the tool's own `execute()`, an `HttpTool` runs its
own HTTP call and an `MCPTool` its own MCP round-trip — identical behavior to
the graph engine, independent of the playbook's internal `http` executor. The
playbook's process layer references each by its `id` (a `ToolSpec` with
`type: python`), the same way a flow's actions reference tools today: you pass
the objects, the artifact wires them in.

- **Graph mode**: `tools=` flows through unchanged.
- **Playbook mode**: bridged as above; no tool type rejected.

## 6. Docs re-lead (positioning)

The just-shipped docs lead with `PlaybookAgent`; they flip to `DialogMachine`
as the one recommended entry point, with a single recurring sentence:
*"DialogMachine runs the Playbook engine by default; pass `engine='flow'` for
the legacy graph runtime."*

- `README.md` Quickstart A → `DialogMachine("playbook.yaml", llm=...)`;
  `PlaybookAgent` becomes an "Advanced: explicit Talker/Director" note.
- `03-embedding-guides.md` → every host section constructs a `DialogMachine`;
  `PlaybookAgent` becomes the advanced aside (inverse of the prior flip).
- `02-api-reference.md` → `DialogMachine` documented as the unified entry (new
  `source`/`engine`/`director_llm` params); `PlaybookAgent` stays as the
  lower-level reference.
- `00-overview` / `01-architecture` → "two engines, one entry point:
  `DialogMachine`"; the CLI `--mode flow` already matches.

## 7. Testing (TDD, offline, scripted fakes)

- `_select_engine` pure table: every `(source type, engine=)` cell + two error
  cases (`engine="flow"` on a playbook; missing `llm` in Playbook mode).
- `DialogMachine("x.yaml", llm=...)` builds a Playbook backend;
  `turn()` / `turn(stream=True)` delegate (CannedLLM / StreamLLM fakes).
- `DialogMachine(flow_obj, llm=...)` still builds the graph backend — the
  existing DialogMachine suite stays green unchanged (the back-compat proof).
- `director_llm=` wires a distinct Director; `tools=[PythonTool / HttpTool /
  MCPTool]` each invoke via `execute()`.
- Flow-JSON path defaults to Playbook; `engine="flow"` on it builds graph.

## 8. Scope

Purely additive: `PlaybookAgent` stays public and unchanged; graph internals
untouched; `chat`/`optimize`/`generate` already behave this way (no CLI
change). The work is the facade selection layer in `DialogMachine.__init__`,
the tool bridge, and the doc re-lead.
