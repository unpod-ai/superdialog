# Unified `DialogMachine` Entry Point — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `DialogMachine` the single entry-level API that drives either engine — the Playbook engine by default, the legacy graph runtime on request — while `PlaybookAgent` stays the advanced lower-level seam.

**Architecture:** `DialogMachine` becomes a thin dispatcher. A pure `_select_engine(source, engine)` resolver picks `"graph"` or `"playbook"` from the constructor inputs; the chosen backend builds lazily on first use (graph = the existing `DialogStateMachine` path, untouched; playbook = a wrapped `PlaybookAgent`). The `Agent`-protocol methods delegate to whichever backend is active. Purely additive: no engine internals change, `PlaybookAgent` stays public.

**Tech Stack:** Python ≥3.10, pydantic v2, `superdialog.playbook` (`Playbook`, `PlaybookAgent`, `provider_adapters`, `httpx_http`), `superdialog.llm.resolver.resolve_llm`, `superdialog.tools` (`Tool`/`PythonTool`/`HttpTool`/`MCPTool`), pytest (`asyncio_mode = "auto"`), uv, ruff, pyrefly.

**Design doc:** `docs/plans/2026-06-12-unified-dialogmachine-entry-design.md` — read it first.

**Verified substrate (cite these, they are real):**
- `src/superdialog/dialog_machine.py` — `DialogMachine.__init__(self, flow: Flow|FlowSet, llm: str, tools=None, memory=None, config=None, traversal_dir=None, adapter="toolcall")` (:53); lazy `_ensure_machine` (:91); `start` (:137); `turn(text, context=None, stream=False) -> Turn | AsyncIterator[StreamChunk]` (:175) delegating to `_run_turn`/`_stream_turn`; `assist` (:262); `chat_ctx` prop (:283); `load_chat_ctx` (:297); `flow_state` prop (:307); `load_flow_state` (:318); `reset` (:327); `set_llm` (:337); `switch_flow` (:347); `state -> dict` (:357); `is_complete` (:368); `seed` (:127).
- `src/superdialog/playbook/agent.py` — `PlaybookAgent(playbook, talker_llm, director_llm, http, python_tools=None, ...)` (:44); `turn(text, *, stream=False) -> TurnResult | AsyncIterator[StreamChunk]` (:76); `assist` (:91); `chat_ctx` (:98); `load_chat_ctx` (:107); `runtime.state` (`.checkpoint_id`, `.ended`, `.slots`, `.slot_value(k)`).
- `src/superdialog/playbook/__init__.py` exports `Playbook`, `PlaybookAgent`, `provider_adapters`, `httpx_http`.
- `src/superdialog/playbook/toolexec.py:38` — `PythonToolFn` protocol: `async __call__(args: dict, state: ConversationState) -> Any`. Python tools dispatch by `spec.id` (:147-148).
- `src/superdialog/tools/base.py:20` — `Tool(ABC)`: `id`, `name`, `description`, `input_schema`, `async execute(args) -> ToolResult`; `ToolResult.data: dict`.
- `src/superdialog/llm/resolver.py:10` — `resolve_llm(uri) -> LLMProvider`.
- `src/superdialog/__init__.py:8` already exports `DialogMachine`.
- Test fakes: `tests/playbook/test_director.py::CannedLLM(payload: dict)`, `tests/playbook/test_talker.py::StreamLLM(chunks)`, `tests/playbook/test_toolexec.py::FakeHttp(responses)`, `tests/playbook/test_models.py::MINIMAL_YAML`, `tests/playbook/test_simple.py::SIMPLE`.

**Conventions for every task:**
- Branch: continue on `main` (this session's working branch).
- TDD: failing test → run it → minimal impl → run green → commit.
- Run `uv run --no-sync pytest <file> -v` (the `--no-sync` avoids the playground extra's unrelated dependency-resolution break).
- Before each commit: `uv run --no-sync ruff format <touched files> && uv run --no-sync ruff check <touched files> && uv run --no-sync pyrefly check <touched source>`. Scope ruff to touched files (legacy debt elsewhere).
- `asyncio_mode = "auto"`: plain `async def` tests, no markers.
- No network in tests — scripted fakes only.
- Commit style: `feat(cli): …` for the facade, `docs: …` for the re-lead.

---

## Task 1: `_select_engine` resolver (pure)

The engine choice is a pure function over `(source, engine)`. Land it first so the constructor just calls it.

**Files:**
- Modify: `src/superdialog/dialog_machine.py`
- Create: `tests/test_dialog_machine_engine.py`

**Step 1: Write the failing test**

`tests/test_dialog_machine_engine.py`:
```python
"""Engine selection + Playbook-mode behavior of the unified DialogMachine."""

import pytest

from superdialog import DialogMachine, Flow
from superdialog.dialog_machine import _select_engine
from superdialog.playbook import Playbook
from tests.playbook.test_models import MINIMAL_YAML


def _flow_obj() -> Flow:
    return Flow.model_validate(
        {
            "id": "t",
            "system_prompt": "s",
            "initial_node": "n",
            "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
        }
    )


def test_flow_object_auto_selects_graph() -> None:
    assert _select_engine(_flow_obj(), "auto") == "graph"


def test_playbook_object_auto_selects_playbook() -> None:
    assert _select_engine(Playbook.from_yaml(MINIMAL_YAML), "auto") == "playbook"


def test_str_path_and_dict_auto_select_playbook() -> None:
    assert _select_engine("booking.yaml", "auto") == "playbook"
    assert _select_engine({"playbook": [{"id": "a"}]}, "auto") == "playbook"


def test_explicit_engine_overrides() -> None:
    assert _select_engine(_flow_obj(), "playbook") == "playbook"  # compile flow
    assert _select_engine("flow.json", "flow") == "graph"


def test_engine_flow_on_playbook_object_is_error() -> None:
    with pytest.raises(ValueError, match="no graph runtime"):
        _select_engine(Playbook.from_yaml(MINIMAL_YAML), "flow")
```

**Step 2: Run to verify failure**

`uv run --no-sync pytest tests/test_dialog_machine_engine.py -v`
Expected: FAIL — `ImportError: cannot import name '_select_engine'`.

**Step 3: Implement**

In `src/superdialog/dialog_machine.py`, add near the top (after imports):
```python
from typing import Literal


def _select_engine(
    source: Any, engine: str = "auto"
) -> Literal["graph", "playbook"]:
    """Resolve the backend engine from the constructor inputs (pure)."""
    from .flow import Flow, FlowSet
    from .playbook import Playbook

    is_flow = isinstance(source, (Flow, FlowSet))
    is_playbook = isinstance(source, Playbook)
    if engine == "flow":
        if is_playbook:
            raise ValueError(
                "engine='flow' but source is a Playbook (no graph runtime "
                "exists for it)"
            )
        return "graph"
    if engine == "playbook":
        return "playbook"
    if engine != "auto":
        raise ValueError(f"unknown engine: {engine!r}")
    # auto: a Flow object keeps the legacy graph engine (back-compat);
    # everything else (Playbook object, path string, parsed dict) runs Playbook.
    return "graph" if is_flow else "playbook"
```

**Step 4: Run to verify pass** — 5 PASS.

**Step 5: Commit**
```bash
uv run --no-sync ruff format src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
uv run --no-sync ruff check src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
git add src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
git commit -m "feat(cli): _select_engine resolver for the unified DialogMachine"
```

---

## Task 2: Tool bridge (pure)

Any `Tool` adapts to the Playbook engine's python-tool seam through its shared `execute()`.

**Files:**
- Modify: `src/superdialog/dialog_machine.py`
- Modify: `tests/test_dialog_machine_engine.py`

**Step 1: Write the failing test**

Append:
```python
from superdialog.dialog_machine import _python_tools_from
from superdialog.tools import PythonTool


async def test_tool_bridge_invokes_execute() -> None:
    calls = {}

    async def impl(city: str) -> dict:
        calls["city"] = city
        return {"ok": True, "city": city}

    tool = PythonTool(impl, name="lookup")
    bridged = _python_tools_from([tool])
    assert set(bridged) == {tool.id}
    fn = bridged[tool.id]
    out = await fn({"city": "Pune"}, None)  # state unused by this bridge
    assert out == {"ok": True, "city": "Pune"}
    assert calls["city"] == "Pune"


def test_tool_bridge_empty_and_none() -> None:
    assert _python_tools_from(None) == {}
    assert _python_tools_from([]) == {}
```

**Step 2: Run to verify failure** — `ImportError: _python_tools_from`.

**Step 3: Implement**

Add to `dialog_machine.py`:
```python
def _python_tools_from(tools: "list[Tool] | None") -> dict[str, Any]:
    """Bridge any Tool to the Playbook engine's PythonToolFn via execute()."""

    def _adapt(tool: "Tool") -> Any:
        async def fn(args: dict[str, Any], state: Any) -> Any:
            return (await tool.execute(args)).data

        return fn

    return {t.id: _adapt(t) for t in (tools or [])}
```
(Import `Tool` under `TYPE_CHECKING`; the runtime body only touches `.id`/`.execute`.)

**Step 4: Run to verify pass** — PASS.

**Step 5: Commit**
```bash
uv run --no-sync ruff format src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
uv run --no-sync ruff check src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
git commit -am "feat(cli): uniform Tool->python-tool bridge for Playbook mode"
```

---

## Task 3: Constructor + Playbook backend + core delegations

Generalize the constructor, build the Playbook backend lazily, and delegate the `Agent`-protocol core. The graph path stays byte-for-byte; Playbook mode is a new branch.

**Files:**
- Modify: `src/superdialog/dialog_machine.py`
- Modify: `tests/test_dialog_machine_engine.py`

**Step 1: Write the failing test**

Append:
```python
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_talker import StreamLLM


async def test_playbook_mode_from_path(tmp_path) -> None:
    p = tmp_path / "play.yaml"
    p.write_text(MINIMAL_YAML)
    dm = DialogMachine(str(p), llm="openai/gpt-4o-mini")
    assert dm._engine == "playbook"  # selected, not yet built
    # inject fakes so no network: build the backend, then patch its seams
    dm._talker_override = StreamLLM(["Hi", " there"])
    dm._director_override = CannedLLM({"slots": {}, "advance": None, "note": None})
    result = await dm.turn("hello")
    assert hasattr(result, "text")  # TurnResult carries .text/.metadata


async def test_flow_object_still_graph_engine() -> None:
    dm = DialogMachine(_flow_obj(), llm="openai/gpt-4o-mini")
    assert dm._engine == "graph"


def test_playbook_mode_requires_llm(tmp_path) -> None:
    p = tmp_path / "play.yaml"
    p.write_text(MINIMAL_YAML)
    with pytest.raises(ValueError, match="needs an llm"):
        DialogMachine(str(p))  # no llm, no director_llm
```

> Note: the `_talker_override` / `_director_override` seams exist purely so
> tests inject fakes without network. Production builds them from
> `provider_adapters(resolve_llm(...))`.

**Step 2: Run to verify failure** — FAIL (`source`/`engine` not accepted; `_engine` missing).

**Step 3: Implement**

Rewrite `DialogMachine.__init__` to:
- Rename the first param to `source` (keep accepting `Flow | FlowSet | Playbook | str | dict`).
- Add `*, engine="auto"`, `director_llm=None`. Keep `llm=None` (now optional).
- `self._engine = _select_engine(source, engine)`.
- **Graph branch** (`self._engine == "graph"`): coerce `source` to a `FlowSet` exactly as today (a path string under `engine="flow"` loads via `Flow.load`), run the existing initialization, and require `llm` as today.
- **Playbook branch**: validate `llm or director_llm` is set, else
  `raise ValueError("DialogMachine needs an llm= for the Playbook engine")`.
  Store `self._pb_source = source`, `self._llm_uri = llm`, `self._director_uri = director_llm`, `self._pb_tools = tools`, `self._pb: PlaybookAgent | None = None`, and the test override seams `self._talker_override = None`, `self._director_override = None`.

Add a lazy builder:
```python
def _ensure_backend(self) -> "PlaybookAgent":
    if self._pb is not None:
        return self._pb
    from .playbook import Playbook, PlaybookAgent, httpx_http, provider_adapters
    from .llm.resolver import resolve_llm

    pb = (
        self._pb_source
        if isinstance(self._pb_source, Playbook)
        else Playbook.load(self._pb_source)
        if isinstance(self._pb_source, str)
        else Playbook.model_validate(self._pb_source)
    )
    director, talker = provider_adapters(resolve_llm(self._llm_uri or self._director_uri))
    if self._director_uri:
        director, _ = provider_adapters(resolve_llm(self._director_uri))
    self._pb = PlaybookAgent(
        playbook=pb,
        talker_llm=self._talker_override or talker,
        director_llm=self._director_override or director,
        http=httpx_http,
        python_tools=_python_tools_from(self._pb_tools),
    )
    return self._pb
```

Branch the core methods on `self._engine`. `turn`:
```python
async def turn(self, text, context=None, stream=False):
    if self._engine == "playbook":
        return await self._ensure_backend().turn(text, stream=bool(stream))
    # existing graph path unchanged:
    if stream:
        return self._stream_turn(text, context)
    return await self._run_turn(text, context)
```
Do the same one-line `if self._engine == "playbook": return self._ensure_backend().<m>(...)` guard at the TOP of `start`, `assist`, `chat_ctx` (property), and `load_chat_ctx`. For `start` in Playbook mode: `return await self._ensure_backend().runtime.start()` if that returns the opening lines, else adapt to the backend's start shape (check `PlaybookAgent`/runtime for the opening-turn method; mirror what `_drive_agent` in `cli/main.py` does — `await agent.runtime.start()`).

**Step 4: Run to verify pass** — new tests PASS; then run the existing suites to prove back-compat:
```bash
uv run --no-sync pytest tests/test_dialog_machine.py tests/test_dialog_machine_stream.py tests/test_dialog_machine_engine.py -v
```
All green (graph path untouched).

**Step 5: Commit**
```bash
uv run --no-sync ruff format src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
uv run --no-sync ruff check src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
uv run --no-sync pyrefly check src/superdialog/dialog_machine.py
git commit -am "feat(cli): DialogMachine drives the Playbook engine by default"
```

---

## Task 4: Read props + graph-only guards in Playbook mode

Map the useful read props to the Playbook runtime; raise a clear, pointed error for genuinely graph-only methods.

**Files:**
- Modify: `src/superdialog/dialog_machine.py`
- Modify: `tests/test_dialog_machine_engine.py`

**Step 1: Write the failing test**

Append:
```python
async def test_playbook_state_and_is_complete(tmp_path) -> None:
    p = tmp_path / "play.yaml"
    p.write_text(MINIMAL_YAML)
    dm = DialogMachine(str(p), llm="openai/gpt-4o-mini")
    dm._talker_override = StreamLLM(["hi"])
    dm._director_override = CannedLLM({"slots": {}, "advance": None, "note": None})
    await dm.turn("hello")
    assert "checkpoint" in dm.state
    assert dm.is_complete in (True, False)


def test_graph_only_methods_raise_in_playbook_mode(tmp_path) -> None:
    p = tmp_path / "play.yaml"
    p.write_text(MINIMAL_YAML)
    dm = DialogMachine(str(p), llm="openai/gpt-4o-mini")
    for call in (
        lambda: dm.switch_flow("x"),
        lambda: dm.load_flow_state({}),
    ):
        with pytest.raises(NotImplementedError, match="PlaybookAgent"):
            call()
```

**Step 2: Run to verify failure** — FAIL (`state` returns graph shape / methods don't guard).

**Step 3: Implement**

Add Playbook branches:
- `state` property → `{"checkpoint": st.checkpoint_id, "slots": {k: v.value for k, v in st.slots.items()}, "ended": st.ended}` where `st = self._ensure_backend().runtime.state`.
- `is_complete` → `self._ensure_backend().runtime.state.ended`.
- `set_llm(uri)` → store `self._llm_uri = uri`; reset `self._pb = None` so the next turn rebuilds (mirrors the graph engine's next-turn semantics).
- `flow_state` property → in Playbook mode return `None` (no flow-graph state) or the event-log JSONL; keep simple: return `None`.
- `switch_flow`, `load_flow_state`, `seed` → in Playbook mode
  `raise NotImplementedError("<name> is a flow-graph concept; use PlaybookAgent / the playbook's journeys for the Playbook engine")`.
- `reset` → in Playbook mode `self._pb = None`.

Each is a guard at the method top: `if self._engine == "playbook": <playbook behavior>`.

**Step 4: Run to verify pass** — PASS; re-run the existing DM suites (still green).

**Step 5: Commit**
```bash
uv run --no-sync ruff format src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
uv run --no-sync ruff check src/superdialog/dialog_machine.py tests/test_dialog_machine_engine.py
uv run --no-sync pyrefly check src/superdialog/dialog_machine.py
git commit -am "feat(cli): map state/is_complete and guard graph-only methods in Playbook mode"
```

---

## Task 5: Docs re-lead

Flip the docs so `DialogMachine` is the recommended entry point; demote `PlaybookAgent` to the advanced aside (inverse of the prior flip). One recurring sentence: *"DialogMachine runs the Playbook engine by default; pass `engine='flow'` for the legacy graph runtime."*

**Files:**
- Modify: `README.md` — Quickstart A leads with `DialogMachine("playbook.yaml", llm=...)`; the explicit-Talker/Director `PlaybookAgent` form becomes an "Advanced" note.
- Modify: `docs/03-embedding-guides.md` — every host section constructs a `DialogMachine`; the `PlaybookAgent` form is the advanced aside. The provider-adapter section keeps the seams for the advanced path.
- Modify: `docs/02-api-reference.md` — document the new `DialogMachine(source, llm, *, engine, director_llm, tools, ...)` signature as the unified entry; keep `PlaybookAgent` as the lower-level reference.
- Modify: `docs/00-overview.md`, `docs/01-architecture.md` — "two engines, one entry point: `DialogMachine`" framing.

**Step 1:** Make the edits. Keep every code example compile-valid; where an example loads a playbook, use `DialogMachine("playbook.yaml", llm="openai/gpt-4.1-mini")`.

**Step 2: Verify** — no em dashes reintroduced; no `PlaybookAgent`-as-default phrasing remains in quickstarts:
```bash
grep -rn "—" README.md docs/*.md || echo "clean"
uv run --no-sync pytest tests/test_dialog_machine_engine.py tests/playbook tests/cli -q
```

**Step 3: Commit**
```bash
git add README.md docs/00-overview.md docs/01-architecture.md docs/02-api-reference.md docs/03-embedding-guides.md
git commit -m "docs: DialogMachine is the recommended entry point (Playbook engine by default)"
```

---

## Honest scope

- Additive: `PlaybookAgent` stays public and unchanged; graph internals untouched; the `chat`/`generate`/`optimize` CLI already behave this way (no CLI change).
- Graph-only methods (`switch_flow`, `seed`, `load_flow_state`) raise a pointed `NotImplementedError` in Playbook mode rather than pretending to support flow-graph concepts. `flow_state` returns `None` in Playbook mode.
- In Playbook mode `turn(stream=False)` returns a `TurnResult` (`.text`/`.metadata`), matching the `Agent` contract; callers reading `.text` are unaffected by the `Turn` vs `TurnResult` distinction.
- The test override seams (`_talker_override`/`_director_override`) keep the offline tests network-free; production builds adapters from `provider_adapters(resolve_llm(...))`.
