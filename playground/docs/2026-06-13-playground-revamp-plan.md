# Playground Revamp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the playground into a two-pane IDE — a left work pane (Preview · Edit · Conversation · Metrics · Traces · Events) and a right control pane (Voice profile dropdown + Chat · Stats · Playbooks) — where Edit is a CodeMirror YAML editor and Chat is an AI builder that rewrites the playbook YAML via the active LLM.

**Architecture:** New harness endpoints (`source` GET/PUT, `validate`, `publish`, `edit`) sit behind a `PlaybookStore` seam. Draft precedence ("save → next Connect runs the draft") lives in `playground/agents/playbooks.py` as the single source of truth; the harness store and the runner both honor it. The frontend restructures `AgentView.tsx` into two panes and adds three new components (EditorPanel, ChatPanel, StatsPanel).

**Tech Stack:** Python 3.12 / FastAPI / pydantic / superdialog (`make_editable`, `resolve_llm`); React 19 + Vite 6 + TypeScript; CodeMirror 6.

**Working context:** Branch `feat/playground-revamp`, **in place** (the `playground/` tree is untracked, so a worktree would lose it). Commit only the exact files each task touches — never `git add -A` (the repo has unrelated untracked WIP: `Taskfile.yml`, `examples/`, `tests/playground/`). Committing our files gradually tracks the playground on this branch, which is intended.

**Test runner:** `bash scripts/run_tests.sh <module>`. Frontend: `cd playground/web && npm run build` (typecheck) and `npm test` (node:test).

---

## Phase 1 — Backend: store, validation, edit, endpoints

### Task 1: Draft precedence in `playbooks.py`

**Files:**
- Modify: `playground/agents/playbooks.py` (add helpers near `_playbooks_dir`; change `build`)
- Test: `tests/playground/test_playbook_paths.py`

**Step 1: Write the failing test**

```python
# tests/playground/test_playbook_paths.py
from pathlib import Path

from playground.agents import playbooks as pb


def _seed(pbdir: Path) -> str:
    """Copy a real, known-valid example playbook into pbdir as demo.yaml."""
    from playground.agents.playbooks import canonical_path, playbook_registry

    infos = playbook_registry()
    text = canonical_path(infos[0].id).read_text(encoding="utf-8")
    (pbdir / "demo.yaml").write_text(text, encoding="utf-8")
    return text


def test_effective_path_prefers_draft(tmp_path, monkeypatch):
    pbdir = tmp_path / "pb"
    pbdir.mkdir()
    draftdir = tmp_path / "drafts"
    text = _seed(pbdir)
    monkeypatch.setenv("PLAYBOOKS_DIR", str(pbdir))
    monkeypatch.setenv("PLAYBOOK_DRAFTS_DIR", str(draftdir))

    # No draft yet → effective == canonical.
    assert pb.effective_path("demo") == pbdir / "demo.yaml"
    assert pb.canonical_path("demo") == pbdir / "demo.yaml"
    assert pb.draft_path("demo") == draftdir / "demo.yaml"

    # Write a draft → effective switches to it.
    draftdir.mkdir(parents=True, exist_ok=True)
    (draftdir / "demo.yaml").write_text(text, encoding="utf-8")
    assert pb.effective_path("demo") == draftdir / "demo.yaml"

    # Unknown id → None.
    assert pb.canonical_path("nope") is None
    assert pb.effective_path("nope") is None
```

**Step 2: Run it to confirm it fails**

Run: `bash scripts/run_tests.sh playground_paths` (after registering) or directly
`uv run pytest tests/playground/test_playbook_paths.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'effective_path'`.

**Step 3: Implement the helpers**

In `playground/agents/playbooks.py`, after `_playbooks_dir()` (line ~52), add:

```python
def _drafts_dir() -> Path:
    """Resolve the local drafts overlay dir (env override, else playground/.drafts)."""
    env = os.getenv("PLAYBOOK_DRAFTS_DIR")
    if env:
        return Path(env).expanduser()
    # This package lives at <repo>/playground/agents/, so .drafts/ is one up.
    return Path(__file__).resolve().parents[1] / ".drafts"


def draft_path(playbook_id: str) -> Path:
    """Path the local draft for ``playbook_id`` would occupy (may not exist)."""
    return _drafts_dir() / f"{playbook_id}.yaml"


def canonical_path(playbook_id: str) -> Path | None:
    """The committed example file for ``playbook_id`` (None if unknown)."""
    entry = _scan().get(playbook_id)
    return entry[0] if entry else None


def effective_path(playbook_id: str) -> Path | None:
    """Draft if one exists, else the canonical file (None if unknown)."""
    dp = draft_path(playbook_id)
    if dp.exists():
        return dp
    return canonical_path(playbook_id)
```

Then change `build()` (lines ~155-164) to load the effective source:

```python
def build(llm: str, playbook_id: str) -> Any:
    """Build a single-playbook ``DialogMachine`` (engine='playbook') for ``id``.

    Loads the effective source (draft if present, else canonical) fresh from
    disk so a call always runs the latest saved playbook.
    """
    src = effective_path(playbook_id)
    if src is None:
        raise KeyError(f"unknown playbook {playbook_id!r}")
    return DialogMachine(Playbook.load(str(src)), llm=llm, engine="playbook")
```

**Step 4: Register the test module, run, confirm pass**

Add a `playground_paths` case to `scripts/run_tests.sh` (see Task 6 for the pattern; do it now for this module). Run:
`uv run pytest tests/playground/test_playbook_paths.py -v` → PASS.

**Step 5: Format + typecheck the touched file only**

```bash
uv run ruff format playground/agents/playbooks.py tests/playground/test_playbook_paths.py
uv run ruff check playground/agents/playbooks.py --fix
uv run pyrefly check   # fix any new errors in the touched files
```

**Step 6: Commit**

```bash
git add playground/agents/playbooks.py tests/playground/test_playbook_paths.py scripts/run_tests.sh
git commit -m "feat(playground): draft-precedence path helpers; build() runs the draft"
```

---

### Task 2: `validate_yaml` in the store module

**Files:**
- Create: `playground/harness/playbook_store.py`
- Test: `tests/playground/test_playbook_store.py`

**Step 1: Write the failing test**

```python
# tests/playground/test_playbook_store.py
from playground.agents.playbooks import canonical_path, playbook_registry
from playground.harness.playbook_store import validate_yaml


def _valid_text() -> str:
    infos = playbook_registry()
    return canonical_path(infos[0].id).read_text(encoding="utf-8")


def test_validate_accepts_a_real_playbook():
    vr = validate_yaml(_valid_text())
    assert vr.valid is True
    assert vr.errors == []
    assert vr.steps >= 1
    assert vr.journey  # non-empty journey name


def test_validate_rejects_garbage():
    vr = validate_yaml("this: [is, not, a, playbook")  # broken YAML
    assert vr.valid is False
    assert vr.errors


def test_validate_rejects_empty_playbook():
    vr = validate_yaml("goal: hi\nplaybook: []\n")  # min_length violation
    assert vr.valid is False
    assert vr.errors
```

**Step 2: Run → FAIL** (`ModuleNotFoundError: playground.harness.playbook_store`).

**Step 3: Implement**

```python
# playground/harness/playbook_store.py
"""Validation + persistence for playground playbooks.

Persistence sits behind a small seam: ``LocalDraftStore`` writes a git-ignored
drafts overlay now; a future ``RemotePlaybookStore`` will target the
speech-service API under the user's account. Routes and the UI depend only on
this module, never on the backing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml
from pydantic import ValidationError
from superdialog.playbook.editable import make_editable

from playground.agents.playbooks import canonical_path, draft_path, effective_path


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a playbook YAML string."""

    valid: bool
    errors: list[str]
    steps: int
    journey: str


def _format_errors(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
    return [str(exc)]


def validate_yaml(text: str) -> ValidationResult:
    """Parse + validate ``text`` via superdialog; never raises."""
    try:
        playbook = make_editable(text).compile()
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        return ValidationResult(False, _format_errors(exc), 0, "")
    journey = next(iter(playbook.journeys), "")
    steps = len(playbook.journeys[journey].checkpoints) if journey else 0
    return ValidationResult(True, [], steps, journey)
```

**Step 4: Run → PASS.** Format + pyrefly the new files.

**Step 5: Commit**

```bash
git add playground/harness/playbook_store.py tests/playground/test_playbook_store.py
git commit -m "feat(playground): validate_yaml via superdialog make_editable"
```

---

### Task 3: `LocalDraftStore` (read / save / publish / has_draft)

**Files:**
- Modify: `playground/harness/playbook_store.py`
- Test: `tests/playground/test_playbook_store.py` (extend)

**Step 1: Add the failing test**

```python
from playground.harness.playbook_store import LocalDraftStore


def _seed(tmp_path, monkeypatch):
    pbdir = tmp_path / "pb"
    pbdir.mkdir()
    draftdir = tmp_path / "drafts"
    text = _valid_text()
    (pbdir / "demo.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("PLAYBOOKS_DIR", str(pbdir))
    monkeypatch.setenv("PLAYBOOK_DRAFTS_DIR", str(draftdir))
    return pbdir, draftdir, text


def test_save_publish_roundtrip(tmp_path, monkeypatch):
    pbdir, draftdir, text = _seed(tmp_path, monkeypatch)
    store = LocalDraftStore()

    assert store.has_draft("demo") is False
    assert store.read("demo") == text

    store.save_draft("demo", text)
    assert store.has_draft("demo") is True
    assert (draftdir / "demo.yaml").exists()
    assert (pbdir / "demo.yaml").read_text(encoding="utf-8") == text  # canonical intact

    store.publish("demo", text)
    assert store.has_draft("demo") is False          # draft cleared
    assert (pbdir / "demo.yaml").read_text(encoding="utf-8") == text  # canonical written


def test_unknown_id_raises(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    store = LocalDraftStore()
    import pytest

    with pytest.raises(KeyError):
        store.read("nope")
    with pytest.raises(KeyError):
        store.publish("nope", "x")
```

**Step 2: Run → FAIL** (`ImportError: cannot import name 'LocalDraftStore'`).

**Step 3: Implement — append to `playbook_store.py`**

```python
class LocalDraftStore:
    """Drafts overlay on the local filesystem (the seam's local impl).

    ``read`` prefers a draft; ``save_draft`` writes the overlay; ``publish``
    writes the canonical example and clears the draft. All path resolution is
    delegated to ``playground.agents.playbooks`` so the runner and this store
    agree on what "the playbook" is.
    """

    def has_draft(self, playbook_id: str) -> bool:
        return draft_path(playbook_id).exists()

    def read(self, playbook_id: str) -> str:
        path = effective_path(playbook_id)
        if path is None:
            raise KeyError(playbook_id)
        return path.read_text(encoding="utf-8")

    def save_draft(self, playbook_id: str, text: str) -> None:
        if canonical_path(playbook_id) is None:
            raise KeyError(playbook_id)
        dp = draft_path(playbook_id)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(text, encoding="utf-8")

    def publish(self, playbook_id: str, text: str) -> None:
        cp = canonical_path(playbook_id)
        if cp is None:
            raise KeyError(playbook_id)
        cp.write_text(text, encoding="utf-8")
        dp = draft_path(playbook_id)
        if dp.exists():
            dp.unlink()
```

**Step 4: Run → PASS.** Format + pyrefly.

**Step 5: Commit**

```bash
git add playground/harness/playbook_store.py tests/playground/test_playbook_store.py
git commit -m "feat(playground): LocalDraftStore read/save/publish overlay"
```

---

### Task 4: `propose_edit` — the AI builder core (LLM injected)

**Files:**
- Create: `playground/harness/playbook_edit.py`
- Test: `tests/playground/test_playbook_edit.py`

**Step 1: Write the failing test**

```python
# tests/playground/test_playbook_edit.py
import anyio

from playground.agents.playbooks import canonical_path, playbook_registry
from playground.harness.playbook_edit import EditProposal, propose_edit


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list = []
        self.metadata: dict = {}


def _valid_text() -> str:
    infos = playbook_registry()
    return canonical_path(infos[0].id).read_text(encoding="utf-8")


def test_propose_edit_parses_summary_and_yaml():
    new_yaml = _valid_text()

    async def fake_complete(messages, *a, **k):
        assert isinstance(messages, list) and messages[-1]["role"] == "user"
        return _FakeResult(f"Added an SMS confirmation step.\n```yaml\n{new_yaml}\n```")

    proposal = anyio.run(
        propose_edit,
        _valid_text(),
        "add an sms confirmation step",
        fake_complete,
    )
    assert isinstance(proposal, EditProposal)
    assert proposal.summary == "Added an SMS confirmation step."
    assert proposal.yaml.strip() == new_yaml.strip()
    assert proposal.valid is True
    assert proposal.errors == []


def test_propose_edit_flags_invalid_yaml_without_raising():
    async def fake_complete(messages, *a, **k):
        return _FakeResult("Broke it.\n```yaml\nplaybook: []\n```")

    proposal = anyio.run(propose_edit, _valid_text(), "break it", fake_complete)
    assert proposal.valid is False
    assert proposal.errors
    assert proposal.yaml.strip() == "playbook: []"  # still returned for inspection
```

**Step 2: Run → FAIL** (module missing).

**Step 3: Implement**

```python
# playground/harness/playbook_edit.py
"""AI playbook builder: rewrite a playbook YAML from a natural-language edit.

Pure + LLM-injected: ``propose_edit`` takes an async ``complete`` callable
(``LitellmProvider.complete``), so tests run without keys or network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from playground.harness.playbook_store import validate_yaml


class _Result(Protocol):
    text: str


Complete = Callable[..., Awaitable[_Result]]

_SYSTEM_PROMPT = (
    "You edit voice-agent playbooks written in superdialog YAML. You are given "
    "the current playbook and an instruction. Return the COMPLETE updated "
    "playbook, preserving the existing format (simple format uses top-level "
    "`goal`, `persona`, `playbook` (a list of steps with id/purpose/say/collect/"
    "done_when), `facts`, `interrupts`; full format uses `journeys` with "
    "`checkpoints`). Keep ids stable where possible and keep it valid.\n\n"
    "Respond with EXACTLY: a one-line summary of what you changed, then the full "
    "updated playbook in a single ```yaml fenced block. Output nothing else."
)

_FENCE = re.compile(r"```(?:ya?ml)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class EditProposal:
    """An LLM-proposed playbook rewrite plus its validation verdict."""

    yaml: str
    summary: str
    valid: bool
    errors: list[str]
    steps: int
    journey: str


def _parse(reply: str) -> tuple[str, str]:
    """Return ``(summary, yaml)`` from the model reply."""
    match = _FENCE.search(reply)
    if not match:
        # No fence — treat the whole reply as YAML, no summary.
        return "Updated the playbook.", reply.strip()
    yaml_text = match.group(1).strip()
    summary = reply[: match.start()].strip().splitlines()
    return (summary[0].strip() if summary else "Updated the playbook."), yaml_text


def _user_prompt(current_yaml: str, instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n\n"
        f"Current playbook:\n```yaml\n{current_yaml}\n```"
    )


async def propose_edit(
    current_yaml: str,
    instruction: str,
    complete: Complete,
) -> EditProposal:
    """Ask the LLM to rewrite ``current_yaml`` per ``instruction`` and validate."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(current_yaml, instruction)},
    ]
    result = await complete(messages)
    summary, new_yaml = _parse(result.text or "")
    vr = validate_yaml(new_yaml)
    return EditProposal(
        yaml=new_yaml,
        summary=summary,
        valid=vr.valid,
        errors=vr.errors,
        steps=vr.steps,
        journey=vr.journey,
    )
```

**Step 4: Run → PASS.** Format + pyrefly. (Note: `anyio` is already a test dep per CLAUDE.md; if missing, `uv add --dev anyio`.)

**Step 5: Commit**

```bash
git add playground/harness/playbook_edit.py tests/playground/test_playbook_edit.py
git commit -m "feat(playground): propose_edit AI builder core (LLM-injected, validated)"
```

---

### Task 5: Wire the five endpoints into `api.py`

**Files:**
- Modify: `playground/harness/api.py` (imports near top; new routes after `/playground/playbooks`; one helper)
- Test: `tests/playground/test_playbook_endpoints.py`

**Step 1: Write the failing test** (uses FastAPI TestClient + a temp dir + a fake LLM)

```python
# tests/playground/test_playbook_endpoints.py
import pytest
from starlette.testclient import TestClient

from playground.agents.playbooks import canonical_path, playbook_registry


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pbdir = tmp_path / "pb"
    pbdir.mkdir()
    text = canonical_path(playbook_registry()[0].id).read_text(encoding="utf-8")
    (pbdir / "demo.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("PLAYBOOKS_DIR", str(pbdir))
    monkeypatch.setenv("PLAYBOOK_DRAFTS_DIR", str(tmp_path / "drafts"))

    # Fake the LLM so /edit needs no keys/network.
    class _Res:
        text = f"Tweaked it.\n```yaml\n{text}\n```"

    class _Provider:
        async def complete(self, messages, *a, **k):
            return _Res()

    import playground.harness.api as api_mod

    monkeypatch.setattr(api_mod, "resolve_llm", lambda uri: _Provider())

    from playground.harness.api import build_app

    with TestClient(build_app()) as c:
        yield c, text


def test_get_source(client):
    c, text = client
    r = c.get("/playground/playbooks/demo/source")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["yaml"] == text and body["valid"] is True


def test_validate(client):
    c, _ = client
    r = c.post("/playground/playbooks/demo/validate", json={"yaml": "playbook: []"})
    body = r.json()
    assert body["valid"] is False and body["errors"]


def test_save_then_publish(client):
    c, text = client
    assert c.put("/playground/playbooks/demo/source", json={"yaml": text}).json()["ok"]
    assert c.get("/playground/playbooks/demo/source").json()["draft"] is True
    assert c.post("/playground/playbooks/demo/publish", json={}).json()["ok"]
    assert c.get("/playground/playbooks/demo/source").json()["draft"] is False


def test_save_rejects_invalid(client):
    c, _ = client
    body = c.put("/playground/playbooks/demo/source", json={"yaml": "playbook: []"}).json()
    assert body["ok"] is False and body["errors"]


def test_edit(client):
    c, text = client
    body = c.post(
        "/playground/playbooks/demo/edit", json={"instruction": "tweak it"}
    ).json()
    assert body["ok"] and body["summary"] == "Tweaked it." and body["valid"] is True
```

**Step 2: Run → FAIL** (routes 404 / `resolve_llm` not importable on api_mod).

**Step 3: Implement**

In `playground/harness/api.py`, add imports near the existing ones (after line ~38):

```python
from superdialog.llm.resolver import resolve_llm

from playground.harness.playbook_edit import propose_edit
from playground.harness.playbook_store import LocalDraftStore, validate_yaml
```

In `build_app()`, after `app.state.registry = SessionRegistry()` (line ~115), add:

```python
    app.state.store = LocalDraftStore()
```

Add a module-level helper (top level, near other helpers):

```python
def _edit_llm_uri() -> str:
    """A resolvable model URI for the AI builder (provider-prefixed)."""
    uri = os.getenv("ACTIVE_LLM", "")
    if "/" in uri:
        return uri
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-haiku-4-5-20251001"
    return "openai/gpt-4.1-mini"
```

Immediately after the `@app.get("/playground/playbooks")` handler (ends ~line 207), add the five routes:

```python
    @app.get("/playground/playbooks/{playbook_id}/source")
    async def get_source(playbook_id: str) -> dict[str, Any]:
        try:
            text = app.state.store.read(playbook_id)
        except KeyError:
            return {"ok": False, "error": f"unknown playbook {playbook_id!r}"}
        vr = validate_yaml(text)
        return {
            "ok": True,
            "id": playbook_id,
            "yaml": text,
            "draft": app.state.store.has_draft(playbook_id),
            "valid": vr.valid,
            "errors": vr.errors,
            "steps": vr.steps,
            "journey": vr.journey,
        }

    @app.post("/playground/playbooks/{playbook_id}/validate")
    async def validate_source(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        vr = validate_yaml(str(body.get("yaml", "")))
        return {
            "valid": vr.valid,
            "errors": vr.errors,
            "steps": vr.steps,
            "journey": vr.journey,
        }

    @app.put("/playground/playbooks/{playbook_id}/source")
    async def save_source(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        text = str(body.get("yaml", ""))
        vr = validate_yaml(text)
        if not vr.valid:
            return {"ok": False, "errors": vr.errors}
        try:
            app.state.store.save_draft(playbook_id, text)
        except KeyError:
            return {"ok": False, "errors": [f"unknown playbook {playbook_id!r}"]}
        return {"ok": True, "draft": True, "steps": vr.steps, "journey": vr.journey}

    @app.post("/playground/playbooks/{playbook_id}/publish")
    async def publish_source(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        text = body.get("yaml")
        if text is None:
            try:
                text = app.state.store.read(playbook_id)
            except KeyError:
                return {"ok": False, "errors": [f"unknown playbook {playbook_id!r}"]}
        vr = validate_yaml(str(text))
        if not vr.valid:
            return {"ok": False, "errors": vr.errors}
        try:
            app.state.store.publish(playbook_id, str(text))
        except KeyError:
            return {"ok": False, "errors": [f"unknown playbook {playbook_id!r}"]}
        return {"ok": True, "published": True}

    @app.post("/playground/playbooks/{playbook_id}/edit")
    async def edit_playbook(playbook_id: str, body: dict[str, Any]) -> dict[str, Any]:
        instruction = str(body.get("instruction", "")).strip()
        if not instruction:
            return {"ok": False, "error": "instruction is required"}
        current = body.get("yaml")
        if current is None:
            try:
                current = app.state.store.read(playbook_id)
            except KeyError:
                return {"ok": False, "error": f"unknown playbook {playbook_id!r}"}
        try:
            provider = resolve_llm(_edit_llm_uri())
            proposal = await propose_edit(
                str(current), instruction, provider.complete
            )
        except Exception as exc:  # noqa: BLE001 — surface any LLM/resolve failure
            logger.warning(f"[playground] edit failed: {exc}")
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "yaml": proposal.yaml,
            "summary": proposal.summary,
            "valid": proposal.valid,
            "errors": proposal.errors,
            "steps": proposal.steps,
            "journey": proposal.journey,
        }
```

Also add a `draft` flag to the existing `list_playbooks` items (helps the UI badge): inside its dict-comprehension add `"draft": app.state.store.has_draft(p.id),`.

**Step 4: Run → PASS.** Format + pyrefly. If the `with TestClient(...)` lifespan fails offline, wrap the lifespan's network calls defensively — but `/config` is the only networked route and is not exercised here.

**Step 5: Commit**

```bash
git add playground/harness/api.py tests/playground/test_playbook_endpoints.py
git commit -m "feat(playground): source/validate/save/publish/edit endpoints"
```

---

### Task 6: Gitignore drafts + register test modules

**Files:**
- Create: `playground/.gitignore`
- Modify: `scripts/run_tests.sh`

**Step 1:** Create `playground/.gitignore`:

```
.drafts/
```

**Step 2:** In `scripts/run_tests.sh`, register the new modules following the existing case pattern (run `bash scripts/run_tests.sh list` first to copy the exact style). Ensure these run:
`tests/playground/test_playbook_paths.py`, `test_playbook_store.py`, `test_playbook_edit.py`, `test_playbook_endpoints.py`.

**Step 3:** Run the whole backend group → all PASS.

**Step 4: Commit**

```bash
git add playground/.gitignore scripts/run_tests.sh
git commit -m "chore(playground): ignore .drafts/, register backend tests"
```

---

## Phase 2 — Frontend shell

### Task 7: `config.ts` — types, API helpers, footer formatter

**Files:**
- Modify: `playground/web/src/config.ts`
- Test: `playground/web/src/config.test.ts`

**Step 1: Write the failing test** (pure formatter only — fetch helpers are smoke-verified later)

```ts
// playground/web/src/config.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { footerLabel } from "./config";

test("footerLabel: valid", () => {
  assert.equal(
    footerLabel({ valid: true, errors: [], steps: 3, journey: "main" }),
    "Valid · 3 steps · journey: main",
  );
});

test("footerLabel: invalid shows first error", () => {
  assert.equal(
    footerLabel({ valid: false, errors: ["boom", "x"], steps: 0, journey: "" }),
    "Invalid · boom",
  );
});
```

**Step 2: Run → FAIL.** `cd playground/web && npm test`.

**Step 3: Implement — append to `config.ts`**

```ts
/** Validation verdict shape returned by the source/validate endpoints. */
export interface PlaybookValidation {
  valid: boolean;
  errors: string[];
  steps: number;
  journey: string;
}

export interface PlaybookSource extends PlaybookValidation {
  ok: boolean;
  id: string;
  yaml: string;
  draft: boolean;
  error?: string;
}

export interface SaveResult {
  ok: boolean;
  draft?: boolean;
  errors?: string[];
}

export interface EditResult extends PlaybookValidation {
  ok: boolean;
  yaml: string;
  summary: string;
  error?: string;
}

export function footerLabel(v: PlaybookValidation): string {
  if (v.valid) return `Valid · ${v.steps} steps · journey: ${v.journey}`;
  return `Invalid · ${v.errors[0] ?? "see editor"}`;
}

export async function fetchSource(id: string): Promise<PlaybookSource> {
  const resp = await fetch(`/playground/playbooks/${id}/source`);
  if (!resp.ok) throw new Error(`source fetch failed: HTTP ${resp.status}`);
  return resp.json() as Promise<PlaybookSource>;
}

export async function validateSource(
  id: string,
  yaml: string,
): Promise<PlaybookValidation> {
  const resp = await fetch(`/playground/playbooks/${id}/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ yaml }),
  });
  return resp.json() as Promise<PlaybookValidation>;
}

export async function saveSource(id: string, yaml: string): Promise<SaveResult> {
  const resp = await fetch(`/playground/playbooks/${id}/source`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ yaml }),
  });
  return resp.json() as Promise<SaveResult>;
}

export async function publishSource(
  id: string,
  yaml?: string,
): Promise<SaveResult> {
  const resp = await fetch(`/playground/playbooks/${id}/publish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(yaml === undefined ? {} : { yaml }),
  });
  return resp.json() as Promise<SaveResult>;
}

export async function editPlaybook(
  id: string,
  instruction: string,
  yaml?: string,
): Promise<EditResult> {
  const resp = await fetch(`/playground/playbooks/${id}/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(yaml === undefined ? { instruction } : { instruction, yaml }),
  });
  return resp.json() as Promise<EditResult>;
}
```

Also extend `PlaybookInfo` with `draft?: boolean;`.

**Step 4: Run → PASS.** Build to typecheck: `npm run build`.

**Step 5: Commit**

```bash
git add playground/web/src/config.ts playground/web/src/config.test.ts
git commit -m "feat(playground/web): playbook source/validate/save/publish/edit client API"
```

---

### Task 8: Two-pane `AgentView` shell

**Files:**
- Modify: `playground/web/src/pages/AgentView.tsx`

This is the structural change. Build + smoke-verify (no unit test for layout).

**Step 1:** Replace the `Tab` type and tab list. New left-pane tabs:

```tsx
type Tab = "preview" | "edit" | "conversation" | "metrics" | "traces" | "events";
type RightTab = "chat" | "stats" | "playbooks";
```

**Step 2:** Add state for the right pane and the active playbook's editor:

```tsx
const [tab, setTab] = useState<Tab>("preview");
const [rightPaneTab, setRightPaneTab] = useState<RightTab>("playbooks");
```

Keep `rightTab`/mode logic but force `mode = "playbook"` on connect (Flows dropped from UI): in `connect()`, replace the `mode` derivation with `const mode = "playbook";` and pass `activePlaybookRef.current`. Remove the `modeRef`/Flows-tab wiring from render (keep the imports/refs harmless or delete `FlowsList` import + `rightTab` state). Simplest: delete `FlowsList` import and the Flows branch; keep `selectPlaybook`.

**Step 3:** Replace the `return (...)` grid with the two-pane layout. The left pane renders the six tabs; **Events** renders `<EventsLog entries={events} />` (no fixed height — it now fills the tab body). The Composer renders only on `preview`/`conversation`.

```tsx
return (
  <div className="shell">
    <TopBar
      hasVoice={!!selectedVoiceProfile}
      hasFlow={hasSelection}
      flowName={selectionName}
      appState={appState}
      onConnect={connect}
      onDisconnect={disconnect}
    />
    <div className="panes" style={{ gridTemplateColumns: `1fr 10px ${rightColWidth}px` }}>
      {/* LEFT — WORK PANE */}
      <section className="workpane">
        <div className="center-tabs">
          {LEFT_TABS.map((t) => (
            <button
              key={t.id}
              className={`ctab${tab === t.id ? " on" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
              {t.id === "preview" && isTurnActive(convState) && <span className="tab-live" />}
            </button>
          ))}
        </div>
        <div className="center-body">
          {tab === "preview" && (
            <PipelinePanel
              live={appState === "active" || appState === "connecting"}
              convState={convState}
              userText={lastUserText}
              replyText={lastAgentText}
              node={currentNode}
            />
          )}
          {tab === "edit" && (
            <EditPanel playbookId={activePlaybook} appState={appState} />
          )}
          {tab === "conversation" && (
            <ConversationPanel turns={turns} appState={appState} />
          )}
          {tab === "metrics" && <MetricsPanel metrics={metrics} />}
          {tab === "traces" && (
            <DashboardPanel
              llmCalls={llmCalls}
              turnTimings={turnTimings}
              metrics={metrics}
              turns={turns}
            />
          )}
          {tab === "events" && <EventsLog entries={events} />}
        </div>
        {(tab === "preview" || tab === "conversation") && (
          <Composer appState={appState} speaking={speaking} />
        )}
      </section>

      <div
        className="col-resize-handle"
        onMouseDown={(e) =>
          startResize(e, "x", rightColWidth, setRightColWidth, -1, 280, 560)
        }
      />

      {/* RIGHT — CONTROL PANE */}
      <aside className="rail right controlpane">
        <div className="controlpane-head">
          <VoiceProfilePanel
            voiceProfiles={config.voice_profiles}
            selected={selectedVoiceProfile}
            onChange={setSelectedVoiceProfile}
            disabled={voiceLocked}
          />
        </div>
        <div className="rail-tabs" role="tablist" aria-label="Control">
          {RIGHT_TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={rightPaneTab === t.id}
              className={`rail-tab${rightPaneTab === t.id ? " on" : ""}`}
              onClick={() => setRightPaneTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="controlpane-body">
          {rightPaneTab === "chat" && (
            <ChatPanel playbookId={activePlaybook} onApplied={() => setTab("edit")} />
          )}
          {rightPaneTab === "stats" && (
            <StatsPanel
              appState={appState}
              agentReady={agentReady}
              convState={convState}
              session={session}
              voiceProfileName={voiceProfileName}
              activeLlm={config.active_llm}
              metrics={metrics}
              botLevel={botLevel}
            />
          )}
          {rightPaneTab === "playbooks" && (
            <PlaybookList
              playbooks={playbooks}
              activePlaybook={activePlaybook}
              appState={appState}
              onSelect={selectPlaybook}
            />
          )}
        </div>
      </aside>
    </div>
  </div>
);
```

Define the tab arrays above `return`:

```tsx
const LEFT_TABS: Array<{ id: Tab; label: string }> = [
  { id: "preview", label: "Preview" },
  { id: "edit", label: "Edit" },
  { id: "conversation", label: "Conversation" },
  { id: "metrics", label: "Metrics" },
  { id: "traces", label: "Traces" },
  { id: "events", label: "Events" },
];
const RIGHT_TABS: Array<{ id: RightTab; label: string }> = [
  { id: "chat", label: "Chat" },
  { id: "stats", label: "Stats" },
  { id: "playbooks", label: "Playbooks" },
];
```

Add imports for `EditPanel`, `ChatPanel`, `StatsPanel` (created in later tasks). Remove the bottom `EventsLog` + `row-resize-handle` block and the `eventsHeight`/`leftColWidth` state that are no longer used. Keep `rightColWidth`.

**Step 3 (stub the new components so it compiles now):** create minimal placeholders so Task 8 builds; later tasks flesh them out.

```tsx
// playground/web/src/components/EditPanel.tsx (placeholder)
import type { AppState } from "../types";
export function EditPanel(_: { playbookId: string; appState: AppState }) {
  return <div style={{ padding: 16 }}>Editor…</div>;
}
```
```tsx
// playground/web/src/components/ChatPanel.tsx (placeholder)
export function ChatPanel(_: { playbookId: string; onApplied: () => void }) {
  return <div style={{ padding: 16 }}>Chat…</div>;
}
```
```tsx
// playground/web/src/components/StatsPanel.tsx (placeholder — props match Task 15)
import type { AppState, MetricSnapshot, SessionInfo } from "../types";
import type { ConvState } from "../state/convState";
export function StatsPanel(_: {
  appState: AppState; agentReady: boolean; convState: ConvState;
  session: SessionInfo | null; voiceProfileName: string; activeLlm: string;
  metrics: MetricSnapshot; botLevel: number;
}) {
  return <div style={{ padding: 16 }}>Stats…</div>;
}
```

**Step 4: Build → PASS** (`npm run build`). Fix any unused-var TS errors from removed state.

**Step 5: Commit**

```bash
git add playground/web/src/pages/AgentView.tsx \
  playground/web/src/components/EditPanel.tsx \
  playground/web/src/components/ChatPanel.tsx \
  playground/web/src/components/StatsPanel.tsx
git commit -m "feat(playground/web): two-pane shell (work pane + control pane)"
```

---

### Task 9: Two-pane CSS

**Files:**
- Modify: `playground/web/src/style.css`

**Step 1:** Replace the `.grid` rule (lines ~155-163) usage by adding a `.panes` layout (keep `.grid` for safety or remove if unused):

```css
.panes {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 10px 360px; /* overridden inline for resize */
  column-gap: 0;
}
.workpane {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}
.workpane > .center-body { flex: 1; min-height: 0; }
.controlpane { display: flex; flex-direction: column; min-height: 0; padding: 0; }
.controlpane-head { padding: 14px 14px 0; overflow: visible; }
.controlpane-body { flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }
.controlpane-body > * { flex: 1; min-height: 0; }
```

**Step 2:** The Events log now lives inside a tab. Override its boxed chrome when nested so it fills the body (it uses primary tokens; neutralize the outer frame):

```css
.center-body > .events {
  height: auto;
  flex: 1;
  border: none;
  border-radius: 0;
  background: transparent;
}
```

**Step 3:** Ensure the voice dropdown isn't clipped: `.controlpane-head { overflow: visible; }` (above) and confirm `.vs-menu` (z-index 40) escapes. Because `.controlpane-body` is `overflow: hidden`, the dropdown lives in `.controlpane-head` which is visible — good.

**Step 4: Build the app and smoke-test:**

```bash
cd playground/web && npm run build && cd ../..
uv run python -m playground.run   # open http://localhost:9100
```
Verify: two panes; six left tabs switch; right pane shows dropdown + three tabs; Events renders in its tab; resize handle works; voice dropdown opens unclipped.

**Step 5: Commit**

```bash
git add playground/web/src/style.css
git commit -m "feat(playground/web): two-pane layout styles"
```

---

## Phase 3 — Edit tab (CodeMirror)

### Task 10: Install CodeMirror + `EditorPanel` (controlled editor)

**Files:**
- Modify: `playground/web/package.json` (+ lockfile)
- Replace: `playground/web/src/components/EditPanel.tsx` placeholder is kept; create a new low-level `CodeEditor.tsx`

**Step 1:** Install:

```bash
cd playground/web
npm install @codemirror/state @codemirror/view @codemirror/commands \
  @codemirror/lang-yaml @codemirror/theme-one-dark
cd ../..
```

**Step 2:** Create the controlled editor:

```tsx
// playground/web/src/components/CodeEditor.tsx
import { useEffect, useRef } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLine } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { yaml } from "@codemirror/lang-yaml";
import { oneDark } from "@codemirror/theme-one-dark";

interface CodeEditorProps {
  value: string;
  onChange: (next: string) => void;
  readOnly?: boolean;
}

export function CodeEditor({ value, onChange, readOnly }: CodeEditorProps) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    if (!host.current) return;
    const state = EditorState.create({
      doc: value,
      extensions: [
        lineNumbers(),
        highlightActiveLine(),
        history(),
        keymap.of([...defaultKeymap, ...historyKeymap]),
        yaml(),
        oneDark,
        EditorView.editable.of(!readOnly),
        EditorView.theme({ "&": { height: "100%" }, ".cm-scroller": { overflow: "auto" } }),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) onChangeRef.current(u.state.doc.toString());
        }),
      ],
    });
    const v = new EditorView({ state, parent: host.current });
    view.current = v;
    return () => v.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push external value changes (e.g. AI edit, playbook switch) into the editor.
  useEffect(() => {
    const v = view.current;
    if (!v) return;
    const cur = v.state.doc.toString();
    if (value !== cur) {
      v.dispatch({ changes: { from: 0, to: cur.length, insert: value } });
    }
  }, [value]);

  return <div className="codeeditor" ref={host} />;
}
```

**Step 3:** Build → PASS. Commit:

```bash
git add playground/web/package.json playground/web/package-lock.json \
  playground/web/src/components/CodeEditor.tsx
git commit -m "feat(playground/web): CodeMirror controlled editor"
```

---

### Task 11: `EditPanel` — load · validate (debounced) · save · publish

**Files:**
- Replace: `playground/web/src/components/EditPanel.tsx`
- Modify: `playground/web/src/style.css` (editor + footer chrome)

**Step 1:** Implement (consumes the Task 7 API + Task 10 editor):

```tsx
// playground/web/src/components/EditPanel.tsx
import { useEffect, useRef, useState } from "react";

import {
  fetchSource,
  validateSource,
  saveSource,
  publishSource,
  footerLabel,
  type PlaybookValidation,
} from "../config";
import type { AppState } from "../types";
import { CodeEditor } from "./CodeEditor";

interface EditPanelProps {
  playbookId: string;
  appState: AppState;
}

const NEUTRAL: PlaybookValidation = { valid: true, errors: [], steps: 0, journey: "" };

export function EditPanel({ playbookId, appState }: EditPanelProps) {
  const [yaml, setYaml] = useState("");
  const [dirty, setDirty] = useState(false);
  const [draft, setDraft] = useState(false);
  const [saving, setSaving] = useState<"" | "saving" | "saved" | "publishing">("");
  const [validation, setValidation] = useState<PlaybookValidation>(NEUTRAL);
  const debounce = useRef<number | undefined>(undefined);

  // Load source whenever the selected playbook changes.
  useEffect(() => {
    if (!playbookId) return;
    let alive = true;
    fetchSource(playbookId).then((s) => {
      if (!alive || !s.ok) return;
      setYaml(s.yaml);
      setDraft(s.draft);
      setDirty(false);
      setValidation({ valid: s.valid, errors: s.errors, steps: s.steps, journey: s.journey });
    });
    return () => {
      alive = false;
    };
  }, [playbookId]);

  function onChange(next: string) {
    setYaml(next);
    setDirty(true);
    setSaving("");
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      validateSource(playbookId, next).then(setValidation);
    }, 400);
  }

  async function onSave() {
    setSaving("saving");
    const res = await saveSource(playbookId, yaml);
    if (res.ok) {
      setDirty(false);
      setDraft(true);
      setSaving("saved");
    } else {
      setValidation((v) => ({ ...v, valid: false, errors: res.errors ?? v.errors }));
      setSaving("");
    }
  }

  async function onPublish() {
    setSaving("publishing");
    const res = await publishSource(playbookId, yaml);
    if (res.ok) {
      setDraft(false);
      setDirty(false);
      setSaving("saved");
    } else {
      setSaving("");
    }
  }

  if (!playbookId) {
    return <div className="edit-empty">Pick a playbook to edit.</div>;
  }

  return (
    <div className="editpanel">
      <div className="editpanel-bar">
        <span className={`edit-status ${validation.valid ? "ok" : "err"}`}>
          {footerLabel(validation)}
        </span>
        <span className="edit-actions">
          {draft && <span className="edit-draft">draft</span>}
          <button className="btn-mini" disabled={!dirty || saving === "saving"} onClick={onSave}>
            {saving === "saving" ? "Saving…" : dirty ? "Save" : "Saved"}
          </button>
          <button
            className="btn-mini publish"
            disabled={!validation.valid || saving === "publishing"}
            onClick={onPublish}
          >
            {saving === "publishing" ? "Publishing…" : "Publish"}
          </button>
        </span>
      </div>
      <CodeEditor value={yaml} onChange={onChange} readOnly={appState === "connecting"} />
    </div>
  );
}
```

**Step 2:** Add CSS:

```css
.editpanel { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.editpanel-bar {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 8px 12px; border-bottom: 1px solid var(--line-1);
  font-family: var(--mono); font-size: 12px;
}
.edit-status.ok { color: var(--mint); }
.edit-status.err { color: var(--rose); }
.edit-actions { display: flex; align-items: center; gap: 8px; }
.edit-draft { font-size: 10px; color: var(--amber); text-transform: uppercase; letter-spacing: 0.08em; }
.btn-mini {
  font: inherit; font-size: 12px; cursor: pointer; color: var(--fg-1);
  background: var(--bg-3); border: 1px solid var(--line-2); border-radius: var(--r-sm);
  padding: 5px 11px;
}
.btn-mini:disabled { opacity: 0.5; cursor: default; }
.btn-mini.publish { background: var(--accent); border-color: var(--accent); color: #fff; }
.codeeditor { flex: 1; min-height: 0; overflow: hidden; }
.edit-empty, .edit-empty { padding: 24px; color: var(--fg-3); }
```

**Step 3:** Build → PASS. Smoke: select a playbook → Edit tab shows its YAML, footer reads "Valid · N steps · journey: main", editing flips to Save, Save shows draft, Connect runs the draft.

**Step 4: Commit**

```bash
git add playground/web/src/components/EditPanel.tsx playground/web/src/style.css
git commit -m "feat(playground/web): Edit tab — load, validate, save draft, publish"
```

---

## Phase 4 — Chat tab (AI builder)

### Task 12: `ChatPanel`

**Files:**
- Replace: `playground/web/src/components/ChatPanel.tsx`
- Modify: `playground/web/src/style.css`

**Step 1:** Implement:

```tsx
// playground/web/src/components/ChatPanel.tsx
import { useState } from "react";

import { editPlaybook } from "../config";

interface ChatMsg {
  role: "user" | "agent";
  text: string;
}

const CHIPS = [
  "Add an SMS confirmation step",
  "Make the agent warmer",
  "Add a callback step",
  "Add a second language",
];

interface ChatPanelProps {
  playbookId: string;
  /** Called after a valid edit is applied (parent switches to the Edit tab). */
  onApplied: (yaml: string) => void;
}

export function ChatPanel({ playbookId, onApplied }: ChatPanelProps) {
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function send(text: string) {
    const instruction = text.trim();
    if (!instruction || busy || !playbookId) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", text: instruction }]);
    setBusy(true);
    const res = await editPlaybook(playbookId, instruction);
    setBusy(false);
    if (!res.ok) {
      setMsgs((m) => [...m, { role: "agent", text: `Couldn't do that: ${res.error ?? "error"}` }]);
      return;
    }
    if (!res.valid) {
      setMsgs((m) => [
        ...m,
        { role: "agent", text: `That produced invalid YAML, so I kept your version. (${res.errors[0] ?? ""})` },
      ]);
      return;
    }
    setMsgs((m) => [...m, { role: "agent", text: res.summary }]);
    onApplied(res.yaml);
  }

  return (
    <div className="chatpanel">
      <div className="chat-head">
        <span className="chat-title">Agent</span>
        <span className="chat-sub">building your playbook</span>
      </div>
      <div className="chat-thread">
        {msgs.length === 0 && (
          <p className="chat-hello">
            Tell me what to change, or pick a suggestion below. I'll edit the plan on the left.
          </p>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`chat-msg chat-msg--${m.role}`}>{m.text}</div>
        ))}
        {busy && <div className="chat-msg chat-msg--agent chat-typing">…</div>}
      </div>
      <div className="chat-chips">
        {CHIPS.map((c) => (
          <button key={c} className="chat-chip" disabled={busy} onClick={() => send(c)}>
            {c}
          </button>
        ))}
      </div>
      <form
        className="chat-composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the agent to change something…"
          disabled={busy || !playbookId}
        />
        <button type="submit" disabled={busy || !input.trim()}>↑</button>
      </form>
    </div>
  );
}
```

Update `AgentView`'s usage: `onApplied` now receives the YAML — the EditPanel reloads from server on `playbookId` change, but the AI edit is a draft only after Save. Simplest correct flow: the `/edit` endpoint returns YAML but does **not** persist; so after `onApplied`, switch to Edit and have EditPanel show the proposed YAML. To pass it through, lift the "pending YAML" to AgentView:

```tsx
const [pendingYaml, setPendingYaml] = useState<string | null>(null);
// ChatPanel onApplied:
onApplied={(y) => { setPendingYaml(y); setTab("edit"); }}
// EditPanel gets an extra optional prop:
<EditPanel playbookId={activePlaybook} appState={appState}
  injectedYaml={pendingYaml} onInjected={() => setPendingYaml(null)} />
```

In `EditPanel`, add `injectedYaml?: string | null; onInjected?: () => void;` and an effect: when `injectedYaml` is non-null, `setYaml(injectedYaml); setDirty(true)`, run validation, then `onInjected()`. (The user then reviews and clicks Save to persist as a draft.)

**Step 2:** Add CSS (thread, chips, composer) using `--bg-2/3`, `--user`, `--agent`, `--accent`, `--r-md`, `--mono`. Mirror the reference image's chip + composer look.

**Step 3:** Build → PASS. Smoke (needs an LLM key, or set `ACTIVE_LLM`): type "add an SMS confirmation step" → agent replies with a summary → Edit tab shows the rewritten YAML → Save persists the draft.

**Step 4: Commit**

```bash
git add playground/web/src/components/ChatPanel.tsx \
  playground/web/src/components/EditPanel.tsx \
  playground/web/src/pages/AgentView.tsx \
  playground/web/src/style.css
git commit -m "feat(playground/web): Chat tab — AI builder rewrites the plan YAML"
```

---

## Phase 5 — Polish

### Task 13: TopBar — Export · Publish · Saved + Connect

**Files:**
- Modify: `playground/web/src/components/TopBar.tsx`, `AgentView.tsx`, `style.css`

Add to `TopBarProps`: `playbookName: string`, `formatBadge?: string`, `saved: "saved" | "draft" | "dirty"`, `onExport: () => void`, `onPublish: () => void`. Render the brand → playbook name + format badge on the left, the `● Saved`/`draft` indicator + `Export`/`Publish` mid/right, keeping `Connect`/`Disconnect`. `onExport` downloads the current YAML (`new Blob([yaml]) → <a download>`); wire `yaml`/`saved` state up from `EditPanel` via a lifted `editorYaml` state in `AgentView` (or keep Export inside EditPanel and only surface Publish here). Build → smoke → commit.

### Task 14: Composer scoping verification

Already scoped in Task 8 (renders on `preview`/`conversation` only). Add a brief code comment explaining why. Build → commit (fold into Task 13 commit if trivial).

### Task 15: `StatsPanel` composition

**Files:**
- Replace: `playground/web/src/components/StatsPanel.tsx`, `style.css`

Compose the existing pieces: render `StatusPanel` (pass through `appState/agentReady/convState/session/voiceProfileName/activeLlm/latencyMs` — derive `latencyMs` from `metrics.ttfa_ms`), a "Bot audio" section wrapping `BotAudioPanel`, and a compact metrics block (reuse `MetricsPanel` or a slim variant). Wrap in a scrollable `.statspanel`. Build → smoke → commit:

```bash
git add playground/web/src/components/StatsPanel.tsx playground/web/src/style.css
git commit -m "feat(playground/web): Stats tab — status + bot audio + metrics"
```

### Task 16: Full verification + smoke

**Step 1:** Backend: `bash scripts/run_tests.sh playground_paths playground_store playground_edit playground_endpoints` (or the registered group) → all PASS.
**Step 2:** Frontend: `cd playground/web && npm test && npm run build` → PASS.
**Step 3:** `uv run python -m playground.run` → exercise every tab; select a playbook; edit + Save (draft) + Connect (runs draft); Chat an edit; Publish (canonical written, draft cleared); Export downloads YAML; Stats shows live data on a call.
**Step 4:** `uv run ruff check playground/ && uv run pyrefly check` clean for touched files.
**Step 5: Final commit** (any doc/cleanup):

```bash
git add -p   # stage only revamp files
git commit -m "chore(playground): revamp polish + verification"
```

Then: **REQUIRED SUB-SKILL** superpowers:finishing-a-development-branch.

---

## Notes & assumptions

- **Draft vs canonical:** Save → draft (Connect runs it); Publish → canonical (draft cleared). The `/playground/playbooks` list metadata reflects canonical until Publish — a known cosmetic lag; the `source` endpoint always returns the effective (draft) content + fresh validation.
- **AI edit does not auto-persist:** `/edit` returns YAML; the user reviews in Edit and clicks Save. Invalid LLM output is reported and the current YAML is preserved.
- **New-playbook creation** (no canonical) is out of scope — Save/Publish require an existing id. Future work, alongside the remote `RemotePlaybookStore` (speech-service API, user account, permissions).
- **Fonts:** Space Grotesk / Hanken Grotesk are referenced but not loaded; out of scope (system fallback), unless we add a `<link>` in a follow-up.
