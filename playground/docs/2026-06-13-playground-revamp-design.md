# Playground Revamp — Design

**Date:** 2026-06-13
**Status:** Validated, ready for planning
**Scope:** `playground/` — frontend (`web/`) + harness (`harness/`)

## 1. Goal

Restructure the playground from a 5-column rail layout into two purposeful
panes: a **work pane** (left) where you preview, author, and inspect a
playbook, and a **control pane** (right) where you pick a voice, build the
playbook by chatting with an AI, and run it. The reference is a playbook
author's IDE — edit the plan, watch it run, ask an agent to change it — not a
debug console.

The visual language does not change: the existing dark/purple tokens
(`--accent #7c3aed`, `--indigo`, Space Grotesk + Hanken Grotesk) already match
the target.

## 2. Layout: two panes

Replaces the `.grid` 5-column layout (`web/src/style.css:155`) and the
full-width bottom Events strip.

```
┌─ TopBar: ◐ <playbook name>  [format badge]   ● Saved  [Export] [Publish] [Connect] ┐
├────────────────────────────────────────────┬─────────────────────────────────────┤
│ LEFT — WORK PANE (flex: 1)                  │ RIGHT — CONTROL PANE (~360px, resize)│
│ Preview · Edit · Conversation · Metrics ·   │ Voice profile ▾    (pinned on top)  │
│ Traces · Events            (tab strip)      │ Chat · Stats · Playbooks   (tabs)   │
│ ┌────────────────────────────────────────┐ │ ┌─────────────────────────────────┐ │
│ │ active tab body                        │ │ │ active tab body                 │ │
│ ├────────────────────────────────────────┤ │ │                                 │ │
│ │ Composer (Preview + Conversation only) │ │ │                                 │ │
│ └────────────────────────────────────────┘ │ └─────────────────────────────────┘ │
└────────────────────────────────────────────┴─────────────────────────────────────┘
```

One resizable handle between the panes (reuse `col-resize-handle`).
`Connect`/`Disconnect` stays in the TopBar, joined by the reference's
`Export` / `Publish` actions and a `● Saved` draft-status indicator.

## 3. Left work pane — six tabs

Every tab maps to an existing component except **Edit** (new).

| Tab            | Component                          | Status        |
| -------------- | ---------------------------------- | ------------- |
| **Preview**    | `PipelinePanel` (pipeline animation) | move as-is; carries the live-turn dot |
| **Edit**       | **NEW** YAML editor                | build         |
| **Conversation** | `ConversationPanel` (transcript) | move as-is    |
| **Metrics**    | `MetricsPanel`                     | move as-is    |
| **Traces**     | `DashboardPanel` (today's "Trace") | move as-is    |
| **Events**     | `EventsLog`                        | relocate from bottom strip into a tab |

The **Composer** stays pinned beneath the left pane but renders only on
**Preview** and **Conversation** (the conversational tabs). It is voice-first /
display-only today; that does not change.

## 4. Right control pane — dropdown + three tabs

- **Voice profile ▾** — `VoiceProfilePanel`, pinned above the tabs, always visible.
- **Chat** — NEW AI playbook-builder agent (§9).
- **Stats** — `StatusPanel` (state/session/voice/LLM/latency pills) +
  `BotAudioPanel` (waveform) + a compact metrics readout. This is the old left
  rail, minus the voice dropdown (now pinned above the tabs).
- **Playbooks** — `PlaybookList`. Selecting a playbook loads its YAML into the
  Edit tab and arms it for the next Connect.

## 5. Dropped / kept

- **Flows dropped from the UI.** `FlowsList.tsx`, `fetchFlows`, `switchFlow`,
  and flow-mode in `runner.py`/`control.py` stay in code (unrendered). Connect
  always sends `mode=playbook`. No engine capability is deleted.
- **Events relocated**, not removed — now the sixth left tab.
- **Composer kept**, scoped to conversational tabs.

## 6. Backend — new harness endpoints

None of these exist today (recon confirmed). Each leans on superdialog
machinery that already exists.

| Endpoint                                   | Does                                                | Built on |
| ------------------------------------------ | --------------------------------------------------- | -------- |
| `GET  /playground/playbooks/{id}/source`   | Return YAML text (draft if present, else canonical) | `PlaybookStore.read` |
| `POST /playground/playbooks/{id}/validate` | `{yaml}` → `{valid, errors[], steps, journey}`      | `make_editable(yaml).compile()` (`editable.py:217`) |
| `PUT  /playground/playbooks/{id}/source`   | `{yaml}` → validate, then save draft                | `PlaybookStore.save_draft` |
| `POST /playground/playbooks/{id}/publish`  | Promote the draft to canonical                      | `PlaybookStore.publish` |
| `POST /playground/playbooks/{id}/edit`     | `{instruction, yaml}` → LLM rewrites → validate → `{yaml, summary, valid, errors}` | `resolve_llm(active_llm).complete()` (`resolver.py:10`) |

Validation reuses the canonical path: `make_editable(text).compile()` raises
`ValidationError` (field/type) or `ValueError` (unresolved references). On
success, `steps = len(pb.journeys[name].checkpoints)` and
`journey = list(pb.journeys)[0]` — the data behind the footer's
"**Valid · N steps · journey: main**".

## 7. Persistence — local drafts now, remote speech-service later

Persistence sits behind a **`PlaybookStore` seam** so the routes and UI never
change when the backing store does.

```python
class PlaybookStore(Protocol):
    def read(self, playbook_id: str) -> str: ...          # draft-if-exists else canonical
    def validate(self, yaml_text: str) -> ValidationResult: ...
    def save_draft(self, playbook_id: str, yaml_text: str) -> None: ...
    def publish(self, playbook_id: str, yaml_text: str) -> None: ...
    def has_draft(self, playbook_id: str) -> bool: ...
```

- **Now — `LocalDraftStore`:** drafts live in `playground/.drafts/<id>.yaml`
  (git-ignored). `read` prefers the draft; `save_draft` writes the overlay;
  `publish` writes back to the canonical `examples/playbooks/<id>.yaml`. The
  curated examples stay pristine until an explicit Publish. The registry's
  mtime cache (`playbooks.py`) means the next Connect runs the latest saved
  file with no restart.
- **Later — `RemotePlaybookStore`:** targets the speech-service API directly,
  scoped to the **current user's account** with **permission checks** and
  **richer metadata** (owner, versioning, publish state). Swapping the store
  binding is the only change; endpoints and frontend are untouched. This is the
  documented future seam, out of scope for this milestone.

## 8. Edit tab — editor + flow

**Editor:** **CodeMirror 6** (`codemirror`, `@codemirror/lang-yaml`, a small
custom dark theme on our tokens) — the single new frontend dependency. Gives
line numbers, YAML highlighting, and real editing; far lighter than Monaco.
(Today the app has no editor lib; LLM JSON renders as `<pre>`.)

**Flow:** select playbook → `GET …/source` loads YAML → edits set a dirty flag
→ debounced `POST …/validate` drives the footer (Valid / inline error markers)
→ **Save** = `PUT …/source` (server re-validates + writes draft) → `● Saved`.
**Publish** = `POST …/publish`. Next **Connect** runs the saved draft.

## 9. Chat tab — AI builder

Type an instruction or tap a suggestion chip → `POST …/edit {instruction,
current_yaml}`. The server builds a **schema-aware prompt** covering both the
*simple* format (`goal / persona / playbook[steps] / facts / interrupts` — what
the reference image uses) and the *full* `journeys / checkpoints` format, calls
the active LLM, and **validates the rewritten YAML before returning**.

The frontend appends the agent's one-line summary to the chat thread and pushes
the new YAML into the **Edit** tab.

**Safety:** if the rewrite fails validation, the endpoint returns it with
errors and the frontend **does not auto-apply** — the current YAML is
preserved and the agent reports it could not produce a valid change. v1 is
non-streaming; streaming the summary via `provider.stream()` is a later
enhancement.

Suggestion chips are canned instructions ("Add SMS confirmation", "Make the
agent warmer", "Add a callback", "Add a language").

## 10. Testing

- **Backend (pytest):** `validate` (valid / invalid / simple + full formats),
  `source` read (draft precedence), `save_draft` (persists, reloads, rejects
  invalid), `publish` (promotes to canonical), `edit` (mock LLM → validated
  YAML; invalid-LLM-output path preserves current YAML). Cover `LocalDraftStore`
  directly. Register modules in `scripts/run_tests.sh`.
- **Frontend:** unit-test the new `config.ts` helpers and the editor's
  validation-debounce reducer; keep `convState.test.ts` green.

## 11. Build sequence

1. **Backend store + endpoints** — `PlaybookStore` protocol, `LocalDraftStore`,
   five endpoints, validation helper, tests. (Independent of UI.)
2. **Frontend shell** — two-pane layout, left 6-tab strip, right dropdown +
   3-tab strip; relocate existing panels; drop Flows from render.
3. **Edit tab** — CodeMirror integration, load/validate/save wiring, footer.
4. **Chat tab** — AI-builder panel, chips, `…/edit` wiring, push-to-Edit.
5. **Polish** — Export/Publish/Saved in TopBar, Composer scoping, tests green,
   `task pg` smoke.

## 12. Open assumptions

- Edit operates on the file's existing format (simple or full); the editor does
  not convert between them.
- `Export` downloads the current YAML; `Publish` promotes the draft (and, in the
  future remote store, pushes to the speech service).
- The AI builder rewrites the full YAML (not the narrow `FullDoc.apply`
  whitelist), since instructions like "add a step" exceed prose-only edits.
