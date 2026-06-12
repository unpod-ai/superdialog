# SuperDialog - Documentation

**Status:** Canonical
**Parent:** [../README.md](../README.md)

SuperDialog is a **standalone open-source framework** for building
conversational brains. Text in, text out. Embeddable anywhere - LiveKit,
PipeCat, FastAPI, CLI, custom. Two engines behind one `Agent` protocol:
**Playbook**, the default - a checkpoint compound runtime (streaming Talker + async
Director) for fluid, outcome-driven conversations, and
**DialogMachine**, the legacy graph-railed dialog state machine
(opt-in via `superdialog chat --mode flow`).

This folder is the canonical documentation set.

---

## Contents

| Doc | Purpose |
|---|---|
| [00-overview.md](00-overview.md) | Positioning - what SuperDialog is, why standalone, why OSS; Playbook as the default engine, DialogMachine as legacy mode |
| [01-architecture.md](01-architecture.md) | Engine internals - the Playbook runtime (event log, Talker/Director, process layer; the default) and the legacy DialogMachine flow graph |
| [02-api-reference.md](02-api-reference.md) | Function signatures and worked examples for the Playbook engine and the legacy DialogMachine |
| [03-embedding-guides.md](03-embedding-guides.md) | How to embed in LiveKit, PipeCat, FastAPI, CLI chatbot, unit tests |
| [04-playbook-guide.md](04-playbook-guide.md) | Playbooks in two parts - Part 1: authoring formats (simple + full); Part 2: technical design (runtime, process layer, evals/optimize) |
| [decisions.md](decisions.md) | OSS-specific decisions: license, repo, governance, roadmap |

---

## Where to start

- **New to SuperDialog?** Read [00-overview.md](00-overview.md), then run a
  quickstart from the [top-level README](../README.md).
- **Writing a new conversation?** Start with the simple playbook format -
  `superdialog generate "describe your agent"` writes one, and
  [04-playbook-guide.md](04-playbook-guide.md) Part 1 is the
  section-by-section reference (simple first, full format when you need
  tools, gates, or typed slots).
- **Operating an existing flow JSON?** It runs on the Playbook engine -
  `Playbook.load` auto-detects flow JSON and compiles it via `compile_flow`;
  `--mode flow` opts into the legacy DialogMachine
  ([04-playbook-guide.md](04-playbook-guide.md)).
- **Embedding into a host?** [03-embedding-guides.md](03-embedding-guides.md) -
  every guide runs on the default Playbook engine; the legacy
  DialogMachine implements the same `Agent` protocol, so each guide applies
  to it too.
- **Looking up a signature?** [02-api-reference.md](02-api-reference.md).

---

## What this is NOT

- **Not a hosted service.** It's a Python library you pip install.
- **Not a voice framework.** It does not handle audio, STT, or TTS.
- **Not coupled to Unpod.** You can use it without ever creating an Unpod
  account.
- **Not a flow UI.** It accepts prompts, playbook YAML, or legacy flow JSON;
  designing conversations in a visual editor is a downstream tool
  (n8n-style, future).
