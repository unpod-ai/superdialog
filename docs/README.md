# SuperDialog — Documentation

**Status:** Canonical
**Parent:** [../README.md](../README.md)

SuperDialog is a **standalone open-source framework** for building
conversational brains. Text in, text out. Embeddable anywhere — LiveKit,
PipeCat, FastAPI, CLI, custom. Two engines behind one `Agent` protocol:
**DialogMachine**, the stable graph-railed dialog state machine, and
**Playbook**, the checkpoint compound runtime (streaming Talker + async
Director) for fluid, outcome-driven conversations.

This folder is the canonical documentation set.

---

## Contents

| Doc | Purpose |
|---|---|
| [00-overview.md](00-overview.md) | Positioning — what SuperDialog is, why standalone, why OSS, where each engine fits |
| [01-architecture.md](01-architecture.md) | Dual-engine internals — flow graph + DialogMachine runtime; Playbook event log, Talker/Director, process layer |
| [02-api-reference.md](02-api-reference.md) | Function signatures and worked examples for both engines |
| [03-embedding-guides.md](03-embedding-guides.md) | How to embed in LiveKit, PipeCat, FastAPI, CLI chatbot, unit tests |
| [04-playbook-guide.md](04-playbook-guide.md) | Playbooks in two parts — Part 1: authoring formats (simple + full); Part 2: technical design (runtime, process layer, evals/optimize) |
| [decisions.md](decisions.md) | OSS-specific decisions: license, repo, governance, roadmap |

---

## Where to start

- **New to SuperDialog?** Read [00-overview.md](00-overview.md), then run a
  quickstart from the [top-level README](../README.md).
- **Writing a new conversation?** Author a playbook —
  [04-playbook-guide.md](04-playbook-guide.md).
- **Operating an existing flow JSON?** It keeps working on DialogMachine;
  `compile_flow` migrates it to a playbook when you are ready
  ([04-playbook-guide.md](04-playbook-guide.md)).
- **Embedding into a host?** [03-embedding-guides.md](03-embedding-guides.md) —
  both engines implement the same `Agent` protocol, so every guide applies to
  either.
- **Looking up a signature?** [02-api-reference.md](02-api-reference.md).

---

## What this is NOT

- **Not a hosted service.** It's a Python library you pip install.
- **Not a voice framework.** It does not handle audio, STT, or TTS.
- **Not coupled to Unpod.** You can use it without ever creating an Unpod
  account.
- **Not a flow UI.** It accepts prompts, flow graphs, or playbook YAML;
  designing conversations in a visual editor is a downstream tool
  (n8n-style, future).
