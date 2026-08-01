# SuperDialog - Documentation

**Status:** Canonical
**Parent:** [../README.md](../README.md)

SuperDialog is a **standalone open-source framework** for building
conversational brains. Text in, text out. Embeddable anywhere - LiveKit,
PipeCat, FastAPI, CLI, custom. Two engines behind one `Agent` protocol:
**Playbook**, the default - a checkpoint compound runtime (streaming Talker + async
Director) for fluid, outcome-driven conversations, and the legacy
**graph engine**, a graph-railed dialog state machine (opt-in via
`superdialog chat --mode flow`). `DialogMachine` is the facade over both,
not a synonym for the legacy engine -
[01-architecture.md](01-architecture.md) §1 disambiguates.

This folder is the canonical documentation set.

---

## Contents

| Doc | Purpose |
|---|---|
| [00-overview.md](00-overview.md) | Positioning - what SuperDialog is, why standalone, why OSS; the terminology canon shared verbatim with the supervoice and unpod-sdk doc sets plus this repo's own names (§2); Playbook as the default engine, the graph engine as legacy mode; the shipped-capability table |
| [01-architecture.md](01-architecture.md) | Engine internals - the Playbook runtime (event log, Talker/Director, process layer; the default), the legacy flow graph, the naming disambiguation callout (§1), and observability + LLM routing (§5) |
| [02-api-reference.md](02-api-reference.md) | Function signatures, the full artifact model (`Playbook` / `Checkpoint` / `SlotSpec` / `ToolSpec` field tables), the CLI table, and worked examples for both engines |
| [03-embedding-guides.md](03-embedding-guides.md) | How to embed in LiveKit, PipeCat, FastAPI, CLI chatbot, unit tests |
| [04-playbook-guide.md](04-playbook-guide.md) | Playbooks in two parts - Part 1: authoring formats (simple + full, `guidelines:`, `llm:`, `strict`, `resolve_from`, multi-entity, pronunciations); Part 2: technical design (runtime, process layer, tool tiers + rewind, the Loop-2 Supervisor, speech control, evals/optimize) |
| [05-eval-guide.md](05-eval-guide.md) | Playbook-vs-vanilla A/B eval - `superdialog eval run`, the two pluggable seams (transport + metric framework), transcript-only scoring, and the two mutually-exclusive RAGAS pins |
| [06-playbook-execution-flow.md](06-playbook-execution-flow.md) | Code-verified execution trace - the two-brain turn loop, every touchpoint cited by symbol, the render steering surface, playbook semantics → behavior, patterns, and measured advantages vs a vanilla LLM |
| [07-running-evals.md](07-running-evals.md) | Operator runbook - setup, the `eval bench` fast path and the explicit gen-dataset/run/serve phases, every flag, how to read the report, dataset format, recipes, and gotchas |
| [08-integrations.md](08-integrations.md) | Integration contracts - a playbook behind an OpenAI-compatible endpoint (the dev `eval serve` door and the production supervoice pool), consuming an endpoint back via `register_llm_provider`, the two unrelated LiveKit integrations disambiguated, and the `SessionInit` / `agent_factory_ctx` embedding seam |
| [decisions.md](decisions.md) | OSS-specific decisions: license, repo, governance, roadmap, and the itemised unversioned shipped wave |

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
  `--mode flow` opts into the legacy graph engine
  ([04-playbook-guide.md](04-playbook-guide.md)).
- **Embedding into a host?** [03-embedding-guides.md](03-embedding-guides.md) -
  every guide runs on the default Playbook engine; the legacy graph
  engine implements the same `Agent` protocol, so each guide applies
  to it too. For the contracts behind those guides - OpenAI-compatible
  serving, custom LLM providers, the two LiveKit integrations, the
  `SessionWorker` seam - see [08-integrations.md](08-integrations.md).
- **Tuning a call that already runs?** [04-playbook-guide.md](04-playbook-guide.md)
  Part 2 - speech control and localized barrier lines (§8), tool tiers and
  rewind (§7), and the opt-in Loop-2 Supervisor for conversations that
  derail across turns (§7).
- **Looking up a signature or a model field?**
  [02-api-reference.md](02-api-reference.md).
- **Auditing whether the engine beats a plain prompt?**
  [05-eval-guide.md](05-eval-guide.md) - A/B a playbook against a vanilla LLM
  handed the same playbook, scored from the transcript alone. To actually
  *run* one, follow [07-running-evals.md](07-running-evals.md).

---

## What this is NOT

- **Not a hosted service.** It's a Python library you pip install.
- **Not a voice framework.** It does not handle audio, STT, or TTS.
- **Not coupled to Unpod.** You can use it without ever creating an Unpod
  account.
- **Not a flow UI.** It accepts prompts, playbook YAML, or legacy flow JSON;
  designing conversations in a visual editor is a downstream tool
  (n8n-style, future).
