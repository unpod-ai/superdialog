# SuperDialog — Decisions

**Status:** Canonical for the SuperDialog product
**Parent:** [README.md](README.md)

This is the decision log for the SuperDialog OSS framework only. Decisions about Voice Infra live in [../voice-infra/wiki/decisions.md](../voice-infra/wiki/decisions.md).

---

## 1. Resolved

| # | Decision | Resolution |
|---|---|---|
| 1 | Standalone library, not a service | **Library.** Python package, in-process. No daemon, no server requirement. |
| 2 | Open source | **Yes.** Permissive license (see #3). |
| 3 | License | **Apache 2.0.** [assumption — confirm with legal] — chosen for patent grant and corporate adoption ease. MIT considered; Apache preferred for the patent clause. |
| 4 | Repo location | **Separate public repo** (e.g. `github.com/unpod/super-dialog`). Not in the main Unpod monorepo. Independent release train. |
| 5 | Coupling to Unpod Voice Infra | **None.** Voice Infra depends on SuperDialog as one option; SuperDialog does not import any Unpod-platform code. |
| 6 | Language | **Python first.** TypeScript later if community asks. |
| 7 | LLM URI scheme | **`provider/model`**, LiveKit/litellm-style. See [01-architecture.md §2.3](01-architecture.md). |
| 8 | Custom LLM provider scope | **Process-global** registry via `register_llm_provider(...)`. One mental model. |
| 9 | Tool interface | **Three shapes, one method:** `PythonTool`, `HttpTool`, `MCPTool`. All register through `DialogMachine(tools=[...])`. |
| 10 | Streaming | **Opt-in via `stream=` flag.** One `turn()` method. `stream=False` returns `Turn`; `stream="text"` returns async iterator. |
| 11 | Mid-conversation model swap | **`set_llm(uri)` applies to next turn.** In-flight streaming continues on old model. |
| 12 | Multi-flow switching | **`FlowSet` + `switch_flow(name)`.** Pattern: many small flows, not one big graph. |
| 13 | Memory backend | **Pluggable.** v0.2: `InMemorySessionStore`, `NullSessionStore` via `SessionWorker`. Distributed backends (`RedisSessionStore`, `FileSessionStore`, `SQLiteSessionStore`) planned v0.3. |
| 14 | Eval harness inclusion | **Yes, first-class — planned v0.3.** `Eval` class is the differentiator but not yet shipped. |
| 15 | CLI chatbot mode | **Yes, first-class.** `superdialog chat <flow.json>` for testing without infrastructure. |
| 16 | Adapters in core or separate packages | **In core.** `superdialog.adapters.{livekit, pipecat, fastapi, websocket}`. Optional dependencies (extras: `pip install superdialog[livekit]`). |
| 17 | Unpod-hosted LLM URI scheme | **`unpod/<vertical>`**, e.g. `unpod/insurance-v1`. Available when registered; not required. |

---

## 2. Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Pre-release** | Hardening existing dialog state machine code, OSS license decisions, repo split | done |
| **v0.1** | Hard-port engine + flow models + LLMProvider + Tool ABC + DialogMachine facade + LiveKit/PipeCat/FastAPI/WS adapters + CLI (chat / flow lint / flow draw / flow generate) + ported dialog_machine test suite | **shipped (this port)** |
| **v0.2** | Port `eval/` (FlowEvaluator, CorpusGenerator, ResponseCache, FlowGraphAnalyzer) + `superdialog eval` CLI | triggered when OSS users ask for an A/B model harness |
| **v0.3** | Persistent memory backends (`RedisMemory`, `FileMemory`, `SQLiteMemory`) | triggered when a long-lived chat use case lands |
| **v0.4** | Q4 flip → A: make `super/core/voice/dialog_machine/__init__.py` a re-export shim; migrate `super_services` voice callers from `SimpleFlowAgent` to `DialogMachineLLM`; full streaming inference via `provider.stream()` | once v0.1 is stable in production via parallel-lives |
| **v0.5** | Decision on `langGraph/` / `langchain/` — drop or port | only if real demand surfaces |
| **v1.0** | API stability commitment, semantic versioning, split to `github.com/unpod/superdialog` | after v0.4 stabilises |

### v0.1 follow-ups carried over

The slim port deferred a handful of dialog_machine adapters and eval modules; the
corresponding tests are collect-ignored in `superdialog/tests/dialog_machine/conftest.py`:

- `superdialog.machine.adapters.simple_agent` — referenced by `test_simple_flow_agent`, `test_scope_build_invariant`, `TestPhase2/Phase3` in `test_language_tracking`.
- `superdialog.machine.adapters.livekit_bridge` — referenced by `test_livekit_bridge`, `test_gated_traversal_e2e`, `TestLivekitBridgeCustomTools`.
- `superdialog.machine.adapters.flow_executor` — referenced by `test_flow_executor_*` suites.
- `superdialog.machine.eval.*` — feeds the v0.2 eval port.

The gate-ordering regressions found by `test_gated_traversal.py` were fixed
in the verification sweep: the criteria/user-spoke gate now runs before the
premature-final guard, auto-proceed source nodes bypass the premature-final
check, and the default `MIN_TURNS_BEFORE_FINAL_NODE` was relaxed to `1`
(matching legacy behaviour; the env var still lets production opt into a
stricter floor).

---

## 3. Anti-goals (will not build)

| Anti-goal | Reason |
|---|---|
| Audio handling, STT, TTS | Out of scope. Belongs to host (Voice Infra, LiveKit, PipeCat). |
| Visual flow editor (n8n-style UI) | Separate product, not part of this library. |
| Multi-modal (vision, audio inputs) at interface level | Text only at the boundary. Multi-modal via tools if needed. |
| Hosted service requiring Unpod account | Library, never a service. |
| Freemium gating of critical features | Fully usable without paying. Paid product is Voice Infra. |
| Tight coupling to a specific LLM vendor | URI scheme is the abstraction; no vendor priority. |

---

## 4. Open questions

| # | Question | Why it matters |
|---|---|---|
| 1 | License choice (Apache 2.0 vs MIT) | Confirm with legal; Apache leans corporate, MIT leans community |
| 2 | Repo: monorepo with `super-dialog/` or own org? | Visibility and contribution friction tradeoff |
| 3 | Telemetry: opt-in usage pings? | Need data on adoption funnel; must not become surveillance |
| 4 | Public benchmarks against LangGraph / LangChain | When is the right time — too early invites comparison before maturity |
| 5 | Governance model | Maintainer team, RFC process, public roadmap — needed before any external contribution |
| 6 | TypeScript port priority | Wait for demand signal or pre-empt? |
| 7 | Memory backend defaults beyond in-memory and Redis | What ships in v0.2? SQLite? Postgres? |
| 8 | Eval corpus formats — JSONL only or also CSV / YAML? | Tooling ecosystem fit |

---

## 5. Cross-references

- Product overview: [00-overview.md](00-overview.md)
- Architecture: [01-architecture.md](01-architecture.md)
- API reference: [02-api-reference.md](02-api-reference.md)
- Embedding guides: [03-embedding-guides.md](03-embedding-guides.md)
- Voice Infra (the other product): [../voice-infra/](../voice-infra/)
- Strategic framing: [../00-two-products.md](../00-two-products.md)

---

## 6. Decision records

### 2026-06-12 — Checkpoint compound architecture (Playbook engine)

**Context.** The graph-railed state machine alone cannot deliver fluid
conversations: users do not follow graphs, and flexibility had accumulated as
~6 stacked escape hatches (`__stay_on_node__`, global edges + intent stack,
`allow_skip`, fallback edges, smart-skip, auto-proceed chains), each patching
one failure mode and interacting subtly with the rest. Every turn cost two
serial LLM calls (route, then speak), so streaming was cosmetic — the user
waited for both calls before hearing anything. Info extraction was split
across multiple mechanisms with no unified schema. Design rationale:
[plans/2026-06-10-checkpoint-compound-architecture-design.md](plans/2026-06-10-checkpoint-compound-architecture-design.md).

**Decision.** Ship a second engine, `superdialog.playbook`:

- A declarative **Playbook** artifact (YAML/JSON or Python): journeys of
  checkpoints — goal, typed slots, guidance, ordered `advance_when` rules —
  plus a process layer (tools, pipelines, handlers, interrupts, policies).
  Checkpoints gate **outcomes, not utterances**.
- A **Talker/Director compound runtime**: a fast Talker streams every spoken
  turn with one LLM call; an async Director extracts slots, judges
  advancement, runs tools, and writes steering/repair notes. Soft checkpoints
  never block; hard gates barrier the Talker until the Director settles.
- An **event-sourced log** (`EventLog` → `ConversationState` fold) as the
  single source of truth — also the audit, replay, and eval artifact.
- `PlaybookAgent` implements the existing `Agent` protocol, so SessionWorker
  and all host adapters run playbooks unchanged.
- The legacy machine (`superdialog.machine`, DialogMachine) stays **fully
  supported in maintenance**; `compile_flow` + `coverage_report` convert
  existing flow JSON into playbooks (validated against a 61-node production
  booking flow) for migration when teams choose to move.

**Consequences.**

- Real token streaming, barge-in safe — TTFT is one fast model's TTFT, not
  route-then-speak.
- Exactly one LLM call on the speech path; judgment and tools move off the
  critical path into the Director.
- The event log becomes a replay/eval substrate (`replay`, `run_session`,
  `run_eval`) — recorded sessions re-run against changed prompts or code.
- Two engines to document and support; positioning must be explicit (flows
  keep working, playbooks are where new investment goes).
- The `superdialog optimize` command (run → eval → improve loop over playbook
  artifacts) is **deferred** — the event log and eval bridge are the
  substrate for it, but the command does not exist yet.

**Alternatives considered.**

- *Patch the machine further* (a seventh escape hatch, smarter routing
  prompts) — **rejected**: the foundation is too rigid; each patch increased
  the interaction surface between the existing six and none removed the two
  serial calls from the speech path.
- *Full agent-harness free-for-all* (drop structure, let one agentic LLM
  loop with tools) — **rejected**: no outcome gating. Business conversations
  need typed extraction, gated irreversible steps, and auditable progression
  — exactly what checkpoints keep and a free-running agent gives up.
