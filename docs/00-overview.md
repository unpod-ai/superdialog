# SuperDialog — Overview

**Status:** Canonical
**Parent:** [README.md](README.md)

---

## 1. What it is

A Python library that turns a prompt, a flow graph, or a checkpoint playbook into
an executable conversation runtime. Pure text in, pure text out. Plays the role of
the "brain" in conversational systems.

Its core engine is the **Playbook** runtime — checkpoints gate outcomes
while the model owns the phrasing — and it is the **default everywhere**:
the unified loader runs full playbooks, simple-format playbooks, and legacy
flow JSON (auto-compiled via `compile_flow`) on the same engine, so users
never have to think about which artifact they hold. The legacy
**DialogMachine** graph engine remains fully supported behind the same
`Agent` protocol as an explicit opt-in (`--mode flow` in the CLI); the
surrounding machinery (sessions, adapters, tools) is shared (see §7).

## 2. Why standalone

Two reasons:

**(a) The brain has natural reuse beyond voice.** A conversation brain that runs a
customer-onboarding journey works the same whether the user is on a phone, a
WhatsApp thread, an Intercom widget, or a CLI test harness. Coupling it to
telephony forecloses every non-voice use case.

**(b) The dependency direction matters.** Voice Infrastructure should depend on
SuperDialog (as one brain option), not the other way around. Putting SuperDialog
inside the platform makes the platform non-modular and the framework non-portable.

> *"उस machine को release करने का more less idea यह है... इस architecture से इन इस infrastructure से उसका कोई लेना देना नहीं है."*

## 3. Why OSS

- **Community pull.** LiveKit and PipeCat owe their adoption to OSS. Releasing a
  strong dialog framework — with good docs, working LiveKit/PipeCat adapters, and
  a CLI chatbot mode for evaluation — creates a top-of-funnel that no closed
  product can match.
- **Lower support burden.** Developers who build complex flows and playbooks will
  keep modifying them. If the framework is theirs to fork, our team is not in the
  loop for every prompt change.
- **Trust.** Buyers who don't want vendor lock-in see an open core and engage
  further. The closed parts (telephony, voice profiles) are the parts they don't
  care about owning.

## 4. Why it ships first

Three reasons:

**(a) It already exists.** The dialog engine code is the most mature part of the
Unpod stack. Polishing it for OSS release is faster than building new telephony
infrastructure.

**(b) Independent shippability.** It needs no telephony, no speech, no media
server, no Room — none of the platform pieces. Therefore nothing on the platform
side gates it.

**(c) Validation channel.** Public release is the cheapest way to learn whether
the framework actually solves the *"developer wants to own their flow"* problem we
hypothesize. If the OSS adoption signal is weak, the Voice Infra GTM (which
depends on the same hypothesis) needs rethinking before we burn cycles on it.

## 5. Positioning

SuperDialog is to **conversation flow** what n8n is to **integration workflow** —
a simple, composable, eval-able runtime for orchestrating turn-by-turn logic.
Where LangChain and LangGraph expose agent primitives, SuperDialog focuses
narrowly on the conversational core: who speaks next, what to say while tools
run, which checkpoint or flow the conversation is in, when to call a tool, when
to escalate, and which outcome the session ended with.

It is intentionally smaller than LangChain in surface area. The pitch is: *"if
your problem is conversation state, this is the right size."*

## 6. Audiences

| Audience | Why they care |
|---|---|
| **Voice developer using LiveKit / PipeCat today** | Drop SuperDialog in as the brain; stop hand-writing turn logic. `PlaybookAgent` gives real token streaming through the same adapters; DialogMachine remains for graph-railed flows |
| **Chatbot developer (text-only)** | Use either engine directly with FastAPI; test DialogMachine flows as a CLI chat; drive playbooks through the `Agent` protocol |
| **Developer with compliance / scripted flows** | Author the flow graph as the spec — every path enumerable and lintable — and run it compiled on the Playbook engine (default), or on DialogMachine via `--mode flow` |
| **Developer whose calls must feel human** | Playbook: checkpoints gate outcomes while the model owns the phrasing; event log gives replay and eval for free |
| **Enterprise dev with their own LLM** | Plug their custom LLM URI (`custom/internal/...`) into DialogMachine, or any streaming/completing model pair into Playbook's `StreamsLLM`/`CompletesLLM` protocols |
| **Unpod Voice Infra customer** | SuperDialog is the default brain Unpod offers; same code runs locally and inside Unpod cloud |

## 7. Two engines

**DialogMachine** (`superdialog.DialogMachine`; engine internals live in
`superdialog.machine`) is the graph-railed state machine: the
flow graph decides what is *possible* and the LLM picks among the outgoing edges.
Every transition is authored, every reachable utterance path is enumerable, and
the CLI can lint and draw the graph. That makes it strong where determinism is
the point — compliance scripts, regulated disclosures, IVR replacement, any flow
where "the agent must never improvise" is a feature.

**Playbook** (`superdialog.playbook`) inverts the ownership: **checkpoints gate
outcomes, not utterances.** A playbook declares journeys of checkpoints — goal,
typed slots, guidance prose, ordered advance rules — plus a process layer of
tools, pipelines, handlers, interrupts, and policies. At runtime a fast **Talker**
streams every spoken turn with a single LLM call, while an async **Director**
extracts slots, judges advancement, runs tools, and writes steering/repair notes
— both over an append-only, event-sourced log that doubles as the audit, replay,
and eval artifact. Soft checkpoints never block; hard gates barrier the Talker at
irreversible moments (payments, identity) until the Director's verdict lands.

Why a second engine exists, honestly: users don't follow graphs. The graph-railed
model accumulated roughly six stacked escape hatches (`__stay_on_node__`, global
edges + intent stack, `allow_skip`, fallback edges, smart-skip, auto-proceed
chains) — each patching one failure mode, each interacting subtly with the rest —
and still cost two serial LLM calls per turn with only cosmetic streaming. Rather
than bolt on a seventh hatch, the Playbook engine moves fluidity to the model and
keeps the framework's authority where it belongs: on outcomes.

| | DialogMachine (legacy, opt-in) | Playbook (default) |
|---|---|---|
| Authoring unit | Graph node + edges | Checkpoint (goal, slots, guidance, advance rules) |
| Who owns fluidity | The graph | The model, inside checkpoints |
| What is gated | Every transition | Outcomes (slots + advance rules; hard gates where irreversible) |
| LLM calls on speech path | Two serial (route, then speak) | One streaming call; Director runs async |
| Streaming | Chunked after the fact | Real token streaming, barge-in safe |
| State | Snapshot context | Event-sourced log (replay, audit, eval) |
| Best for | Deterministic compliance flows | Conversations that must feel human |

Both engines sit behind the existing `superdialog.agent.Agent` protocol, so
`SessionWorker` and the host adapters run either one unchanged:

```python
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

agent = PlaybookAgent(
    playbook=Playbook.load("booking.yaml"),
    talker_llm=talker,      # StreamsLLM: stream(messages) -> AsyncIterator[str]
    director_llm=director,  # CompletesLLM: await complete(messages) -> str
    http=httpx_http,
)
result = await agent.turn("hello")
```

**Migration, not replacement.** Existing flows keep working — by default
they now run **compiled on the Playbook engine**: `Playbook.load` detects
flow JSON and converts it via `compile_flow`, with `coverage_report(flow,
pb)` proving every node, edge, and action mapped (validated against a
61-node production booking flow). DialogMachine remains fully supported
for anyone who wants the original graph runtime — pass `--mode flow`. New
investment goes into the Playbook engine.

**Roadmap (future, not built yet):** a `superdialog optimize` command closing the
run → eval → improve loop over playbook artifacts; a playbook CLI chat mode; host
plumbing that feeds voice events (silence timeouts, barge-in signals) into the
event log as external events. Today the text path through the `Agent` protocol
works with all existing adapters.

## 8. What it does well

| Capability | Status |
|---|---|
| Prompt → flow: `await create_dialog_flow(prompt=..., llm=...)` | shipped (v0.1) |
| Turn execution: `await dialog_machine.turn(text)` | shipped (v0.1) |
| LLM provider abstraction (model URIs) | shipped (v0.1) |
| Tools: Python callables, HTTP endpoints, MCP servers | shipped (v0.1) |
| Mid-conversation flow switching (`FlowSet`, `switch_flow`) | shipped (v0.1) |
| CLI: `chat`, `flow lint / draw / generate` (DialogMachine) | shipped (v0.1) |
| Adapters: LiveKit `DialogMachineLLM`, PipeCat `make_processor`, FastAPI, WebSocket | shipped (v0.1) |
| `Agent` Protocol + `Session` + `SessionWorker` (multi-conversation lifecycle, in-process persistence, per-session locking) | shipped (v0.2) |
| `LLMAgent`, `LangChainAgent` (non-DM brains usable in SessionWorker; LangChainAgent via the `langchain` extra, `superdialog.agents.langchain_agent`) | shipped (v0.2) |
| `assist(text)` (renamed from `inject_system`) | shipped (v0.2) |
| Playbook engine: checkpoints, Talker/Director compound, event-sourced log (`superdialog.playbook`) | shipped |
| Real token streaming, barge-in safe, behind the `Agent` protocol (`PlaybookAgent`) | shipped |
| Sandboxed declarative tools and Director pipelines (HTTP + python, retry/middleware) | shipped |
| Flow → playbook migration: `compile_flow`, `coverage_report` | shipped |
| Replay + persona eval bridge: `replay`, `run_session`, `run_eval` | shipped |
| Distributed stores (`RedisSessionStore`, `FileSessionStore`, `SQLiteSessionStore`) + `RedisLockBackend` | planned |
| Pluggable HTTP auth (`BearerAuth`, `BasicAuth`, callable) | planned |
| `Eval` harness + `superdialog eval` CLI | planned |
| `superdialog optimize` (playbook run → eval → improve loop) | roadmap (future) |

## 9. What it explicitly is not

- **Not a UI flow designer.** That belongs to a downstream tool (future,
  n8n-style).
- **Not a voice framework.** Audio/STT/TTS are out of scope. (Playbook's Talker
  streams text tokens; the host turns them into speech.)
- **Not multi-modal.** Text only at the interface. (Vision/audio inputs through
  tools, if needed.)
- **Not a hosted service.** A library. Hosting is offered by Voice Infra for
  those who want it.

## 10. Success criteria

- **GitHub stars and forks.** Baseline target TBD, but real numbers — not vanity
  metrics.
- **Adapter usage.** Are developers actually plugging SuperDialog into LiveKit
  and PipeCat? Telemetry from optional usage pings if they opt in.
- **Eval adoption.** Are developers running evals — the playbook replay/eval
  bridge today, the full harness later — or just using the runtime? The eval is
  part of what differentiates this from "yet another agent framework."
- **Playbook uptake.** Of new projects, how many author playbooks (or compile
  flows into them) versus staying on raw graphs? This signals whether the
  checkpoint model earns its keep.
- **Issue and PR volume.** OSS health.
- **Unpod Voice Infra trial conversion.** Of the OSS users who try Voice Infra,
  what fraction stick? This is the funnel justification for releasing the
  framework freely.

## 11. Anti-goals

We will refuse to:
- Add features that only matter on a phone call (audio handling, RTP, SIP, etc.).
- Tie OSS releases to Unpod account creation.
- Use the OSS as a freemium ladder where critical features are paid. The
  framework is fully usable without ever paying Unpod.

The paid product is the Speech Pipe. The framework is the loss leader.
