# SuperDialog

**Standalone dialog framework. Pure text in, pure text out.**

SuperDialog is the **brain** layer for conversational systems. It ships two
engines behind one text interface: **DialogMachine**, the graph-railed dialog
state machine that executes flow graphs deterministically, and **Playbook**,
a checkpoint-based compound runtime — a fast streaming Talker plus an async
Director — for fluid, outcome-driven conversations. Both manage turn-by-turn
logic, tool calls, transitions, and conversation memory; both speak the same
`Agent` protocol, so every host adapter runs either one unchanged.

```
User text → agent.turn() → reply text
```

Audio, STT, TTS, telephony, and media servers are out of scope - those belong to
voice infrastructure like LiveKit, PipeCat, or the Unpod Voice Platform.
SuperDialog ends at text in, text out.

> SuperDialog is to **conversation flow** what n8n is to **integration
> workflow** - a small, composable, eval-able runtime for orchestrating
> turn-by-turn logic. Where LangChain and LangGraph expose general agent
> primitives, SuperDialog focuses narrowly on the conversational runtime:
> who speaks next, what to extract, when to call a tool, when to escalate.

---

## Why standalone

**The brain has natural reuse beyond voice.** A dialog brain that runs a
customer-onboarding conversation works the same whether the user is on a phone,
a WhatsApp thread, an Intercom widget, or a CLI test harness. Coupling it to
telephony forecloses every non-voice use case.

**The dependency direction matters.** Voice infrastructure should depend on
SuperDialog (as one brain option), not the other way around - keeping the
framework portable and the platform composable.

Because the interface is text-only, **every dialog is a unit-testable function.**
No audio fixtures, no API keys, no phone number to test a conversation.

## Install

```bash
pip install superdialog
```

Install only the extras you need:

```bash
pip install superdialog[livekit]    # LiveKit adapter
pip install superdialog[pipecat]    # PipeCat adapter
pip install superdialog[fastapi]    # FastAPI adapter + uvicorn
pip install superdialog[ws]         # WebSocket runner
pip install superdialog[mcp]        # MCP tool support
pip install superdialog[langchain]  # LangChainAgent
```

## Quickstart A — Playbook engine (recommended)

A **Playbook** declares a conversation as journeys of checkpoints — goal, typed
slots, guidance prose, advance rules — plus a process layer of tools, pipelines,
handlers, and policies. Checkpoints gate **outcomes**, not utterances: a fast
Talker LLM streams every spoken turn while the async Director extracts slots,
judges advancement, and runs tools, both over an append-only event log that
doubles as the audit/replay artifact.

```yaml
# booking.yaml
persona: "You are a booking assistant."
env: {API_BASE_URL: "https://api.example.com"}
journeys:
  booking:
    checkpoints:
      - id: collect
        goal: "Have city and date"
        slots:
          city: {type: str, required: true}
          date: {type: date, required: true}
        guidance: "Collect naturally."
        advance_when:
          - {when: "details complete", judge: llm, to: booking.confirm,
             requires: [city, date]}
      - id: confirm
        gate: hard
        say_verbatim: "Your booking is held."
        pipeline: confirm_and_hold
        advance_when:
          - {when: "pipeline.ok", judge: expr, to: booking.close}
      - id: close
        terminal: true
        outcome: confirmed
tools:
  - id: hold_slot
    method: POST
    url: "{{ env.API_BASE_URL }}/slots/hold"
    store_response_as: hold_result
pipelines:
  - id: confirm_and_hold
    steps:
      - tool: hold_slot
        on: {ok: continue, failed: {retry: 1, on_exhaust: booking.collect}}
```

```python
import asyncio
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

agent = PlaybookAgent(
    playbook=Playbook.load("booking.yaml"),
    talker_llm=talker,      # StreamsLLM: stream(messages) -> AsyncIterator[str]
    director_llm=director,  # CompletesLLM: async complete(messages) -> str
    http=httpx_http,        # HTTP executor for declared tools (templates render in a Jinja sandbox)
)

async def chat():
    reply = await agent.turn("Hi, I'd like to book something.")
    print(reply.text)

asyncio.run(chat())
```

Streaming is real, not cosmetic: `await agent.turn(text, stream=True)` yields
`StreamChunk`s as the Talker produces tokens, and barge-in (aborting the
stream) interrupts speech without losing the Director's decision. The event
log replays offline — `superdialog.playbook.replay` re-runs the Director over
a recorded session and diffs every decision.

Adapting providers is two lambdas: the Director wants plain text — wrap a
provider as `(await provider.complete(messages)).text`; the Talker wants raw
tokens — yield `chunk.text` from `provider.stream(messages)`.

## Quickstart B — DialogMachine (flow graphs)

The original engine: a flow graph executed as a deterministic state machine.

```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine, Flow

# 1. Bootstrap a flow from a prompt (one-shot LLM call at construction).
#    The build LLM is used ONLY here - never at runtime.
async def build():
    flow = await create_dialog_flow(
        prompt="Confirm appointment. Ask if Friday 4pm works; offer 5pm if not.",
        llm="openai/gpt-5.1",
    )
    flow.save("appointment.json")        # JSON, version-controllable

asyncio.run(build())

# 2. Build the runtime machine (runtime model can differ from the build model).
dialog_machine = DialogMachine(
    flow=Flow.load("appointment.json"),
    llm="anthropic/claude-haiku-4-5",
)

# 3. Run a conversation.
async def chat():
    reply = await dialog_machine.turn("Hi, I'm calling about my appointment.")
    print(reply.text)

asyncio.run(chat())
```

Or skip the Python and use the bundled CLI:

```bash
superdialog chat --flow appointment.json
```

Tools plug in as `PythonTool` / `HttpTool` / `MCPTool`; models are picked per
machine with litellm-style URIs (`openai/gpt-5.1`, `vllm/<model>@<host>`,
`custom/<name>/<model>`, …) and swapped at runtime with `set_llm(uri)`. See
[docs/02-api-reference.md](docs/02-api-reference.md).

## Which engine?

| You are building | Use |
|---|---|
| IVR-style scripts: deterministic, graph-railed, every path enumerable | **DialogMachine** |
| Fluid conversations where the model owns fluidity and checkpoints gate outcomes | **Playbook** |
| Real token streaming with a compound Talker/Director turn model | **Playbook** |
| Event-sourced audit log, deterministic replay, persona-driven eval | **Playbook** |
| An existing flow JSON in production | **DialogMachine** — it keeps working |

Flows keep working; playbooks are where new investment goes. Existing flow
graphs compile down losslessly:

```python
from superdialog import Flow
from superdialog.playbook import compile_flow, coverage_report

flow = Flow.load("appointment.json")
pb = compile_flow(flow)               # ConversationFlow -> Playbook
report = coverage_report(flow, pb)    # proves every node/edge/action mapped
```

## Deploy anywhere

`PlaybookAgent` and `DialogMachine` implement the same
`superdialog.agent.Agent` protocol (`turn` / `assist` / `chat_ctx`), so the
same object drops into every host. The host varies; the SuperDialog code is
identical.

| Host | Adapter | Approx. LoC |
|---|---|---|
| **CLI** | none - `superdialog chat` (flows) or an `input()`/`print()` loop | ~5 |
| **LiveKit** | `superdialog.adapters.livekit.DialogMachineLLM` (`Agent(llm=...)` plugin) | ~6 |
| **PipeCat** | `superdialog.adapters.pipecat.make_processor(dm)` | ~2 |
| **FastAPI** | direct `/turn` route, or a `SessionWorker` for multi-user | ~6 |
| **Unpod Voice** | `superdialog.adapters.websocket.WebSocketRunner` | ~6 |
| **Slack / Discord / IRC / etc.** | none - direct callback | ~3 |

```python
# LiveKit - same agent object, ~6 lines
from livekit.agents import Agent, AgentSession
from superdialog.adapters.livekit import DialogMachineLLM

async def entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(dialog_machine))
    await AgentSession().start(agent=agent, room=ctx.room)
```

When a conversation must outlive the process, hand any agent factory to a
`SessionWorker` — it builds one agent per session and multiplexes N concurrent
sessions, serializing same-session requests via a per-session lock. `LLMAgent` and `LangChainAgent`
are drop-in non-state-machine brains for the same machinery.

## What it is not

- **Not a UI flow designer** - that belongs to a downstream tool.
- **Not a voice framework** - audio, STT, TTS are out of scope.
- **Not multi-modal** - text only at the interface (vision/audio via tools).
- **Not a hosted service** - a library. Hosting is offered by the Unpod Voice
  Platform for those who want it.

## Documentation

| Doc | Contents |
|---|---|
| [docs/00-overview.md](docs/00-overview.md) | What it is, why standalone, positioning |
| [docs/01-architecture.md](docs/01-architecture.md) | Dual-engine internals, contracts, data shapes |
| [docs/02-api-reference.md](docs/02-api-reference.md) | Every class and method |
| [docs/03-embedding-guides.md](docs/03-embedding-guides.md) | Host-by-host integration walkthroughs |
| [docs/04-playbook-guide.md](docs/04-playbook-guide.md) | Authoring playbooks, compiling flows, replay/eval |

## Roadmap

Future work — none of this is in the current release:

- `superdialog optimize` - a run → eval → improve loop over recorded sessions
- Playbook mode for the bundled CLI
- Voice-event plumbing (silence, barge-in signals) from live adapters into
  playbook external events
- Distributed session stores (Redis / File / SQLite)

## License

Apache-2.0. See [LICENSE](LICENSE).
