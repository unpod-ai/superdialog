# SuperDialog

**Standalone dialog framework. Pure text in, pure text out.**

SuperDialog is the **brain** layer for conversational systems. It ships two
engines behind one text interface: **Playbook**, the default - a
checkpoint-based compound runtime (a fast streaming Talker plus an async
Director) for fluid, outcome-driven conversations - and **DialogMachine**,
the supported legacy engine, a graph-railed dialog state machine that
executes flow graphs deterministically. Both manage turn-by-turn
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

## Quickstart A - Playbook engine (default)

A **Playbook** declares a conversation as journeys of checkpoints - goal, typed
slots, guidance prose, advance rules - plus a process layer of tools, pipelines,
handlers, and policies. Checkpoints gate **outcomes**, not utterances: a fast
Talker LLM streams every spoken turn while the async Director extracts slots,
judges advancement, and runs tools, both over an append-only event log that
doubles as the audit/replay artifact.

Start with the **simple format** - prose steps, a structured persona, and
reference data as plain YAML. It is what `superdialog generate` produces,
and every loader and command accepts it directly:

```yaml
# playbook.yaml (simple format)
goal: "Book a haircut and confirm it."
persona:
  name: Mira
  language: ["en", "hi"]
  identity: "You are Mira, a booking assistant for Glow Studio."
  voice_style: "Warm and brief. One question at a time."
playbook:
  - id: greet
    purpose: "Open the call."
    say: "Greet the caller and ask how you can help."
    done_when: "Caller is ready to book."
  - id: collect
    purpose: "Get the booking details."
    say: "Ask for their name and preferred service."
    collect: [name, service]
    done_when: "Name and service are captured."
  - id: confirm
    purpose: "Confirm and close."
    say: "Read back the booking and confirm."
    done_when: "Caller has confirmed."
facts:
  canonical_pricing: {haircut: "₹400"}
boundaries: ["NEVER invent prices."]
interrupts:
  - {when: "Caller says goodbye.", to: main.confirm}
```

When you need precision the simple format can't express - tools, pipelines,
hard gates, typed slots, multiple outcomes - graduate to the **full format**
(same engine, same loader; see
[docs/04-playbook-guide.md](docs/04-playbook-guide.md) Part 1):

```yaml
# booking.yaml (full format)
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

`DialogMachine` is the recommended entry point: one class, one model URI. It
runs the Playbook engine by default; pass `engine="flow"` for the legacy graph
runtime.

```python
import asyncio
from superdialog import DialogMachine

agent = DialogMachine("booking.yaml", llm="openai/gpt-4.1-mini")

async def chat():
    reply = await agent.turn("Hi, I'd like to book something.")
    print(reply.text)

asyncio.run(chat())
```

The single `llm=` is the Talker, and the Director too unless you split them
with `director_llm=` (the cheap-Talker / strong-Director latency split). Pass
any `Tool` via `tools=` - each runs its own `execute()`, both engines.

Streaming is real, not cosmetic: `await agent.turn(text, stream=True)` yields
`StreamChunk`s as the Talker produces tokens, and barge-in (aborting the
stream) interrupts speech without losing the Director's decision. The event
log replays offline - `superdialog.playbook.replay` re-runs the Director over
a recorded session and diffs every decision.

**Advanced: explicit Talker/Director.** Drop to `PlaybookAgent` when you need
to supply scripted LLMs, a custom HTTP executor, or inspect the runtime
directly. Adapting providers is two lambdas - or use the bundled
`superdialog.playbook.provider_adapters(provider)`, which returns the
`(director, talker)` pair for any `LLMProvider`:

```python
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

agent = PlaybookAgent(
    playbook=Playbook.load("booking.yaml"),
    talker_llm=talker,      # StreamsLLM: stream(messages) -> AsyncIterator[str]
    director_llm=director,  # CompletesLLM: async complete(messages) -> str
    http=httpx_http,        # HTTP executor for declared tools (Jinja-sandboxed templates)
)
```

Or skip the Python entirely - generate a playbook from a description and
chat against it. The Playbook engine is the default for every format
(full, simple, and flow JSON, which is compiled automatically):

```bash
superdialog generate "Book a demo call; capture day and time."   # -> playbook.yaml
superdialog chat                              # picks up ./playbook.yaml
superdialog chat --playbook booking.yaml      # explicit (any format)
superdialog chat --flow appointment.json      # flow JSON, compiled onto the engine
superdialog chat --flow appointment.json --mode flow   # legacy DialogMachine
superdialog optimize --playbook playbook.yaml  # reflective prose optimizer
```

## Quickstart B - the legacy graph engine

The original engine: a flow graph executed as a deterministic state machine.
Still driven through `DialogMachine` - pass `engine="flow"` (or `--mode flow`
on the CLI) to select it. New agents should start with Quickstart A.

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
#    engine="flow" selects the legacy graph runtime; the default is Playbook.
dialog_machine = DialogMachine(
    Flow.load("appointment.json"),
    llm="anthropic/claude-haiku-4-5",
    engine="flow",
)

# 3. Run a conversation.
async def chat():
    reply = await dialog_machine.turn("Hi, I'm calling about my appointment.")
    print(reply.text)

asyncio.run(chat())
```

Or skip the Python and use the bundled CLI - legacy mode is an explicit opt-in:

```bash
superdialog chat --flow appointment.json --mode flow
```

Tools plug in as `PythonTool` / `HttpTool` / `MCPTool`; models are picked per
machine with litellm-style URIs (`openai/gpt-5.1`, `vllm/<model>@<host>`,
`custom/<name>/<model>`, …) and swapped at runtime with `set_llm(uri)`. See
[docs/02-api-reference.md](docs/02-api-reference.md).

## Which engine?

| You are building | Use |
|---|---|
| IVR-style scripts: deterministic, graph-railed, every path enumerable | **DialogMachine** (legacy, opt-in via `--mode flow`) |
| Fluid conversations where the model owns fluidity and checkpoints gate outcomes | **Playbook** |
| Real token streaming with a compound Talker/Director turn model | **Playbook** |
| Event-sourced audit log, deterministic replay, persona-driven eval | **Playbook** |
| An existing flow JSON in production | **Playbook** - every loader auto-compiles flow JSON; `--mode flow` keeps legacy DialogMachine behaviour |

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

`DialogMachine` (and the lower-level `PlaybookAgent`) implement the same
`superdialog.agent.Agent` protocol (`turn` / `assist` / `chat_ctx`), so the
same object drops into every host. The host varies; the SuperDialog code is
identical.

| Host | Adapter | Approx. LoC |
|---|---|---|
| **CLI** | none - `superdialog chat` (playbooks or flows) or an `input()`/`print()` loop | ~5 |
| **LiveKit** | `superdialog.adapters.livekit.DialogMachineLLM` (`Agent(llm=...)` plugin) | ~6 |
| **PipeCat** | `superdialog.adapters.pipecat.make_processor(agent)` | ~2 |
| **FastAPI** | direct `/turn` route, or a `SessionWorker` for multi-user | ~6 |
| **Unpod Voice** | `superdialog.adapters.websocket.WebSocketRunner` | ~6 |
| **Slack / Discord / IRC / etc.** | none - direct callback | ~3 |

```python
# LiveKit - same agent object, ~6 lines
from livekit.agents import Agent, AgentSession
from superdialog.adapters.livekit import DialogMachineLLM

async def entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(playbook_agent))
    await AgentSession().start(agent=agent, room=ctx.room)
```

When a conversation must outlive the process, hand any agent factory to a
`SessionWorker` - it builds one agent per session and multiplexes N concurrent
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
| [docs/01-architecture.md](docs/01-architecture.md) | Engine internals - the Playbook runtime (default) and the legacy DialogMachine; contracts, data shapes |
| [docs/02-api-reference.md](docs/02-api-reference.md) | Every class and method |
| [docs/03-embedding-guides.md](docs/03-embedding-guides.md) | Host-by-host integration walkthroughs |
| [docs/04-playbook-guide.md](docs/04-playbook-guide.md) | Part 1: authoring formats (simple + full); Part 2: technical design - compiling flows, replay/eval |

## Roadmap

Future work - none of this is in the current release:

- Voice-event plumbing (silence, barge-in signals) from live adapters into
  playbook external events
- Distributed session stores (Redis / File / SQLite)

## License

Apache-2.0. See [LICENSE](LICENSE).
