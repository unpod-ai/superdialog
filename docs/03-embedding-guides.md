# SuperDialog - Embedding Guides

**Status:** Canonical
**Parent:** [README.md](README.md)
**Purpose:** Show how SuperDialog drops into each common host environment.

---

## The shape of every embedding

In every host, three things stay the same:

1. **Construct an engine** - a `PlaybookAgent` (the default engine:
   playbook + Talker/Director LLMs). `Playbook.load` accepts full
   playbooks, simple-format playbooks, *and* legacy flow JSON
   (auto-compiled), so you don't pick a format - you just point it at your
   artifact. The legacy `DialogMachine` (flow + LLM URI + tools) remains
   for hosts that explicitly want the original graph runtime
   (`--mode flow` in the CLI).
2. **Route inbound text** to `engine.turn(text)`.
3. **Send the reply text** back to the host's output channel.

Both engines implement the same `superdialog.agent.Agent` protocol
(`turn` / `assist` / `chat_ctx` / `load_chat_ctx`), so every adapter below
accepts either one. The host varies; the SuperDialog code is identical.
The `DialogMachine` examples below apply to the opt-in legacy mode.

### Provider adapters for the Playbook engine

`DialogMachine` takes a model URI string. `PlaybookAgent` instead takes two
small LLM seams, defined in `superdialog.playbook`:

- **Director** (`CompletesLLM`): `async complete(messages) -> str` - one
  structured call per user utterance (extract, judge, steer).
- **Talker** (`StreamsLLM`): `stream(messages) -> AsyncIterator[str]` - one
  streaming call per spoken turn, tokens straight to the host.

Any `superdialog.llm.LLMProvider` (e.g. the litellm-backed one behind model
URIs) adapts in a few lines:

```python
from typing import Any, AsyncIterator

from superdialog.llm import LLMProvider, resolve_llm


class TextLLM:
    """Adapt an LLMProvider to the Talker/Director text protocols."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return (await self._provider.complete(messages, **kwargs)).text

    async def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        async for chunk in self._provider.stream(messages, **kwargs):
            if chunk.text:
                yield chunk.text


talker = TextLLM(resolve_llm("anthropic/claude-haiku-4-5"))   # fast: speaks
director = TextLLM(resolve_llm("anthropic/claude-opus-4-7"))  # strong: judges
```

`CompletionResult.text` carries the completion; `StreamChunk.text` carries
each token delta (it is `None` on tool-call frames, hence the guard). The
complete FastAPI example in §4 uses these two objects. Authoring playbooks
themselves is covered in [04-playbook-guide.md](04-playbook-guide.md).

---

## 1. CLI chatbot (testing / dev loop)

Zero infrastructure. Useful for prompt tuning, eval prep, and demos.
The default loop is the bundled CLI: `superdialog generate "Confirm KYC."`
writes a validated simple-format `playbook.yaml`, and `superdialog chat`
picks it up (then `./flow.json`), auto-detects any format, and runs it on
the Playbook engine (default `--llm openai/gpt-4.1-mini`). In code, the
same loop is a `PlaybookAgent` driven from `input()` (with `talker` and
`director` from the provider-adapter section above):

```python
import asyncio
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

agent = PlaybookAgent(
    playbook=Playbook.load("playbook.yaml"),  # full, simple, or flow JSON
    talker_llm=talker,
    director_llm=director,
    http=httpx_http,
)

async def main():
    while True:
        user = input("> ")
        if user.strip() in ("quit", "exit"):
            break
        reply = await agent.turn(user)
        print(reply.text)

asyncio.run(main())
```

The event log (`agent.event_log.to_jsonl()`) is the audit artifact: every
utterance, slot write, advance, and tool call, replayable offline - useful
for inspecting behaviour, building eval datasets, and auditing production
conversations.

> **Legacy DialogMachine:** `superdialog chat kyc.json --mode flow` runs
> the original graph engine. In code, construct
> `DialogMachine(flow=..., llm=..., traversal_dir="./traversal_history")`
> and drive the same loop; `traversal_dir` writes a timestamped JSON per
> completed session (full node path, every turn, collected slots) - the
> DialogMachine equivalent of the event log, accepted by any host.

**When to use:** during initial playbook (or legacy flow) design, before any voice infrastructure is set up.

---

## 2. LiveKit

SuperDialog ships a `DialogMachineLLM` plugin (named for the legacy engine,
but it accepts any superdialog `Agent`) that wires an agent into a LiveKit
`Agent` via the `llm=` parameter (the same shape LiveKit's own
`livekit-plugins-langchain` uses).

```python
from livekit.agents import Agent, AgentSession
from superdialog.adapters.livekit import DialogMachineLLM
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

playbook_agent = PlaybookAgent(
    playbook=Playbook.load("kyc.yaml"),   # full, simple, or legacy flow JSON
    talker_llm=talker,                    # see provider adapters above
    director_llm=director,
    http=httpx_http,
)

async def entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(playbook_agent))
    await AgentSession().start(agent=agent, room=ctx.room)
```

LiveKit's `AgentSession` drives the conversation; `DialogMachineLLM`
translates between LiveKit's `ChatContext` and SuperDialog's `turn()` API.
With a `PlaybookAgent`, streaming is real: the Talker's tokens reach TTS
as they are generated, and a barge-in (the host aborting the stream
mid-utterance) interrupts speech, never the state machine - the Director's
decision still lands. Voice-event plumbing (feeding silence timeouts into
`agent.runtime.on_external`) is roadmap; today the adapter covers the text
path.

> **Legacy DialogMachine:** pass a
> `DialogMachine(flow=Flow.load("kyc.json"), llm="anthropic/claude-opus-4-7")`
> in place of the `PlaybookAgent` - same adapter, same wiring.

**When to use:** you're already on LiveKit for media routing and want SuperDialog to manage turn-by-turn logic.

---

## 3. PipeCat

PipeCat's `FrameProcessor` base class shifts between releases, so
SuperDialog ships a factory rather than a subclass: `make_processor(agent)`
synthesises a concrete `FrameProcessor` against whichever PipeCat is
installed. It accepts any superdialog `Agent`:

```python
from superdialog.adapters.pipecat import make_processor

processor = make_processor(playbook_agent)   # built as in §2

# Compose into a PipeCat pipeline
pipeline = Pipeline([
    stt_processor,
    processor,
    tts_processor,
])
```

> **Legacy DialogMachine:**
> `make_processor(DialogMachine(flow=Flow.load("kyc.json"), llm=...))` -
> same factory, same pipeline position.

**When to use:** PipeCat-based voice stack; SuperDialog replaces hand-written LLM logic between STT and TTS.

---

## 4. FastAPI (text chatbot / REST endpoint)

For single-user or stateless `/turn` endpoints, use an agent directly
(with `talker` and `director` from the provider-adapter section above):

```python
from fastapi import FastAPI
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

app = FastAPI()
agent = PlaybookAgent(
    playbook=Playbook.load("kyc.yaml"),   # full, simple, or legacy flow JSON
    talker_llm=talker,
    director_llm=director,
    http=httpx_http,
)

@app.post("/turn")
async def turn(payload: dict):
    return {"reply": (await agent.turn(payload["text"])).text}
```

For **multi-user** or **multi-worker** deployments, route per-conversation
state through a `SessionWorker` so any request can land on any worker and
resume the right conversation. It multiplexes N concurrent sessions, each
with its own agent, sharing the immutable playbook by reference; requests
for different `session_id`s run in parallel, requests for the same id
serialise via a per-session lock:

```python
from fastapi import FastAPI
from superdialog import InMemorySessionStore, SessionWorker
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

app = FastAPI()
playbook = Playbook.load("booking.yaml")

worker = SessionWorker(
    agent_factory=lambda: PlaybookAgent(
        playbook=playbook,
        talker_llm=talker,        # StreamsLLM - see provider adapters above
        director_llm=director,    # CompletesLLM
        http=httpx_http,          # sandboxed HTTP executor for playbook tools
    ),
    store=InMemorySessionStore(),  # swap in a distributed SessionStore in
                                   # production (RedisSessionStore is planned;
                                   # implement the SessionStore protocol today)
)

@app.post("/turn")
async def turn(payload: dict):
    async with worker.acquire(payload["session_id"]) as h:
        result = await h.turn(payload["text"])
    return {"reply": result.text}
```

> **Legacy DialogMachine:** identical wiring - swap the factory for
> `lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1")` with a shared
> `flow = Flow.load("kyc.json")`; the store and lock semantics are
> unchanged.

`result.metadata` carries `checkpoint`, `version`, `ended`, and (on terminal
checkpoints) `outcome`. External events - webhooks, timers, silence - go to
`agent.runtime.on_external(...)` from your own endpoints; automatic
voice-event plumbing through the host adapters is roadmap.

One caveat: the in-process `SessionWorker` works as-is because agents stay
cache-resident, but durable or multi-worker resume requires persisting
`agent.event_log.to_jsonl()` yourself and restoring via `load_event_log` -
`SessionWorker`'s `SessionRecord` persists `chat_ctx`/`flow_state` only,
which loses playbook state fidelity.

Mount on Intercom-style chat widget, WhatsApp webhook, SMS gateway, or anywhere HTTP fits.

**When to use:** non-voice deployments - text-only chatbot, support widget, async messaging.

---

## 5. Unpod Voice Infrastructure

This is the production voice path. SuperDialog runs on the developer's machine via a WebSocket runner; Unpod's infra connects to it.

```python
import os

from superdialog.adapters.websocket import WebSocketRunner

WebSocketRunner(
    agent=playbook_agent,           # any Agent - built as in §2 or §4
    agent_id="kerali-kyc-bot",      # registers with Unpod
    api_key=os.environ["UNPOD_API_KEY"],
).serve(port=8080)
```

For multi-tenant serving, pass `worker=SessionWorker(...)` instead of
`agent=`; every inbound frame then carries a `session_id`. A legacy
`DialogMachine` (or a `SessionWorker` of them) drops in unchanged.

Then on Unpod side, the Identity binds the inbound number to this agent. When a call lands, Unpod connects to your WSS endpoint, streams text in, and sends agent text out for TTS. See the Unpod voice platform docs for the full picture.

**When to use:** you want voice + numbers + speech infrastructure without writing telephony code.

---

## 6. Unit tests

`turn` is async; tests use `pytest-asyncio` (or `anyio`). Construct a
`PlaybookAgent` with stub LLMs (no network) and assert on
`agent.runtime.state` - slots, checkpoint, ended/outcome:

```python
import pytest
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

@pytest.mark.asyncio
async def test_kyc_collects_aadhaar():
    agent = PlaybookAgent(
        playbook=Playbook.load("kyc.yaml"),
        talker_llm=stub_talker,       # scripted StreamsLLM
        director_llm=stub_director,   # scripted CompletesLLM
        http=httpx_http,
    )
    reply = await agent.turn("मेरा आधार 1234 से शुरू होता है")
    assert reply.text
    assert agent.runtime.state.slot_value("aadhaar_last_4") == "1234"
```

Playbooks additionally support LLM-free replay regression and
persona-driven evals over the event log; see
[04-playbook-guide.md](04-playbook-guide.md) Part 2 §9.

> **Legacy DialogMachine:** same pattern -
> `DialogMachine(flow=Flow.load("kyc.json"), llm=...)`, then assert on
> `machine.state["slots"]` (its `state` property returns
> `{"node_id": ..., "slots": ...}`).

**When to use:** always. Because SuperDialog is text-only, every dialog is a unit-testable function. This is the killer feature vs voice-coupled frameworks where tests need audio fixtures.

---

## 7. Custom integration (anything else)

The interface is minimal: pass text in, get text out. **Note that
`turn(...)` is always async** - wrap it in an event loop for sync hosts.

```python
import asyncio

# IRC (sync handler)
def on_message(msg):
    reply = asyncio.run(agent.turn(msg.body))
    return reply.text

# Slack (sync handler)
@slack_app.message(...)
def handle(message, say):
    reply = asyncio.run(agent.turn(message["text"]))
    say(reply.text)

# Discord (async handler - preferred)
@discord_bot.event
async def on_message(message):
    reply = await agent.turn(message.content)
    await message.channel.send(reply.text)
```

`agent` is any superdialog `Agent` - a `PlaybookAgent` (default) or a
legacy `DialogMachine` - the `Agent` protocol is the only contract.

For high-throughput hosts that hand you many concurrent conversations,
prefer a `SessionWorker` per process and route per-conversation state
through `worker.acquire(session_id)` - see §4 above.

> **Note on sync hosts:** wrapping every `agent.turn(...)` in
> `asyncio.run` creates a fresh event loop per call. For sustained traffic
> this is wasteful; either route through `SessionWorker` from an existing
> async runtime, or maintain a single long-lived loop. A dedicated
> `SyncDialogMachine` wrapper is on the roadmap.

---

## Summary

| Host | Adapter needed | LoC |
|---|---|---|
| CLI | None - direct `input()`/`print()` loop or `superdialog chat` | ~5 |
| LiveKit | `DialogMachineLLM` (accepts any Agent) | ~8 |
| PipeCat | `make_processor` (accepts any Agent) | ~12 |
| FastAPI | None - direct route or `SessionWorker` | ~6 |
| Unpod Voice Infra | `WebSocketRunner` | ~6 |
| Unit test | None - direct calls | ~3 |
| Custom (Slack, Discord, IRC, etc.) | None - direct callback | ~3 |

Every row holds for both engines: the default `PlaybookAgent` and the
legacy `DialogMachine` are
interchangeable behind the `Agent` protocol. The library does one thing
well: text in, text out. Everything else is host code.
