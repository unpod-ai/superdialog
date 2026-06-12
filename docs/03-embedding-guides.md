# SuperDialog — Embedding Guides

**Status:** Canonical
**Parent:** [README.md](README.md)
**Purpose:** Show how SuperDialog drops into each common host environment.

---

## The shape of every embedding

In every host, three things stay the same:

1. **Construct an engine** — a `PlaybookAgent` (the default engine:
   playbook + Talker/Director LLMs). `Playbook.load` accepts full
   playbooks, simple-format playbooks, *and* legacy flow JSON
   (auto-compiled), so you don't pick a format — you just point it at your
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

- **Director** (`CompletesLLM`): `async complete(messages) -> str` — one
  structured call per user utterance (extract, judge, steer).
- **Talker** (`StreamsLLM`): `stream(messages) -> AsyncIterator[str]` — one
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

```python
import asyncio
from superdialog import create_dialog_flow, DialogMachine

async def main():
    flow = await create_dialog_flow(prompt="Confirm KYC.", llm="openai/gpt-5.1")
    dialog_machine = DialogMachine(
        flow=flow,
        llm="anthropic/claude-haiku-4-5",
        traversal_dir="./traversal_history",  # saves a JSON per session on completion
    )
    while True:
        user = input("> ")
        if user.strip() in ("quit", "exit"):
            break
        reply = await dialog_machine.turn(user)
        print(reply.text)

asyncio.run(main())
```

Pass `traversal_dir` to any `DialogMachine` — CLI, LiveKit, FastAPI, or any other host. A timestamped JSON file is written to that directory each time a session reaches a terminal node. Each file captures the full node path, every turn, and all collected slot values. Useful for inspecting flow behaviour, building eval datasets, and auditing production conversations.

Or use the bundled CLI:

```
superdialog chat kyc.json
```

> **Using the Playbook engine:** `superdialog chat` runs flows only — there
> is no playbook CLI mode yet. For a playbook dev loop, drive a
> `PlaybookAgent` from the same `input()` loop; its event log
> (`agent.event_log.to_jsonl()`) is the audit artifact, replacing
> `traversal_dir`.

**When to use:** during initial flow design, before any voice infrastructure is set up.

---

## 2. LiveKit

SuperDialog ships a `DialogMachineLLM` plugin that wires a `DialogMachine`
into a LiveKit `Agent` via the `llm=` parameter (the same shape LiveKit's
own `livekit-plugins-langchain` uses).

```python
from livekit.agents import Agent, AgentSession
from superdialog import DialogMachine, Flow
from superdialog.adapters.livekit import DialogMachineLLM

dialog_machine = DialogMachine(
    flow=Flow.load("kyc.json"),
    llm="anthropic/claude-opus-4-7",
)

async def entrypoint(ctx):
    agent = Agent(llm=DialogMachineLLM(dialog_machine))
    await AgentSession().start(agent=agent, room=ctx.room)
```

LiveKit's `AgentSession` drives the conversation; `DialogMachineLLM`
translates between LiveKit's `ChatContext` and SuperDialog's `turn()` API.

> **Using the Playbook engine:** `DialogMachineLLM(agent)` accepts any
> superdialog `Agent` — pass a `PlaybookAgent` and streaming becomes real:
> the Talker's tokens reach TTS as they are generated, and a barge-in
> (the host aborting the stream mid-utterance) interrupts speech, never
> the state machine — the Director's decision still lands. Voice-event
> plumbing (feeding silence timeouts into `agent.runtime.on_external`)
> is roadmap; today the adapter covers the text path.

**When to use:** you're already on LiveKit for media routing and want SuperDialog to manage turn-by-turn logic.

---

## 3. PipeCat

PipeCat's `FrameProcessor` base class shifts between releases, so
SuperDialog ships a factory rather than a subclass: `make_processor(dm)`
synthesises a concrete `FrameProcessor` against whichever PipeCat is
installed.

```python
from superdialog import DialogMachine, Flow
from superdialog.adapters.pipecat import make_processor

dialog_machine = DialogMachine(flow=Flow.load("kyc.json"), llm="openai/gpt-5.1")
processor = make_processor(dialog_machine)

# Compose into a PipeCat pipeline
pipeline = Pipeline([
    stt_processor,
    processor,
    tts_processor,
])
```

> **Using the Playbook engine:** `make_processor(agent)` accepts any
> superdialog `Agent`, including a `PlaybookAgent` — construct it as in §4
> and pass it in place of the `DialogMachine`.

**When to use:** PipeCat-based voice stack; SuperDialog replaces hand-written LLM logic between STT and TTS.

---

## 4. FastAPI (text chatbot / REST endpoint)

For single-user or stateless `/turn` endpoints, use a `DialogMachine` directly:

```python
from fastapi import FastAPI
from superdialog import DialogMachine, Flow

app = FastAPI()
dialog_machine = DialogMachine(flow=Flow.load("kyc.json"), llm="openai/gpt-5.1")

@app.post("/turn")
async def turn(payload: dict):
    return {"reply": (await dialog_machine.turn(payload["text"])).text}
```

For **multi-user** or **multi-worker** deployments, route per-conversation
state through a `SessionWorker` so any request can land on any worker and
resume the right conversation:

```python
from fastapi import FastAPI
from superdialog import DialogMachine, Flow, SessionWorker, InMemorySessionStore

app = FastAPI()
flow = Flow.load("kyc.json")
worker = SessionWorker(
    agent_factory=lambda: DialogMachine(flow=flow, llm="openai/gpt-5.1"),
    store=InMemorySessionStore(),    # swap in a distributed SessionStore in
                                     # production (RedisSessionStore is planned;
                                     # implement the SessionStore protocol today)
)

@app.post("/turn")
async def turn(payload: dict):
    async with worker.acquire(payload["session_id"]) as h:
        result = await h.turn(payload["text"])
    return {"reply": result.text}
```

The `SessionWorker` multiplexes N concurrent sessions, each with its own
`DialogMachine`, sharing the immutable `Flow` by reference. Concurrent
requests for different `session_id`s run in parallel; concurrent requests
for the same id serialise via a per-session lock.

### Using the Playbook engine

`PlaybookAgent` implements the same `Agent` protocol, so the wiring is
identical — only the factory changes. Complete example (with `TextLLM`,
`talker`, and `director` from the provider-adapter section above):

```python
from fastapi import FastAPI
from superdialog import InMemorySessionStore, SessionWorker
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http

app = FastAPI()
playbook = Playbook.load("booking.yaml")

worker = SessionWorker(
    agent_factory=lambda: PlaybookAgent(
        playbook=playbook,
        talker_llm=talker,        # StreamsLLM — see provider adapters above
        director_llm=director,    # CompletesLLM
        http=httpx_http,          # sandboxed HTTP executor for playbook tools
    ),
    store=InMemorySessionStore(),
)

@app.post("/turn")
async def turn(payload: dict):
    async with worker.acquire(payload["session_id"]) as h:
        result = await h.turn(payload["text"])
    return {"reply": result.text}
```

`result.metadata` carries `checkpoint`, `version`, `ended`, and (on terminal
checkpoints) `outcome`. External events — webhooks, timers, silence — go to
`agent.runtime.on_external(...)` from your own endpoints; automatic
voice-event plumbing through the host adapters is roadmap.

One caveat: the in-process `SessionWorker` works as-is because agents stay
cache-resident, but durable or multi-worker resume requires persisting
`agent.event_log.to_jsonl()` yourself and restoring via `load_event_log` —
`SessionWorker`'s `SessionRecord` persists `chat_ctx`/`flow_state` only,
which loses playbook state fidelity.

Mount on Intercom-style chat widget, WhatsApp webhook, SMS gateway, or anywhere HTTP fits.

**When to use:** non-voice deployments — text-only chatbot, support widget, async messaging.

---

## 5. Unpod Voice Infrastructure

This is the production voice path. SuperDialog runs on the developer's machine via a WebSocket runner; Unpod's infra connects to it.

```python
import os

from superdialog import DialogMachine, Flow
from superdialog.adapters.websocket import WebSocketRunner

dialog_machine = DialogMachine(flow=Flow.load("kyc.json"), llm="anthropic/claude-opus-4-7")

WebSocketRunner(
    agent=dialog_machine,           # any Agent: DialogMachine or PlaybookAgent
    agent_id="kerali-kyc-bot",      # registers with Unpod
    api_key=os.environ["UNPOD_API_KEY"],
).serve(port=8080)
```

For multi-tenant serving, pass `worker=SessionWorker(...)` instead of
`agent=`; every inbound frame then carries a `session_id`. A `PlaybookAgent`
(or a `SessionWorker` of them) drops in unchanged.

Then on Unpod side, the Identity binds the inbound number to this agent. When a call lands, Unpod connects to your WSS endpoint, streams text in, and sends agent text out for TTS. See the Unpod voice platform docs for the full picture.

**When to use:** you want voice + numbers + speech infrastructure without writing telephony code.

---

## 6. Unit tests

`DialogMachine.turn` is async; tests use `pytest-asyncio` (or `anyio`).
The `state` property returns `{"node_id": ..., "slots": ...}` — read
collected data through `state["slots"]`.

```python
import pytest
from superdialog import DialogMachine, Flow

@pytest.mark.asyncio
async def test_kyc_flow_collects_aadhaar():
    machine = DialogMachine(
        flow=Flow.load("kyc.json"),
        llm="anthropic/claude-haiku-4-5",
    )
    reply = await machine.turn("मेरा आधार 1234 से शुरू होता है")
    assert "धन्यवाद" in reply.text or "thank" in reply.text.lower()
    assert machine.state["slots"].get("aadhaar_last_4") == "1234"
```

> **Using the Playbook engine:** the same pattern applies — construct a
> `PlaybookAgent` with stub LLMs and assert on
> `agent.runtime.state.slots` / `.checkpoint_id`. Playbooks additionally
> support LLM-free replay regression and persona-driven evals; see
> [04-playbook-guide.md](04-playbook-guide.md) §6.

**When to use:** always. Because SuperDialog is text-only, every dialog is a unit-testable function. This is the killer feature vs voice-coupled frameworks where tests need audio fixtures.

---

## 7. Custom integration (anything else)

The interface is minimal: pass text in, get text out. **Note that
`turn(...)` is always async** — wrap it in an event loop for sync hosts.

```python
import asyncio

# IRC (sync handler)
def on_message(msg):
    reply = asyncio.run(dialog_machine.turn(msg.body))
    return reply.text

# Slack (sync handler)
@slack_app.message(...)
def handle(message, say):
    reply = asyncio.run(dialog_machine.turn(message["text"]))
    say(reply.text)

# Discord (async handler — preferred)
@discord_bot.event
async def on_message(message):
    reply = await dialog_machine.turn(message.content)
    await message.channel.send(reply.text)
```

Every snippet above works verbatim with a `PlaybookAgent` in place of the
`dialog_machine` — the `Agent` protocol is the only contract.

For high-throughput hosts that hand you many concurrent conversations,
prefer a `SessionWorker` per process and route per-conversation state
through `worker.acquire(session_id)` — see §4 above.

> **Note on sync hosts:** wrapping every `dialog_machine.turn(...)` in
> `asyncio.run` creates a fresh event loop per call. For sustained traffic
> this is wasteful; either route through `SessionWorker` from an existing
> async runtime, or maintain a single long-lived loop. A dedicated
> `SyncDialogMachine` wrapper is on the roadmap.

---

## Summary

| Host | Adapter needed | LoC |
|---|---|---|
| CLI | None — direct `input()`/`print()` loop or `superdialog chat` | ~5 |
| LiveKit | `DialogMachineLLM` (accepts any Agent) | ~8 |
| PipeCat | `make_processor` (accepts any Agent) | ~12 |
| FastAPI | None — direct route or `SessionWorker` | ~6 |
| Unpod Voice Infra | `WebSocketRunner` | ~6 |
| Unit test | None — direct calls | ~3 |
| Custom (Slack, Discord, IRC, etc.) | None — direct callback | ~3 |

Every row holds for both engines: `DialogMachine` and `PlaybookAgent` are
interchangeable behind the `Agent` protocol. The library does one thing
well: text in, text out. Everything else is host code.
