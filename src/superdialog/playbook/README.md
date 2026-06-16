# Playbook engine

A **Playbook** declares a conversation as journeys of checkpoints (goal, typed
slots, guidance prose, advance rules) plus a process layer (tools, pipelines,
handlers, interrupts, policies). At runtime a fast **Talker** LLM streams every
spoken turn while an async **Director** extracts slots, judges advancement, and
runs tools — both over an append-only, event-sourced log that doubles as the
audit/replay artifact. Legacy flow graphs compile down losslessly via
`compile_flow`.

## A minimal playbook

```yaml
persona: "You are a booking assistant."
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

## Usage

```python
from superdialog.playbook import Playbook, PlaybookAgent
from superdialog.playbook.toolexec import httpx_http

agent = PlaybookAgent(
    playbook=Playbook.load("booking.yaml"),
    talker_llm=talker,      # stream(messages) -> AsyncIterator[str]
    director_llm=director,  # async complete(messages) -> str
    http=httpx_http,
)
result = await agent.turn("hello")
```

Provider adapters: the Director wants plain text — wrap a real provider with
`(await provider.complete(messages)).text`; the Talker wants raw tokens —
yield `chunk.text` from `provider.stream(messages)`.

## Per-slot gate policy

Confirmation gating is decided **per slot**, not per whole turn. A checkpoint's
`gate` (`soft` | `hard`) is the default for its slots; a slot may override it
with its own `gate`:

```yaml
- id: collect
  slots:
    name:  {type: str}                 # inherits the checkpoint gate (soft)
    phone: {type: str, gate: hard}     # risky slot: confirm-then-speak
  advance_when:
    - {when: "both given", judge: llm, to: intake.done, requires: [name, phone]}
```

* **Hard** slots must be `confirmed` before they gate advancement or are spoken;
  a single Director verdict only marks them `provisional` (it cannot self-attest
  through a hard gate). Confirmation comes from a tool, an expr `set:`, a later
  turn, or a high-confidence fast verdict (below).
* **Soft** slots advance on a `filled` (provisional) value.
* Unannotated slots/playbooks behave exactly as before (default-soft).

**Split-utterance streaming.** At a gated turn the Talker streams a
commitment-free *onset* (a value-independent template — never an interpolated
slot value) as its first token(s), then barriers only the committal payload on
the Director. This keeps time-to-first-token off the barrier path while
guaranteeing nothing unconfirmed is asserted. Set `Talker(split_utterance=False)`
to restore barrier-before-first-token (filler-on-expiry) behavior.

**Fast-classifier release.** When `Director(fast_release=True)`, the verdict
carries a per-slot `confidence`; a hard slot with confidence ≥ threshold
(default `0.85`) is confirmed in one shot (releasing the barrier), falling
through to the normal re-confirmation loop when uncertain. Known hard gates
(phone/email/payment/card/cvv/otp/ssn/account/routing/iban, by name) are always
denied fast release; tune with `fast_release_allow` / `fast_release_deny`.
Disabled by default — hard slots stay provisional until separately confirmed.

## Compiling legacy flows

```python
from superdialog.playbook import compile_flow, coverage_report

pb = compile_flow(flow)               # ConversationFlow -> Playbook
report = coverage_report(flow, pb)    # proves every node/edge/action mapped
```

Design rationale and the full architecture live in
`docs/plans/2026-06-10-checkpoint-compound-architecture-design.md`.
