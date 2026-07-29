# SuperDialog - Playbook Execution Flow

**Status:** Canonical
**Parent:** [README.md](README.md)

A code-verified trace of the Playbook engine: the layered design, the
two-brain turn loop, every touchpoint a user turn passes through (with
`file.py::symbol` citations into `src/superdialog/`), the prompt-assembly
steering surface, and how each playbook construct changes conversational
behavior. [01-architecture.md](01-architecture.md) explains the engine's
concepts; this doc walks the actual execution path.

---

## 1. The big picture

```
                     HOSTS (text in ── text out; audio/STT/TTS is NOT superdialog's job)
   ┌────────────┬────────────┬────────────┬────────────┬────────────┐
   │  LiveKit   │  PipeCat   │  FastAPI   │  WebSocket │  CLI REPL  │   adapters/*
   └──────┬─────┴──────┬─────┴──────┬─────┴──────┬─────┴──────┬─────┘
          └────────────┴─────┬──────┴────────────┴────────────┘
                             ▼
              ╔══════════════════════════════════╗
              ║  Agent Protocol  agent.py::Agent ║   the ONE embedding seam:
              ║   turn() assist() chat_ctx()     ║   structural typing, no ABC —
              ║   load_chat_ctx()                ║   any brain that fits, plugs in
              ╚═══════════════╤══════════════════╝
                              ▼
              ┌──────────────────────────────────┐
              │ DialogMachine (facade, strategy) │  dialog_machine.py::DialogMachine
              │   _select_engine() picks:        │  dialog_machine.py::_select_engine
              └───────┬──────────────────┬───────┘
                      ▼                  ▼
        ┌──────────────────────┐   ┌─────────────────────────┐
        │ ENGINE B: PLAYBOOK   │   │ ENGINE A: graph (legacy)│
        │ (default)            │   │ DialogStateMachine      │
        │                      │   │ node──edge──node rails, │
        │  ┌────────┐ ┌──────┐ │   │ 2 serial LLM calls/turn │
        │  │ TALKER │ │DIRECT│ │   └─────────────────────────┘
        │  │ fast,  │ │ OR   │ │
        │  │ streams│ │slow, │ │      flow.json compiles ONTO the
        │  │ speech │ │judges│ │      playbook engine (compile_flow),
        │  └───┬────┘ └──┬───┘ │      --mode flow opts into legacy
        │      └────┬────┘     │
        │           ▼          │
        │   ┌──────────────┐   │
        │   │  EVENT LOG   │   │  append-only, versioned —
        │   │ (single      │   │  the ONLY source of truth
        │   │  source of   │   │  state = fold(log)
        │   │  truth)      │   │  (state.py::ConversationState.fold)
        │   └──────────────┘   │
        └──────────────────────┘
                 ▲
                 │ loads
   ┌─────────────┴──────────────────────────────┐
   │ PLAYBOOK (one IR, three frontends)         │
   │  simple.yaml ──► simple_to_playbook ─┐     │
   │  flow.json ────► compile_flow ───────┼──►  │  Playbook model
   │  full.yaml ────► Playbook._from_doc ─┘     │  (validated at load:
   │                                            │   bad refs = load error,
   │  persona · checkpoints · slots · gates ·   │   not runtime deadlock)
   │  advance rules · interrupts · KB · tools   │
   └────────────────────────────────────────────┘
```

---

## 2. One user turn — two brains, one log

The core trick: **the Talker speaks speculatively while the Director thinks**.
Latency of one streaming call, judgment of a structured second opinion.
(Diagram citations use bare `Class.method` names; §3 gives the full
`file.py::symbol` paths.)

```
 USER TEXT
    │
    ▼                                                            time ──►
 ┌──[1] append UtteranceEvent to log                 ◄─ log FIRST, so both brains see it
 │     (PlaybookAgent._stream_turn)
 ├────────────────────── fork ───────────────────────────────┐
 ▼  TALKER lane (speech path — streams NOW)                   ▼  DIRECTOR lane (shielded task,
 │                                                            │  survives barge-in:
 │  gated checkpoint?  (hard gate / hard slot)                │  PlaybookAgent._stream_turn)
 │   ├─ yes ─► BARRIER: wait ≤barrier_timeout ─► filler       │  1. judge:expr rules first —
 │   │         ─► wait ≤hold_timeout ─► polite hold-line      │     deterministic, ZERO LLM
 │   │         (Talker.speak)  then RE-SNAPSHOT fresh state   │     (Director._expr_advance)
 │   └─ no ──► speak from snapshot immediately                │  2. else ONE structured JSON
 │                                                            │     verdict call (Director.evaluate)
 │  say_verbatim step? ─► speak template EXACTLY,             │  3. slot writes (typed, coerced,
 │                        NO LLM AT ALL (Talker.speak)        │     provisional at hard gates)
 │                                                            │  4. interrupt > advance
 │  else: render_view() builds the prompt  ──── §4 ────       │  5. advance if requires met,
 │        ONE streaming LLM call (Talker.speak)               │     else steer-note "cannot
 │        chunks yield live to host                           │     move on yet"
 │        (PlaybookAgent._stream_turn)                        │  6. quiesce loop ≤8 hops:
 │                                                            │     pipelines·auto·resume
 │  barge-in? GeneratorExit kills SPEECH ONLY ────────────────┤     (PlaybookRuntime._quiesce)
 │                                                            │
 ├────────────────────── join ────────────────────────────────┘
 ▼
 shielded finally (PlaybookAgent._stream_turn): await Director — its decision ALWAYS lands
    │
    ├─ Director advanced checkpoint mid-turn + has pass-through?
    │     └─ SUPPRESS the Talker's stale speculative reply (PlaybookAgent._stream_turn)
    │        Director's verbatim lines are authoritative
    │
    ├─ check_repairs (PlaybookRuntime.check_repairs): did we re-ask something already answered?
    │     └─ append "Correction from supervisor" note ─► feeds NEXT turn's prompt
    │
    ▼
 Turn{text, metadata:{checkpoint, version, ended, outcome}} ──► host
```

Barrier timing note: the Talker default is `barrier_timeout=0.4s` /
`hold_timeout=4s` when wired through `DialogMachine`
(talker.py::Talker.__init__, dialog_machine.py::DialogMachine.__init__); a
directly-embedded `PlaybookAgent` defaults to `barrier_timeout=4.0`
(playbook/agent.py::PlaybookAgent.__init__).

---

## 3. Full path — every touchpoint

Citation convention: bare engine files (`director.py`, `talker.py`,
`runtime.py`, `render.py`, `state.py`, `events.py`, `expr.py`, `models.py`,
`_guidelines.py`) live under `src/superdialog/playbook/`; `dialog_machine.py`,
the top-level `agent.py` (the protocol), and `cli/main.py` sit directly under
`src/superdialog/`. `playbook/agent.py` is the engine.

```
BOOTSTRAP (once per session)
 B1 host calls turn/start ......................... agent.py::Agent (protocol)
 B2 DialogMachine.start() ......................... dialog_machine.py::DialogMachine.start
 B3 _ensure_backend: load playbook, resolve LLMs,
    split provider → Director+Talker adapters,
    wire tracing + billing ........................ dialog_machine.py::DialogMachine._ensure_backend
 B4 greet(): Talker speaks opening checkpoint,
    greeting logged, double-greet guard armed ..... playbook/agent.py::PlaybookAgent.greet

PER TURN
  1 dispatch stream/non-stream .................... playbook/agent.py::PlaybookAgent.turn
  2 _ensure_started (cold runtime → start()) ...... playbook/agent.py::PlaybookAgent._ensure_started
  3 user UtteranceEvent appended, version=N+1 ..... playbook/agent.py::PlaybookAgent._stream_turn
                                                    · events.py::EventLog.append
  4 entry checkpoint snapshotted .................. playbook/agent.py::PlaybookAgent._stream_turn
  5 Director launched (detached, shielded) ........ playbook/agent.py::PlaybookAgent._stream_turn
      ── DIRECTOR lane ──────────────────────────────────────────
  6   state = fold(log), cached by log version .... state.py::ConversationState.fold
                                                    · runtime.py::PlaybookRuntime.state
  7   judge:expr rules — first match, no LLM ...... director.py::Director._expr_advance
  8   verdict prompt: slots, rules, interrupts,
      date anchor, anti-injection, last 12 turns .. director.py::_verdict_prompt
  9   ONE structured completion (json_mode) ....... director.py::Director.evaluate
 10   slot extraction: declared-only, type-coerced,
      provisional at hard gates ................... director.py::Director.evaluate
 11   interrupt beats advance (anti-regression) ... director.py::Director.evaluate
 12   advance: requires met → SlotWrites+Advance;
      unmet → "cannot move on yet" steer note ..... director.py::Director.evaluate
 13   resume-over-drift: return to interrupted
      step via resume_stack ....................... runtime.py::PlaybookRuntime.on_user_text
 14   checkpoint being left says its verbatim ..... runtime.py::PlaybookRuntime._speak_verbatim
 15   _enter: on_enter tools; terminal → outcome
      + SessionEndEvent ........................... runtime.py::PlaybookRuntime._enter
 16   turn-budget policy: wrap-up note, then
      force-advance to on_failure ................. runtime.py::PlaybookRuntime._apply_turn_budget
 17   quiesce ≤8 hops: pipeline→expr→auto→resume .. runtime.py::PlaybookRuntime._quiesce
 18   quiescent.set() → director_done resolves .... playbook/agent.py::PlaybookAgent._stream_turn
      ── TALKER lane (concurrent with 6–18) ────────────────────
 19   double-greet guard / settle_before_speak .... playbook/agent.py::PlaybookAgent._stream_turn
 20   speak_state snapshot; Talker.speak() ........ playbook/agent.py::PlaybookAgent._stream_turn
 21   hard gate? barrier on director_done,
      filler → hold-line, re-snapshot on settle ... talker.py::Talker.speak
 22   say_verbatim? exact template, NO LLM ........ talker.py::Talker.speak
 23   render_view() — the steering surface (§4) ... render.py::render_view
                                                    (from talker.py::Talker.speak)
 24   ONE streaming call; chunks stamped with
      spoke_from_version .......................... talker.py::Talker.speak
 25   chunks yield live; barge-in kills speech .... playbook/agent.py::PlaybookAgent._stream_turn
      ── JOIN ──────────────────────────────────────────────────
 26 shielded finally: await Director ............... playbook/agent.py::PlaybookAgent._stream_turn
 27 stale-speech suppression ....................... playbook/agent.py::PlaybookAgent._stream_turn
 28 check_repairs → "Correction" note .............. runtime.py::PlaybookRuntime.check_repairs
 29 pass-through verbatim yielded; done chunk
    carries final Turn + metadata ................. playbook/agent.py::PlaybookAgent._stream_turn
 30 observer.on_flow_node on checkpoint change ..... dialog_machine.py::DialogMachine.turn
 31 host prints/plays; reads state.ended ........... cli/main.py::_drive_agent
```

Two wiring notes worth knowing:

- `DialogMachine`'s playbook path (dialog_machine.py::DialogMachine.turn)
  calls `pb.turn(text, stream=...)` without the per-turn language parameter —
  bridge-detected language flows only on direct `PlaybookAgent` embedding.
- `runtime.on_user_text` runs with `record=False` inside the Director task
  (runtime.py::PlaybookRuntime.on_user_text) because the agent already
  appended the utterance at step 3 — this avoids a double append.

---

## 4. The steering surface — how the prompt is built, every turn

This is where the playbook grabs the steering wheel. `render.py::render_view`
and its system-block builder `render.py::_system_block` assemble the Talker's
entire world in fixed order:

```
┌─ SYSTEM BLOCK — render.py::_system_block ───────────────────┐
│ ╔═ CACHEABLE PREFIX (session-constant → provider cache) ═╗  │
│ ║ 1. PERSONA — always leads                              ║  │
│ ║ 2. GUIDELINE SPINE (composed per config)               ║  │
│ ║    grounding ALWAYS · voice_core · tone ·              ║  │
│ ║    language_accent · gender · domain ·                 ║  │
│ ║    end_discipline ...                                  ║  │
│ ║         (_guidelines.py::compose_guidelines)           ║  │
│ ║ 3. DATE ANCHOR + DATE_DISCIPLINE                       ║  │
│ ║         (_guidelines.py::datetime_anchor_line)         ║  │
│ ╚═════════════════════════════════════════════════════════╝  │
│ ── VOLATILE TAIL (changes per turn) ──                       │
│ 4. "## Direction from supervisor"   ◄─ Director steer note   │
│         (BEFORE guidance — guidance wins conflicts)          │
│ 5. "## Current step: {id} / Goal" + step guidance            │
│ 6. "Still needed: <unfilled required slots>"                 │
│ 7. "Never say: …"  hard negative constraints                 │
│ 8. "## Correction from supervisor"  ◄─ repair note           │
│         (AFTER guidance — correction OVERRIDES)              │
│ 9. "## Caller language (this turn)" (outside cache!)         │
│ 10. "## Known information" (collected slots)                 │
│ 11. "## Reference data" (computed views)                     │
│ 12. "## Earlier in this conversation" + memory guard         │
│ 13. "## Knowledge base" — ONLY if this step's guidance       │
│     references it; + "answer briefly, steer back"            │
│ 14. grounding close: "only state facts present in …"         │
└──────────────────────────────────────────────────────────────┘
┌─ TRANSCRIPT — render.py::render_view ───────────────────────┐
│ newest-first knapsack; 20% of budget RESERVED for history   │
│ so a fat KB can never starve the Talker of context          │
│ (render.py::_TRANSCRIPT_BUDGET_FRACTION)                    │
└──────────────────────────────────────────────────────────────┘
```

Items 4–14 are appended in exactly this order inside
`render.py::_system_block`.

Observability is built in: every turn emits `[guidelines] fed=[...]` and
`[turn-trace] side=brain version=N checkpoint=... slots=<keys only>` as
DEBUG-level log lines (render.py::_system_block) — slot keys only, never
values, so traces stay PII-safe.

---

## 5. Playbook semantics → conversation behavior

Each authoring construct maps to a distinct enforcement mechanism, ordered by
strength:

```
 CONSTRUCT            MECHANISM                                BEHAVIOR CHANGE
 ─────────────────    ─────────────────────────────────────    ──────────────────────────
 say_verbatim         bypasses LLM entirely                    regulated lines spoken
                      (talker.py::Talker.speak)                word-for-word, zero drift
 gate: hard           speech BARRIER on director_done          model literally cannot
                      (talker.py::Talker.speak)                speak past unconfirmed facts
 provisional writes   verdict can't confirm its own            prompt injection can't
                      requires in one shot                     advance a hard gate
                      (director.py::Director._write_status)
 judge: expr          sandboxed AST eval, no LLM               deterministic routing the
                      (expr.py::evaluate, via                  model can't talk around
                      director.py::Director._expr_advance)
 requires             advance blocked + "still need …"         agent circles back instead
                      steer note                               of skipping steps
                      (director.py::Director._steer_text)
 interrupts+resume    detour stack, auto-return                digressions answered, then
                      (runtime.py::PlaybookRuntime._hop)       back to the step it left
 turn_budget          wrap-up note → force on_failure          conversations can't stall
                      (runtime.py::PlaybookRuntime._apply_turn_budget)  forever on one step
 guidance             "Current step / Goal" section            model owns PHRASING inside
                      (render.py::_system_block)               a framework-owned OUTCOME
 steer/repair notes   Direction (weak) / Correction            supervisor nudges mid-
                      (strong) positioning                     conversation, self-healing
                      (render.py::_system_block)
 never_say            prompt line + DETERMINISTIC stream       authored phrases are
                      filter excising phrases before TTS       excised whatever the LLM
                      (talker.py::_filter_never_say)           emits — a guarantee
 uses_kb / KB gate    explicit per-checkpoint flag; None =     step-scoped facts, no
                      guidance-substring heuristic             off-step KB rambling
                      (models.py::Checkpoint.uses_kb,
                      render.py::_system_block)
 goodbye guard        terminal interrupt with required slots   a terse caller's "bye"
                      unfilled AND capture ≥2/3 complete →     can't silently skip
                      ONE steer to collect, then the next      nearly-complete required
                      goodbye passes                           capture
                      (director.py::Director.evaluate)
 repair loop          spoke_from_version diff → correction     never re-asks an answered
                      (runtime.py::PlaybookRuntime.check_repairs)  question twice
```

The one-line thesis: **checkpoints gate outcomes, not utterances** — the
framework owns where the conversation must go; the model owns only how it
sounds getting there.

---

## 6. Patterns it follows

- **Event sourcing + CQRS-lite** — append-only versioned `EventLog`; state is
  a pure fold, cached by log version
  (runtime.py::PlaybookRuntime.state). Replay = free persistence, audit, and
  eval: every conversation is immediately replayable.
- **Supervisor/actor two-LLM split** — a structured slow judge (Director)
  supervises a streaming fast speaker (Talker) that never decides anything.
- **Speculative execution with barrier** — optimistic speech, hard-gate
  barriers only at irreversible moments.
- **Hexagonal / ports & adapters** — `Agent` Protocol port;
  LiveKit/PipeCat/FastAPI/CLI adapters; `CompletesLLM`/`StreamsLLM` provider
  seams (fakes inject trivially for evals).
- **Facade + Strategy** — `DialogMachine` hides two engines behind one API.
- **Compiler-to-one-IR** — three authoring frontends lower to one validated
  `Playbook` model; references cross-checked at load, not at runtime.
- **FSM with audited transitions** — every advance carries a rule string
  (`expr:`, `llm:`, `policy:turn_budget`, `interrupt:`, `resume`) saying why.
- **Graceful-degradation ladder** — Director dies → `DegradedEvent`, Talker
  continues solo, LLM-free policies still enforce; Talker dies → retry →
  recovery line. Every rung is an event.
- **Sandboxing at every author-input seam** — Jinja `SandboxedEnvironment`,
  AST-whitelisted expr evaluator, untrusted-transcript hardening in the
  verdict prompt.
- **Cache-prefix split** — session-constant prompt head tagged for provider
  caching (render.py::render_view); the per-turn language directive lives
  deliberately outside it so a language switch never invalidates the cache.

---

## 7. What it buys over a vanilla LLM (same playbook, flat prompt)

```
                     VANILLA (1 prompt)          PLAYBOOK ENGINE
 outcome control     hope + prose                checkpoints, gates, requires — enforced
 regulated speech    can paraphrase/refuse       say_verbatim: exact, no LLM in path
 prompt injection    one context = one attack    provisional writes; verdict can't
                     surface                     self-confirm; transcript pinned untrusted
 context drift       whole history every call,   priority-packed view, 20% history
                     drowns in KB                reserve, step-scoped KB
 digressions         wanders, forgets task       interrupt → answer → auto-resume
 stalls              loops politely forever      turn_budget → wrap-up → on_failure
 self-repair         none                        spoke_from_version repair loop
 auditability        one blob of text            versioned event per decision, with
                                                 the rule that caused it
 latency             1 call                      1 streaming call on speech path;
                                                 judgment runs concurrent, not serial
```

Measured, not just claimed: the A/B harness ([05-eval-guide.md](05-eval-guide.md))
on a 17-step enquiry-qualification playbook (4 personas, gpt-4o-mini agent /
gpt-4o director / gpt-4.1-mini judge, after the audit-fix waves: call-end
detection, require-subset gating, expr fast-advance, Director cache prefix,
step-scoped KB) scored playbook **task_success 0.950 vs vanilla 0.850**,
**slot_accuracy 0.917 vs 0.792**, at **5,959 input tok/turn vs 11,415 (−48%)**
and lower p50 latency — the priority-packed view beats re-sending the raw
playbook + full history every call on cost AND quality. Two cautionary tales
from the same tuning loop: before the Talker history-starvation fix the
playbook scored task_success 0.05 (starve the Talker of transcript and no
amount of checkpointing saves the conversation), and naively requiring every
collected slot deadlocked a 14-slot branchy step (gate focused capture steps,
never multi-path qualifiers). The render surface (§4) *is* the behavior.
