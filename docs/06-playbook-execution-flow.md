# SuperDialog - Playbook Execution Flow

**Status:** Canonical
**Parent:** [README.md](README.md)

A code-verified trace of the Playbook engine: the layered design, the
two-brain turn loop, every touchpoint a user turn passes through (with
file:line citations into `src/superdialog/`), the prompt-assembly steering
surface, and how each playbook construct changes conversational behavior.
[01-architecture.md](01-architecture.md) explains the engine's concepts;
this doc walks the actual execution path.

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
              ║   Agent Protocol  (agent.py:38)  ║   the ONE embedding seam:
              ║   turn() assist() chat_ctx()     ║   structural typing, no ABC —
              ║   load_chat_ctx()                ║   any brain that fits, plugs in
              ╚═══════════════╤══════════════════╝
                              ▼
              ┌──────────────────────────────────┐
              │ DialogMachine (facade, strategy) │  dialog_machine.py:80
              │   _select_engine(:46) picks:     │
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
        │   │  source of   │   │  state = fold(log)  state.py:96
        │   │  truth)      │   │
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

```
 USER TEXT
    │
    ▼                                                            time ──►
 ┌──[1] append UtteranceEvent to log (agent.py:284)  ◄─ log FIRST, so both brains see it
 │
 ├────────────────────── fork ───────────────────────────────┐
 ▼  TALKER lane (speech path — streams NOW)                   ▼  DIRECTOR lane (shielded task,
 │                                                            │  survives barge-in) agent.py:292
 │  gated checkpoint?  (hard gate / hard slot)                │
 │   ├─ yes ─► BARRIER: wait ≤barrier_timeout ─► filler       │  1. judge:expr rules first —
 │   │         ─► wait ≤hold_timeout ─► polite hold-line      │     deterministic, ZERO LLM
 │   │         (talker.py:102)  then RE-SNAPSHOT fresh state  │     (director.py:345)
 │   └─ no ──► speak from snapshot immediately                │  2. else ONE structured JSON
 │                                                            │     verdict call (director.py:401)
 │  say_verbatim step? ─► speak template EXACTLY,             │  3. slot writes (typed, coerced,
 │                        NO LLM AT ALL (talker.py:124)       │     provisional at hard gates)
 │                                                            │  4. interrupt > advance
 │  else: render_view() builds the prompt  ──── §4 ────       │  5. advance if requires met,
 │        ONE streaming LLM call (talker.py:141)              │     else steer-note "cannot
 │        chunks yield live to host (agent.py:316)            │     move on yet"
 │                                                            │  6. quiesce loop ≤8 hops:
 │  barge-in? GeneratorExit kills SPEECH ONLY ────────────────┤     pipelines·auto·resume
 │                                                            │     (runtime.py:214)
 ├────────────────────── join ────────────────────────────────┘
 ▼
 shielded finally (agent.py:328): await Director — its decision ALWAYS lands
    │
    ├─ Director advanced checkpoint mid-turn + has pass-through?
    │     └─ SUPPRESS the Talker's stale speculative reply (agent.py:347)
    │        Director's verbatim lines are authoritative
    │
    ├─ check_repairs (runtime.py:169): did we re-ask something already answered?
    │     └─ append "Correction from supervisor" note ─► feeds NEXT turn's prompt
    │
    ▼
 Turn{text, metadata:{checkpoint, version, ended, outcome}} ──► host
```

Barrier timing note: the Talker default is `barrier_timeout=0.4s` /
`hold_timeout=4s` when wired through `DialogMachine` (talker.py:47-48,
dialog_machine.py:101); a directly-embedded `PlaybookAgent` defaults to
`barrier_timeout=4.0` (playbook/agent.py:107).

---

## 3. Full path — every touchpoint

Paths relative to `src/superdialog/`.

```
BOOTSTRAP (once per session)
 B1 host calls turn/start ......................... agent.py:38 (protocol)
 B2 DialogMachine.start() ......................... dialog_machine.py:339
 B3 _ensure_backend: load playbook, resolve LLMs,
    split provider → Director+Talker adapters,
    wire tracing + billing ........................ dialog_machine.py:225
 B4 greet(): Talker speaks opening checkpoint,
    greeting logged, double-greet guard armed ..... playbook/agent.py:380

PER TURN
  1 dispatch stream/non-stream .................... playbook/agent.py:154
  2 _ensure_started (cold runtime → start()) ...... playbook/agent.py:258
  3 user UtteranceEvent appended, version=N+1 ..... playbook/agent.py:284 · events.py:143
  4 entry checkpoint snapshotted .................. playbook/agent.py:287
  5 Director launched (detached, shielded) ........ playbook/agent.py:292
      ── DIRECTOR lane ──────────────────────────────────────────
  6   state = fold(log), cached by log version .... state.py:119 · runtime.py:74
  7   judge:expr rules — first match, no LLM ...... director.py:389,345
  8   verdict prompt: slots, rules, interrupts,
      date anchor, anti-injection, last 12 turns .. director.py:107
  9   ONE structured completion (json_mode) ....... director.py:401
 10   slot extraction: declared-only, type-coerced,
      provisional at hard gates ................... director.py:421
 11   interrupt beats advance (anti-regression) ... director.py:458
 12   advance: requires met → SlotWrites+Advance;
      unmet → "cannot move on yet" steer note ..... director.py:476
 13   resume-over-drift: return to interrupted
      step via resume_stack ....................... runtime.py:147
 14   checkpoint being left says its verbatim ..... runtime.py:153
 15   _enter: on_enter tools; terminal → outcome
      + SessionEndEvent ........................... runtime.py:398
 16   turn-budget policy: wrap-up note, then
      force-advance to on_failure ................. runtime.py:329
 17   quiesce ≤8 hops: pipeline→expr→auto→resume .. runtime.py:214
 18   quiescent.set() → director_done resolves .... playbook/agent.py:271
      ── TALKER lane (concurrent with 6–18) ────────────────────
 19   double-greet guard / settle_before_speak .... playbook/agent.py:298
 20   speak_state snapshot; Talker.speak() ........ playbook/agent.py:312
 21   hard gate? barrier on director_done,
      filler → hold-line, re-snapshot on settle ... talker.py:102
 22   say_verbatim? exact template, NO LLM ........ talker.py:124
 23   render_view() — the steering surface (§4) ... render.py:319 (from talker.py:137)
 24   ONE streaming call; chunks stamped with
      spoke_from_version .......................... talker.py:141
 25   chunks yield live; barge-in kills speech .... playbook/agent.py:316
      ── JOIN ──────────────────────────────────────────────────
 26 shielded finally: await Director ............... playbook/agent.py:328
 27 stale-speech suppression ....................... playbook/agent.py:347
 28 check_repairs → "Correction" note .............. runtime.py:169
 29 pass-through verbatim yielded; done chunk
    carries final Turn + metadata ................. playbook/agent.py:370
 30 observer.on_flow_node on checkpoint change ..... dialog_machine.py:423
 31 host prints/plays; reads state.ended ........... cli/main.py:103
```

Two wiring notes worth knowing:

- `DialogMachine`'s playbook path (dialog_machine.py:421) calls
  `pb.turn(text, stream=...)` without the per-turn language parameter —
  bridge-detected language flows only on direct `PlaybookAgent` embedding.
- `runtime.on_user_text` runs with `record=False` inside the Director task
  (runtime.py:126-127) because the agent already appended the utterance at
  step 3 — this avoids a double append.

---

## 4. The steering surface — how the prompt is built, every turn

This is where the playbook grabs the steering wheel. `render_view`
(render.py:188-358) assembles the Talker's entire world in fixed order:

```
┌─ SYSTEM BLOCK ──────────────────────────────────────────────┐
│ ╔═ CACHEABLE PREFIX (session-constant → provider cache) ═╗  │
│ ║ 1. PERSONA — always leads                 render.py:239 ║  │
│ ║ 2. GUIDELINE SPINE (composed per config)                ║  │
│ ║    grounding ALWAYS · voice_core · tone ·               ║  │
│ ║    language_accent · gender · domain ·                  ║  │
│ ║    end_discipline ...          _guidelines.py:204       ║  │
│ ║ 3. DATE ANCHOR + DATE_DISCIPLINE          render.py:242 ║  │
│ ╚═════════════════════════════════════════════════════════╝  │
│ ── VOLATILE TAIL (changes per turn) ──                       │
│ 4. "## Direction from supervisor"   ◄─ Director steer note   │
│         (BEFORE guidance — guidance wins conflicts) :255     │
│ 5. "## Current step: {id} / Goal" + step guidance    :257    │
│ 6. "Still needed: <unfilled required slots>"         :260    │
│ 7. "Never say: …"  hard negative constraints         :265    │
│ 8. "## Correction from supervisor"  ◄─ repair note           │
│         (AFTER guidance — correction OVERRIDES)      :269    │
│ 9. "## Caller language (this turn)" (outside cache!) :272    │
│ 10. "## Known information" (collected slots)         :274    │
│ 11. "## Reference data" (computed views)             :276    │
│ 12. "## Earlier in this conversation" + memory guard :281    │
│ 13. "## Knowledge base" — ONLY if this step's guidance       │
│     references it; + "answer briefly, steer back"    :295    │
│ 14. grounding close: "only state facts present in …" :312    │
└──────────────────────────────────────────────────────────────┘
┌─ TRANSCRIPT ────────────────────────────────────────────────┐
│ newest-first knapsack; 20% of budget RESERVED for history   │
│ so a fat KB can never starve the Talker of context          │
│ (_TRANSCRIPT_BUDGET_FRACTION)                   render.py:330│
└──────────────────────────────────────────────────────────────┘
```

Observability is built in: every turn prints `[guidelines] fed=[...]` and
`[turn-trace] side=brain version=N checkpoint=... slots=<keys only>`
(render.py:220-235) — slot keys only, never values, so traces stay PII-safe.

---

## 5. Playbook semantics → conversation behavior

Each authoring construct maps to a distinct enforcement mechanism, ordered by
strength:

```
 CONSTRUCT            MECHANISM                                BEHAVIOR CHANGE
 ─────────────────    ─────────────────────────────────────    ──────────────────────────
 say_verbatim         bypasses LLM entirely (talker.py:124)    regulated lines spoken
                                                               word-for-word, zero drift
 gate: hard           speech BARRIER on director_done          model literally cannot
                      (talker.py:102)                          speak past unconfirmed facts
 provisional writes   verdict can't confirm its own            prompt injection can't
                      requires in one shot (director.py:421)   advance a hard gate
 judge: expr          sandboxed AST eval, no LLM               deterministic routing the
                      (director.py:345)                        model can't talk around
 requires             advance blocked + "still need …"         agent circles back instead
                      steer note (director.py:516)             of skipping steps
 interrupts+resume    detour stack, auto-return                digressions answered, then
                      (runtime.py:147)                         back to the step it left
 turn_budget          wrap-up note → force on_failure          conversations can't stall
                      (runtime.py:329)                         forever on one step
 guidance             "Current step / Goal" section            model owns PHRASING inside
                      (render.py:257)                          a framework-owned OUTCOME
 steer/repair notes   Direction (weak) / Correction            supervisor nudges mid-
                      (strong) positioning (render.py:255,269) conversation, self-healing
 never_say            prompt line + DETERMINISTIC stream       authored phrases are
                      filter excising phrases before TTS       excised whatever the LLM
                      (talker.py _filter_never_say)            emits — a guarantee
 uses_kb / KB gate    explicit per-checkpoint flag; None =     step-scoped facts, no
                      guidance-substring heuristic             off-step KB rambling
                      (models.py uses_kb, render.py:295)
 goodbye guard        terminal interrupt with required slots   a terse caller's "bye"
                      unfilled → ONE steer to collect, then    can't silently skip
                      the next goodbye passes (director.py)    required capture
 repair loop          spoke_from_version diff → correction     never re-asks an answered
                      (runtime.py:169)                         question twice
```

The one-line thesis: **checkpoints gate outcomes, not utterances** — the
framework owns where the conversation must go; the model owns only how it
sounds getting there.

---

## 6. Patterns it follows

- **Event sourcing + CQRS-lite** — append-only versioned `EventLog`; state is
  a pure fold, cached by log version (runtime.py:74-77). Replay = free
  persistence, audit, and eval: every conversation is immediately replayable.
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
  caching (render.py:352); the per-turn language directive lives deliberately
  outside it so a language switch never invalidates the cache.

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
