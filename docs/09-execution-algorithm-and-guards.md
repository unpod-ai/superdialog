# Playbook Execution Algorithm & Framework Guards

**Scope:** the per-turn execution algorithm of the playbook engine
(`src/superdialog/playbook/`) as of continuity v2, and the inventory of
deterministic guards the framework layers on top of its LLMs.
Companion to `06-playbook-execution-flow.md` (component walkthrough) and
`docs/plans/2026-08-07-playbook-continuity-v2-design.md` (design rationale).

## Design rule

> **Guards bound structure, evidence, and blast radius — semantics stay with
> the LLM.**

The engine never second-guesses what the model *understood* (a lexical filter
that junked a caller's "No." as a placeholder killed a production call — see
G4's history). It instead enforces: allowlists and types (structure), what
counts as *confirmed* and what advances silently (evidence), and what a wrong
decision is allowed to cost (blast radius).

## Authority split

| loop | runs | decides | share of authority |
|---|---|---|---|
| Talker | every turn (streaming) | wording only | speech |
| Director | every turn (one JSON verdict) | slots, advance, interrupt — **proposals** | ~60% |
| Engine | every event | validation, classification, audit | ~25% |
| Supervisor | only on watermarked trigger evidence | trajectory repair (inject/redirect/rewind/discard/handover) | ~15% |

`legacy_continuity: true` on a playbook restores pre-v2 semantics (guards
marked *v2* below become inert; defect-class guards stay on).

## The turn algorithm

```
TURN(user_text):
  1. APPEND   UtteranceEvent → EventLog (append-only, versioned)
  2. FOLD     state = fold(EventLog)          # pure; replay = same state
  3. DIRECTOR one LLM call → {slots, advance, note, interrupt}
  4. ENGINE   validate the verdict (guards G1-G18 below)
  5. QUIESCE  expr rules / pipelines / auto hops until stable (G19-G21, G36)
  6. TALKER   one streaming call from the settled state (G22-G28)
  7. SUPERVISOR trajectory review iff a trigger has NEW evidence (G29-G32)
  always:     repair detection, injection clamps, post-terminal silence (G33-G35)
```

## Guard inventory

### Director / verdict validation

| id | guard | class |
|---|---|---|
| G1 | LLM call retried once before degrading | resilience |
| G2 | json_mode + fence-strip; malformed → `DegradedEvent`, Talker continues solo | fail-loud |
| G3 | slot allowlist: declared-on-any-checkpoint (v2; multi-entity stays checkpoint-strict); `authoritative` slots are tool-only | structure |
| G4 | junk guard filters **structure only**: JSON `null`, empty/whitespace, literal `"null"`. Semantic strings (`none`, `unknown`, `n/a`) are trusted — the verdict prompt's decline-convention makes them stated answers. *(v2; redesigned after a lexical filter re-ask-looped a caller's "No." until hangup)* | structure |
| G5 | type/enum/date/time coercion; invalid → skipped, never stored; str values whitespace-collapsed + clamped (prompt-forgery defense) | structure |
| G6 | `resolve_from` slots must be a **live candidate id** — raw caller text never stored as an id | structure |
| G7 | churn dampener: identical confirmed re-extraction → no event *(v2)* | evidence |
| G8 | hard-gate slots stay **provisional** from a verdict (it can never confirm its own `requires`); fast-release deny-list (phone/otp/pan/…) always escalates | evidence |
| G9 | language-named slots filled from the **bridge-detected** language, not LLM inference *(v2)* | structure |

### Interrupt handling

| id | guard | class |
|---|---|---|
| G10 | interrupt targeting the current checkpoint / an active detour → **hold detour open** (steer, no self-loop, no stack corruption) | never-loop |
| G11 | unknown interrupt id → `DegradedEvent` (loud) | fail-loud |
| G12 | deterministic goodbye backstop (literal bye tokens) + false-goodbye and manipulation guards ("pretend the flow is over" ≠ goodbye) | injection defense |
| G13 | terminal-interrupt wrap guard: capture ≥2/3 complete → ONE wrap question (one-shot marker), then the close proceeds | evidence |
| G14 | close-class interrupt into a terminal checkpoint → **closing steer** ("brief warm close — no questions, no offers") *(v2)* | speech honesty |

### Advance handling

| id | guard | class |
|---|---|---|
| G15 | corroboration classifier: `requires` met / expr fired / same-turn **caller-derived** write (a rule's own `set:` never self-vouches). Uncorroborated → advance still lands **plus** a steer into the target; stamped `corroborated: false` *(v2)* | evidence |
| G16 | `requires` gate: hard → confirmed, soft → filled; unmet → steer, no advance | evidence |
| G17 | unknown advance target → `DegradedEvent` (was a silent stall) | fail-loud |
| G18 | end-on-frustration: repair note in flight + terminal advance → recover instead of hanging up | evidence |

### Quiescence / runtime

| id | guard | class |
|---|---|---|
| G19 | `max_hops` = 8 per quiesce; exhaustion → `DegradedEvent` | never-wedge |
| G20 | turn budget → wrap-up steer → grace → **forced advance** (a call can never camp forever) | never-wedge |
| G21 | resume integrity: nested detours unwind level-by-level; stack entries expire after 6 advances; a **corroborated** advance beats the forced return; held detours skip resume *(v2 pieces flagged)* | never-loop |
| G36 | tool interception ladder: reversible = free · compensable = expr guard · irreversible = confirmed slots + fail-**closed** classifier; deny steers deduped | blast radius |

### Talker / speech

| id | guard | class |
|---|---|---|
| G22 | hard-gate barrier: wait → filler → hold line; Director down → polite degrade, never hang | resilience |
| G23 | `say_verbatim` bypasses generation; `strict` without a script → recovery line, never improvised | speech honesty |
| G24 | `never_say`: deterministic token excision on the stream AND the filler (punctuation-only remainder → skipped) | speech honesty |
| G25 | filler/hold lines tagged and **excluded from the logged transcript** (spoken, never poisoning Director context) | speech honesty |
| G26 | stale-speech suppression when the turn advanced and verbatim pass-through exists | speech honesty |
| G27 | stream retry ×1 → recovery line; inner stream closed on barge-in | resilience |
| G28 | token budget with a reserved transcript floor; KB injected only on steps that use it | structure |

### Supervisor (Loop 2 — default-on under v2)

| id | guard | class |
|---|---|---|
| G29 | trigger-gated: free detectors every turn, LLM verdict only past a 2-turn cooldown; **sticky triggers watermarked** — no new qualifying evidence since the last review → silent | cost bound |
| G30 | redirect is **forward-only**: terminal targets blocked AND completed-checkpoint re-entry blocked (a caller's restart demand is never honored) | never-loop |
| G31 | rewind: version bounds; compensable effects need a two-step caller confirmation; refuses across irreversible effects; partial compensation → loud | blast radius |
| G32 | malformed supervisor verdict → `DegradedEvent`; the call is never disturbed | fail-loud |

### Always-on

| id | guard | class |
|---|---|---|
| G33 | `check_repairs`: Talker re-asked an already-answered slot → repair note (idempotent) | evidence |
| G34 | injection clamps: notes/steers whitespace-collapsed and length-clamped; all three prompts declare the transcript **untrusted** | injection defense |
| G35 | post-terminal short-circuit: turns after session end are recorded but never resurrect the call; first-turn double-greeting guard | never-loop |

## Failure-mode → guard map (from production evidence)

| production failure | guard(s) that now prevent it |
|---|---|
| self-interrupt corrupted resume stack, caller's place lost (westgate2) | G10, G21 |
| interrupts dead after one visit | G10 (detour-scoped suppression) |
| `configuration='None'` fabrications confirmed as truth | G4 (structural) + prompt omit-convention + G7/G8 |
| caller's "No." junked into a re-ask loop until hangup (ENDSESSION) | G4 redesign (semantics returned to the LLM) |
| conveyor rules desyncing state from dialogue | G15 (uncorroborated steer + audit) |
| goodbye turn free-wheeling into offers (Rohan-1) | G14 |
| supervisor honoring restart demands (restart-demander) | G30 |
| filler polluting transcripts in the wrong language | G24, G25 |
| prompt-injection "pretend the flow is over" closing the call | G12 |
| calls wedged on unsatisfiable gates | G19, G20 |

## Verification status (2026-08-08)

711 unit tests; five behavioral eval suites green (composites .80–.92) with
the goodbye/no-reentry checks live and falsifiable; the two historical
breaking points (prompt-injection, goodbye-turn close) at ts 1.0 / 0.9–1.0.
