# Playbook Continuity v2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the continuity v2 design (`docs/plans/2026-08-07-playbook-continuity-v2-design.md`): fix the four continuity-destroying engine defects unconditionally, then add the soft-gated advance semantics, junk-slot guard, supervisor default-on, and observability upgrades behind the `legacy_continuity` escape hatch.

**Architecture:** All engine work lands in `src/superdialog/playbook/` (this repo, branch `feat/playbook-continuity-v2`). The director/runtime stay a single-LLM-call-per-turn design; the engine gains a corroboration classifier and detour-continuation signal. Host fixes (Phase F) live in the sibling `super` repo and ship separately.

**Tech Stack:** Python 3.12, pydantic v2, pytest (async via anyio, already configured), `uv` for everything.

**Test idioms to reuse** (read these before starting):
- `tests/playbook/test_runtime.py` — `_runtime()` helper pattern: `PlaybookRuntime(pb, director_llm=CannedLLM(payload), http=FakeHttp([]))`
- `tests/playbook/test_director.py` — `CannedLLM` (returns one canned JSON verdict forever)
- `tests/playbook/test_models.py` — `MINIMAL_YAML` inline playbook fixture
- Run everything with `uv run pytest tests/playbook -q` (601 tests green at baseline).

---

## Shared test fixture (used by many tasks below)

Create `tests/playbook/continuity_fixtures.py`:

```python
"""Shared fixtures for continuity-v2 tests."""

import json
import textwrap

CONTINUITY_YAML = textwrap.dedent("""
    persona: "Test assistant."
    journeys:
      main:
        checkpoints:
          - id: ask_location
            goal: "Capture location"
            gate: soft
            slots:
              location: {type: str}
            guidance: "Ask for location."
            advance_when:
              - {when: "location given", judge: llm, to: main.pitch,
                 requires: [location]}
          - id: pitch
            goal: "Pitch the product"
            gate: soft
            guidance: "Pitch."
            advance_when:
              - {when: "caller responded in any way", judge: llm,
                 to: main.ask_budget}
          - id: ask_budget
            goal: "Capture budget"
            gate: soft
            slots:
              budget: {type: str}
            guidance: "Ask for budget."
            advance_when:
              - {when: "budget given", judge: llm, to: main.close,
                 requires: [budget]}
          - id: pricing_faq
            goal: "Answer pricing questions"
            gate: soft
            guidance: "Answer pricing."
          - id: close
            terminal: true
            outcome: done
    interrupts:
      - {id: price_guardrail, when: "caller asks about price", judge: llm,
         to: main.pricing_faq, resume: true}
""")


class SeqLLM:
    """Director stub returning a DIFFERENT canned verdict per call.

    Unlike CannedLLM (same verdict forever), this pops from a sequence —
    needed for multi-turn scenarios (interrupt, then self-interrupt, then
    plain turn). Falls back to an empty verdict when exhausted.
    """

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)

    async def complete(self, messages, **kwargs) -> str:
        if self._payloads:
            return json.dumps(self._payloads.pop(0))
        return json.dumps({"slots": {}, "advance": None, "note": None})
```

Commit this file together with Task 1 (it has no standalone test).

---

## Phase A — Unconditional engine fixes

These apply regardless of `legacy_continuity`. They are defect fixes.

### Task 1: E1+E2 — detour-continuation guard for interrupts

The core westgate2 bug. Two defects, one mechanism:
- **E1:** an interrupt targeting the *current* checkpoint pushes the checkpoint onto its own resume stack, then "resumes" to itself, stranding the real return point.
- **E2:** `already_handled = spec.to in state.completed` uses the append-only whole-call list, so an interrupt whose target was ever exited is dead for the rest of the call.

Fix: replace the `completed` check with an *active-detour* check (`spec.to == cp_ref or spec.to in state.resume_stack`). When it hits, hold the detour open: emit a steer, and signal the runtime (new `DirectorDecision.detour_continues`) to skip the forced resume this turn — in **both** `on_user_text` and `_hop`.

**Files:**
- Create: `tests/playbook/continuity_fixtures.py` (content above)
- Create: `tests/playbook/test_continuity_interrupts.py`
- Modify: `src/superdialog/playbook/director.py` (interrupt block in `evaluate`, ~line 571; `DirectorDecision` class ~line 37)
- Modify: `src/superdialog/playbook/runtime.py` (`on_user_text` resume branch ~line 197; `_quiesce`/`_hop` ~lines 496-550)

**Step 1: Write the failing tests**

```python
# tests/playbook/test_continuity_interrupts.py
from superdialog.playbook.events import AdvanceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.continuity_fixtures import CONTINUITY_YAML, SeqLLM
from tests.playbook.test_toolexec import FakeHttp

INTERRUPT = {"slots": {}, "advance": None, "note": None,
             "interrupt": "price_guardrail"}
PLAIN = {"slots": {}, "advance": None, "note": None}


def _rt(payloads: list[dict]) -> PlaybookRuntime:
    return PlaybookRuntime(
        Playbook.from_yaml(CONTINUITY_YAML),
        director_llm=SeqLLM(payloads),
        http=FakeHttp([]),
    )


async def test_self_interrupt_holds_detour_instead_of_self_looping() -> None:
    """westgate2 steps 10-12: a second price question while already inside
    pricing_faq must NOT re-fire the interrupt onto itself."""
    rt = _rt([INTERRUPT, INTERRUPT, PLAIN])
    await rt.start()
    await rt.on_user_text("what's the price?")          # detour in
    assert rt.state.checkpoint_id == "main.pricing_faq"
    assert rt.state.resume_stack == ["main.ask_location"]

    await rt.on_user_text("and any discounts?")          # self-interrupt turn
    # stays in the detour: no self-loop advance, stack unchanged
    assert rt.state.checkpoint_id == "main.pricing_faq"
    assert rt.state.resume_stack == ["main.ask_location"]
    self_loops = [
        e for e in rt.log.events
        if isinstance(e, AdvanceEvent)
        and e.from_checkpoint == e.to_checkpoint == "main.pricing_faq"
    ]
    assert self_loops == []
    assert rt.state.steering_note  # the hold-open steer is live

    await rt.on_user_text("ok thanks")                   # detour done
    assert rt.state.checkpoint_id == "main.ask_location"  # resumed correctly
    assert rt.state.resume_stack == []


async def test_interrupt_can_refire_after_detour_completes() -> None:
    """E2: a completed detour must not kill the interrupt for the whole call."""
    rt = _rt([INTERRUPT, PLAIN, INTERRUPT])
    await rt.start()
    await rt.on_user_text("price?")                      # detour in
    await rt.on_user_text("thanks")                      # resume out
    assert rt.state.checkpoint_id == "main.ask_location"

    await rt.on_user_text("wait, price again?")          # must fire AGAIN
    assert rt.state.checkpoint_id == "main.pricing_faq"
```

**Step 2: Run to verify both fail**

Run: `uv run pytest tests/playbook/test_continuity_interrupts.py -v`
Expected: FAIL — first test sees a `pricing_faq→pricing_faq` advance (or wrong resume target); second test stays at `ask_location` because `completed` suppression eats the interrupt.

**Step 3: Implement**

In `src/superdialog/playbook/director.py`:

3a. Add the field to `DirectorDecision`:

```python
class DirectorDecision(BaseModel):
    """Outcome of one Director evaluation: events to append, or degraded."""

    events: list[Event] = Field(default_factory=list)
    degraded: bool = False  # LLM failed; Talker continues solo
    detail: str = ""  # why degraded: llm_error | json_parse_error | non_dict_verdict
    # The interrupt that fired targets the checkpoint we are already at (or a
    # detour we are already inside): the runtime must hold the detour open
    # this turn instead of forcing the resume return.
    detour_continues: bool = False
```

3b. In `evaluate`, replace the `already_handled` logic inside `if interrupt_id:`:

```python
        if interrupt_id:
            spec = next((i for i in self._pb.interrupts if i.id == interrupt_id), None)
            if spec is not None and (
                spec.to == cp_ref or spec.to in state.resume_stack
            ):
                # Already at (or inside the detour of) this interrupt's target.
                # Re-firing would push the current step onto its own resume
                # stack and later "resume" back to itself, stranding the real
                # return point (westgate2 steps 10-12). Hold the detour open:
                # keep handling the topic here this turn.
                events.append(
                    SteeringNoteEvent(
                        text=(
                            "The caller is continuing the same topic — keep "
                            "handling it at this step before resuming the flow."
                        ),
                        kind="steer",
                    )
                )
                return DirectorDecision(events=events, detour_continues=True)
            if spec is not None:
                # ... existing terminal-interrupt slot guard + AdvanceEvent
                #     body unchanged, just de-indented from the old
                #     `not already_handled` condition ...
```

Delete the `already_handled` variable entirely.

3c. In `src/superdialog/playbook/runtime.py`:

- `on_user_text` resume branch gains the new condition:

```python
        if (
            state.entered_via_resume
            and state.resume_stack
            and not is_interrupt
            and not decision.detour_continues
        ):
```

- Thread the hold through quiescence. Change `_quiesce` and `_hop` signatures:

```python
    async def _quiesce(self, hold_resume: bool = False) -> list[str]:
        ...
            if not await self._hop(pass_through, hold_resume=hold_resume):
        ...

    async def _hop(self, pass_through: list[str], hold_resume: bool = False) -> bool:
```

and guard `_hop`'s resume branch:

```python
        if (
            not hold_resume
            and state.entered_via_resume
            and state.resume_stack
            and state.user_turns_in_checkpoint >= 1
        ):
```

- In `on_user_text`, the final quiesce call becomes
  `pass_through.extend(await self._quiesce(hold_resume=decision.detour_continues))`.
  All other `_quiesce()` call sites keep the default.

**Step 4: Run the new tests, then the whole suite**

Run: `uv run pytest tests/playbook/test_continuity_interrupts.py -v`
Expected: PASS.
Run: `uv run pytest tests/playbook -q`
Expected: mostly green. `grep -rn "completed" tests/playbook/test_director.py` — any test asserting the old completed-path suppression (the `global_card_not_received` scenario in the deleted comment) must be updated to assert the new detour-scoped behavior instead; justify in the commit message.

**Step 5: Commit**

```bash
git add tests/playbook/continuity_fixtures.py tests/playbook/test_continuity_interrupts.py src/superdialog/playbook/director.py src/superdialog/playbook/runtime.py
git commit -m "fix(playbook): hold detour open on self-interrupt; scope interrupt suppression to active detour

E1: an interrupt targeting the current checkpoint pushed it onto its own
resume stack and 'resumed' to itself, stranding the real return point
(westgate2 traversal steps 10-12). E2: already_handled checked the
append-only completed list, killing an interrupt for the whole call after
one visit."
```

### Task 2: E3 — unknown advance target logs a DegradedEvent

**Files:**
- Modify: `src/superdialog/playbook/director.py` (advance block, ~line 618)
- Test: `tests/playbook/test_continuity_interrupts.py` (append)

**Step 1: Failing test**

```python
async def test_unknown_advance_target_is_logged_not_silent() -> None:
    from superdialog.playbook.events import DegradedEvent
    rt = _rt([{"slots": {}, "advance": "main.nonexistent", "note": None}])
    await rt.start()
    await rt.on_user_text("hello")
    assert rt.state.checkpoint_id == "main.ask_location"  # no advance
    assert any(
        isinstance(e, DegradedEvent)
        and e.detail == "unknown_advance_target:main.nonexistent"
        for e in rt.log.events
    )
```

**Step 2:** Run: `uv run pytest tests/playbook/test_continuity_interrupts.py::test_unknown_advance_target_is_logged_not_silent -v` — FAIL (no DegradedEvent).

**Step 3: Implement.** In `evaluate`, the advance block currently does `rule = next(...)` then `if rule is not None:`. Add the else:

```python
            if rule is not None:
                ...
            else:
                # The verdict named a target no llm rule declares. Silent
                # no-ops here looked like caller-visible stalls with zero log
                # evidence — make them auditable.
                events.append(
                    DegradedEvent(
                        component="director",
                        detail=f"unknown_advance_target:{target}",
                    )
                )
```

Add `DegradedEvent` to the `from .events import (...)` list in director.py.

**Step 4:** Run the test (PASS), then `uv run pytest tests/playbook -q` (green).

**Step 5: Commit** — `fix(playbook): log DegradedEvent when verdict names an unknown advance target`

### Task 3: E9a — barrier filler respects never_say

**Files:**
- Modify: `src/superdialog/playbook/talker.py` (`speak`, filler yield ~line 163)
- Test: `tests/playbook/test_talker.py` (append; reuse its existing fakes/playbook helpers — read the file first)

**Step 1: Failing test** (adapt helper names to what `test_talker.py` already uses):

```python
async def test_filler_respects_never_say() -> None:
    """golf lists the engine's own filler in never_say; it must be excised."""
    # Build a playbook whose gated checkpoint has
    #   never_say: ["One moment, let me confirm that"]
    # and a director_done that resolves only AFTER the barrier timeout.
    # Assert: no yielded chunk contains "One moment"; and when the excised
    # filler is empty/punctuation-only, no filler chunk is yielded at all.
```

Concretely: instantiate `Talker(pb, llm, barrier_timeout=0.01, hold_timeout=0.5)` with an async `director_done` that `await anyio.sleep(0.1)` before returning the state; collect chunks from `speak(state, director_done=director_done)`; assert `all("One moment" not in c.text for c in chunks)`.

**Step 2:** Run — FAIL (filler chunk contains the phrase).

**Step 3: Implement.** In `Talker.speak`, the gated branch currently yields the filler directly. Route it through the same excision the stream gets:

```python
            if fresh is None:
                filler_text = self._resolve_line(self._filler, state, FILLER)
                folded = [p.casefold() for p in (cp.never_say or []) if p]
                if folded:
                    filler_text = _excise(filler_text, folded)
                # Skip a filler that excision reduced to punctuation/space —
                # speaking "…" alone is worse than silence.
                if re.sub(r"[\W_]+", "", filler_text, flags=re.UNICODE):
                    yield SpeechChunk(
                        text=filler_text + " ",
                        spoke_from_version=state.version,
                        filler=True,   # Task 4 adds this field
                    )
                with anyio.move_on_after(self._hold_timeout):
                    fresh = await director_done()
```

Add `import re` at the top of talker.py. (Add the `filler=True` kwarg only in Task 4; keep this task compiling by omitting it here if you do the tasks strictly in order.)

**Step 4:** Test PASS; full suite green (`test_talker.py` has filler assertions — update any that assert the raw FILLER text is always yielded, per the design).

**Step 5: Commit** — `fix(playbook): run barrier filler through never_say excision`

### Task 4: E9b — filler chunks excluded from the logged utterance

**Files:**
- Modify: `src/superdialog/playbook/talker.py` (`SpeechChunk` + filler/hold-line yields)
- Modify: `src/superdialog/playbook/agent.py` (`talker_text` join, ~line 431)
- Test: `tests/playbook/test_agent.py` (append; reuse its agent-construction helpers)

**Step 1: Failing test** — construct a `PlaybookAgent` on a hard-gated playbook whose director stub sleeps past `barrier_timeout` (so the filler fires), run one full `turn()`, then assert the logged assistant `UtteranceEvent` does NOT start with the filler text while the streamed chunks DID include it.

**Step 2:** FAIL — the log contains "One moment, let me confirm that…" as the utterance prefix (exactly what westgate1 steps 11-12 show).

**Step 3: Implement.**
- `SpeechChunk` gains `filler: bool = False`.
- Both the barrier-filler yield and the hold-line yield in `Talker.speak` set `filler=True`.
- `agent.py` `_stream_turn` finally block:

```python
                talker_text = "".join(
                    c.text for c in talker_chunks if not c.filler
                ).strip()
```

The filler is still streamed/spoken (the caller hears it); it just stops polluting the transcript the director and talker reason over.

**Step 4:** PASS + full suite green.

**Step 5: Commit** — `fix(playbook): keep barrier filler out of the logged transcript`

---

## Phase B — v2 semantics behind `legacy_continuity`

### Task 5: the flag

**Files:**
- Modify: `src/superdialog/playbook/models.py` (`Playbook` model)
- Modify: `src/superdialog/playbook/simple.py` (simple-format model + conversion, ~line 489)
- Test: `tests/playbook/test_models.py` (append)

**Step 1: Failing test**

```python
def test_legacy_continuity_flag_defaults_off() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    assert pb.legacy_continuity is False
    legacy = Playbook.from_yaml(MINIMAL_YAML + "\nlegacy_continuity: true\n")
    assert legacy.legacy_continuity is True
```

**Step 2:** FAIL (unknown field or attribute error).

**Step 3:** Add to `Playbook`:

```python
    # Continuity v2 escape hatch: true restores pre-v2 semantics (no junk-slot
    # rejection, no churn dampener, no uncorroborated-advance steer, supervisor
    # stays opt-in). New playbooks get v2 by default.
    legacy_continuity: bool = False
```

Mirror the field on the simple-format model in `simple.py` and pass it through in the simple→Playbook conversion (find where `supervisor=sp.supervisor` is forwarded, ~line 489, and forward `legacy_continuity=` alongside).

**Step 4:** PASS + suite green. **Step 5: Commit** — `feat(playbook): legacy_continuity escape hatch flag`

### Task 6: junk-slot guard (+ junk_rejected audit signal)

**Files:**
- Modify: `src/superdialog/playbook/director.py`
- Test: `tests/playbook/test_director.py` (append)

**Step 1: Failing tests**

```python
async def test_junk_slot_values_are_rejected_under_v2() -> None:
    # CannedLLM verdict: {"slots": {"city": "None"}, "advance": None}
    # against MINIMAL_YAML (v2 by default).
    # Assert: no SlotWriteEvent for city; a DegradedEvent with
    # detail == "junk_rejected:city" is in the decision events.

async def test_junk_slot_values_pass_under_legacy() -> None:
    # Same verdict against MINIMAL_YAML + legacy_continuity: true.
    # Assert: SlotWriteEvent city="None" written (byte-identical old behavior).
```

Write them via `Director(pb, CannedLLM(...)).evaluate(state)` directly (see existing `test_director.py` patterns for building a folded state).

**Step 2:** FAIL. **Step 3: Implement** in director.py:

```python
#: Values a verdict sometimes emits that mean "nothing extracted". Writing
#: them as confirmed truth poisoned the Talker prompt every turn
#: (configuration='None', city='' in production traversals).
_JUNK_VALUES = {"", "none", "null", "n/a", "na", "not specified", "unknown"}


def _is_junk(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _JUNK_VALUES
```

In the `evaluate` slot loop, before coercion:

```python
        for key, value in (verdict.get("slots") or {}).items():
            slot_spec = cp.slots.get(key)
            if slot_spec is None or slot_spec.authoritative:
                continue
            if not self._pb.legacy_continuity and _is_junk(value):
                # Auditable rejection: repeated junk on one key is a
                # supervisor trigger (junk_rejected:<key>).
                events.append(
                    DegradedEvent(
                        component="director", detail=f"junk_rejected:{key}"
                    )
                )
                continue
            coerced = _coerce_slot(value, slot_spec, state.now)
            ...
```

**Step 4:** PASS + suite. **Step 5: Commit** — `feat(playbook): reject junk slot values under v2, audit as junk_rejected`

### Task 7: churn dampener

**Files:** `src/superdialog/playbook/director.py`; test in `tests/playbook/test_director.py`.

**Step 1: Failing test** — state already holds `city="Pune"` confirmed; verdict re-extracts `{"city": "Pune"}`; assert NO new `SlotWriteEvent` in decision events under v2, and one IS emitted under `legacy_continuity: true`.

**Step 2:** FAIL. **Step 3:** In the same loop, after coercion:

```python
            existing = state.slots.get(_ekey(cp.entity, key))
            if (
                not self._pb.legacy_continuity
                and existing is not None
                and existing.status == "confirmed"
                and existing.value == coerced
            ):
                continue  # identical confirmed value: no event, no churn
```

(`_ekey` is already imported in director.py.)

**Step 4:** PASS + suite (westgate2 wrote `staying` 4× identically — this ends that). **Step 5: Commit** — `feat(playbook): drop identical-value confirmed slot rewrites under v2`

### Task 8: corroboration classification + uncorroborated steer

**Files:**
- Modify: `src/superdialog/playbook/events.py` (`AdvanceEvent`)
- Modify: `src/superdialog/playbook/director.py` (`_expr_advance` + advance block)
- Test: `tests/playbook/test_continuity_advances.py` (create)

**Step 1: Failing tests**

```python
# tests/playbook/test_continuity_advances.py
from superdialog.playbook.events import AdvanceEvent, SteeringNoteEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.continuity_fixtures import CONTINUITY_YAML, SeqLLM
from tests.playbook.test_toolexec import FakeHttp


def _rt(payloads):
    return PlaybookRuntime(
        Playbook.from_yaml(CONTINUITY_YAML),
        director_llm=SeqLLM(payloads),
        http=FakeHttp([]),
    )


async def test_requires_backed_advance_is_corroborated_and_silent() -> None:
    rt = _rt([{"slots": {"location": "Pune"}, "advance": "main.pitch"}])
    await rt.start()
    await rt.on_user_text("I'm in Pune")
    adv = [e for e in rt.log.events if isinstance(e, AdvanceEvent)
           and e.to_checkpoint == "main.pitch"]
    assert adv and adv[-1].corroborated is True
    assert rt.state.steering_note is None  # silent


async def test_prose_only_advance_is_uncorroborated_and_steered() -> None:
    rt = _rt([
        {"slots": {"location": "Pune"}, "advance": "main.pitch"},
        {"slots": {}, "advance": "main.ask_budget"},   # conveyor rule, no evidence
    ])
    await rt.start()
    await rt.on_user_text("Pune")
    await rt.on_user_text("hmm what?")
    adv = [e for e in rt.log.events if isinstance(e, AdvanceEvent)
           and e.to_checkpoint == "main.ask_budget"]
    assert adv and adv[-1].corroborated is False
    assert rt.state.steering_note is not None
    assert "without confirmed input" in rt.state.steering_note


async def test_prose_only_advance_not_steered_under_legacy() -> None:
    pb = Playbook.from_yaml(CONTINUITY_YAML).model_copy(
        update={"legacy_continuity": True}
    )
    rt = PlaybookRuntime(pb, director_llm=SeqLLM([
        {"slots": {"location": "Pune"}, "advance": "main.pitch"},
        {"slots": {}, "advance": "main.ask_budget"},
    ]), http=FakeHttp([]))
    await rt.start()
    await rt.on_user_text("Pune")
    await rt.on_user_text("hmm what?")
    assert rt.state.steering_note is None
```

**Step 2:** FAIL (no `corroborated` field). **Step 3: Implement.**

`events.py` — `AdvanceEvent` gains:

```python
    # v2 accountability: True = slot evidence / expr backed this advance;
    # False = prose-only (steered); None = unclassified (interrupts, policy,
    # resume, pre-v2 logs).
    corroborated: bool | None = None
```

`director.py`:
- `_expr_advance`: stamp `corroborated=True` on its `AdvanceEvent`.
- LLM advance block — replace the `events.append(AdvanceEvent(...))` with:

```python
                    slot_written_this_turn = any(
                        isinstance(e, SlotWriteEvent) and e.key in cp.slots
                        for e in events
                    )
                    corroborated = bool(rule.requires) or slot_written_this_turn
                    events.append(
                        AdvanceEvent(
                            from_checkpoint=cp_ref,
                            to_checkpoint=rule.to,
                            rule=rule.rule_id,
                            corroborated=corroborated,
                        )
                    )
                    if not corroborated and not self._pb.legacy_continuity:
                        # Appended AFTER the AdvanceEvent so the fold's
                        # advance-time steering reset doesn't clear it: the
                        # steer belongs to the TARGET step.
                        events.append(
                            SteeringNoteEvent(
                                text=(
                                    "Entering this step without confirmed "
                                    "input — address the caller's last "
                                    "utterance first, then pursue this "
                                    "step's goal."
                                ),
                                kind="steer",
                            )
                        )
```

Interrupt advances stay `corroborated=None`.

> **Snippet flaw** — `slot_written_this_turn` ordering: as written above it would also count the rule's own `set:` writes already in `events` (fixed in implementation: computed before rule.set — see design doc deviations, d30edb7).

**Step 4:** PASS + suite. Note: `runtime._emit_exit_say` composes with a same-turn note — verify `test_exit_say_composes_with_same_turn_director_note` still passes; if the uncorroborated steer collides with exit_say composition, the exit_say line wins and the uncorroborated text rides its `(Also: …)` suffix — that is acceptable, update the assertion accordingly.

**Step 5: Commit** — `feat(playbook): classify advances as corroborated/uncorroborated; steer prose-only advances under v2`

### Task 9: corroborated advance wins over forced resume

**Files:** `src/superdialog/playbook/runtime.py` (`on_user_text` resume branch); test in `tests/playbook/test_continuity_advances.py`.

**Step 1: Failing test** — caller inside the `pricing_faq` detour answers the detour AND gives their budget… simpler with this fixture: at the detour, verdict returns `{"slots": {"location": "Pune"}, "advance": "main.pitch"}`? `location` is not declared on `pricing_faq`, so the write is rejected. Use the requires-backed path instead: while in the detour (resume_stack=[main.ask_location]), the verdict proposes `advance: "main.pitch"` — but `pricing_faq` has no advance rules, so no rule matches (that path logs unknown target). Therefore test via a fixture tweak: give `pricing_faq` one rule `- {when: "caller names location", judge: llm, to: main.pitch, requires: [location]}` and declare `location: {type: str}` in its slots (edit `CONTINUITY_YAML` — additive, no other test relies on pricing_faq having no rules; re-run Phase A tests after). Then:

```python
async def test_corroborated_advance_beats_forced_resume() -> None:
    rt = _rt([
        {"slots": {}, "interrupt": "price_guardrail"},
        {"slots": {"location": "Pune"}, "advance": "main.pitch"},
    ])
    await rt.start()
    await rt.on_user_text("price?")
    assert rt.state.checkpoint_id == "main.pricing_faq"
    await rt.on_user_text("it's for Pune by the way")
    # v2: the evidence-backed advance is honored, NOT the forced return
    assert rt.state.checkpoint_id == "main.pitch"
```

Also add the inverse: under `legacy_continuity: true` the forced resume still wins (checkpoint back to `main.ask_location`).

**Step 2:** FAIL. **Step 3: Implement** — in `on_user_text`, inside the resume branch:

```python
        if (
            state.entered_via_resume
            and state.resume_stack
            and not is_interrupt
            and not decision.detour_continues
        ):
            corroborated_advance = (
                advance is not None
                and not self._pb.legacy_continuity
                and getattr(advance, "corroborated", None) is True
            )
            if not corroborated_advance:
                self._apply([...])                # existing body unchanged
                await self._advance(...)
                ...
                return pass_through
            # else fall through: the evidence-backed advance below wins.
            # The stack entry it strands is reaped by the fold expiry (Task 10).
```

**Step 4:** PASS + suite. **Step 5: Commit** — `feat(playbook): honor corroborated advances over forced detour resume under v2`

### Task 10: resume-stack expiry in the fold

**Files:** `src/superdialog/playbook/state.py` (`ConversationState` + fold); test in `tests/playbook/test_state.py` (append).

**Step 1: Failing test** — build an `EventLog` by hand: a resume-push interrupt advance, then 7 further AdvanceEvents (any rule), fold, assert `resume_stack == []`. And a control: 5 advances → entry survives.

**Step 2:** FAIL. **Step 3: Implement.**

```python
#: A resume-stack entry older than this many subsequent advances is stale:
#: "resuming" to it would teleport the caller minutes backward. Reaped by the
#: fold (pure — no event needed; the non-resume is visible in traversals).
_RESUME_STACK_MAX_AGE = 6
```

`ConversationState` gains `resume_stack_seq: list[int] = Field(default_factory=list)` (advance ordinal at push; parallel to `resume_stack`). In the fold's `AdvanceEvent` branch, maintain an `adv_seq` local counter (increment per AdvanceEvent), append the seq on push, pop it on resume, and after each advance:

```python
                while s.resume_stack and adv_seq - s.resume_stack_seq[0] > _RESUME_STACK_MAX_AGE:
                    s.resume_stack = s.resume_stack[1:]
                    s.resume_stack_seq = s.resume_stack_seq[1:]
```

(Deliberate deviation from the design doc: expiry is fold-pure and silent — the fold cannot append a DegradedEvent. The dropped resume is visible in traversals as the absent return.)

**Step 4:** PASS + suite. **Step 5: Commit** — `feat(playbook): expire stale resume-stack entries after 6 advances`

---

## Phase C — Supervisor default-on

### Task 11: supervisor resolves from the v2 flag

**Files:** `src/superdialog/playbook/models.py` (Guidelines), `src/superdialog/playbook/agent.py` (~line 141); tests in `tests/playbook/test_agent.py`.

**Step 1: Failing test** — construct `PlaybookAgent` on `MINIMAL_YAML` (v2): assert `agent._supervisor is not None`; on `MINIMAL_YAML + legacy_continuity: true`: assert `None`; with explicit `guidelines: {supervisor: false}` under v2: assert `None` (explicit wins).

**Step 2:** FAIL. **Step 3:**

- `models.py`: `supervisor: bool | None = None` (docstring: None = resolved from `legacy_continuity`; explicit value wins either way).
- `agent.py`:

```python
        _sup_flag = playbook.guidelines.supervisor
        _sup_on = _sup_flag if _sup_flag is not None else not playbook.legacy_continuity
        sup_llm = supervisor_llm or (director_llm if _sup_on else None)
```

**Step 4:** Run FULL suite. This flips the default for every existing `PlaybookAgent` test on a v2 playbook — tests whose scripted director LLM would now also serve supervisor verdicts may need `guidelines: {supervisor: false}` pinned in their fixture YAML (do that; do NOT weaken the new default). **Step 5: Commit** — `feat(playbook): supervisor default-on under v2, explicit guidelines.supervisor wins`

### Task 12: new supervisor triggers

**Files:** `src/superdialog/playbook/supervisor.py` (`detect_triggers`); test in `tests/playbook/test_supervisor.py`.

**Step 1: Failing tests** — build logs by hand (see existing `detect_triggers` tests):
- two consecutive director `AdvanceEvent`s with `corroborated=False` (rules `llm:x`) → `"uncorroborated_streak"` in triggers; one → not.
- two `DegradedEvent(component="director", detail="junk_rejected:city")` → `"junk_rejected:city"` in triggers.

**Step 2:** FAIL. **Step 3:** append to `detect_triggers`:

```python
    directed = [
        e for e in log.events
        if isinstance(e, AdvanceEvent)
        and e.rule.startswith(("llm:", "expr:"))
    ]
    if len(directed) >= 2 and all(
        e.corroborated is False for e in directed[-2:]
    ):
        triggers.append("uncorroborated_streak")
    junk_counts: dict[str, int] = {}
    for e in log.events:
        if isinstance(e, DegradedEvent) and e.detail.startswith("junk_rejected:"):
            key = e.detail.split(":", 1)[1]
            junk_counts[key] = junk_counts.get(key, 0) + 1
    for key, n in junk_counts.items():
        if n >= 2:
            triggers.append(f"junk_rejected:{key}")
```

**Step 4:** PASS + suite. **Step 5: Commit** — `feat(playbook): supervisor triggers for uncorroborated streaks and repeated junk rejections`

---

## Phase D — Observability

### Task 13: traversal records dwell turns

**Files:** `src/superdialog/playbook/traversal.py`; tests in `tests/playbook/test_traversal.py`.

**Step 1: Failing test** — log with: advance into cp A, user turn 1 (no advance), assistant reply, user turn 2 + advance to B. Assert the traversal has a step for turn 1 with `from_checkpoint == to_checkpoint == A`, `advance_rule == "dwell"`, `advance_by is None`, correct `bot_message`/`user_message`, and the advance step for turn 2 unchanged.

**Step 2:** FAIL (turn 1 invisible). **Step 3: Implement** — after building `traversal_steps` from windows, synthesize dwell steps: walk events tracking `cur_cp` (updated at each AdvanceEvent) and the current turn number (incremented at each user utterance); for every user turn number absent from `_turn_for_advance.values()`, emit

```python
        {
            "step": 0,  # renumbered below
            "from_checkpoint": cur_cp, "to_checkpoint": cur_cp,
            "advance_rule": "dwell", "advance_by": None,
            "version": <user utterance version>,
            "goal": <cp goal>, "bot_message": <next assistant utterance text or None>,
            "user_message": <this user utterance text>,
            "slots_written": {}, "tool_calls": [],
            "degraded": False, "turn": <turn>,
            "director_ms": <per-turn lookup>, "talker_ms": <per-turn lookup>,
        }
```

then merge with the advance steps sorted by `version` and renumber `step` 1..N. Existing tests asserting step counts change deliberately — update them (the schema is unchanged; there are only more steps).

**Step 4:** PASS + suite. **Step 5: Commit** — `feat(playbook): traversal records dwell turns (from==to, rule=dwell)`

### Task 14: TTFT metric

**Files:** `src/superdialog/playbook/agent.py` (`_LLMTimer`); test in `tests/playbook/test_agent.py`.

**Step 1: Failing test** — wrap a fake stream that sleeps 30 ms before its first token and 0 ms after; assert `timer.stats["ttft_p95_ms"] >= 25` and `ttft` < total stream ms.

**Step 2:** FAIL. **Step 3:** `_LLMTimer` gains `self.ttft_ms: list[float] = []`; in `stream()`, record `(perf_counter()-t0)*1000` on the first yielded chunk; `stats` adds, when `self.ttft_ms` is non-empty:

```python
            "ttft_mean_ms": round(sum(t) / len(t), 1),
            "ttft_p95_ms": round(sorted(t)[int(len(t) * 0.95)], 1),
```

**Step 4:** PASS + suite. **Step 5: Commit** — `feat(playbook): record talker time-to-first-token in latency stats`

### Task 15: traversal source prefers playbook.source_path

**Files:** `src/superdialog/playbook/agent.py` (~line 172); test in `tests/playbook/test_agent.py`.

**Step 1: Failing test** — `PlaybookAgent(pb_with_source_path, ..., traversal_source="unpod-prod-general-agent-v3")` → `agent._traversal_source == "<basename of source_path>"`; with `source_path=None` the explicit arg still wins.

**Step 2:** FAIL. **Step 3:**

```python
        self._traversal_source = (
            (playbook.source_path and Path(playbook.source_path).name)
            or traversal_source
            or ""
        )
```

**Step 4:** PASS + suite. **Step 5: Commit** — `fix(playbook): traversal source names the actual playbook file, not the host's static label`

---

## Phase E — Scenario regressions (production sequences)

### Task 16: end-to-end continuity scenarios

**Files:** Create `tests/playbook/test_continuity_scenarios.py`.

Two integration tests through `PlaybookRuntime` + `SeqLLM` on `CONTINUITY_YAML`, mirroring the production failures step-for-step:

1. **westgate2 self-interrupt sequence** (already unit-covered in Task 1; here assert the full trajectory list): advances are exactly `init → ask_location`, `interrupt → pricing_faq`, `resume → ask_location` — no `pricing_faq → pricing_faq` edge, empty final resume stack, and a later interrupt still fires (E2).
2. **golfai conveyor bounce**: verdicts `[{slots:{location:"Pune"}, advance: main.pitch}, {slots:{}, advance: main.ask_budget}, {slots:{}, advance: main.ask_budget}]` — assert the second/third advances carry `corroborated is False`, the steer is live at `ask_budget`, and `detect_triggers` returns `uncorroborated_streak`.

Steps: write both tests → run (they should PASS if Tasks 1-12 are correct; any failure here is an integration gap — fix before proceeding) → `uv run pytest tests/playbook -q` full green → commit `test(playbook): production-derived continuity scenario regressions`.

### Task 17: update the design doc + final sweep

- Amend `docs/plans/2026-08-07-playbook-continuity-v2-design.md`: resume-stack expiry is fold-pure/silent (no DegradedEvent), and the junk signal is a `DegradedEvent(detail="junk_rejected:<key>")` rather than a new event type.
- Run: `uv run pytest tests/playbook -q` (expect ~630+ green) and `uv run pytest tests -q` for the whole repo.
- Run: `uv run ruff check src/superdialog/playbook tests/playbook` and fix.
- Commit — `docs(plans): reconcile continuity v2 design with implementation`

---

## Phase F — Host truthfulness (separate repo: `super`, branch `feat/playbook-continuity-host`)

Not executable from this repo — one branch in `/Users/parvbhullar/Drives/Vault/Projects/Unpod/super`, after Phases A-E ship and the superdialog pin bumps. Anchors:

| Fix | Where | What |
|---|---|---|
| E7 dropped turns | `super/core/voice/livekit/lite_v2/agent.py:796-821` | When `_pb_turn_lock` is held, queue the utterance; on release, append it to `pb_agent.runtime.log` as a user `UtteranceEvent` (in order) before the next turn instead of skipping the reply entirely. |
| E5 barge-in | `lite_v2` interruption path (grep `resume_false_interruption`, `conversation_item_added`) | Call `pb_agent.mark_interrupted(heard_prefix)` with the truncated transcript LiveKit reports. |
| E8 language | `lite_v2` handler + `superdialog/adapters/livekit.py:108` | Thread the bridge-detected language into `agent.turn(text, language=...)`; the adapter accepts an optional language extractor. |
| Tests | super repo `scripts/run_tests.sh` | Extend the `playbook-adapter` module with tests for all three. |

Each is its own TDD task in a plan written in that repo (the discovery steps — exact hook points — belong there).

---

## Execution notes

- Task order is dependency order: 1→4 (Phase A) are independent of the flag; 5 must precede 6-12; 8 must precede 9 and 12; 13-15 are independent of each other.
- After every task: full `uv run pytest tests/playbook -q`. Baseline is 601 passed / 0 failed — never leave it red between commits.
- When an existing test contradicts a deliberate semantic change (Task 1 completed-suppression, Task 11 supervisor default, Task 13 step counts), update the test and say why in the commit body — never silently delete assertions.
