# Playbook v3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Implement the surviving v3 changes from
`2026-08-08-playbook-v3-first-principles-design.md`: close the `never_say`
canned-line hole, dedupe the rewind bounds check, replace the wrap
fraction with a constant-free predicate, add substring-anchored slot
writes (shadow-first), make barge-in truncation append-only, and delete
`legacy_continuity`.

**Architecture:** All changes live in `src/superdialog/playbook/`. Each
task is independently committable and ordered safest-first. The anchor
check ships in `shadow` mode by default (log, don't enforce) mirroring
the CRITIC_MODE rollout pattern.

**Tech Stack:** Python 3.12, pydantic v2, pytest (`uv run pytest`),
pre-commit (black/isort/flake8). Never `ruff format` repo-wide — scope to
touched files.

**Regression floor:** `uv run pytest tests/playbook -q` green after every
task. The docs/09 failure-mode map must keep its guards.

**Dissolved at implementation depth (do NOT implement):** rewind/discard
merger (nothing shared), `@6 advances` counter deletion (load-bearing
backstop — multi-checkpoint detours strand entries by design and the
counter reaps them), `note` → reason codes (the note has a defined
consumer: the Talker steer, clamped at director.py:992-999), 12-turn
window trim (deferred — lexical reference detection is G4-class
second-guessing). The design doc records these corrections.

---

### Task 1: `never_say` excision for hold + recovery lines

The token stream and the barrier filler are excised (talker.py:417-428,
481-494); the hold line (talker.py:431-437) and both recovery-line sites
(talker.py:452-458 and the stream-failure recovery near the end of
`speak`) are not. Route every canned line through one helper.

**Files:**
- Modify: `src/superdialog/playbook/talker.py`
- Test: `tests/playbook/test_talker.py`

**Step 1: Write the failing tests**

Model fixtures on `test_hard_gate_hold_line_when_director_never_comes`
(test_talker.py:357) — same construction, but set
`never_say: ["bear with me"]` on the gated checkpoint (and a phrase from
`RECOVERY_LINE` for the strict test):

```python
async def test_hold_line_respects_never_say() -> None:
    # gated checkpoint with never_say containing a phrase of the hold line
    ...
    spoken = "".join(c.text for c in chunks)
    assert "bear with me" not in spoken.casefold()


async def test_recovery_line_respects_never_say() -> None:
    # strict checkpoint, no say_verbatim, never_say=["say that again"]
    ...
    assert "say that again" not in spoken.casefold()
```

**Step 2: Run to verify both fail**

Run: `uv run pytest tests/playbook/test_talker.py -q -k never_say`
Expected: 2 FAIL (phrases present in output).

**Step 3: Implement**

Add beside `_excise` in talker.py:

```python
def _excise_line(text: str, folded: list[str]) -> str:
    """Excise never_say phrases from a canned line.

    Returns '' when excision leaves only punctuation/space — speaking
    "…" alone is worse than silence. Callers fall back to their built-in
    default line (itself excised) before accepting silence.
    """
    if not folded:
        return text
    cleaned = _excise(text, folded)
    return cleaned if re.sub(r"[\W_]+", "", cleaned, flags=re.UNICODE) else ""
```

Refactor the filler site to use it (behavior identical). At the hold
site, excise the resolved hold line; when it collapses, fall back to
`_excise_line(HOLD_LINE, folded)`; when that also collapses, yield the
empty final chunk (existing pattern at talker.py:534). Same treatment
for both recovery sites with `RECOVERY_LINE`. `say_verbatim` stays
unexcised — authored script, author intent wins.

**Step 4: Run** `uv run pytest tests/playbook/test_talker.py -q` — all pass.

**Step 5: Commit** `fix(talker): route hold and recovery lines through never_say excision`

---

### Task 2: rewind bounds check — single source of truth

supervisor.py:310 duplicates the range check runtime.rewind raises on
(runtime.py:309-312). Keep runtime's; the supervisor catches instead of
pre-checking. The `None` check stays in the supervisor (schema concern,
not a range concern).

**Files:**
- Modify: `src/superdialog/playbook/supervisor.py:308-316`
- Test: `tests/playbook/test_supervisor.py` (existing `rewind_bad_version`
  coverage keeps passing — that IS the test; extend it with a
  `to_version=None` case if not present)

**Step 1: Confirm existing coverage**

Run: `uv run pytest tests/playbook/test_supervisor.py -q -k rewind`
Expected: PASS (baseline).

**Step 2: Implement**

```python
if decision.action == "rewind":
    v = decision.to_version
    if v is None:
        runtime.log.append(
            DegradedEvent(
                component="supervisor", detail="rewind_bad_version:None"
            )
        )
        return []
    pending_confirm = (runtime.state.steering_note or "").startswith(
        COMPENSATE_MARKER
    )
    try:
        outcome = await runtime.rewind(
            v,
            decision.reason or "supervisor rewind",
            by="supervisor",
            confirmed=decision.confirmed and pending_confirm,
            repair_note=note or None,
        )
    except ValueError:
        runtime.log.append(
            DegradedEvent(
                component="supervisor", detail=f"rewind_bad_version:{v}"
            )
        )
        return []
```

**Step 3: Run** `uv run pytest tests/playbook/test_supervisor.py tests/playbook/test_rewind.py -q` — all pass.

**Step 4: Commit** `refactor(supervisor): delegate rewind bounds to runtime.rewind`

---

### Task 3: wrap guard — ⅔ fraction → "one wrap completes capture"

Replace `_capture_nearly_complete` (director.py:161-181). New predicate:
wrap iff (a) something required is already captured (the caller has
invested — an early goodbye still closes immediately), and (b) the
current checkpoint's missing required slots are the ONLY missing required
slots playbook-wide (one wrap question finishes the capture; otherwise a
wrap can't finish the job, so honor the close).

**Files:**
- Modify: `src/superdialog/playbook/director.py:161-181` and call site
  director.py:864-868
- Test: `tests/playbook/test_goodbye_backstop.py` /
  `tests/playbook/test_continuity_interrupts.py` (wherever the existing
  ⅔ tests live — `grep -rn "nearly_complete\|2/3" tests/playbook/`)

**Step 1: Write the failing tests**

```python
def test_wrap_fires_when_one_question_completes_capture() -> None:
    # all required slots filled except the current checkpoint's one
    assert _wrap_would_complete(pb, cp, state) is True


def test_no_wrap_when_other_checkpoints_still_missing_slots() -> None:
    # current cp missing one, a later cp missing two more
    assert _wrap_would_complete(pb, cp, state) is False


def test_no_wrap_when_nothing_captured_yet() -> None:
    # single-required-slot playbook, first ask, caller says goodbye
    assert _wrap_would_complete(pb, cp, state) is False
```

**Step 2: Run to verify fail** (`ImportError: _wrap_would_complete`).

**Step 3: Implement**

```python
def _wrap_would_complete(
    pb: Playbook, cp: Checkpoint, state: ConversationState
) -> bool:
    """One wrap question would finish the playbook's required capture.

    The terminal-interrupt investment signal, constant-free: wrap only
    when (a) the caller has already given something required (early
    goodbyes close immediately — a deflected close is a lost close) and
    (b) the current checkpoint's missing required slots are the ONLY
    ones missing playbook-wide, so the wrap can actually finish the job.
    Default (caller) entity throughout — a multi-entity playbook errs
    toward closing, never toward deflecting.
    """
    required = {
        k
        for j in pb.journeys.values()
        for c in j.checkpoints
        for k, s in c.slots.items()
        if s.required
    }
    if not required:
        return False
    missing_all = {k for k in required if not state.filled([k])}
    missing_here = {
        k for k, s in cp.slots.items() if s.required and not state.filled([k])
    }
    captured_some = len(missing_all) < len(required)
    return bool(missing_here) and captured_some and missing_all <= missing_here
```

Delete `_capture_nearly_complete`; update the call site to
`_wrap_would_complete(self._pb, cp, peek)`. Update any old ⅔ tests to
the new semantics (the ⅔ cases map: near-complete-capture → still wraps
because only current-cp slots remain; early-call → still closes because
nothing captured).

**Step 4: Run** `uv run pytest tests/playbook -q -k "goodbye or wrap or interrupt"` — pass.

**Step 5: Commit** `refactor(director): wrap guard fires iff one wrap completes capture (no 2/3 constant)`

---

### Task 4: substring anchor — shadow mode

The Director points at evidence; the Engine verifies it. Additive
verdict field `"spans": {<key>: "<caller's words>"}` beside `"slots"`.
Engine check per slot write: the span (or, spanless, the value itself)
must appear in the current user utterance; date/time values must
re-derive from the span. Modes: `off | shadow | enforce`, default
**shadow** — mismatches log `anchor_miss`, writes still land.

**Files:**
- Modify: `src/superdialog/playbook/director.py` (prompt, parse,
  validation loop, `__init__`)
- Modify: `src/superdialog/playbook/agent.py` (constructor pass-through)
- Test: `tests/playbook/test_director.py`

**Step 1: Write the failing tests**

Model on existing verdict tests in test_director.py (stub LLM returning
a fixed verdict dict):

```python
async def test_anchor_shadow_logs_fabricated_date_but_writes() -> None:
    # utterance: "any time later in the week is fine"
    # verdict: {"slots": {"appointment_date": "2026-08-14"}, "spans": {}}
    # shadow (default): SlotWriteEvent present AND DegradedEvent
    # detail == "anchor_miss:appointment_date" present


async def test_anchor_enforce_rejects_fabricated_date() -> None:
    # same verdict, Director(anchor="enforce"):
    # no SlotWriteEvent; DegradedEvent anchor_miss present


async def test_anchor_enforce_accepts_spanned_date() -> None:
    # utterance: "next friday works"; spans: {"appointment_date": "next friday"}
    # normalize_date("next friday", now) == verdict value → write lands


async def test_anchor_value_fallback_without_span() -> None:
    # utterance: "make it premium"; slots: {"tier": "premium"}, no spans
    # value appears in utterance → write lands, no anchor_miss


async def test_anchor_exempts_resolve_from_slots() -> None:
    # bare-affirmation candidate resolution keeps working: the span
    # lives in the ASSISTANT's prior turn, so resolve_from slots skip
    # the anchor entirely (G6 already guards them via the live list)
```

**Step 2: Run to verify fail.**

**Step 3: Implement**

`__init__` gains `anchor: Literal["off", "shadow", "enforce"] = "shadow"`
(store as `self._anchor`). `PlaybookAgent.__init__` gains the same
parameter, passed through.

Prompt (inside the SLOT RULE block of `cache_prefix`, stable text):

```text
For every slot you extract, also copy the exact words of THIS user turn
you heard the value in into "spans": {<key>: "<words>"}. Omit a key you
have no words for.
```

Parse: `spans = verdict.get("spans") or {}`.

Validation — in the slot loop, after coercion and the resolve_from
check, before the churn check:

```python
if not self._anchor_ok(spans.get(key), value, coerced, slot_spec, state):
    events.append(
        DegradedEvent(
            component="director",
            detail=f"anchor_miss:{_ekey(cp.entity, key)}",
        )
    )
    if self._anchor == "enforce":
        continue
```

Helper (module-level `_norm`, method `_anchor_ok`):

```python
def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _anchor_ok(self, span, value, coerced, spec, state) -> bool:
    """Evidence check: the write must be anchored in the caller's turn.

    resolve_from slots are exempt (their evidence may be the assistant's
    own prior offer; the live-candidate check is their guard). Spanless
    writes fall back to the value itself appearing in the utterance —
    dates/times can't (normalized form differs), so they effectively
    require a span. str values legitimately differ from their span
    (decline convention: "No" → "none"), so a present-and-anchored span
    is sufficient for them; only date/time re-derive.
    """
    if self._anchor == "off" or spec.resolve_from is not None:
        return True
    utterance = _norm(_last_user_text(state))
    if not utterance:
        return True  # no user turn this round (silence policy etc.)
    if span:
        if _norm(span) not in utterance:
            return False
        if spec.type in ("date", "time"):
            return _coerce_slot(span, spec, state.now) == coerced
        return True
    return _norm(value) in utterance or _norm(coerced) in utterance
```

The F2 language fill and rule `set:` writes never pass through this loop
— deterministic/authored provenance, exempt by construction.

**Step 4: Run** `uv run pytest tests/playbook/test_director.py -q` — pass.
Also run the full suite: existing verdict tests emit no spans → the
value-fallback keeps non-date tests green; any date-slot test that now
logs `anchor_miss` in shadow is EXPECTED (assert-adjust, don't weaken).

**Step 5: Commit** `feat(director): substring-anchored slot writes, shadow-first (anchor_miss audit)`

---

### Task 5: append-only barge-in truncation

`mark_interrupted` (agent.py:240-270) rewrites a frozen event in place —
the one mutation of the log. Replace with `SpeechCorrectionEvent`; the
fold applies it to the transcript. The `[interrupted by caller]` tag
remains the truncation marker the Director already reads.

**Files:**
- Modify: `src/superdialog/playbook/events.py` (new event + union)
- Modify: `src/superdialog/playbook/state.py` (fold branch)
- Modify: `src/superdialog/playbook/agent.py:240-270`
- Test: `tests/playbook/test_events.py`, `tests/playbook/test_state.py`,
  `tests/playbook/test_agent.py` (existing mark_interrupted tests —
  `grep -n mark_interrupted tests/playbook/test_agent.py`)

**Step 1: Write the failing tests**

```python
def test_speech_correction_truncates_transcript() -> None:
    # log: user, assistant("full generated reply"),
    #      SpeechCorrectionEvent(utterance_version=2,
    #                            heard_text="full gen [interrupted by caller]")
    # fold: transcript[1].text == heard_text; log.events[1].text unchanged


def test_mark_interrupted_appends_not_rewrites() -> None:
    # after agent.mark_interrupted("heard prefix"):
    # last event is SpeechCorrectionEvent; the original UtteranceEvent
    # object is unchanged; agent.state.transcript shows the heard prefix
```

**Step 2: Run to verify fail.**

**Step 3: Implement**

events.py:

```python
class SpeechCorrectionEvent(_Base):
    """Correct an assistant utterance to what the caller actually heard.

    Append-only barge-in truncation: the original UtteranceEvent stays
    in the log (what was GENERATED); the fold's transcript shows this
    text (what was DELIVERED). Its presence is the completed=False
    marker for the corrected utterance.
    """

    type: Literal["speech_correction"] = "speech_correction"
    utterance_version: int
    heard_text: str
```

Add to the `Event` union. state.py fold, new branch:

```python
elif isinstance(e, SpeechCorrectionEvent):
    for i in range(len(s.transcript) - 1, -1, -1):
        if s.transcript[i].version == e.utterance_version:
            s.transcript[i] = TranscriptEntry(
                role=s.transcript[i].role,
                text=e.heard_text,
                version=s.transcript[i].version,
            )
            break
```

agent.py `mark_interrupted`: same scan loop, but instead of
`events[i] = event.model_copy(...)` append
`SpeechCorrectionEvent(utterance_version=event.version,
heard_text=f"{base} [interrupted by caller]")` via
`self.runtime.log.append(...)`; keep the cache invalidation line.
Update the docstring (drop "rewrite ... in place").

**Step 4: Run** `uv run pytest tests/playbook -q` — pass (serialization
round-trip via `to_jsonl`/`from_jsonl` covered by test_events patterns).

**Step 5: Commit** `feat(events): SpeechCorrectionEvent — append-only barge-in truncation`

---

### Task 6: delete `legacy_continuity`

v3 semantics become the only semantics. The field stays declared for one
release as accepted-but-inert (parse-time warning) so existing playbook
YAML doesn't explode; every branch dies.

**Files:**
- Modify: `src/superdialog/playbook/models.py:354-357` (deprecation note
  + `model_validator` warning via `logging`)
- Modify: `src/superdialog/playbook/director.py` — branches at 677
  (condition becomes `slot_spec is None and not self._pb.multi_entity`),
  701, 728, 750, 888, 961 (drop the flag from each condition)
- Modify: `src/superdialog/playbook/runtime.py:205` (drop flag term)
- Modify: `src/superdialog/playbook/agent.py:152-155`
  (`_sup_on = _sup_flag if _sup_flag is not None else True`)
- Modify: `src/superdialog/playbook/simple.py:144-145, 503` (drop
  propagation; keep SimpleDoc field inert + warn)
- Test: update `tests/playbook/test_continuity_advances.py`,
  `test_continuity_interrupts.py`, `test_director.py`, `test_models.py`,
  `test_simple.py`, `test_agent.py`, `test_supervisor.py` — delete
  legacy-mode cases, keep v3-mode cases; add one test that a playbook
  setting the flag parses, warns, and gets v3 behavior.

**Step 1: Inventory** `grep -rn legacy_continuity src/ tests/` (expect
~12 source + ~17 test references).

**Step 2: Write the failing test**

```python
def test_legacy_continuity_flag_is_inert_and_warns(caplog) -> None:
    pb = _pb_with(legacy_continuity=True)
    # v3 behavior anyway: junk guard active, churn dampener active
    # caplog contains "legacy_continuity is deprecated and ignored"
```

**Step 3: Implement** source changes above; each deleted branch keeps
its v3 arm only.

**Step 4: Update tests** — legacy-parametrized cases deleted; every
remaining test asserts v3 behavior unconditionally.

**Step 5: Run** `uv run pytest tests/playbook -q` — full pass.

**Step 6: Commit** `feat(playbook)!: legacy_continuity is inert — v3 semantics only`

---

### Task 7: docs sync + full verification

**Files:**
- Modify: `docs/09-execution-algorithm-and-guards.md` — G4/G7/G14/G15
  drop their v2 flags (only mode now); G13 → wrap-completes-capture
  predicate; G21 → counter documented as secondary fallback; G24 →
  covers hold/recovery lines; add G37 (anchor check, class: evidence,
  mode-noted shadow/enforce) and G38 (SpeechCorrection append-only
  truncation, class: speech honesty); drop the `legacy_continuity`
  paragraph.
- Verify: design doc corrections already recorded (dissolved items).

**Steps:** edit doc → `uv run pytest tests/playbook -q` (711+ green) →
commit `docs: guard inventory v3 — anchor (G37), speech correction (G38), legacy flag removed`.

---

## Rollout notes

- Anchor stays `shadow` until production shows a near-zero `anchor_miss`
  rate on legitimate traffic (grep `anchor_miss` in DegradedEvents);
  then hosts flip `anchor="enforce"` per deployment.
- The five behavioral eval suites (goodbye/no-reentry, disconnect, DNC)
  are the wrap-guard regression gate — run them before merging Task 3.
- `pre-commit` runs black/isort/flake8 on commit; do not reformat files
  outside the diff.
