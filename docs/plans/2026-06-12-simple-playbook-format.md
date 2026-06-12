# Simple Playbook Authoring Format Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a NEW first-class authoring format — the "simple playbook" — that authors write by hand (prose steps, persona, facts, objections, boundaries) and that a loader compiles into the existing `superdialog.playbook.Playbook`. The simple format is to playbooks what `compile_flow` is to legacy flows: a friendlier surface that lowers to the validated runtime artifact. No runtime changes — the compiled `Playbook` drives the same Talker/Director engine unchanged.

**Architecture:** A `SimplePlaybook` pydantic model mirrors the author-facing YAML (`name`, `goal`, `persona`, `opening`, `closing`, `playbook` step list, `facts`, `objections`, `boundaries`, `fallback_actions`). `simple_to_playbook(doc: dict) -> Playbook` performs the PROVEN mapping (encoded below as the loader spec): one rich `persona` string folds identity + voice + goal + facts + objections + boundaries + fallbacks + closing; each step becomes a `Checkpoint` in a single journey `"main"`; `collect` lists become `str` slots; `done_when` becomes a single `judge: llm` advance rule to the next step; the last step is `terminal`. Detection is structural: a doc is "simple" when its top-level `playbook` key is a LIST of step dicts (a real `Playbook` carries `journeys:`; a flow carries `nodes`/`initial_node`). The produced `Playbook` is validated through the existing `Playbook` model, so reference validation guards dangling refs for free.

**Tech Stack:** Python ≥3.10, pydantic v2, PyYAML (declared), pytest (`asyncio_mode = "auto"`), uv, ruff, pyrefly. All deps already in `pyproject.toml` — no new dependencies.

**Reference design (READ before starting):**
- The author format — read `/Users/parvbhullar/Downloads/woodspring.yaml` IN FULL for the exact keys and nesting (top-level `name`/`goal`/`persona{name,language,voice_style,identity}`/`opening`/`closing`/`playbook[]`/`facts{...,canonical_pricing}`/`objections[{trigger,handle}]`/`boundaries[]`/`fallback_actions{name:desc}`; each step has `id`, `purpose`, `say`, optional `collect`, `done_when`).
- The target schema — `src/superdialog/playbook/models.py` (`Playbook`, `Journey`, `Checkpoint`, `SlotSpec`, `AdvanceRule`; `Playbook.from_yaml/from_json/load`; the `_check_references` validator).
- The compile-to-Playbook precedent — `src/superdialog/playbook/compiler.py` (`compile_flow`); the existing committed example `examples/playbooks/woodspring.yaml` is the *already-converted Playbook form* this loader reproduces from the authored simple form.

**Conventions for every task:**
- Run `uv run pytest <test file> -v` after each test/impl step; run `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check` before each commit; fix what they flag. Use `uv` only — never `pip`.
- `pyproject.toml` sets `asyncio_mode = "auto"`; this feature is synchronous (a loader), so tests are plain `def`. No async needed.
- New module: `src/superdialog/playbook/simple.py`. It imports only from `superdialog.playbook.models` (the `Playbook` artifact) and `yaml` — it must NOT import `runtime`, `director`, `talker`, `agent`, or the old `superdialog.machine`.
- Type hints on every function; public functions get docstrings; 88-char lines; f-strings; snake_case.
- Commit after every green test, conventional style: `feat(playbook): …`. Work on branch `feat/playbook-engine` (continue the existing engine branch).

**Branch setup (first step, once):**
```bash
git checkout feat/playbook-engine 2>/dev/null || git checkout -b feat/playbook-engine
```

---

## Task 1: SimplePlaybook schema + `simple_to_playbook` loader

**Files:**
- Create: `src/superdialog/playbook/simple.py`
- Create: `tests/playbook/test_simple.py`

The loader is the contract. Encode the PROVEN mapping exactly:

**Persona string (ONE rich block).** `simple_to_playbook` builds the Talker's
standing `persona` by joining these blocks with blank lines, in order, skipping
any that are empty:
1. `persona.identity` (verbatim).
2. `Voice & manner: {persona.voice_style}` (omit the line if `voice_style` is empty).
3. `Overall goal: {goal}` (omit if `goal` empty).
4. `## Reference facts (never invent beyond these)` followed by
   `yaml.safe_dump(facts, sort_keys=False, allow_unicode=True)` — the canonical
   pricing and all nested reference data live here (omit the block if `facts`
   is empty/absent).
5. `## Objection handling` followed by one bullet per objection:
   `- If {trigger} -> {handle}` (omit block if no objections).
6. `## Hard boundaries` followed by `- {boundary}` bullets (omit if none).
7. `## Fallback actions` followed by `- {name}: {desc}` bullets
   (omit if `fallback_actions` empty).
8. `## Closing line` followed by `closing` (omit if `closing` empty).

RATIONALE (document this in the module docstring): the Talker sees the persona
every turn but `Playbook` has no dedicated fields for facts/objections/
boundaries, and the `env` lane is NEVER rendered to the Talker. So this
reference material MUST live in `persona`, not `env` — putting facts in `env`
would hide them from speech.

**Steps -> checkpoints (single journey `"main"`).** For each step in order:
- `id = step.id`
- `goal = step.purpose`
- `guidance = step.say` (prose; kept as-is — Jinja-safe since no `{{ }}`).
- `slots = {c: SlotSpec(type="str", description="") for c in (step.collect or [])}`
  (v1: all slots are `str` — type inference is deferred).
- `advance_when`: for every NON-last step, one rule:
  `AdvanceRule(when=step.done_when, judge="llm", to="main.<next step id>",
  requires=step.collect or [])`. OMIT `requires` (leave it `[]`) when the step
  has no `collect`. Use `done_when` verbatim as `when`; if `done_when` is empty,
  default `when` to `"step complete"`.
- The LAST step: `terminal=True`, `outcome="closed"`, and NO `advance_when`
  (terminal checkpoints end the session).
- All gates stay `soft` (the default). No tools, pipelines, interrupts, or
  policies are emitted in v1.

**Opening fallback.** `opening` and `name` are informational. Use `opening`
ONLY to seed the FIRST checkpoint's `guidance` when that step has no `say`.
For woodspring the first step `greeting` already has a `say`, so `opening` is
redundant there — document that `opening` is a fallback for the initial
checkpoint only.

**Validation.** Return `Playbook.model_validate({...})` so the existing
`_check_references` validator runs (it guards dangling `advance_when` targets,
duplicate ids, etc.). Because every non-last step routes to `main.<next id>`
and the last step is terminal, all targets resolve by construction.

**Detection.** Add `is_simple_playbook(doc: dict) -> bool`: True when `doc` is a
mapping whose top-level `playbook` value is a non-empty LIST (a real `Playbook`
has `journeys:` as a dict and no top-level `playbook` list; a flow has
`nodes`/`initial_node`). Be tolerant: non-dict input -> False.

**Convenience loaders.** Add module-level `load_simple(path: str) -> Playbook`
(reads YAML/JSON by extension, mirrors `Playbook.load`, then
`simple_to_playbook`). Also add a `Playbook.from_simple` is NOT added to the
model (keep `simple.py` free of model edits); `load_simple` is the public entry.

**Step 1: Write failing test**

`tests/playbook/test_simple.py`:
```python
import textwrap

import pytest

from superdialog.playbook.models import Playbook
from superdialog.playbook.simple import (
    SimplePlaybook,
    SimpleStep,
    is_simple_playbook,
    simple_to_playbook,
)

SIMPLE = textwrap.dedent("""
    name: "Tiny Booking Bot"
    goal: "Book a haircut and confirm it."
    persona:
      name: Mira
      language: English
      voice_style: "Warm and brief. One question at a time."
      identity: "You are Mira, a booking assistant for Glow Studio."
    opening: "Greet the caller warmly."
    closing: "Thank them and say goodbye."
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
      services: [haircut, massage, facial]
      canonical_pricing:
        haircut: "₹400"
        massage: "₹900"
    objections:
      - trigger: "Caller says it's too expensive."
        handle: "Acknowledge and mention the value; offer the cheapest option."
    boundaries:
      - "NEVER invent prices."
    fallback_actions:
      callback: "Offer to call back at a convenient time."
""")


def test_detection_simple_vs_playbook() -> None:
    import yaml
    assert is_simple_playbook(yaml.safe_load(SIMPLE)) is True
    # a real Playbook has 'journeys', not a 'playbook' list
    assert is_simple_playbook({"journeys": {"j": {"checkpoints": []}}}) is False
    # a flow has nodes/initial_node
    assert is_simple_playbook({"nodes": [], "initial_node": "a"}) is False
    assert is_simple_playbook("not a dict") is False
    assert is_simple_playbook({"playbook": []}) is False  # empty list


def test_compiles_to_valid_playbook_with_expected_checkpoints() -> None:
    import yaml
    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert isinstance(pb, Playbook)
    ids = pb.checkpoint_ids()
    assert ids == {"main.greet", "main.collect", "main.confirm"}
    assert pb.initial_checkpoint_id == "main.greet"


def test_steps_chain_to_next_and_last_is_terminal() -> None:
    import yaml
    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    greet = pb.checkpoint("main.greet")
    assert [r.to for r in greet.advance_when] == ["main.collect"]
    assert greet.advance_when[0].judge == "llm"
    assert greet.advance_when[0].when == "Caller is ready to book."
    collect = pb.checkpoint("main.collect")
    assert collect.advance_when[0].to == "main.confirm"
    confirm = pb.checkpoint("main.confirm")
    assert confirm.terminal is True
    assert confirm.outcome == "closed"
    assert confirm.advance_when == []


def test_collect_maps_to_str_slots_and_requires() -> None:
    import yaml
    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    collect = pb.checkpoint("main.collect")
    assert set(collect.slots) == {"name", "service"}
    assert all(s.type == "str" for s in collect.slots.values())
    # collect -> requires on the step's own advance rule
    assert collect.advance_when[0].requires == ["name", "service"]
    # a step with no collect has empty requires
    assert pb.checkpoint("main.greet").advance_when[0].requires == []


def test_guidance_is_the_say_prose() -> None:
    import yaml
    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert pb.checkpoint("main.greet").guidance == (
        "Greet the caller and ask how you can help."
    )


def test_persona_folds_facts_objections_boundaries_fallbacks_closing() -> None:
    import yaml
    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    persona = pb.persona
    assert "Mira" in persona                       # identity
    assert "Voice & manner: Warm and brief" in persona
    assert "Overall goal: Book a haircut" in persona
    assert "## Reference facts" in persona
    assert "canonical_pricing" in persona and "₹400" in persona  # facts dumped
    assert "## Objection handling" in persona
    assert "If Caller says it's too expensive. ->" in persona
    assert "## Hard boundaries" in persona and "NEVER invent prices." in persona
    assert "## Fallback actions" in persona and "callback:" in persona
    assert "## Closing line" in persona and "Thank them and say goodbye." in persona


def test_facts_not_in_env() -> None:
    import yaml
    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    assert pb.env == {}  # facts live in persona, never env


def test_opening_seeds_first_guidance_only_when_say_missing() -> None:
    import yaml
    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][0].pop("say")  # greet now has no say
    pb = simple_to_playbook(doc)
    assert pb.checkpoint("main.greet").guidance == "Greet the caller warmly."
    # and when say is present, opening is ignored (already covered above)


def test_empty_done_when_defaults_to_step_complete() -> None:
    import yaml
    doc = yaml.safe_load(SIMPLE)
    doc["playbook"][0].pop("done_when")
    pb = simple_to_playbook(doc)
    assert pb.checkpoint("main.greet").advance_when[0].when == "step complete"


def test_simpleplaybook_model_round_trips_keys() -> None:
    import yaml
    sp = SimplePlaybook.model_validate(yaml.safe_load(SIMPLE))
    assert sp.name == "Tiny Booking Bot"
    assert sp.persona.identity.startswith("You are Mira")
    assert [s.id for s in sp.playbook] == ["greet", "collect", "confirm"]
    assert isinstance(sp.playbook[1], SimpleStep)
    assert sp.playbook[1].collect == ["name", "service"]


def test_compiled_playbook_round_trips_through_from_yaml() -> None:
    import yaml
    pb = simple_to_playbook(yaml.safe_load(SIMPLE))
    # dump the compiled Playbook to YAML and reload via the real model
    dumped = yaml.safe_dump(pb.model_dump(mode="json"), sort_keys=False)
    reloaded = Playbook.from_yaml(dumped)
    assert reloaded.checkpoint_ids() == pb.checkpoint_ids()
    assert reloaded.persona == pb.persona


def test_single_step_playbook_is_terminal_with_no_rules() -> None:
    pb = simple_to_playbook({
        "persona": {"identity": "Solo."},
        "playbook": [{"id": "only", "purpose": "p", "say": "Say hi.",
                      "done_when": "done"}],
    })
    only = pb.checkpoint("main.only")
    assert only.terminal is True and only.advance_when == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/playbook/test_simple.py -v`
Expected: FAIL — `ModuleNotFoundError: superdialog.playbook.simple`

**Step 3: Implement**

`src/superdialog/playbook/simple.py` — structure to follow (write it fully;
keep each helper small and pure):

```python
"""Simple authoring format -> Playbook compiler.

The simple format is a human-friendly surface for authoring playbooks: prose
steps, a nested persona, and reference data (facts/objections/boundaries/
fallbacks). `simple_to_playbook` lowers it to the validated `Playbook` runtime
artifact, the same way `compile_flow` lowers legacy flows.

Facts, objections, boundaries, fallbacks, and the closing line are folded into
ONE rich `persona` string. The Talker sees `persona` every turn but the `env`
lane is never rendered to it, so this reference material must live in persona —
NOT env — to stay visible during speech.
"""

from __future__ import annotations

import json
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .models import AdvanceRule, Checkpoint, Journey, Playbook, SlotSpec


class SimplePersona(BaseModel):
    name: str = ""
    language: str = ""
    voice_style: str = ""
    identity: str = ""


class SimpleStep(BaseModel):
    id: str
    purpose: str = ""
    say: str = ""
    collect: list[str] = Field(default_factory=list)
    done_when: str = ""


class SimpleObjection(BaseModel):
    trigger: str
    handle: str


class SimplePlaybook(BaseModel):
    name: str = ""
    goal: str = ""
    persona: SimplePersona = Field(default_factory=SimplePersona)
    opening: str = ""
    closing: str = ""
    playbook: list[SimpleStep] = Field(min_length=1)
    facts: dict[str, Any] = Field(default_factory=dict)
    objections: list[SimpleObjection] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    fallback_actions: dict[str, str] = Field(default_factory=dict)


def is_simple_playbook(doc: Any) -> bool:
    """True when ``doc`` is a simple playbook: top-level ``playbook`` is a list."""
    return (
        isinstance(doc, dict)
        and isinstance(doc.get("playbook"), list)
        and len(doc["playbook"]) > 0
    )


def _build_persona(sp: SimplePlaybook) -> str:
    parts: list[str] = []
    if sp.persona.identity.strip():
        parts.append(sp.persona.identity.strip())
    if sp.persona.voice_style.strip():
        parts.append(f"Voice & manner: {sp.persona.voice_style.strip()}")
    if sp.goal.strip():
        parts.append(f"Overall goal: {sp.goal.strip()}")
    if sp.facts:
        dumped = yaml.safe_dump(sp.facts, sort_keys=False, allow_unicode=True)
        parts.append(
            "## Reference facts (never invent beyond these)\n" + dumped.strip()
        )
    if sp.objections:
        bullets = "\n".join(
            f"- If {o.trigger} -> {o.handle.strip()}" for o in sp.objections
        )
        parts.append("## Objection handling\n" + bullets)
    if sp.boundaries:
        bullets = "\n".join(f"- {b}" for b in sp.boundaries)
        parts.append("## Hard boundaries\n" + bullets)
    if sp.fallback_actions:
        bullets = "\n".join(f"- {k}: {v}" for k, v in sp.fallback_actions.items())
        parts.append("## Fallback actions\n" + bullets)
    if sp.closing.strip():
        parts.append("## Closing line\n" + sp.closing.strip())
    return "\n\n".join(parts)


def _step_to_checkpoint(
    step: SimpleStep, next_id: str | None, opening: str
) -> Checkpoint:
    guidance = step.say.strip() or (opening.strip() if not step.say else "")
    slots = {c: SlotSpec(type="str", description="") for c in step.collect}
    if next_id is None:
        return Checkpoint(
            id=step.id, goal=step.purpose, guidance=guidance, slots=slots,
            terminal=True, outcome="closed",
        )
    rule = AdvanceRule(
        when=step.done_when.strip() or "step complete",
        judge="llm", to=next_id, requires=list(step.collect),
    )
    return Checkpoint(
        id=step.id, goal=step.purpose, guidance=guidance, slots=slots,
        advance_when=[rule],
    )


def simple_to_playbook(doc: dict[str, Any]) -> Playbook:
    """Compile a simple-format dict into a validated ``Playbook``."""
    sp = SimplePlaybook.model_validate(doc)
    checkpoints: list[Checkpoint] = []
    for i, step in enumerate(sp.playbook):
        is_last = i == len(sp.playbook) - 1
        next_id = None if is_last else f"main.{sp.playbook[i + 1].id}"
        # opening only seeds the FIRST step when it has no say
        opening = sp.opening if i == 0 else ""
        checkpoints.append(_step_to_checkpoint(step, next_id, opening))
    return Playbook(
        persona=_build_persona(sp),
        journeys={"main": Journey(checkpoints=checkpoints)},
    )


def load_simple(path: str) -> Playbook:
    """Load a simple-format file (YAML or JSON) and compile it to a Playbook."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    doc = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    return simple_to_playbook(doc)
```

NOTE on the `_step_to_checkpoint` guidance line: `step.say.strip()` is truthy
when a `say` exists, so the opening fallback only applies when `say` is empty.
Verify against `test_opening_seeds_first_guidance_only_when_say_missing`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/playbook/test_simple.py -v` — all PASS

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add src/superdialog/playbook/simple.py tests/playbook/test_simple.py
git commit -m "feat(playbook): simple authoring format schema and simple_to_playbook loader"
```

---

## Task 2: Golden fixture + woodspring example round-trip

**Files:**
- Create: `tests/fixtures/playbooks/simple_booking.yaml` (a trimmed 3–4 step
  simple-format booking playbook with `facts`/`objections`/`boundaries`)
- Create: `examples/playbooks/woodspring.simple.yaml` (COPY the authored simple
  form — see step 1)
- Test: append to `tests/playbook/test_simple.py`

The `Downloads/woodspring.yaml` source is OUTSIDE the repo, so we cannot load it
from tests. Instead: (a) a self-contained golden fixture committed under
`tests/fixtures/playbooks/`, and (b) the authored woodspring simple form
committed under `examples/playbooks/woodspring.simple.yaml` so the existing
converted `examples/playbooks/woodspring.yaml` has its authored source alongside
it and a test proves the loader reproduces a valid Playbook from it.

**Step 1: Create the fixture and the example**

Create `tests/fixtures/playbooks/simple_booking.yaml` — a small but complete
simple playbook (4 steps, real `facts.canonical_pricing`, 2 objections, 2
boundaries, 1 fallback). Keep it under ~50 lines:

```yaml
name: "Glow Studio Booking"
goal: "Book a salon appointment and confirm it."
persona:
  name: Mira
  language: English
  voice_style: "Warm, brief, one question at a time."
  identity: "You are Mira, the booking assistant for Glow Studio."
opening: "Greet the caller warmly and ask how you can help."
closing: "Thank the caller and wish them a great day."
playbook:
  - id: greeting
    purpose: "Open the call."
    say: "Greet the caller as Mira from Glow Studio and ask how you can help."
    done_when: "Caller indicates they want to book."
  - id: collect_details
    purpose: "Capture name and service."
    say: "Ask for the caller's name and which service they'd like."
    collect: [name, service]
    done_when: "Name and service are captured."
  - id: present_price
    purpose: "Share the price for the chosen service."
    say: "Share the canonical price for the chosen service and ask if it works."
    collect: [budget_ok]
    done_when: "Caller has responded on the price."
  - id: confirm_booking
    purpose: "Confirm and close."
    say: "Read back the booking, confirm it, and close warmly."
    done_when: "Caller has confirmed the booking."
facts:
  services: [haircut, massage, facial]
  canonical_pricing:
    haircut: "₹400"
    massage: "₹900"
    facial: "₹1200"
  mandatory_disclaimer: "Prices exclude taxes; the desk confirms the final bill."
objections:
  - trigger: "Caller says the price is too high."
    handle: "Acknowledge, mention the value, and offer the most affordable service."
  - trigger: "Caller asks for a discount."
    handle: "Pricing is fixed; the desk can explain current offers."
boundaries:
  - "NEVER invent prices — use only facts.canonical_pricing."
  - "NEVER promise same-day availability without checking."
fallback_actions:
  callback: "Offer to call back at a convenient time with availability."
```

Then COPY the authored simple woodspring source into the repo as the example:
```bash
cp /Users/parvbhullar/Downloads/woodspring.yaml \
   examples/playbooks/woodspring.simple.yaml
```
Add a one-line comment header to the copied file (first line) noting it is the
AUTHORED simple form that compiles to `examples/playbooks/woodspring.yaml`:
```
# Authored simple-playbook form. Compiles via superdialog.playbook.load_simple
# to the Playbook in examples/playbooks/woodspring.yaml.
```
(Edit the file to prepend those two comment lines; everything else is the
source verbatim.)

**Step 2: Write failing test (append to `tests/playbook/test_simple.py`)**

```python
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "playbooks"
EXAMPLES = (
    Path(__file__).resolve().parents[2] / "examples" / "playbooks"
)


def test_golden_fixture_compiles_and_validates() -> None:
    from superdialog.playbook.simple import load_simple
    pb = load_simple(str(FIXTURES / "simple_booking.yaml"))
    ids = pb.checkpoint_ids()
    assert ids == {
        "main.greeting", "main.collect_details",
        "main.present_price", "main.confirm_booking",
    }
    # collect -> slots + requires
    cd = pb.checkpoint("main.collect_details")
    assert set(cd.slots) == {"name", "service"}
    assert cd.advance_when[0].requires == ["name", "service"]
    assert cd.advance_when[0].to == "main.present_price"
    # last step terminal
    assert pb.checkpoint("main.confirm_booking").terminal is True
    # facts/objections/boundaries surfaced in persona
    assert "## Reference facts" in pb.persona
    assert "canonical_pricing" in pb.persona and "₹400" in pb.persona
    assert "## Objection handling" in pb.persona
    assert "If Caller says the price is too high. ->" in pb.persona
    assert "## Hard boundaries" in pb.persona
    assert "NEVER invent prices" in pb.persona
    # env stays empty
    assert pb.env == {}


def test_golden_fixture_round_trips_through_from_yaml() -> None:
    import yaml
    from superdialog.playbook.models import Playbook
    from superdialog.playbook.simple import load_simple
    pb = load_simple(str(FIXTURES / "simple_booking.yaml"))
    dumped = yaml.safe_dump(pb.model_dump(mode="json"), sort_keys=False)
    reloaded = Playbook.from_yaml(dumped)
    assert reloaded.checkpoint_ids() == pb.checkpoint_ids()


def test_woodspring_simple_example_compiles() -> None:
    from superdialog.playbook.models import Playbook
    from superdialog.playbook.simple import load_simple
    pb = load_simple(str(EXAMPLES / "woodspring.simple.yaml"))
    assert isinstance(pb, Playbook)
    # woodspring's first step `greeting` has a `say`, so opening is redundant
    assert pb.initial_checkpoint_id == "main.greeting"
    assert pb.checkpoint("main.greeting").guidance  # came from `say`, not opening
    # the last authored step is terminal
    last_id = sorted(pb.checkpoint_ids())  # any membership check is fine
    assert "main.deliver_closing" in pb.checkpoint_ids()
    assert pb.checkpoint("main.deliver_closing").terminal is True
    # facts incl. canonical_pricing folded into persona
    assert "canonical_pricing" in pb.persona
    assert "Hard boundaries" in pb.persona
```

**Step 3: Run to verify failure, then pass**

Run: `uv run pytest tests/playbook/test_simple.py -v`
The new tests fail until the fixture + example file exist (they were created in
Step 1) — confirm they now PASS. If the woodspring last-step id differs, read
`examples/playbooks/woodspring.simple.yaml` and assert against the actual final
step id (it should be `deliver_closing` per the source).

**Step 4: Format, typecheck**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
```

**Step 5: Commit**

```bash
git add tests/fixtures/playbooks/simple_booking.yaml \
        examples/playbooks/woodspring.simple.yaml tests/playbook/test_simple.py
git commit -m "test(playbook): golden simple-playbook fixture and woodspring example round-trip"
```

---

## Task 3: CLI auto-detection + `--simple`

**Files:**
- Modify: `src/superdialog/cli/main.py`
- Test: append to `tests/cli/test_chat.py`

Mirror the existing `_looks_like_playbook` (which checks for `journeys`). Add
detection for the simple format and wire it into `_cmd_chat`. The simple
playbook compiles to a `Playbook`, then drives the SAME `_run_playbook_repl`.

**Detection precedence** (document it in `_cmd_chat`):
1. top-level `journeys` (dict) -> real Playbook -> `_run_playbook_repl`
2. top-level `playbook` (list) -> simple playbook -> compile -> `_run_playbook_repl`
3. `nodes` / `initial_node` -> flow -> `DialogMachine` REPL

`journeys` wins over `playbook` if both appear (a hand-authored Playbook never
has a top-level `playbook` list, so the ordering is just defensive).

**Step 1: Write failing tests (append to `tests/cli/test_chat.py`)**

```python
_SIMPLE_PLAYBOOK = """\
name: "Tiny"
persona:
  identity: "You are a tiny demo agent."
playbook:
  - id: only
    purpose: "Say hi and stop."
    say: "Hi there!"
    done_when: "greeted"
"""


def _write_simple(tmp_path: Path) -> Path:
    path = tmp_path / "simple.yaml"
    path.write_text(_SIMPLE_PLAYBOOK)
    return path


def test_chat_detects_simple_playbook(tmp_path: Path) -> None:
    """A --flow path whose content has a top-level 'playbook' list compiles + runs."""
    path = _write_simple(tmp_path)
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--flow", str(path)])
    assert rc == 0
    mock_flow.assert_not_called()
    mock_play.assert_not_called()
    mock_simple.assert_called_once()
    assert mock_simple.call_args[0][0] == str(path)


def test_chat_explicit_simple_flag(tmp_path: Path) -> None:
    """--simple PATH compiles a simple playbook and runs its REPL."""
    path = _write_simple(tmp_path)
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--simple", str(path)])
    assert rc == 0
    mock_flow.assert_not_called()
    mock_simple.assert_called_once()
    assert mock_simple.call_args[0][0] == str(path)


def test_chat_journeys_wins_over_playbook_list(tmp_path: Path) -> None:
    """A doc with 'journeys' is a real Playbook even if it also had a playbook key."""
    path = tmp_path / "p.yaml"
    path.write_text(_MINIMAL_PLAYBOOK)  # has 'journeys', no 'playbook' list
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_playbook_repl") as mock_play,
    ):
        rc = main(["chat", "--flow", str(path)])
    assert rc == 0
    mock_simple.assert_not_called()
    mock_play.assert_called_once()


def test_chat_missing_simple_file(capsys: pytest.CaptureFixture) -> None:
    rc = main(["chat", "--simple", "/nope-simple.yaml"])
    assert rc == 1
    assert "/nope-simple.yaml" in capsys.readouterr().err


def test_chat_malformed_simple_exits_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A file with a 'playbook' list but invalid schema exits 1, no traceback."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("playbook:\n  - not_a_step_mapping\n")
    with patch.object(_cli_main_module, "_run_simple_repl") as mock_simple:
        rc = main(["chat", "--flow", str(bad)])
    assert rc == 1
    mock_simple.assert_not_called()
    assert "Invalid simple playbook" in capsys.readouterr().err


def test_chat_still_runs_flow_after_simple_detection(tmp_path: Path) -> None:
    """A flow JSON (nodes/initial_node) still routes to the DialogMachine REPL."""
    flow_data = {
        "id": "t", "system_prompt": "s", "initial_node": "n",
        "nodes": [{"id": "n", "name": "N", "edges": [], "is_final": True}],
    }
    flow_file = tmp_path / "flow.json"
    flow_file.write_text(json.dumps(flow_data))
    with (
        patch.object(_cli_main_module, "_run_simple_repl") as mock_simple,
        patch.object(_cli_main_module, "_run_chat_repl") as mock_flow,
    ):
        rc = main(["chat", "--flow", str(flow_file)])
    assert rc == 0
    mock_simple.assert_not_called()
    mock_flow.assert_called_once()
```

**Step 2: Run to verify failure** — FAIL (`--simple` arg / helpers missing).

**Step 3: Implement** in `src/superdialog/cli/main.py`:

1. Add `_looks_like_simple_playbook(path: str) -> bool` next to
   `_looks_like_playbook`, reusing the same tolerant parse, returning
   `is_simple_playbook(doc)` from `superdialog.playbook.simple`:
   ```python
   def _looks_like_simple_playbook(path: str) -> bool:
       """True when ``path`` parses to a mapping with a top-level 'playbook' list."""
       from ..playbook.simple import is_simple_playbook
       try:
           doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
       except (OSError, yaml.YAMLError):
           return False
       return is_simple_playbook(doc)
   ```

2. Add `_run_simple_repl(simple_path: str, llm: str) -> None` mirroring
   `_run_playbook_repl`, but building the Playbook via the simple loader:
   ```python
   def _run_simple_repl(simple_path: str, llm: str) -> None:
       """Blocking REPL driving a Playbook compiled from a simple-format file."""
       from ..llm.resolver import resolve_llm
       from ..playbook import PlaybookAgent, httpx_http, provider_adapters
       from ..playbook.simple import load_simple

       provider = resolve_llm(llm)
       director, talker = provider_adapters(provider)
       agent = PlaybookAgent(
           playbook=load_simple(simple_path),
           talker_llm=talker, director_llm=director, http=httpx_http,
       )
       # ... identical loop body to _run_playbook_repl ...
   ```
   To avoid duplicating the loop, refactor the loop body of
   `_run_playbook_repl` into a private `_drive_agent(agent) -> None` and have
   both `_run_playbook_repl` and `_run_simple_repl` build the agent then call
   `_drive_agent`. (Keep functions focused; DRY.)

3. Add `_chat_simple(path: str, llm: str) -> int` mirroring `_chat_playbook`:
   pre-flight `load_simple(path)` inside try/except, printing
   `f"Invalid simple playbook {path}: {exc}"` on failure and returning 1, else
   `_run_simple_repl(path, llm)` and return 0.

4. In `_cmd_chat`:
   - After the existing `--playbook` handling, add `--simple` handling:
     ```python
     simple_path = getattr(args, "simple", None)
     if simple_path:
         if not Path(simple_path).exists():
             print(f"No simple playbook found at: {simple_path}", file=sys.stderr)
             return 1
         return _chat_simple(simple_path, llm)
     ```
   - In the auto-detect block on `flow_path`, the precedence becomes:
     ```python
     if _looks_like_playbook(flow_path):       # journeys -> Playbook
         return _chat_playbook(flow_path, llm)
     if _looks_like_simple_playbook(flow_path):  # playbook list -> simple
         return _chat_simple(flow_path, llm)
     # else fall through to Flow.load (nodes/initial_node)
     ```

5. In `_build_parser`, add the `--simple` argument to the `chat` subparser and
   extend the `--flow` help text to mention simple-playbook auto-detection:
   ```python
   chat.add_argument(
       "--simple", default=None,
       help="Path to a simple-format playbook (YAML/JSON); compiles then runs",
   )
   ```

**Step 4: Run to verify pass** — `uv run pytest tests/cli/test_chat.py -v` all PASS

**Step 5: Format, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add src/superdialog/cli/main.py tests/cli/test_chat.py
git commit -m "feat(playbook): CLI auto-detects simple playbooks and adds --simple flag"
```

---

## Task 4: Exports + docs

**Files:**
- Modify: `src/superdialog/playbook/__init__.py`
- Modify: `docs/04-playbook-guide.md`
- Modify: `README.md`

**Step 1: Write failing export test (append to `tests/playbook/test_simple.py`)**

```python
def test_public_exports() -> None:
    from superdialog.playbook import (
        is_simple_playbook,
        load_simple,
        simple_to_playbook,
    )
    assert callable(is_simple_playbook)
    assert callable(load_simple)
    assert callable(simple_to_playbook)
```

**Step 2: Run to verify failure** — ImportError.

**Step 3: Implement**

In `src/superdialog/playbook/__init__.py`, add the import and `__all__` entries
(keep `__all__` sorted, as it currently is):
```python
from .simple import is_simple_playbook, load_simple, simple_to_playbook
```
Add `"is_simple_playbook"`, `"load_simple"`, `"simple_to_playbook"` to
`__all__` in sorted position.

**Step 4: Docs**

(a) `docs/04-playbook-guide.md` — add a new section
`## 8. Simple authoring format` BEFORE the current `## 7. Roadmap` (renumber
Roadmap to `## 9`, or keep Roadmap last and number the new section to fit;
prefer inserting as `## 8` immediately before Roadmap and bumping Roadmap to
`## 9`). The section must include:
- A short intro: the simple format is the easiest way to author a playbook;
  `load_simple(path)` (or `superdialog chat --simple PATH`) compiles it to a
  `Playbook`, the same artifact `compile_flow` produces from flows.
- A compact woodspring-style YAML example (8–12 lines: `name`, `goal`,
  `persona{identity,voice_style}`, a 2–3 step `playbook`, `facts`,
  one `objection`, one `boundary`).
- The mapping table:

  | Simple key | Compiles to |
  | --- | --- |
  | `persona.identity` + `voice_style` + `goal` + `facts` + `objections` + `boundaries` + `fallback_actions` + `closing` | one rich `Playbook.persona` string |
  | each `playbook` step | a `Checkpoint` in journey `main` |
  | `step.id` | checkpoint id |
  | `step.purpose` | `checkpoint.goal` |
  | `step.say` | `checkpoint.guidance` |
  | `step.collect` | `str` slots + the step rule's `requires` |
  | `step.done_when` | the step's single `judge: llm` advance rule `when`, `to` the next step |
  | last step | `terminal: true`, `outcome: closed` |
  | `opening` | seeds the first step's guidance only if it has no `say` |

- A one-paragraph "What's NOT in v1" pointer to the Explicitly deferred list
  below (linear sequences only; objections live in persona prose, not as
  interrupt checkpoints; all slots are `str`).

(b) `README.md` — update the auto-detect note (around line 136–142) so it reads
that `chat` runs flows, playbooks, AND simple playbooks, and show the third
detection. Add a line to the CLI block:
```bash
superdialog chat --simple woodspring.yaml    # simple format -> compiled playbook
```
and amend the prose "auto-detecting a playbook by its top-level `journeys` key"
to also mention "or a simple playbook by its top-level `playbook` list".

**Step 5: Verify, format, commit**

```bash
uv run pytest tests/playbook/test_simple.py -v   # exports test passes
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add src/superdialog/playbook/__init__.py docs/04-playbook-guide.md README.md \
        tests/playbook/test_simple.py
git commit -m "feat(playbook): export simple loader and document the simple authoring format"
```

---

## Task 5: Full-suite green gate

**Files:** none (verification only).

**Step 1:** Run the whole repo suite — the simple format is additive and must
not regress the engine, compiler, or CLI:
```bash
uv run pytest -q
```
Expected: all green (the existing playbook/CLI/flow/machine suites untouched).

**Step 2:** Final lint/type sweep:
```bash
uv run ruff format . && uv run ruff check . && uv run pyrefly check
```

**Step 3:** Push and open a draft PR titled
"feat: simple playbook authoring format (compiles to Playbook)":
```bash
git push -u origin feat/playbook-engine
gh pr create --draft \
  --title "feat: simple playbook authoring format (compiles to Playbook)" \
  --body "$(cat <<'EOF'
Adds a human-friendly "simple playbook" authoring format that compiles to the
existing Playbook runtime artifact — the same way compile_flow lowers legacy
flows. Authors write prose steps + a nested persona + reference data
(facts/objections/boundaries/fallbacks); the loader folds reference data into
one rich persona string and lowers each step to a Checkpoint in a single
journey. CLI auto-detects a simple playbook by its top-level `playbook` list
(precedence: journeys -> Playbook, playbook-list -> simple, nodes -> flow) and
adds --simple PATH.

Loader spec, golden fixture, woodspring example round-trip, CLI detection
tests, and the docs mapping table are all included. v1 maps linear step
sequences only; branching, objection interrupts, structured fact views,
tools/pipelines, and slot type inference are deferred (see plan).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(No commit on this task; it is the gate. If anything is red, fix in the
relevant task's module and amend that task's commit before pushing.)

---

## Explicitly deferred (do NOT build in this plan)

- **Branching** — a step routing to a non-adjacent step. v1 maps strictly
  LINEAR sequences: each `done_when` advances to the immediately following step.
  Authors who need branches drop to the raw `Playbook` format (or compile a
  flow). Modeling `done_when`/route pairs per step is a v2 schema extension.
- **Objection steps as interrupts.** Objections live in `persona` PROSE in v1;
  the Director/Talker handle them conversationally turn-to-turn. They are NOT
  compiled into `InterruptSpec`s or dedicated objection checkpoints.
- **Facts as structured `views`.** `facts` is YAML-dumped into the persona text
  rather than extracted into `Playbook.views` computed expressions. Structured
  fact views (so guidance can `{{ views.price }}`) are a v2 concern.
- **Tools / pipelines / handlers / policies in the simple format.** The simple
  format emits no process layer. Side-effecting playbooks author in raw
  `Playbook` form. Silence policy, webhook/timer handlers, and middleware are
  out of scope here.
- **Type inference for `collect` slots.** Every collected slot is `str` in v1.
  Inferring `int`/`date`/`enum` from names or hints (and threading `values:`
  enums) is deferred.
- **`Playbook.from_simple` classmethod on the model.** Kept out to avoid
  coupling `models.py` to the simple format; `load_simple` /
  `simple_to_playbook` are the public entry points. Add the classmethod later
  only if a model-method ergonomic is wanted.
- **Round-tripping a Playbook BACK to simple form** (a `playbook_to_simple`
  decompiler) — one-directional authoring is enough for v1.
- **Optimizer integration** — the simple format participates in
  `superdialog optimize` only via its compiled `Playbook`; no simple-form-aware
  optimization (see `docs/plans/2026-06-12-playbook-optimize-command.md`).
