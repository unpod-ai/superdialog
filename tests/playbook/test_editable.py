"""Tests for the EditableDoc abstraction (FullDoc / SimpleDoc)."""

import pytest
import yaml as _yaml

from superdialog.playbook.editable import (
    Edit,
    FullDoc,
    MutationError,
    SimpleDoc,
    make_editable,
)
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_simple import SIMPLE

_GUIDANCE = "journeys.booking.checkpoints.collect.guidance"


def test_fields_enumerates_exactly_the_whitelist() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    addrs = {f.address for f in doc.fields()}
    assert "persona" in addrs
    assert _GUIDANCE in addrs
    assert "journeys.booking.checkpoints.collect.goal" in addrs
    assert "journeys.booking.checkpoints.collect.slots.city.description" in addrs
    # the collect rule is llm-judged -> editable
    assert "journeys.booking.checkpoints.collect.advance_when[0].when" in addrs
    # confirm's rules are expr-judged -> frozen
    assert "journeys.booking.checkpoints.confirm.advance_when[0].when" not in addrs
    # say_verbatim editable only where present
    assert "journeys.booking.checkpoints.confirm.say_verbatim" in addrs
    assert "journeys.booking.checkpoints.collect.say_verbatim" not in addrs
    # structure is unreachable
    assert "journeys.booking.checkpoints.confirm.gate" not in addrs


def test_apply_returns_new_doc_and_compiles() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    edited = doc.apply([Edit(address=_GUIDANCE, new_text="Collect warmly.")])
    assert edited.compile().checkpoint("booking.collect").guidance == "Collect warmly."
    # the original is untouched (apply is functional)
    assert doc.compile().checkpoint("booking.collect").guidance == "Collect naturally."


def test_emit_diff_touches_only_the_edited_line() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    edited = doc.apply([Edit(address=_GUIDANCE, new_text="Collect warmly.")])
    before = doc.emit().splitlines()
    after = edited.emit().splitlines()
    assert len(before) == len(after)
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == 1
    assert "Collect warmly." in changed[0][1]


def test_apply_rejects_non_whitelisted_addresses() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    for bad in (
        "journeys.booking.checkpoints.confirm.gate",  # structure
        "journeys.booking.checkpoints.confirm.advance_when[0].when",  # expr
        "journeys.booking.checkpoints.collect.say_verbatim",  # absent -> no add
        "journeys.booking.checkpoints.nope.guidance",  # unknown checkpoint
        "tools",  # structure
    ):
        with pytest.raises(MutationError):
            doc.apply([Edit(address=bad, new_text="x")])


def test_never_say_entries_may_be_added_but_not_removed() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    addr = "journeys.booking.checkpoints.collect.never_say"
    grown = doc.apply([Edit(address=addr, new_text=["never promise refunds"])])
    cp = grown.compile().checkpoint("booking.collect")
    assert cp.never_say == ["never promise refunds"]
    with pytest.raises(MutationError):
        grown.apply([Edit(address=addr, new_text=[])])  # shrinking is removal


def test_string_field_requires_string_payload() -> None:
    doc = FullDoc.from_text(MINIMAL_YAML)
    with pytest.raises(MutationError):
        doc.apply([Edit(address=_GUIDANCE, new_text=["not", "a", "string"])])


def test_pipeline_on_keys_survive_the_round_trip() -> None:
    # MINIMAL_YAML's pipeline uses an `on:` key; YAML 1.1 would load it as a
    # boolean. FullDoc must parse with the models loader, not yaml.safe_load.
    doc = FullDoc.from_text(MINIMAL_YAML)
    reparsed = FullDoc.from_text(doc.emit())
    assert reparsed.compile().pipeline("confirm_and_hold").steps[0].on


def test_simple_fields_enumerate_step_and_persona_prose() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    addrs = {f.address for f in doc.fields()}
    assert {
        "opening",
        "closing",
        "persona.identity",
        "persona.voice_style",
        "steps.collect.say",
        "steps.collect.done_when",
        "steps.collect.purpose",
    } <= addrs
    # reference data is frozen
    assert not any(
        a.startswith(("facts", "objections", "boundaries", "fallback_actions"))
        for a in addrs
    )


def test_simple_apply_recompiles_and_emits_simple_format() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    edited = doc.apply(
        [Edit(address="steps.collect.say", new_text="Warmly ask their name.")]
    )
    cp = edited.compile().checkpoint("main.collect")
    assert cp.guidance == "Warmly ask their name."
    out = _yaml.safe_load(edited.emit())
    assert "playbook" in out and "journeys" not in out  # still simple format


def test_simple_facts_survive_prose_edits() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    edited = doc.apply(
        [Edit(address="persona.voice_style", new_text="Bubbly and quick.")]
    )
    persona = edited.compile().persona
    assert "₹400" in persona  # canonical pricing intact
    assert "NEVER invent prices" in persona  # boundaries intact


def test_simple_apply_rejects_frozen_addresses() -> None:
    doc = SimpleDoc.from_text(SIMPLE)
    for bad in (
        "facts.canonical_pricing.haircut",
        "boundaries",
        "steps.collect.collect",
        "steps.nope.say",
    ):
        with pytest.raises(MutationError):
            doc.apply([Edit(address=bad, new_text="x")])


def test_make_editable_routes_by_format() -> None:
    assert isinstance(make_editable(SIMPLE), SimpleDoc)
    assert isinstance(make_editable(MINIMAL_YAML), FullDoc)
