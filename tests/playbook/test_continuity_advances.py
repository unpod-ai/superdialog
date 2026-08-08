"""Continuity-advance semantics: corroboration classification, uncorroborated
steer, and corroborated advances beating a forced detour resume."""

from superdialog.playbook.events import AdvanceEvent
from superdialog.playbook.models import Playbook
from tests.playbook.continuity_fixtures import CONTINUITY_YAML, make_runtime

_rt = make_runtime


async def test_requires_backed_advance_is_corroborated_and_silent() -> None:
    rt = _rt([{"slots": {"location": "Pune"}, "advance": "main.pitch"}])
    await rt.start()
    await rt.on_user_text("I'm in Pune")
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.to_checkpoint == "main.pitch"
    ]
    assert adv and adv[-1].corroborated is True
    assert rt.state.steering_note is None  # silent


async def test_prose_only_advance_is_uncorroborated_and_steered() -> None:
    rt = _rt(
        [
            {"slots": {"location": "Pune"}, "advance": "main.pitch"},
            {"slots": {}, "advance": "main.ask_budget"},  # conveyor, no evidence
        ]
    )
    await rt.start()
    await rt.on_user_text("Pune")
    await rt.on_user_text("hmm what?")
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.to_checkpoint == "main.ask_budget"
    ]
    assert adv and adv[-1].corroborated is False
    assert rt.state.steering_note is not None
    assert "without confirmed input" in rt.state.steering_note


async def test_slot_written_this_turn_corroborates_rule_without_requires() -> None:
    # Strip requires from ask_budget's rule so corroboration can ONLY come
    # from the same-turn declared-slot write (budget is declared on the
    # checkpoint, written by this turn's verdict).
    yaml = CONTINUITY_YAML.replace(
        'to: main.close,\n             requires: [budget]}',
        "to: main.close}",
    )
    assert yaml != CONTINUITY_YAML
    rt = _rt(
        [
            {"slots": {"location": "Pune"}, "advance": "main.pitch"},
            {"slots": {}, "advance": "main.ask_budget"},
            {"slots": {"budget": "50L"}, "advance": "main.close"},
        ],
        pb=Playbook.from_yaml(yaml),
    )
    await rt.start()
    await rt.on_user_text("Pune")
    await rt.on_user_text("ok")
    await rt.on_user_text("budget is 50L")
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.to_checkpoint == "main.close"
    ]
    assert adv and adv[-1].corroborated is True


async def test_rule_set_write_does_not_self_corroborate() -> None:
    # A prose-only rule whose set: targets a slot DECLARED on the current
    # checkpoint must not count its own bookkeeping stamp as evidence —
    # clause (c) is caller-derived verdict extraction only.
    yaml = CONTINUITY_YAML.replace(
        "to: main.ask_budget}",
        "to: main.ask_budget, set: {pitch_ack: done}}",
    ).replace(
        'gate: soft\n        guidance: "Pitch."',
        "gate: soft\n        slots:\n"
        "          pitch_ack: {type: str}\n"
        '        guidance: "Pitch."',
    )
    assert yaml != CONTINUITY_YAML
    rt = _rt(
        [
            {"slots": {"location": "Pune"}, "advance": "main.pitch"},
            {"slots": {}, "advance": "main.ask_budget"},
        ],
        pb=Playbook.from_yaml(yaml),
    )
    await rt.start()
    await rt.on_user_text("Pune")
    await rt.on_user_text("hmm what?")
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.to_checkpoint == "main.ask_budget"
    ]
    assert adv and adv[-1].corroborated is False
    assert rt.state.steering_note is not None
    assert "without confirmed input" in rt.state.steering_note


async def test_expr_advance_is_corroborated() -> None:
    # Insert an expr rule ahead of ask_location's llm rule; once the verdict
    # writes the location slot, the quiescence loop fires the expr rule.
    yaml = CONTINUITY_YAML.replace(
        '- {when: "location given"',
        '- {when: "slots.location is not None", judge: expr, '
        "to: main.pitch, requires: [location]}\n"
        '          - {when: "location given"',
    )
    assert yaml != CONTINUITY_YAML
    rt = _rt(
        [{"slots": {"location": "Pune"}, "advance": None}],
        pb=Playbook.from_yaml(yaml),
    )
    await rt.start()
    await rt.on_user_text("I'm in Pune")
    adv = [
        e for e in rt.log.events if isinstance(e, AdvanceEvent) and e.by == "expr"
    ]
    assert adv and adv[-1].to_checkpoint == "main.pitch"
    assert adv[-1].corroborated is True


async def test_prose_only_advance_not_steered_under_legacy() -> None:
    pb = Playbook.from_yaml(CONTINUITY_YAML).model_copy(
        update={"legacy_continuity": True}
    )
    rt = _rt(
        [
            {"slots": {"location": "Pune"}, "advance": "main.pitch"},
            {"slots": {}, "advance": "main.ask_budget"},
        ],
        pb=pb,
    )
    await rt.start()
    await rt.on_user_text("Pune")
    await rt.on_user_text("hmm what?")
    assert rt.state.steering_note is None
    # corroborated is still STAMPED under legacy (harmless metadata);
    # only the steer is gated.
    adv = [
        e
        for e in rt.log.events
        if isinstance(e, AdvanceEvent) and e.to_checkpoint == "main.ask_budget"
    ]
    assert adv and adv[-1].corroborated is False


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


async def test_corroborated_advance_defers_to_resume_under_legacy() -> None:
    pb = Playbook.from_yaml(CONTINUITY_YAML).model_copy(
        update={"legacy_continuity": True}
    )
    rt = _rt([
        {"slots": {}, "interrupt": "price_guardrail"},
        {"slots": {"location": "Pune"}, "advance": "main.pitch"},
    ], pb=pb)
    await rt.start()
    await rt.on_user_text("price?")
    await rt.on_user_text("it's for Pune by the way")
    # legacy: forced return still wins
    assert rt.state.checkpoint_id == "main.ask_location"


async def test_uncorroborated_advance_defers_to_resume_under_v2() -> None:
    """The resume-override is EVIDENCE-gated: a prose-only advance proposed
    inside a detour must still be dropped in favor of the forced return.

    pricing_faq's rule is stripped of its requires so the director genuinely
    EMITS an AdvanceEvent (corroborated=False). With requires intact and no
    location written, _requires_met fails and no advance is emitted at all —
    the resume branch would then run because advance is None, leaving the
    corroborated gate unpinned (an always-fall-through mutant survives).
    """
    yaml = CONTINUITY_YAML.replace(
        '- {when: "caller names location", judge: llm, to: main.pitch,\n'
        "             requires: [location]}",
        '- {when: "caller names location", judge: llm, to: main.pitch}',
    )
    assert yaml != CONTINUITY_YAML
    rt = _rt(
        [
            {"slots": {}, "interrupt": "price_guardrail"},
            {"slots": {}, "advance": "main.pitch"},  # prose-only, no evidence
        ],
        pb=Playbook.from_yaml(yaml),
    )
    await rt.start()
    await rt.on_user_text("price?")
    assert rt.state.checkpoint_id == "main.pricing_faq"
    await rt.on_user_text("hmm okay")
    assert rt.state.checkpoint_id == "main.ask_location"  # forced resume wins


# -- F1/F2: volunteered-fact recall (extraction sub-algorithm) ---------------------


async def test_volunteered_slot_from_other_checkpoint_is_written_under_v2() -> None:
    """F1: rajesh gave his name in a 2-turn DNC call; the current checkpoint
    didn't declare the slot, extraction dropped it. A slot declared on ANY
    checkpoint is authored vocabulary — accept it under the same discipline."""
    rt = _rt([{"slots": {"budget": "50L"}, "advance": None, "note": None}])
    await rt.start()  # at ask_location; `budget` is declared on ask_budget
    await rt.on_user_text("my budget is 50L by the way")
    assert rt.state.slot_value("budget") == "50L"


async def test_volunteered_slot_rejected_under_legacy_and_when_undeclared() -> None:
    from superdialog.playbook.models import Playbook
    from tests.playbook.continuity_fixtures import CONTINUITY_YAML

    pb = Playbook.from_yaml(CONTINUITY_YAML).model_copy(
        update={"legacy_continuity": True}
    )
    rt = _rt([{"slots": {"budget": "50L"}, "advance": None, "note": None}], pb=pb)
    await rt.start()
    await rt.on_user_text("my budget is 50L")
    assert rt.state.slot_value("budget") is None  # legacy: strict cp scoping
    # undeclared-anywhere keys stay rejected under v2
    rt2 = _rt([{"slots": {"favourite_color": "red"}, "advance": None}])
    await rt2.start()
    await rt2.on_user_text("i like red")
    assert rt2.state.slot_value("favourite_color") is None


async def test_volunteered_junk_still_rejected_and_multi_entity_stays_strict() -> None:
    from superdialog.playbook.models import Playbook
    from tests.playbook.continuity_fixtures import CONTINUITY_YAML

    rt = _rt([{"slots": {"budget": "None"}, "advance": None}])
    await rt.start()
    await rt.on_user_text("hmm")
    assert rt.state.slot_value("budget") is None  # junk guard still applies
    pb = Playbook.from_yaml(CONTINUITY_YAML).model_copy(update={"multi_entity": True})
    rt2 = _rt([{"slots": {"budget": "50L"}, "advance": None}], pb=pb)
    await rt2.start()
    await rt2.on_user_text("my budget is 50L")
    assert rt2.state.slot_value("budget") is None  # entity ambiguity: strict


async def test_language_slot_filled_from_bridge_signal_under_v2() -> None:
    """F2: callers SPEAK Hindi but rarely SAY 'Hindi'; the SLOT RULE forbids
    inference, so language slots structurally miss. The engine fills them
    deterministically from the bridge-detected sticky language."""
    from superdialog.playbook.models import Playbook
    from tests.playbook.continuity_fixtures import CONTINUITY_YAML

    yaml = CONTINUITY_YAML.replace(
        "          location: {type: str}",
        "          location: {type: str}\n"
        "          preferred_language: {type: str}",
        1,  # ask_location's declaration only (pricing_faq shares the line)
    )
    assert yaml != CONTINUITY_YAML
    rt = _rt([{"slots": {}, "advance": None}], pb=Playbook.from_yaml(yaml))
    await rt.start()
    await rt.on_user_text("haan bataiye", language="hi")
    assert rt.state.slot_value("preferred_language") == "Hindi"
    # legacy: inert
    pb = Playbook.from_yaml(yaml).model_copy(update={"legacy_continuity": True})
    rt2 = _rt([{"slots": {}, "advance": None}], pb=pb)
    await rt2.start()
    await rt2.on_user_text("haan bataiye", language="hi")
    assert rt2.state.slot_value("preferred_language") is None
