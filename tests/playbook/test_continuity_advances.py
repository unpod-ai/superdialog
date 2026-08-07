"""Task 8: corroboration classification for advances + uncorroborated steer."""

from superdialog.playbook.events import AdvanceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.continuity_fixtures import CONTINUITY_YAML, SeqLLM
from tests.playbook.test_toolexec import FakeHttp


def _rt(payloads, pb=None):
    return PlaybookRuntime(
        pb or Playbook.from_yaml(CONTINUITY_YAML),
        director_llm=SeqLLM(payloads),
        http=FakeHttp([]),
    )


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
