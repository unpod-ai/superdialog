from superdialog.eval.endpoints.in_process import InProcessVanilla
from superdialog.eval.runner import drive_journey
from superdialog.playbook.eval.models import PersonaSpec
from tests.eval.fakes import FakeProvider, FakeSpeaksUser


async def test_drive_journey_produces_transcript_with_latencies():
    agent = InProcessVanilla(playbook_text="bot", llm=FakeProvider({"*": "ok"}))
    user = FakeSpeaksUser(["I want a haircut", "Saturday", "bye"])
    persona = PersonaSpec(name="Sam", traits="eager", goal="book", max_turns=3)
    t = await drive_journey(agent, persona, user)
    assert t.records[0].role == "assistant"  # greeting first
    assert t.turn_count() <= 3  # capped by max_turns
    assert all(ms >= 0 for ms in t.assistant_latencies_ms())


async def test_journey_ends_when_persona_hangs_up():
    agent = InProcessVanilla(playbook_text="bot", llm=FakeProvider({"*": "ok"}))
    user = FakeSpeaksUser(
        ["I want a haircut", "Saturday, thanks — bye! <END_CALL>", "never sent"]
    )
    persona = PersonaSpec(name="Sam", traits="eager", goal="book", max_turns=10)
    t = await drive_journey(agent, persona, user)
    # journey stopped at the hang-up, well before max_turns
    assert t.turn_count() == 2
    # the goodbye WAS fed to the agent (so goodbye interrupts can fire) but
    # the sentinel never appears in the transcript
    texts = [r.text for r in t.records if r.role == "user"]
    assert texts[-1] == "Saturday, thanks — bye!"
    assert all("<END_CALL>" not in r.text for r in t.records)


async def test_hangup_with_bare_token_feeds_default_goodbye():
    agent = InProcessVanilla(playbook_text="bot", llm=FakeProvider({"*": "ok"}))
    user = FakeSpeaksUser(["<END_CALL>"])
    persona = PersonaSpec(name="Sam", traits="", goal="book", max_turns=5)
    t = await drive_journey(agent, persona, user)
    texts = [r.text for r in t.records if r.role == "user"]
    assert texts == ["Thanks, that's everything — goodbye!"]
