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
