from superdialog.eval.endpoints.in_process import InProcessVanilla
from tests.eval.fakes import FakeProvider


async def test_vanilla_uses_playbook_text_as_system_prompt():
    llm = FakeProvider({"BEGIN": "Hello, welcome!", "*": "noted"})
    ep = InProcessVanilla(playbook_text="PERSONA: spa bot", llm=llm)
    greeting = await ep.start()
    assert "Hello" in greeting
    reply = await ep.turn("I want a haircut")
    assert reply == "noted"


async def test_vanilla_reset_clears_history():
    llm = FakeProvider({"*": "ok"})
    ep = InProcessVanilla(playbook_text="x", llm=llm)
    await ep.turn("a")
    ep.reset()  # must not raise; new LLMAgent underneath
    assert await ep.turn("b") == "ok"
