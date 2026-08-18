from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.eval.vanilla import VanillaSessionMetrics, run_vanilla_session


class _FakeTurnResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeVanillaAgent:
    """Stands in for LLMAgent: scripted replies, ends after N turns."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    async def turn(self, text: str) -> _FakeTurnResult:
        self.calls.append(text)
        reply = self._replies.pop(0) if self._replies else "Goodbye!"
        return _FakeTurnResult(reply)


class _FakeUserLLM:
    """Scripted persona replies; ends the call on the last one."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        return self._lines.pop(0) if self._lines else ""


async def test_run_vanilla_session_drives_turns_and_records_transcript():
    agent = _FakeVanillaAgent(["Sure, what date?", "Booked for Friday!"])
    persona = PersonaSpec(
        name="p1", traits="brisk", goal="book a slot", opening="I want to book",
        max_turns=5,
    )
    user_llm = _FakeUserLLM(["Friday please", ""])  # second empty line ends the call
    metrics = await run_vanilla_session(agent, persona, user_llm)
    assert isinstance(metrics, VanillaSessionMetrics)
    assert metrics.persona == "p1"
    assert metrics.turns == 2
    assert [t.text for t in metrics.transcript if t.role == "assistant"] == [
        "Sure, what date?",
        "Booked for Friday!",
    ]
    assert [t.text for t in metrics.transcript if t.role == "user"] == [
        "I want to book",
        "Friday please",
    ]


async def test_run_vanilla_session_stops_at_max_turns():
    agent = _FakeVanillaAgent(["r1", "r2", "r3", "r4", "r5"])
    persona = PersonaSpec(
        name="p2", traits="t", goal="g", opening="start", max_turns=2,
    )
    user_llm = _FakeUserLLM(["still going", "still going", "still going"])
    metrics = await run_vanilla_session(agent, persona, user_llm)
    assert metrics.turns == 2
