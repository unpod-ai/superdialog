# tests/playbook/eval/test_runner.py
"""Session runner tests (migrated from test_eval_bridge.py)."""

from typing import Any

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.eval.runner import run_eval, run_session
from superdialog.playbook.events import DegradedEvent, EventLog, SteeringNoteEvent
from superdialog.playbook.models import Playbook
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE_VERDICT: dict = {"slots": {}, "advance": None, "note": None}
_ADVANCING_VERDICT: dict = {
    "slots": {"city": "Pune", "date": "2026-06-12"},
    "advance": "booking.confirm",
    "note": None,
}
_HOLD_OK = (200, {"data": {"hold_id": "h1"}})


class ScriptedUser:
    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(messages)
        if not self.lines:
            return ""
        return self.lines.pop(0) if len(self.lines) > 1 else self.lines[0]


def _agent(
    verdict: dict | None = None,
    http_responses: list[tuple[int, dict]] | None = None,
) -> PlaybookAgent:
    return PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=StreamLLM(["Which", " city?"]),
        director_llm=CannedLLM(verdict or _IDLE_VERDICT),
        http=FakeHttp(http_responses or []),
    )


async def test_session_completes_and_measures() -> None:
    agent = _agent(verdict=_ADVANCING_VERDICT, http_responses=[_HOLD_OK])
    persona = PersonaSpec(
        name="eager",
        traits="impatient",
        goal="book a tee time in Pune",
        ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
    )
    metrics = await run_session(agent, persona, ScriptedUser(["Pune on 2026-06-12"]))
    assert metrics.completed is True
    assert metrics.slot_accuracy == 1.0
    assert metrics.turns >= 1


async def test_slot_mismatch_measured() -> None:
    agent = _agent(
        verdict={"slots": {"city": "Mumbai", "date": "2026-06-12"}, "advance": "booking.confirm", "note": None},
        http_responses=[_HOLD_OK],
    )
    persona = PersonaSpec(
        name="mismatched", traits="terse", goal="book in Pune",
        ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
    )
    metrics = await run_session(agent, persona, ScriptedUser(["anything"]))
    assert metrics.slot_accuracy == 0.5
    assert metrics.slot_diffs == {"city": ("Pune", "Mumbai")}


async def test_max_turns_caps_incomplete_session() -> None:
    agent = _agent(verdict=_IDLE_VERDICT)
    persona = PersonaSpec(name="stuck", traits="vague", goal="unclear", max_turns=3)
    metrics = await run_session(agent, persona, ScriptedUser(["um", "hmm", "well"]))
    assert metrics.completed is False
    assert metrics.turns == 3


async def test_run_eval_aggregates() -> None:
    agents = [
        _agent(verdict=_ADVANCING_VERDICT, http_responses=[_HOLD_OK]),
        _agent(verdict=_IDLE_VERDICT),
    ]

    def factory() -> PlaybookAgent:
        return agents.pop(0)

    personas = [
        PersonaSpec(name="closer", traits="direct", goal="book in Pune",
                    ground_truth_slots={"city": "Pune", "date": "2026-06-12"}),
        PersonaSpec(name="rambler", traits="vague", goal="chatting", max_turns=2,
                    ground_truth_slots={"city": "Goa"}),
    ]
    report = await run_eval(factory, personas, ScriptedUser(["anything"]), n=1)
    assert len(report.sessions) == 2
    assert report.completion_rate == 0.5
    assert report.mean_slot_accuracy == 0.5