"""Persona eval driver: scripted sessions measured end to end."""

from typing import Any

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.eval_bridge import (
    PersonaSpec,
    run_eval,
    run_session,
)
from superdialog.playbook.events import (
    DegradedEvent,
    EventLog,
    SteeringNoteEvent,
)
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
    """Persona LLM that pops scripted lines, repeating the last one forever."""

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
        traits="impatient, gives all details at once",
        goal="book a tee time in Pune tomorrow",
        ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
    )
    user = ScriptedUser(["Pune on 2026-06-12 please", "anything"])
    metrics = await run_session(agent, persona, user)
    assert metrics.completed is True
    assert metrics.outcome == "confirmed"
    assert metrics.slot_accuracy == 1.0
    assert metrics.slot_diffs == {}
    assert metrics.turns >= 1
    assert metrics.turns_per_checkpoint["booking.collect"] >= 1
    assert metrics.repair_count == 0
    assert metrics.degraded_count == 0
    restored = EventLog.from_jsonl(metrics.event_log_jsonl)
    assert restored.version == agent.runtime.log.version


async def test_slot_mismatch_measured() -> None:
    agent = _agent(
        verdict={
            "slots": {"city": "Mumbai", "date": "2026-06-12"},
            "advance": "booking.confirm",
            "note": None,
        },
        http_responses=[_HOLD_OK],
    )
    persona = PersonaSpec(
        name="mismatched",
        traits="terse",
        goal="book a tee time in Pune",
        ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
    )
    metrics = await run_session(agent, persona, ScriptedUser(["anything"]))
    assert metrics.slot_accuracy == 0.5
    assert metrics.slot_diffs == {"city": ("Pune", "Mumbai")}


async def test_max_turns_caps_incomplete_session() -> None:
    agent = _agent(verdict=_IDLE_VERDICT)
    persona = PersonaSpec(
        name="stuck", traits="vague, evasive", goal="unclear", max_turns=3
    )
    user = ScriptedUser(["um", "hmm", "well"])
    metrics = await run_session(agent, persona, user)
    assert metrics.completed is False
    assert metrics.outcome is None
    assert metrics.turns == 3
    assert metrics.slot_accuracy == 1.0  # no ground truth declared
    assert metrics.turns_per_checkpoint == {"booking.collect": 3}
    assert len(user.calls) == 2  # generated turns 2 and 3 only
    # persona prompt carries traits/goal and flipped transcript roles
    system = user.calls[0][0]
    assert system["role"] == "system"
    assert "vague, evasive" in system["content"]
    assert any(m["role"] == "assistant" for m in user.calls[0][1:])


async def test_repair_and_degraded_counted() -> None:
    agent = _agent(verdict=_IDLE_VERDICT)
    persona = PersonaSpec(
        name="bumpy", traits="confused", goal="book something", max_turns=2
    )

    class InjectingUser:
        """Appends audit events between turns, as a live runtime would."""

        async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            agent.runtime.log.append(
                SteeringNoteEvent(kind="repair", text="already have city")
            )
            agent.runtime.log.append(
                DegradedEvent(component="director", detail="llm_error")
            )
            return "still here"

    metrics = await run_session(agent, persona, InjectingUser())
    assert metrics.repair_count == 1
    assert metrics.degraded_count == 1
    assert metrics.turns == 2


async def test_run_eval_aggregates() -> None:
    agents = [
        _agent(verdict=_ADVANCING_VERDICT, http_responses=[_HOLD_OK]),
        _agent(verdict=_IDLE_VERDICT),
    ]

    def factory() -> PlaybookAgent:
        return agents.pop(0)

    personas = [
        PersonaSpec(
            name="closer",
            traits="direct",
            goal="book in Pune",
            ground_truth_slots={"city": "Pune", "date": "2026-06-12"},
        ),
        PersonaSpec(
            name="rambler",
            traits="vague",
            goal="just chatting",
            max_turns=2,
            ground_truth_slots={"city": "Goa"},
        ),
    ]
    report = await run_eval(factory, personas, ScriptedUser(["anything"]), n=1)
    assert len(report.sessions) == 2
    assert [s.persona for s in report.sessions] == ["closer", "rambler"]
    assert report.completion_rate == 0.5
    # closer: 2/2 correct (1.0); rambler: city never extracted (0.0)
    assert report.mean_slot_accuracy == 0.5
    assert report.sessions[1].slot_diffs == {"city": ("Goa", None)}
