# tests/playbook/eval/test_auditor.py
"""SessionAuditor: post-session path + slot + quality analysis."""

import json
from typing import Any

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.eval.auditor import SessionAuditor
from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.eval.runner import run_session
from superdialog.playbook.models import Playbook
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_ADVANCING_VERDICT = {
    "slots": {"city": "Pune", "date": "2026-06-12"},
    "advance": "booking.confirm",
    "note": None,
}
_HOLD_OK = (200, {"data": {"hold_id": "h1"}})


class ScriptedUser:
    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        if not self.lines:
            return ""
        return self.lines.pop(0) if len(self.lines) > 1 else self.lines[0]


class JudgeLLM:
    """Returns a fixed quality score as JSON {"score": N}."""

    def __init__(self, score: int = 4) -> None:
        self.score = score

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        return json.dumps({"score": self.score})


def _make_agent(verdict: dict, http_responses: list) -> PlaybookAgent:
    return PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=StreamLLM(["Which city?"]),
        director_llm=CannedLLM(verdict),
        http=FakeHttp(http_responses),
    )


async def test_audit_complete_session_path_valid() -> None:
    agent = _make_agent(_ADVANCING_VERDICT, [_HOLD_OK])
    persona = PersonaSpec(name="eager", traits="direct", goal="book in Pune")
    await run_session(agent, persona, ScriptedUser(["Pune on 2026-06-12"]))

    playbook = Playbook.from_yaml(MINIMAL_YAML)
    auditor = SessionAuditor(playbook=playbook, judge_llm=JudgeLLM(score=4))
    report = await auditor.audit(agent, session_id="test-1")

    assert report.session_id == "test-1"
    assert report.path_valid is True
    assert len(report.checkpoint_path) >= 1
    assert report.path_violations == []


async def test_audit_slot_completeness_all_filled() -> None:
    agent = _make_agent(_ADVANCING_VERDICT, [_HOLD_OK])
    persona = PersonaSpec(name="eager", traits="direct", goal="book in Pune")
    await run_session(agent, persona, ScriptedUser(["Pune on 2026-06-12"]))

    playbook = Playbook.from_yaml(MINIMAL_YAML)
    auditor = SessionAuditor(playbook=playbook, judge_llm=JudgeLLM(score=5))
    report = await auditor.audit(agent)

    assert "city" in report.slot_coverage
    assert "date" in report.slot_coverage
    assert report.slot_coverage["city"] is True
    assert report.slot_coverage["date"] is True
    assert report.slot_completeness == 1.0


async def test_audit_slot_completeness_missing_slots() -> None:
    idle_verdict = {"slots": {}, "advance": None, "note": None}
    agent = _make_agent(idle_verdict, [])
    persona = PersonaSpec(name="stuck", traits="vague", goal="nothing", max_turns=1)
    await run_session(agent, persona, ScriptedUser(["hello"]))

    playbook = Playbook.from_yaml(MINIMAL_YAML)
    auditor = SessionAuditor(playbook=playbook, judge_llm=JudgeLLM(score=3))
    report = await auditor.audit(agent)

    assert report.slot_completeness < 1.0
    assert any(not v for v in report.slot_coverage.values())


async def test_audit_response_quality_uses_judge() -> None:
    agent = _make_agent(_ADVANCING_VERDICT, [_HOLD_OK])
    persona = PersonaSpec(name="eager", traits="direct", goal="book in Pune")
    await run_session(agent, persona, ScriptedUser(["Pune on 2026-06-12"]))

    playbook = Playbook.from_yaml(MINIMAL_YAML)
    auditor = SessionAuditor(playbook=playbook, judge_llm=JudgeLLM(score=5))
    report = await auditor.audit(agent)

    # judge returns 5 for every utterance → response_quality == 1.0
    assert report.response_quality == 1.0


async def test_audit_overall_score_computed() -> None:
    agent = _make_agent(_ADVANCING_VERDICT, [_HOLD_OK])
    persona = PersonaSpec(name="eager", traits="direct", goal="book in Pune")
    await run_session(agent, persona, ScriptedUser(["Pune on 2026-06-12"]))

    playbook = Playbook.from_yaml(MINIMAL_YAML)
    auditor = SessionAuditor(playbook=playbook, judge_llm=JudgeLLM(score=4))
    report = await auditor.audit(agent)

    assert 0.0 <= report.overall_score <= 1.0