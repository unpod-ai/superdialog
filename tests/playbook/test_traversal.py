# tests/playbook/test_traversal.py
"""Playbook traversal recorder tests."""

import json
import tempfile
from pathlib import Path
from typing import Any

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.eval.runner import run_session
from superdialog.playbook.models import Playbook
from superdialog.playbook.traversal import build_playbook_traversal, save_playbook_traversal
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_ADVANCING_VERDICT: dict = {
    "slots": {"city": "Pune", "date": "2026-06-12"},
    "advance": "booking.confirm",
    "note": None,
}
_IDLE_VERDICT: dict = {"slots": {}, "advance": None, "note": None}
_HOLD_OK = (200, {"data": {"hold_id": "h1"}})


class ScriptedUser:
    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        if not self.lines:
            return ""
        return self.lines.pop(0) if len(self.lines) > 1 else self.lines[0]


def _agent(verdict: dict, http_responses: list) -> PlaybookAgent:
    return PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=StreamLLM(["Great,"]),
        director_llm=CannedLLM(verdict),
        http=FakeHttp(http_responses),
    )


async def test_traversal_has_all_top_level_keys() -> None:
    agent = _agent(_ADVANCING_VERDICT, [_HOLD_OK])
    await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML))
    for key in ("session_id", "playbook_file", "model", "started_at", "ended_at",
                "is_complete", "outcome", "checkpoints", "traversal", "final_slots", "graph"):
        assert key in t, f"missing key: {key}"


async def test_traversal_is_complete_on_terminal() -> None:
    agent = _agent(_ADVANCING_VERDICT, [_HOLD_OK])
    await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML))
    assert t["is_complete"] is True
    assert t["outcome"] == "confirmed"  # SessionEndEvent.outcome = Checkpoint.outcome


async def test_traversal_steps_ordered_and_have_required_fields() -> None:
    agent = _agent(_ADVANCING_VERDICT, [_HOLD_OK])
    await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML))
    steps = t["traversal"]
    assert len(steps) >= 1
    for i, step in enumerate(steps, 1):
        assert step["step"] == i
        for field in ("from_checkpoint", "to_checkpoint", "advance_rule", "advance_by",
                      "version", "goal", "bot_message", "user_message",
                      "slots_written", "tool_calls", "degraded"):
            assert field in step, f"step missing field: {field}"


async def test_traversal_slots_written_recorded() -> None:
    agent = _agent(_ADVANCING_VERDICT, [_HOLD_OK])
    await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML))
    all_slots: dict = {}
    for step in t["traversal"]:
        all_slots.update(step["slots_written"])
    assert "city" in all_slots
    assert all_slots["city"]["value"] == "Pune"


async def test_traversal_final_slots_match_session() -> None:
    agent = _agent(_ADVANCING_VERDICT, [_HOLD_OK])
    await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML))
    assert "city" in t["final_slots"]
    assert t["final_slots"]["city"]["value"] == "Pune"


async def test_traversal_graph_contains_checkpoints_and_edges() -> None:
    agent = _agent(_ADVANCING_VERDICT, [_HOLD_OK])
    await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML))
    graph = t["graph"]
    assert len(graph["checkpoints"]) >= 1
    assert len(graph["advance_edges"]) >= 1

    # Traversed edges have traversed_at_step set
    traversed = [e for e in graph["advance_edges"] if e["traversed"]]
    assert len(traversed) >= 1
    for e in traversed:
        assert e["traversed_at_step"] is not None


async def test_traversal_incomplete_session_not_complete() -> None:
    agent = _agent(_IDLE_VERDICT, [])
    persona = PersonaSpec(name="stuck", traits="", goal="nothing", max_turns=1)
    await run_session(agent, persona, ScriptedUser(["hello"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML))
    assert t["is_complete"] is False
    assert t["outcome"] is None


async def test_traversal_dir_on_agent_auto_saves_on_session_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = PlaybookAgent(
            playbook=Playbook.from_yaml(MINIMAL_YAML),
            talker_llm=StreamLLM(["Great,"]),
            director_llm=CannedLLM(_ADVANCING_VERDICT),
            http=FakeHttp([_HOLD_OK]),
            traversal_dir=tmp,
            traversal_source="hotel.yaml",
            traversal_model="gpt-4o-mini",
        )
        await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

        files = list(Path(tmp).glob("traversal_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["is_complete"] is True
        assert data["playbook_file"] == "hotel.yaml"
        assert data["model"] == "gpt-4o-mini"


async def test_save_playbook_traversal_writes_json() -> None:
    agent = _agent(_ADVANCING_VERDICT, [_HOLD_OK])
    await run_session(agent, PersonaSpec(name="p", traits="", goal="book"), ScriptedUser(["Pune 2026-06-12"]))

    t = build_playbook_traversal(agent.event_log, Playbook.from_yaml(MINIMAL_YAML), source="hotel.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        path = save_playbook_traversal(t, tmp)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["session_id"] == t["session_id"]
        assert loaded["playbook_file"] == "hotel.yaml"