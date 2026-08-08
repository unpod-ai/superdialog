# tests/playbook/test_traversal.py
"""Playbook traversal recorder tests."""

import json
import tempfile
from pathlib import Path
from typing import Any

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.eval.runner import run_session
from superdialog.playbook.events import (
    AdvanceEvent,
    DegradedEvent,
    EventLog,
    SlotWriteEvent,
    SpeechCorrectionEvent,
    UtteranceEvent,
)
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


# --- dwell steps: every user turn yields a step -------------------------


def _log(*events: Any) -> EventLog:
    log = EventLog()
    for e in events:
        log.append(e)
    return log


def test_dwell_turns_get_steps() -> None:
    # init advance into collect; turn 1 (no advance); turn 2 (advance to
    # confirm); turn 3 (no advance, dwelling in confirm).
    log = _log(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init"),
        UtteranceEvent(role="assistant", text="Hi! Where would you like to go?"),
        UtteranceEvent(role="user", text="I want a hotel"),  # turn 1
        UtteranceEvent(role="assistant", text="Which city?"),
        UtteranceEvent(role="user", text="Pune 2026-06-12"),  # turn 2
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director"),
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="r1",
        ),
        UtteranceEvent(role="assistant", text="Your booking is held."),
        UtteranceEvent(role="user", text="thanks"),  # turn 3
        UtteranceEvent(role="assistant", text="You're welcome"),
    )
    t = build_playbook_traversal(log, Playbook.from_yaml(MINIMAL_YAML))
    steps = t["traversal"]
    assert [s["step"] for s in steps] == [1, 2, 3, 4]

    init, dwell_a, adv, dwell_b = steps
    assert init["advance_rule"] == "init"
    assert init["turn"] is None
    assert init["bot_message"] == "Hi! Where would you like to go?"

    # turn 1: no advance → dwell step in collect
    assert dwell_a["from_checkpoint"] == dwell_a["to_checkpoint"] == "booking.collect"
    assert dwell_a["advance_rule"] == "dwell"
    assert dwell_a["advance_by"] is None
    assert dwell_a["turn"] == 1
    assert dwell_a["user_message"] == "I want a hotel"
    assert dwell_a["bot_message"] == "Which city?"  # the reply that FOLLOWED turn 1
    assert dwell_a["goal"] == "Have city and date"
    assert dwell_a["slots_written"] == {}
    assert dwell_a["tool_calls"] == []
    assert dwell_a["degraded"] is False

    # turn 2: advance — no dwell step for this turn (no double-counting)
    assert adv["advance_rule"] == "r1"
    assert adv["from_checkpoint"] == "booking.collect"
    assert adv["to_checkpoint"] == "booking.confirm"
    assert adv["turn"] == 2
    assert adv["user_message"] == "Pune 2026-06-12"
    assert adv["bot_message"] == "Your booking is held."
    assert "city" in adv["slots_written"]

    # turn 3: dwell in confirm
    assert dwell_b["from_checkpoint"] == dwell_b["to_checkpoint"] == "booking.confirm"
    assert dwell_b["advance_rule"] == "dwell"
    assert dwell_b["turn"] == 3
    assert dwell_b["user_message"] == "thanks"
    assert dwell_b["bot_message"] == "You're welcome"


def test_speech_correction_reaches_bot_message() -> None:
    """G38: exports must carry what the caller HEARD, not what was generated —
    covers both bot_message reads (turn-0 greeting bucket and per-turn window)."""
    log = _log(
        AdvanceEvent(
            from_checkpoint=None, to_checkpoint="booking.collect", rule="init"
        ),
        UtteranceEvent(role="assistant", text="Hi! Where would you like to go?"),
        SpeechCorrectionEvent(
            utterance_version=2, heard_text="Hi! [interrupted by caller]"
        ),
        UtteranceEvent(role="user", text="I want a hotel"),  # turn 1
        UtteranceEvent(role="assistant", text="Which city and what date?"),
        SpeechCorrectionEvent(
            utterance_version=5, heard_text="Which city [interrupted by caller]"
        ),
    )
    t = build_playbook_traversal(log, Playbook.from_yaml(MINIMAL_YAML))
    steps = t["traversal"]
    assert steps[0]["bot_message"] == "Hi! [interrupted by caller]"
    assert steps[1]["bot_message"] == "Which city [interrupted by caller]"
    assert steps[1]["user_message"] == "I want a hotel"  # user text untouched


def test_dwell_step_carries_window_events_and_latency() -> None:
    # Dwell buckets carry the slot writes / degraded flag that landed during
    # their turn window — junk_rejected and no-advance extractions visible.
    log = _log(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init"),
        UtteranceEvent(role="user", text="junk input"),  # turn 1, no advance
        SlotWriteEvent(key="city", value="Pune", status="provisional", by="director"),
        DegradedEvent(component="director", detail="junk_rejected"),
        UtteranceEvent(role="assistant", text="Could you repeat that?"),
    )
    latency = {"director": {"per_turn_ms": [111]}, "talker": {"per_turn_ms": [222]}}
    t = build_playbook_traversal(log, Playbook.from_yaml(MINIMAL_YAML), latency=latency)
    steps = t["traversal"]
    assert len(steps) == 2
    dwell = steps[1]
    assert dwell["advance_rule"] == "dwell"
    assert dwell["slots_written"]["city"]["value"] == "Pune"
    assert dwell["degraded"] is True
    assert dwell["director_ms"] == 111
    assert dwell["talker_ms"] == 222


def test_multi_advance_turn_produces_one_step_per_advance_no_dwell() -> None:
    # Quiescence chain: one turn, two advances → two steps, no dwell step.
    log = _log(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init"),
        UtteranceEvent(role="user", text="Pune 2026-06-12"),  # turn 1
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="r1",
        ),
        AdvanceEvent(
            from_checkpoint="booking.confirm",
            to_checkpoint="booking.close",
            rule="pipeline",
            by="expr",
        ),
        UtteranceEvent(role="assistant", text="Done."),
    )
    t = build_playbook_traversal(log, Playbook.from_yaml(MINIMAL_YAML))
    rules = [s["advance_rule"] for s in t["traversal"]]
    assert rules == ["init", "r1", "pipeline"]
    assert [s["turn"] for s in t["traversal"]] == [None, 1, 1]


def test_scripted_user_replays_each_turn_once() -> None:
    # Dwell turns replay too; a quiescence chain's repeated user_message
    # contributes one line, not one per advance step.
    from superdialog.playbook.eval.from_traversal import traversal_to_scripted_user

    log = _log(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init"),
        UtteranceEvent(role="user", text="hello"),  # turn 1: dwell
        UtteranceEvent(role="assistant", text="hi"),
        UtteranceEvent(role="user", text="Pune 2026-06-12"),  # turn 2: chain
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="r1",
        ),
        AdvanceEvent(
            from_checkpoint="booking.confirm",
            to_checkpoint="booking.close",
            rule="pipeline",
            by="expr",
        ),
    )
    t = build_playbook_traversal(log, Playbook.from_yaml(MINIMAL_YAML))
    user = traversal_to_scripted_user(t)
    assert user._messages == ["hello", "Pune 2026-06-12"]


def test_steps_carry_corroborated() -> None:
    # Advance steps surface AdvanceEvent.corroborated (design §5: prose-share
    # per step); dwell and init/turn-0 steps carry None.
    log = _log(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init"),
        UtteranceEvent(role="user", text="hello"),  # turn 1: dwell
        UtteranceEvent(role="assistant", text="hi"),
        UtteranceEvent(role="user", text="Pune 2026-06-12"),  # turn 2: evidence
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="llm:r1",
            corroborated=True,
        ),
        UtteranceEvent(role="user", text="yes book it"),  # turn 3: prose only
        AdvanceEvent(
            from_checkpoint="booking.confirm",
            to_checkpoint="booking.close",
            rule="llm:r2",
            corroborated=False,
        ),
    )
    t = build_playbook_traversal(log, Playbook.from_yaml(MINIMAL_YAML))
    by_rule = {s["advance_rule"]: s["corroborated"] for s in t["traversal"]}
    assert by_rule == {"init": None, "dwell": None, "llm:r1": True, "llm:r2": False}


def test_dwell_does_not_touch_visits_or_graph_edges() -> None:
    log = _log(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init"),
        UtteranceEvent(role="user", text="hello"),  # dwell turn
        UtteranceEvent(role="assistant", text="hi"),
    )
    t = build_playbook_traversal(log, Playbook.from_yaml(MINIMAL_YAML))
    collect = next(c for c in t["checkpoints"] if c["id"] == "booking.collect")
    assert collect["visit_count"] == 1  # init only; dwell never increments
    assert not any(e["rule"] == "dwell" for e in t["graph"]["advance_edges"])