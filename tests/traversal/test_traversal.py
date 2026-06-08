"""Tests for traversal history module."""
from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from superdialog import DialogMachine, Flow
from superdialog.flow.models import ActionTriggerType
from superdialog.machine.actions import ActionExecutor
from superdialog.machine.models import ActionRecord, FlowContext

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "flow"


class _ScriptedProvider:
    def __init__(self, responses):
        self._responses = list(responses)

    async def complete(self, messages, tools=None, **opts):
        from superdialog.llm.provider import CompletionResult
        text = self._responses.pop(0) if self._responses else "{}"
        return CompletionResult(text=text, tool_calls=[], metadata={})

    async def stream(self, messages, tools=None, **opts):
        result = await self.complete(messages)
        from superdialog.llm.provider import StreamChunk as SC
        yield SC(text=result.text, tool_call_delta=None, done=True)


def _criteria(edge, response="ok", all_met=True):
    return json.dumps({
        "criteria_met": {},
        "extracted_slots": {},
        "all_required_met": all_met,
        "user_insisting": False,
        "recommended_edge_id": edge,
        "reason": "test",
        "response": response,
    })


def _make_fake_machine(nodes, transition_log=None, action_log=None, is_complete=True):
    """Build a minimal fake DialogMachine for build_traversal."""
    machine = MagicMock()
    machine.is_complete = is_complete
    machine.state = {"node_id": nodes[-1]["id"] if nodes else "", "slots": {}}

    ctx = MagicMock()
    ctx.transition_log = transition_log or []
    ctx.visit_count = {n["id"]: 1 for n in nodes}
    ctx.action_log = action_log or []

    inner = MagicMock()
    inner.context = ctx
    inner.is_complete = is_complete
    machine._machine = inner

    return machine


def _make_flow(node_ids):
    """Build a minimal fake ConversationFlow."""
    flow = MagicMock()
    flow_nodes = []
    for nid in node_ids:
        n = MagicMock()
        n.id = nid
        n.name = nid.replace("_", " ").title()
        n.instruction = f"Instruction for {nid}"
        n.static_text = None
        n.is_final = (nid == node_ids[-1])
        n.edges = []
        flow_nodes.append(n)
    flow.nodes = flow_nodes
    flow.global_edges = []
    return flow


def _wrap(engine):
    """Wrap a raw DialogStateMachine so build_traversal can read it."""
    facade = MagicMock()
    facade._machine = engine
    facade.is_complete = engine.is_complete
    return facade


def test_build_traversal_schema():
    from superdialog.machine.models import TransitionRecord
    from superdialog.traversal import build_traversal

    nodes = [{"id": "greet"}, {"id": "collect_name"}, {"id": "done"}]
    flow = _make_flow(["greet", "collect_name", "done"])

    tr = TransitionRecord(
        from_node="greet",
        to_node="collect_name",
        edge_id="greet_to_collect",
        criteria_met={"name_asked": True},
        skipped=False,
        timestamp=1700000001.0,
    )

    machine = _make_fake_machine(nodes, transition_log=[tr], is_complete=True)
    chat_turns = [
        {"step": 1, "bot": "Hello!", "user": None, "node": "greet", "ts": "2026-05-23T00:00:00Z"},
        {"step": 2, "bot": "What is your name?", "user": "Ankit", "node": "collect_name", "ts": "2026-05-23T00:00:05Z"},
    ]
    started_at = datetime(2026, 5, 23, 0, 0, 0, tzinfo=timezone.utc)

    result = build_traversal(machine, chat_turns, flow, "kyc.json", "openai/gpt-4.1-mini", started_at)

    assert result["flow_file"] == "kyc.json"
    assert result["model"] == "openai/gpt-4.1-mini"
    assert result["is_complete"] is True
    assert len(result["traversal"]) == 2
    step1 = result["traversal"][0]
    assert step1["step"] == 1
    assert step1["bot_message"] == "Hello!"
    assert step1["user_message"] is None
    assert step1["from_node"] is None
    step2 = result["traversal"][1]
    assert step2["step"] == 2
    assert step2["from_node"] == "greet"
    assert step2["to_node"] == "collect_name"
    assert step2["edge_id"] == "greet_to_collect"
    assert step2["user_message"] == "Ankit"
    assert "actions" in step2

    assert "graph" in result
    graph_node_ids = [n["id"] for n in result["graph"]["nodes"]]
    assert "greet" in graph_node_ids
    assert result["graph"]["edges"][0]["traversed"] is True


def test_save_traversal_creates_file():
    from superdialog.traversal import save_traversal

    traversal = {
        "session_id": "20260523_000000_000000",
        "flow_file": "test.json",
        "model": "test",
        "started_at": "2026-05-23T00:00:00+00:00",
        "ended_at": "2026-05-23T00:01:00+00:00",
        "is_complete": True,
        "nodes": [],
        "traversal": [],
        "graph": {"nodes": [], "edges": []},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_traversal(traversal, Path(tmpdir))
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["session_id"] == "20260523_000000_000000"


def test_action_record_fields():
    rec = ActionRecord(
        action_id="action-auth-token",
        node_id="greeting",
        trigger="on_enter",
        url="https://api.example.com/auth/token",
        method="POST",
        status=200,
        success=True,
        result_data={"access_token": "abc123"},
    )
    assert rec.action_id == "action-auth-token"
    assert rec.trigger == "on_enter"
    assert rec.result_data == {"access_token": "abc123"}
    assert rec.timestamp > 0


def test_flow_context_action_log_starts_empty():
    ctx = FlowContext()
    assert ctx.action_log == []


def _make_action_trigger(action_id: str):
    """Build a minimal action trigger matching what ActionExecutor expects."""
    trigger = MagicMock()
    trigger.action_id = action_id
    trigger.trigger_type = ActionTriggerType.ON_ENTER
    return trigger


def _make_custom_action(action_id: str):
    action = MagicMock()
    action.id = action_id
    action.store_response_as = "auth_result"
    action.env_updates = []
    action.run_once = False
    return action


def test_action_executor_records_action_log():
    ctx = FlowContext()
    ctx.state.current_node_id = "greeting"

    custom_action = _make_custom_action("action-auth-token")
    action_map = {"action-auth-token": custom_action}

    adapter = MagicMock()
    adapter.execute_action = AsyncMock(return_value={
        "status": 200,
        "success": True,
        "data": {"access_token": "tok123"},
        "_rendered_url": "https://api.example.com/auth/token",
        "_method": "POST",
    })

    executor = ActionExecutor(adapter=adapter, action_map=action_map)
    trigger = _make_action_trigger("action-auth-token")

    asyncio.run(executor.execute([trigger], ctx, trigger_type=ActionTriggerType.ON_ENTER))

    assert len(ctx.action_log) == 1
    rec = ctx.action_log[0]
    assert rec.action_id == "action-auth-token"
    assert rec.node_id == "greeting"
    assert rec.trigger == "on_enter"
    assert rec.url == "https://api.example.com/auth/token"
    assert rec.method == "POST"
    assert rec.status == 200
    assert rec.success is True
    assert rec.result_data == {"access_token": "tok123"}


def test_flow_context_action_log_append():
    ctx = FlowContext()
    rec = ActionRecord(
        action_id="action-players-search",
        node_id="greeting",
        trigger="on_enter",
        url="https://api.example.com/players/search",
        method="GET",
        status=400,
        success=False,
        result_data={"detail": {"error": "invalid_request"}},
    )
    ctx.action_log.append(rec)
    assert len(ctx.action_log) == 1
    assert ctx.action_log[0].action_id == "action-players-search"


@pytest.mark.asyncio
async def test_dialog_machine_auto_saves_traversal():
    flow = Flow.load(FIXTURE_DIR / "kyc.json")

    responses = [
        _criteria("greet_to_name", "What's your name?"),       # greet → collect_name
        "Please tell me your full name.",                        # collect_name entry reply
        _criteria("name_to_dob", "Date of birth?"),             # collect_name eval
        "Thanks. What is your date of birth?",                   # collect_dob entry reply
        _criteria("dob_to_pan", "PAN number?"),                 # collect_dob eval
        "Now I need your PAN number.",                           # collect_pan entry reply
        _criteria("pan_to_done", "Thank you!", all_met=True),   # collect_pan eval
        "KYC complete.",                                         # done entry reply
    ]
    provider = _ScriptedProvider(responses)

    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DialogMachine(
            flow=flow,
            llm="openai/gpt-4.1-mini",
            traversal_dir=tmpdir,
        )
        dm._llm = provider

        await dm.start()
        await dm.turn("Ankit")           # greet → collect_name
        await dm.turn("01/01/1990")      # collect_name → collect_dob
        await dm.turn("ABCDE1234F")      # collect_dob → collect_pan
        await dm.turn("confirmed")       # collect_pan → done (pan_to_done)

        files = list(Path(tmpdir).glob("traversal_*.json"))
        assert len(files) == 1, f"Expected 1 traversal file, got {files}"

        data = json.loads(files[0].read_text())
        assert data["is_complete"] is True
        assert len(data["traversal"]) >= 1
        assert data["model"] == "openai/gpt-4.1-mini"


def test_build_traversal_actions_per_step():
    """Actions from action_log should appear in the correct traversal step."""
    from superdialog.traversal import build_traversal

    nodes = [{"id": "greeting"}, {"id": "done"}]
    flow = _make_flow(["greeting", "done"])

    rec = ActionRecord(
        action_id="action-auth-token",
        node_id="greeting",
        trigger="on_enter",
        url="https://api.example.com/auth/token",
        method="POST",
        status=200,
        success=True,
        result_data={"access_token": "abc123"},
    )

    machine = _make_fake_machine(nodes, action_log=[rec], is_complete=True)
    chat_turns = [
        {"step": 1, "bot": "Hello!", "user": None, "node": "greeting", "ts": "2026-05-23T00:00:00Z"},
    ]
    started_at = datetime(2026, 5, 23, 0, 0, 0, tzinfo=timezone.utc)

    result = build_traversal(machine, chat_turns, flow, "test.json", "openai/gpt-4.1-mini", started_at)

    step1 = result["traversal"][0]
    assert len(step1["actions"]) == 1
    action = step1["actions"][0]
    assert action["action_id"] == "action-auth-token"
    assert action["trigger"] == "on_enter"
    assert action["url"] == "https://api.example.com/auth/token"
    assert action["method"] == "POST"
    assert action["status"] == 200
    assert action["success"] is True
    assert action["result_data"] == {"access_token": "abc123"}


def test_build_traversal_prefers_record_messages_on_chained_turns():
    """One user turn -> two transition records, but only one chat_turns entry.
    build_traversal must read messages from records, not mis-pair positionally."""
    from superdialog.machine.models import TransitionRecord
    from superdialog.traversal import build_traversal

    flow = _make_flow(["a", "r", "c"])
    log = [
        TransitionRecord(
            from_node="a",
            to_node="r",
            edge_id="a_to_r",
            criteria_met={"x": True},
            user_message="hello",
            bot_message="",
            timestamp=1.0,
        ),
        TransitionRecord(
            from_node="r",
            to_node="c",
            edge_id="r_to_c",
            criteria_met={},
            user_message=None,
            bot_message="Goodbye!",
            timestamp=2.0,
        ),
    ]
    machine = _make_fake_machine(
        [{"id": "a"}, {"id": "r"}, {"id": "c"}],
        transition_log=log,
    )
    # Desync: the chained hop has NO chat_turns entry (the real bug).
    chat_turns = [
        {"step": 1, "bot": "Hi", "user": None, "node": "a", "ts": ""},
        {"step": 2, "bot": "", "user": "hello", "node": "r", "ts": ""},
    ]
    started_at = datetime(2026, 6, 8, tzinfo=timezone.utc)

    result = build_traversal(machine, chat_turns, flow, "f.json", "m", started_at)

    step_ar = result["traversal"][1]
    assert step_ar["user_message"] == "hello"
    step_rc = result["traversal"][2]  # would be blank under positional pairing
    assert step_rc["user_message"] is None
    assert step_rc["bot_message"] == "Goodbye!"


def test_build_traversal_met_true_when_no_criteria():
    """A criteria-free transition that succeeded should report met=True, not the
    old bool({}) -> False wart."""
    from superdialog.machine.models import TransitionRecord
    from superdialog.traversal import build_traversal

    flow = _make_flow(["a", "b"])
    log = [
        TransitionRecord(
            from_node="a",
            to_node="b",
            edge_id="a_to_b",
            criteria_met={},
            skipped=False,
            timestamp=1.0,
        )
    ]
    machine = _make_fake_machine([{"id": "a"}, {"id": "b"}], transition_log=log)
    chat_turns = [{"step": 1, "bot": "Hi", "user": None, "node": "a", "ts": ""}]
    started_at = datetime(2026, 6, 8, tzinfo=timezone.utc)

    result = build_traversal(machine, chat_turns, flow, "f.json", "m", started_at)
    assert result["traversal"][1]["criteria"]["met"] is True


@pytest.mark.anyio
async def test_linear_flow_traversal_messages_align() -> None:
    from superdialog.flow.models import ConversationFlow, Edge, FlowNode
    from superdialog.machine.machine import DialogStateMachine
    from superdialog.machine.testing.mock_adapter import MockAdapter
    from superdialog.traversal import build_traversal

    flow = ConversationFlow(
        system_prompt="t", initial_node="g",
        nodes=[
            FlowNode(id="g", name="g", instruction="Greet.",
                     edges=[Edge(id="g_to_c", condition="ok", target_node_id="c")]),
            FlowNode(id="c", name="c", instruction="Confirm.", is_final=True),
        ],
    )
    machine = await DialogStateMachine.from_flow(
        flow=flow, adapter=MockAdapter(["g_to_c"]),
    )
    await machine.process_turn("yes please")

    # Minimal chat_turns (just the greeting step, as DialogMachine records it).
    chat_turns = [{"step": 1, "bot": "Hi", "user": None, "node": "g", "ts": ""}]
    tr = build_traversal(
        _wrap(machine), chat_turns, flow, "lin.json", "mock",
        datetime(2026, 6, 8, tzinfo=timezone.utc),
    )
    step2 = tr["traversal"][1]
    assert step2["from_node"] == "g" and step2["to_node"] == "c"
    assert step2["user_message"] == "yes please"
    assert step2["bot_message"] == "mock reply for c"
    assert tr["is_complete"] is True
