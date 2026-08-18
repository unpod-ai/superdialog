import json

from superdialog.playbook.eval.events import (
    any_real_success,
    assistant_utterances,
    parse_events,
    slot_writes,
    tool_calls_and_results,
)


def _jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def test_parse_events_splits_jsonl_into_dicts():
    log = _jsonl(
        {"type": "utterance", "role": "assistant", "text": "Hi"},
        {"type": "advance", "from_checkpoint": None, "to_checkpoint": "greet", "rule": "init"},
    )
    events = parse_events(log)
    assert events == [
        {"type": "utterance", "role": "assistant", "text": "Hi"},
        {"type": "advance", "from_checkpoint": None, "to_checkpoint": "greet", "rule": "init"},
    ]


def test_parse_events_skips_blank_lines():
    assert parse_events("\n\n") == []
    assert parse_events("") == []


def test_assistant_utterances_filters_role_and_type():
    events = [
        {"type": "utterance", "role": "user", "text": "book me a slot"},
        {"type": "utterance", "role": "assistant", "text": "Sure, when?"},
        {"type": "slot_write", "key": "date", "value": "Friday"},
        {"type": "utterance", "role": "assistant", "text": "Booked!"},
    ]
    assert assistant_utterances(events) == ["Sure, when?", "Booked!"]


def test_tool_calls_and_results_pairs_by_index_order():
    events = [
        {"type": "tool_call", "tool": "availability", "args": {}},
        {"type": "tool_result", "tool": "availability", "ok": True, "status": 200, "data": {}},
        {"type": "tool_call", "tool": "confirm_booking", "args": {}},
        {"type": "tool_result", "tool": "confirm_booking", "ok": False, "status": 404, "data": None},
    ]
    pairs = tool_calls_and_results(events)
    assert len(pairs) == 2
    assert pairs[0][0]["tool"] == "availability" and pairs[0][1]["ok"] is True
    assert pairs[1][0]["tool"] == "confirm_booking" and pairs[1][1]["ok"] is False


def test_slot_writes_filters_type_only():
    events = [
        {"type": "slot_write", "key": "date", "value": "Friday", "status": "confirmed", "by": "director"},
        {"type": "utterance", "role": "assistant", "text": "x"},
    ]
    assert slot_writes(events) == [
        {"key": "date", "value": "Friday", "status": "confirmed", "by": "director"}
    ]


def test_any_real_success_true_when_a_tool_result_ok():
    events = [
        {"type": "tool_result", "tool": "confirm_booking", "ok": False, "status": 404},
        {"type": "tool_result", "tool": "confirm_booking", "ok": True, "status": 200},
    ]
    assert any_real_success(events) is True


def test_any_real_success_false_with_no_ok_tool_result():
    events = [
        {"type": "utterance", "role": "assistant", "text": "Your booking is confirmed!"},
        {"type": "tool_result", "tool": "confirm_booking", "ok": False, "status": 404},
    ]
    assert any_real_success(events) is False


def test_any_real_success_false_with_no_tool_calls_at_all():
    events = [{"type": "utterance", "role": "assistant", "text": "Booked!"}]
    assert any_real_success(events) is False


def test_any_real_success_can_scope_to_specific_tool_ids():
    events = [
        {"type": "tool_result", "tool": "availability", "ok": True, "status": 200},
        {"type": "tool_result", "tool": "confirm_booking", "ok": False, "status": 500},
    ]
    # availability succeeding is not booking succeeding
    assert any_real_success(events, tool_ids={"confirm_booking"}) is False
    assert any_real_success(events, tool_ids={"availability"}) is True
