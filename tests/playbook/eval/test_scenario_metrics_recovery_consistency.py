from superdialog.playbook.eval.scenario_metrics import (
    conversational_consistency,
    failure_recovery_rate,
)


def test_failure_recovery_rate_full_when_agent_never_claims_success_after_failure():
    events = [
        {"type": "tool_call", "tool": "confirm_booking", "args": {}},
        {"type": "tool_result", "tool": "confirm_booking", "ok": False, "status": 404, "error": "not found"},
        {"type": "utterance", "role": "assistant", "text": "That slot isn't available anymore, let me check others."},
    ]
    result = failure_recovery_rate(events)
    assert result.value == 1.0
    assert result.extra["failures"] == 1
    assert result.extra["recoveries"] == 1


def test_failure_recovery_rate_zero_when_agent_fabricates_success_after_failure():
    events = [
        {"type": "tool_call", "tool": "confirm_booking", "args": {}},
        {"type": "tool_result", "tool": "confirm_booking", "ok": False, "status": 500, "error": "server error"},
        {"type": "utterance", "role": "assistant", "text": "All done, your booking is confirmed!"},
    ]
    result = failure_recovery_rate(events)
    assert result.value == 0.0
    assert result.extra["failures"] == 1
    assert result.extra["recoveries"] == 0


def test_failure_recovery_rate_skipped_when_no_failures_occurred():
    events = [
        {"type": "tool_result", "tool": "confirm_booking", "ok": True, "status": 200},
    ]
    result = failure_recovery_rate(events)
    assert result.skipped is True
    assert result.value is None


def test_conversational_consistency_full_when_no_slot_ever_changes():
    events = [
        {"type": "slot_write", "key": "date", "value": "Friday", "status": "confirmed", "by": "director"},
        {"type": "slot_write", "key": "players", "value": "4", "status": "confirmed", "by": "director"},
    ]
    result = conversational_consistency(events)
    assert result.value == 1.0
    assert result.extra["contradictions"] == []


def test_conversational_consistency_flags_a_changed_slot_value():
    events = [
        {"type": "slot_write", "key": "date", "value": "Friday", "status": "confirmed", "by": "director"},
        {"type": "slot_write", "key": "date", "value": "Saturday", "status": "confirmed", "by": "director"},
    ]
    result = conversational_consistency(events)
    assert result.value == 0.0
    assert result.extra["contradictions"] == [
        {"key": "date", "from": "Friday", "to": "Saturday"}
    ]


def test_conversational_consistency_ignores_repeated_identical_value():
    """Re-confirming the SAME value is not a contradiction."""
    events = [
        {"type": "slot_write", "key": "date", "value": "Friday", "status": "provisional", "by": "director"},
        {"type": "slot_write", "key": "date", "value": "Friday", "status": "confirmed", "by": "director"},
    ]
    result = conversational_consistency(events)
    assert result.value == 1.0
    assert result.extra["contradictions"] == []
