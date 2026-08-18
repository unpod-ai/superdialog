from superdialog.playbook.eval.scenario_metrics import (
    evidence_gated_task_success,
    pass_at_1,
)


def test_task_success_true_when_completed_and_real_tool_success():
    events = [
        {"type": "tool_result", "tool": "confirm_booking", "ok": True, "status": 200},
        {"type": "utterance", "role": "assistant", "text": "You're all set!"},
    ]
    result = evidence_gated_task_success(completed=True, events=events)
    assert result.value == 1.0
    assert result.passed is True


def test_task_success_false_when_completed_but_no_real_tool_call():
    """The golfai S1 incident shape: agent claims success, zero real calls."""
    events = [
        {"type": "utterance", "role": "assistant", "text": "Payment link sent!"},
    ]
    result = evidence_gated_task_success(completed=True, events=events)
    assert result.value == 0.0
    assert result.passed is False
    assert "no real tool-call evidence" in result.reason.lower()


def test_task_success_false_when_tool_ran_but_state_never_completed():
    events = [
        {"type": "tool_result", "tool": "confirm_booking", "ok": True, "status": 200},
    ]
    result = evidence_gated_task_success(completed=False, events=events)
    assert result.value == 0.0
    assert result.passed is False


def test_task_success_false_for_vanilla_by_construction():
    """Vanilla never makes real tool calls -- it can never pass this gate."""
    events: list = []
    result = evidence_gated_task_success(completed=True, events=events)
    assert result.value == 0.0
    assert result.passed is False


def test_pass_at_1_true_with_evidence_and_no_retries():
    events = [
        {"type": "advance", "from_checkpoint": "collect", "to_checkpoint": "confirm", "rule": "director"},
        {"type": "tool_result", "tool": "confirm_booking", "ok": True, "status": 200},
    ]
    result = pass_at_1(completed=True, events=events)
    assert result.value == 1.0
    assert result.passed is True


def test_pass_at_1_false_when_same_checkpoint_revisited():
    """A retry/backtrack shows up as the SAME to_checkpoint advanced-to twice."""
    events = [
        {"type": "advance", "from_checkpoint": "collect", "to_checkpoint": "confirm", "rule": "director"},
        {"type": "advance", "from_checkpoint": "confirm", "to_checkpoint": "collect", "rule": "on_failure"},
        {"type": "advance", "from_checkpoint": "collect", "to_checkpoint": "confirm", "rule": "director"},
        {"type": "tool_result", "tool": "confirm_booking", "ok": True, "status": 200},
    ]
    result = pass_at_1(completed=True, events=events)
    assert result.value == 0.0
    assert result.passed is False
    assert "retry" in result.reason.lower() or "revisit" in result.reason.lower()


def test_pass_at_1_false_without_evidence_even_with_no_retries():
    events = [
        {"type": "advance", "from_checkpoint": "collect", "to_checkpoint": "confirm", "rule": "director"},
    ]
    result = pass_at_1(completed=True, events=events)
    assert result.value == 0.0
    assert result.passed is False
