import json

from superdialog.eval.dataset.models import EvalCase
from superdialog.playbook.eval.bench_scenarios import ScenarioResult, score_case_both_modes
from superdialog.playbook.eval.models import PersonaSpec, SessionMetrics
from superdialog.playbook.eval.vanilla import TranscriptEntry, VanillaSessionMetrics


class _FakeJudge:
    async def complete(self, messages, **kwargs) -> str:
        return json.dumps({"violated": False, "reason": "clean"})


def _pb_metrics(event_log_jsonl: str) -> SessionMetrics:
    return SessionMetrics(
        persona="p1", completed=True, outcome="booked", turns=3,
        turns_per_checkpoint={}, slot_accuracy=1.0, slot_diffs={},
        repair_count=0, degraded_count=0, event_log_jsonl=event_log_jsonl,
    )


async def test_score_case_both_modes_returns_one_result_per_mode():
    case = EvalCase(
        id="c1", playbook="x.yaml",
        persona=PersonaSpec(name="p1", traits="t", goal="g", opening="hi"),
    )
    events = [
        {"type": "tool_result", "tool": "confirm_booking", "ok": True, "status": 200},
        {"type": "utterance", "role": "assistant", "text": "Booked!"},
    ]
    pb_metrics = _pb_metrics("\n".join(json.dumps(e) for e in events))
    vanilla_metrics = VanillaSessionMetrics(
        persona="p1", turns=2,
        transcript=[
            TranscriptEntry(role="user", text="book me a slot"),
            TranscriptEntry(role="assistant", text="Sure, you're all booked!"),
        ],
    )
    results = await score_case_both_modes(
        case=case,
        playbook_metrics=pb_metrics,
        vanilla_metrics=vanilla_metrics,
        never_say=[],
        judge_llm=_FakeJudge(),
    )
    assert set(results.keys()) == {"playbook", "vanilla"}
    assert isinstance(results["playbook"], ScenarioResult)
    assert results["playbook"].metrics["task_success"].value == 1.0
    # vanilla never makes a real tool call -- must fail the evidence gate.
    assert results["vanilla"].metrics["task_success"].value == 0.0
