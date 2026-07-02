"""Task 4.3: SlotAccuracyJudge scoring, diffs, and errored/no-call paths."""

from typing import Any

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.metrics.custom import SlotAccuracyJudge
from tests.eval.fakes import FakeProvider


class _RaisingJudge:
    """A judge that fails if invoked — proves the no-ground-truth short-circuit."""

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        raise AssertionError("judge should not be called")


def _sample(slots: dict[str, str]) -> EvalSample:
    return EvalSample(
        kind="conversation",
        user_input=[{"role": "user", "content": "book haircut sat"}],
        metadata={"ground_truth_slots": slots},
    )


async def test_slot_accuracy_all_captured():
    judge = FakeProvider(
        {"*": '{"per_slot": {"day": true}, "accuracy": 1.0, "diffs": {}}'}
    )
    m = SlotAccuracyJudge(judge)
    r = await m.score(_sample({"day": "Saturday"}))
    assert r.value == 1.0 and not r.errored


async def test_slot_accuracy_missing_slot_has_diff():
    judge = FakeProvider(
        {
            "*": '{"per_slot": {"day": true, "time": false}, '
            '"accuracy": 0.5, "diffs": {"time": ["3pm", null]}}'
        }
    )
    m = SlotAccuracyJudge(judge)
    r = await m.score(_sample({"day": "Saturday", "time": "3pm"}))
    assert r.value == 0.5
    assert r.extra["diffs"] == {"time": ["3pm", None]}


async def test_slot_accuracy_errored_on_bad_json():
    m = SlotAccuracyJudge(FakeProvider({"*": "not json"}))
    r = await m.score(_sample({"day": "Saturday"}))
    assert r.errored and r.value is None


async def test_slot_accuracy_no_ground_truth_skips_judge():
    m = SlotAccuracyJudge(_RaisingJudge())
    r = await m.score(_sample({}))
    assert r.value == 1.0 and not r.errored
