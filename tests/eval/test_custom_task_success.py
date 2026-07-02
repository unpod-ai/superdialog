"""Task 4.2: TaskSuccessJudge parses judge JSON and is errored-safe."""

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.metrics.custom import TaskSuccessJudge
from tests.eval.fakes import FakeProvider


async def test_task_success_parses_judge_json():
    judge = FakeProvider(
        {"*": '{"completed": true, "graded": 0.9, "reason": "booked"}'}
    )
    m = TaskSuccessJudge(judge)
    s = EvalSample(
        kind="conversation",
        user_input=[{"role": "user", "content": "book haircut sat"}],
        reference="book a haircut on saturday",
    )
    r = await m.score(s)
    assert r.value == 0.9 and r.passed is True


async def test_task_success_errored_on_bad_json():
    judge = FakeProvider({"*": "not json"})
    m = TaskSuccessJudge(judge)
    s = EvalSample(kind="conversation", user_input=[], reference="x")
    r = await m.score(s)
    assert r.errored and r.value is None
