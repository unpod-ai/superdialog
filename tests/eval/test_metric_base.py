"""Task 4.1: Metric protocol + MetricResult + MetricSuite behaviour."""

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.metrics.base import Metric, MetricResult, MetricSuite


class _Const:
    name = "const"

    async def score(self, sample):
        return MetricResult(name=self.name, value=1.0, passed=True, reason="always")


class _Boom:
    name = "boom"

    async def score(self, sample):
        raise RuntimeError("judge crashed")


class _Convo:
    name = "convo_only"
    applies_to = ("conversation",)

    async def score(self, sample):
        return MetricResult(name=self.name, value=0.5)


async def test_suite_scores_applicable_samples():
    suite = MetricSuite([_Const()])
    s = EvalSample(kind="probe", user_input="q", response="a")
    results = await suite.score(s)
    assert results[0].value == 1.0 and results[0].passed
    assert isinstance(_Const(), Metric)  # runtime_checkable


async def test_suite_skips_non_applicable_kind():
    suite = MetricSuite([_Convo()])
    probe = EvalSample(kind="probe", user_input="q", response="a")
    assert await suite.score(probe) == []
    convo = EvalSample(kind="conversation", user_input=[], response="a")
    assert (await suite.score(convo))[0].value == 0.5


async def test_metric_raise_yields_errored_not_zero():
    suite = MetricSuite([_Boom()])
    s = EvalSample(kind="probe", user_input="q", response="a")
    r = (await suite.score(s))[0]
    assert r.errored and r.value is None
