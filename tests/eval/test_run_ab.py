"""run_ab: every case x mode x repeat, aggregated into one ModeResult per mode."""

from __future__ import annotations

from superdialog.eval.dataset.models import EvalCase, EvalDataset
from superdialog.eval.endpoints.in_process import InProcessVanilla
from superdialog.eval.metrics.registry import build_suite
from superdialog.eval.runner import run_ab
from superdialog.playbook.eval.models import PersonaSpec

from tests.eval.fakes import FakeProvider, FakeSpeaksUser

_METRICS = ["task_success", "efficiency"]
# Only the task-success judge calls the LLM here; perfect completion for all.
_JUDGE = '{"completed": true, "graded": 1.0, "reason": "ok"}'


def _case(cid: str) -> EvalCase:
    return EvalCase(
        id=cid,
        playbook="pb",
        persona=PersonaSpec(name="p", traits="terse", goal="get help", max_turns=1),
        reference="the assistant should help the caller",
    )


async def test_run_ab_single_mode_two_cases_two_repeats() -> None:
    """One mode, populated aggregates, composite in [0,1], metrics echoed back."""
    dataset = EvalDataset(playbook="pb", cases=[_case("c1"), _case("c2")])
    suite = build_suite(_METRICS, FakeProvider({"*": _JUDGE}), "m")

    result = await run_ab(
        dataset,
        modes=["vanilla"],
        endpoint_factories={
            "vanilla": lambda case: InProcessVanilla("bot", FakeProvider({"*": "ok"}))
        },
        suite=suite,
        user_llm=FakeSpeaksUser(["hi"]),
        metric_names=_METRICS,
        repeats=2,
    )

    assert result.metrics == _METRICS
    assert len(result.modes) == 1
    mode = result.modes[0]
    assert mode.mode == "vanilla"
    # 2 cases x 2 repeats = 4 case results.
    assert len(mode.case_results) == 4
    assert set(mode.aggregates) >= {"task_success", "efficiency"}
    for agg in mode.aggregates.values():
        # each metric is scored on every conversation sample -> n >= 1, except
        # data-dependent metrics (token_cost) that skip when the endpoint
        # reports no usage — a skipped-only aggregate row is still informative.
        assert agg.n >= 1 or agg.skipped >= 1
    assert 0.0 <= mode.composite_mean <= 1.0
