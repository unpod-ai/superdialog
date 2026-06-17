# tests/playbook/eval/test_scorer.py
"""scorer: score_report (moved from optimize) + run_multi_model."""

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.eval.models import EvalReport, PersonaSpec, SessionMetrics
from superdialog.playbook.eval.scorer import ObjectiveBreakdown, run_multi_model, score_report
from superdialog.playbook.models import Playbook
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE = {"slots": {}, "advance": None, "note": None}
_ADVANCING = {
    "slots": {"city": "Pune", "date": "2026-06-12"},
    "advance": "booking.confirm",
    "note": None,
}
_HOLD_OK = (200, {"data": {"hold_id": "h1"}})


def _make_session(completed: bool, slot_acc: float = 1.0, repairs: int = 0) -> SessionMetrics:
    return SessionMetrics(
        persona="test",
        completed=completed,
        outcome="confirmed" if completed else None,
        turns=3,
        turns_per_checkpoint={"booking.collect": 2, "booking.confirm": 1} if completed else {"booking.collect": 3},
        slot_accuracy=slot_acc,
        slot_diffs={},
        repair_count=repairs,
        degraded_count=0,
        event_log_jsonl="",
    )


def test_score_report_empty_returns_zeros() -> None:
    report = EvalReport(sessions=[])
    bd = score_report(report)
    assert bd.objective == 0.0
    assert bd.completion_rate == 0.0
    assert bd.slot_accuracy == 0.0


def test_score_report_perfect_session() -> None:
    report = EvalReport(sessions=[_make_session(True, 1.0, 0)])
    bd = score_report(report)
    assert isinstance(bd, ObjectiveBreakdown)
    assert bd.completion_rate == 1.0
    assert bd.slot_accuracy == 1.0
    assert 0.0 < bd.objective <= 1.0


def test_score_report_incomplete_session() -> None:
    report = EvalReport(sessions=[_make_session(False, 0.0, 0)])
    bd = score_report(report)
    assert bd.completion_rate == 0.0
    assert bd.objective < score_report(EvalReport(sessions=[_make_session(True, 1.0)])).objective


def test_score_report_repair_penalises_objective() -> None:
    clean = score_report(EvalReport(sessions=[_make_session(True, 1.0, repairs=0)]))
    noisy = score_report(EvalReport(sessions=[_make_session(True, 1.0, repairs=5)]))
    assert clean.objective > noisy.objective


async def test_run_multi_model_returns_one_score_per_model() -> None:
    class ScriptedUser:
        async def complete(self, messages, **kw):
            return "anything"

    def factory_advancing() -> PlaybookAgent:
        return PlaybookAgent(
            playbook=Playbook.from_yaml(MINIMAL_YAML),
            talker_llm=StreamLLM(["ok"]),
            director_llm=CannedLLM(_ADVANCING),
            http=FakeHttp([_HOLD_OK]),
        )

    def factory_idle() -> PlaybookAgent:
        return PlaybookAgent(
            playbook=Playbook.from_yaml(MINIMAL_YAML),
            talker_llm=StreamLLM(["hmm"]),
            director_llm=CannedLLM(_IDLE),
            http=FakeHttp([]),
        )

    personas = [PersonaSpec(name="p1", traits="direct", goal="book", max_turns=2)]
    report = await run_multi_model(
        {"model-a": factory_advancing, "model-b": factory_idle},
        personas,
        ScriptedUser(),
        n=1,
    )
    assert len(report.models) == 2
    model_ids = {s.model_id for s in report.models}
    assert model_ids == {"model-a", "model-b"}


async def test_run_multi_model_better_model_scores_higher() -> None:
    class ScriptedUser:
        async def complete(self, messages, **kw):
            return "anything"

    personas = [PersonaSpec(name="p1", traits="direct", goal="book", max_turns=2,
                             ground_truth_slots={"city": "Pune", "date": "2026-06-12"})]

    def good() -> PlaybookAgent:
        return PlaybookAgent(
            playbook=Playbook.from_yaml(MINIMAL_YAML),
            talker_llm=StreamLLM(["ok"]),
            director_llm=CannedLLM(_ADVANCING),
            http=FakeHttp([_HOLD_OK]),
        )

    def bad() -> PlaybookAgent:
        return PlaybookAgent(
            playbook=Playbook.from_yaml(MINIMAL_YAML),
            talker_llm=StreamLLM(["hmm"]),
            director_llm=CannedLLM(_IDLE),
            http=FakeHttp([]),
        )

    report = await run_multi_model({"good": good, "bad": bad}, personas, ScriptedUser())
    scores = {s.model_id: s.objective for s in report.models}
    assert scores["good"] > scores["bad"]
