"""Tests for the optimize loop: scoring, reflection, paired rounds."""

from superdialog.playbook.eval_bridge import EvalReport, SessionMetrics
from superdialog.playbook.optimize import ObjectiveBreakdown, score_report


def _session(**kw) -> SessionMetrics:
    base = dict(
        persona="p",
        completed=True,
        outcome="confirmed",
        turns=4,
        turns_per_checkpoint={"booking.collect": 2, "booking.confirm": 2},
        slot_accuracy=1.0,
        slot_diffs={},
        repair_count=0,
        degraded_count=0,
        event_log_jsonl="",
    )
    base.update(kw)
    return SessionMetrics(**base)


def test_breakdown_dimensions_match_metrics() -> None:
    report = EvalReport(sessions=[_session(), _session(completed=False, outcome=None)])
    b = score_report(report)
    assert isinstance(b, ObjectiveBreakdown)
    assert b.completion_rate == 0.5
    assert b.slot_accuracy == 1.0
    # smoothness proxy: mean turns/checkpoint over COMPLETED sessions only
    assert b.mean_turns_per_checkpoint == 2.0
    assert b.repair_rate == 0.0


def test_scalar_objective_is_weighted_sum_in_unit_range() -> None:
    good = score_report(EvalReport(sessions=[_session()]))
    bad = score_report(
        EvalReport(
            sessions=[
                _session(
                    completed=False,
                    outcome=None,
                    slot_accuracy=0.0,
                    repair_count=3,
                    turns_per_checkpoint={"a": 8},
                ),
            ]
        )
    )
    assert 0.0 <= bad.objective < good.objective <= 1.0


def test_empty_report_scores_zero() -> None:
    b = score_report(EvalReport(sessions=[]))
    assert b.objective == 0.0
    assert b.completion_rate == 0.0


def test_smoothness_rewards_fewer_turns_per_checkpoint() -> None:
    smooth = score_report(
        EvalReport(sessions=[_session(turns_per_checkpoint={"a": 1, "b": 1})])
    )
    bumpy = score_report(
        EvalReport(sessions=[_session(turns_per_checkpoint={"a": 6, "b": 6})])
    )
    assert smooth.objective > bumpy.objective


def test_incomplete_sessions_earn_no_smoothness_credit() -> None:
    # A fail-fast incomplete session must not raise the smoothness term.
    failing = score_report(
        EvalReport(
            sessions=[
                _session(
                    completed=False,
                    outcome=None,
                    slot_accuracy=0.0,
                    turns_per_checkpoint={"a": 1},
                )
            ]
        )
    )
    assert failing.mean_turns_per_checkpoint == 0.0  # nothing completed
