"""Suite runner: behavioral gates over bench runs (suites.py).

All tests are LLM-free: the bench invocation is monkeypatched; assertions run
against synthetic logs/reports shaped like real `eval bench` output.
"""

import json
from pathlib import Path

import pytest
import yaml

from superdialog.eval import suites as su
from superdialog.eval.suites import (
    Suite,
    SuiteExpect,
    evaluate_expectations,
    run_suite,
    split_log_by_case,
)

LOG = """\
[eval-progress] mode=playbook case 1/2 rep 1/1 id=bye-case
[DIRECTOR] verdict advance=STAY cp=main.greeting
[DIRECTOR] verdict advance=interrupt:global_goodbye cp=main.greeting
[eval-progress] mode=playbook case 2/2 rep 1/1 id=control-case
[DIRECTOR] verdict advance=llm:main.next cp=main.greeting
[DIRECTOR] verdict advance=STAY cp=main.next
"""


def _report(ts_bye: float = 0.9, ts_ctl: float = 0.7, comp: float = 0.8) -> dict:
    def case(cid: str, ts: float) -> dict:
        return {
            "case_id": cid,
            "mode": "playbook",
            "metric_results": {"task_success": [{"name": "task_success", "value": ts}]},
        }

    return {
        "modes": [
            {
                "mode": "playbook",
                "composite_mean": comp,
                "case_results": [
                    case("bye-case", ts_bye),
                    case("control-case", ts_ctl),
                ],
            }
        ]
    }


def _suite(**over) -> Suite:
    base = dict(
        name="s",
        playbook="pb.yaml",
        dataset="ds.yaml",
        models=["m"],
        expect={
            "bye-case": SuiteExpect(goodbye="fired", min_task_success=0.7),
            "control-case": SuiteExpect(goodbye="absent", min_task_success=0.5),
        },
        min_composite=0.6,
    )
    base.update(over)
    return Suite(**base)


def test_split_log_by_case() -> None:
    sections = split_log_by_case(LOG)
    assert set(sections) == {"bye-case", "control-case"}
    assert "interrupt:global_goodbye" in sections["bye-case"]
    assert "interrupt:global_goodbye" not in sections["control-case"]


def test_expectations_all_pass() -> None:
    checks = evaluate_expectations(_suite(), _report(), LOG)
    assert checks and all(c.passed for c in checks)


def test_goodbye_absent_expectation_fails_when_fired() -> None:
    s = _suite(expect={"bye-case": SuiteExpect(goodbye="absent")}, min_composite=None)
    checks = evaluate_expectations(s, _report(), LOG)
    assert [c.passed for c in checks] == [False]


def test_task_success_floor_fails_below() -> None:
    s = _suite(
        expect={"control-case": SuiteExpect(min_task_success=0.9)}, min_composite=None
    )
    checks = evaluate_expectations(s, _report(ts_ctl=0.7), LOG)
    assert [c.passed for c in checks] == [False]
    assert "got 0.7" in checks[0].detail


def test_composite_floor() -> None:
    s = _suite(expect={}, min_composite=0.9)
    checks = evaluate_expectations(s, _report(comp=0.8), LOG)
    assert [c.passed for c in checks] == [False]


def test_missing_case_is_a_failure() -> None:
    s = _suite(expect={"ghost-case": SuiteExpect(goodbye="fired")}, min_composite=None)
    checks = evaluate_expectations(s, _report(), LOG)
    assert [c.passed for c in checks] == [False]
    assert checks[0].check == "present"


def test_custom_goodbye_signal() -> None:
    log = LOG.replace("interrupt:global_goodbye", "interrupt:global_caller_goodbye")
    s = _suite(
        goodbye_signal="interrupt:global_caller_goodbye",
        expect={"bye-case": SuiteExpect(goodbye="fired")},
        min_composite=None,
    )
    checks = evaluate_expectations(s, _report(), log)
    assert [c.passed for c in checks] == [True]


# -- run_suite orchestration (bench monkeypatched) --------------------------------


def _write_inputs(tmp_path: Path) -> tuple[str, str]:
    pb = tmp_path / "pb.yaml"
    ds = tmp_path / "ds.yaml"
    pb.write_text("name: test\n")
    ds.write_text(
        yaml.safe_dump(
            {"cases": [{"id": "bye-case"}, {"id": "control-case"}]},
            allow_unicode=True,
        )
    )
    return str(pb), str(ds)


def _fake_bench(tmp_path: Path, calls: list[tuple[str, str]]):
    def fake(suite: Suite, dataset: str, out_dir: Path, judge: str, user: str) -> str:
        calls.append((judge, user))
        mdir = su._model_dir(out_dir, suite.models[0])
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "report.json").write_text(json.dumps(_report()))
        return LOG

    return fake


def test_run_suite_passes_and_stamps_hash(tmp_path, monkeypatch) -> None:
    pb, ds = _write_inputs(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(su, "_run_bench", _fake_bench(tmp_path, calls))
    s = _suite(playbook=pb, dataset=ds)
    r1 = run_suite(s, tmp_path / "out")
    assert r1.status == "passed" and r1.composite == 0.8
    # second run: inputs unchanged -> skipped without invoking bench again
    r2 = run_suite(s, tmp_path / "out")
    assert r2.status == "skipped"
    assert len(calls) == 1
    # touching the dataset invalidates the stamp
    Path(ds).write_text(Path(ds).read_text() + "\n# changed\n")
    r3 = run_suite(s, tmp_path / "out")
    assert r3.status == "passed"
    assert len(calls) == 2


def test_run_suite_failed_expectations_do_not_stamp(tmp_path, monkeypatch) -> None:
    pb, ds = _write_inputs(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(su, "_run_bench", _fake_bench(tmp_path, calls))
    s = _suite(
        playbook=pb, dataset=ds, expect={"bye-case": SuiteExpect(goodbye="absent")}
    )
    assert run_suite(s, tmp_path / "out").status == "failed"
    # failure must NOT stamp: rerun executes the bench again
    assert run_suite(s, tmp_path / "out").status == "failed"
    assert len(calls) == 2


def test_quota_fallback_retries_with_fallback_models(tmp_path, monkeypatch) -> None:
    pb, ds = _write_inputs(tmp_path)
    calls: list[tuple[str, str]] = []
    inner = _fake_bench(tmp_path, calls)

    def flaky(suite: Suite, dataset: str, out_dir: Path, judge: str, user: str) -> str:
        if judge == "openai/gpt-4.1-mini":
            raise RuntimeError("Error code: 429 - insufficient_quota")
        return inner(suite, dataset, out_dir, judge, user)

    monkeypatch.setattr(su, "_run_bench", flaky)
    s = _suite(
        playbook=pb,
        dataset=ds,
        judge="openai/gpt-4.1-mini",
        fallback_judge="livekit/openai/gpt-4o-mini",
    )
    r = run_suite(s, tmp_path / "out")
    assert r.status == "passed"
    assert calls == [("livekit/openai/gpt-4o-mini", "livekit/openai/gpt-4o-mini")]


def test_quota_error_without_fallback_errors_suite(tmp_path, monkeypatch) -> None:
    pb, ds = _write_inputs(tmp_path)

    def always_quota(*a, **kw):
        raise RuntimeError("insufficient_quota")

    monkeypatch.setattr(su, "_run_bench", always_quota)
    s = _suite(playbook=pb, dataset=ds, fallback_judge=None)
    r = run_suite(s, tmp_path / "out")
    assert r.status == "errored" and "insufficient_quota" in r.detail


def test_smoke_tier_subsets_dataset(tmp_path, monkeypatch) -> None:
    pb, ds = _write_inputs(tmp_path)
    seen: dict[str, list[str]] = {}

    def fake(suite: Suite, dataset: str, out_dir: Path, judge: str, user: str) -> str:
        doc = yaml.safe_load(Path(dataset).read_text())
        seen["ids"] = [c["id"] for c in doc["cases"]]
        mdir = su._model_dir(out_dir, suite.models[0])
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "report.json").write_text(json.dumps(_report()))
        return LOG

    monkeypatch.setattr(su, "_run_bench", fake)
    s = _suite(
        playbook=pb,
        dataset=ds,
        smoke_cases=["bye-case"],
        expect={"bye-case": SuiteExpect(goodbye="fired")},
        min_composite=None,
    )
    assert run_suite(s, tmp_path / "out", tier="smoke").status == "passed"
    assert seen["ids"] == ["bye-case"]


def test_smoke_unknown_case_errors(tmp_path, monkeypatch) -> None:
    pb, ds = _write_inputs(tmp_path)
    monkeypatch.setattr(su, "_run_bench", _fake_bench(tmp_path, []))
    s = _suite(playbook=pb, dataset=ds, smoke_cases=["nope"])
    r = run_suite(s, tmp_path / "out", tier="smoke")
    assert r.status == "errored" and "nope" in r.detail


def test_load_suites_registry() -> None:
    doc = su.load_suites(
        Path(__file__).parents[3] / "examples" / "datasets" / "suites.yaml"
    )
    names = [s.name for s in doc]
    assert "realestate-disconnect" in names
    kair = next(s for s in doc if s.name == "kairali-derailment")
    assert kair.goodbye_signal == "interrupt:global_caller_goodbye"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


def test_smoke_tier_filters_expectations_to_ran_cases(tmp_path, monkeypatch) -> None:
    # Expectations for cases outside the smoke subset must be skipped, not
    # failed as "absent from run".
    pb, ds = _write_inputs(tmp_path)
    monkeypatch.setattr(su, "_run_bench", _fake_bench(tmp_path, []))
    s = _suite(
        playbook=pb,
        dataset=ds,
        smoke_cases=["bye-case"],
        expect={
            "bye-case": SuiteExpect(goodbye="fired"),
            "control-case": SuiteExpect(goodbye="absent"),  # not in smoke run
        },
        min_composite=None,
    )
    r = run_suite(s, tmp_path / "out", tier="smoke")
    assert r.status == "passed"
    assert [c.case_id for c in r.checks] == ["bye-case"]


def test_quota_fallback_detects_chained_cause(tmp_path, monkeypatch) -> None:
    # Real shape: LLMResilienceError("failed after 3 attempts") from
    # RateLimitError("... insufficient_quota ...") — the signal is in the
    # CAUSE, not the top-level message.
    pb, ds = _write_inputs(tmp_path)
    calls: list[tuple[str, str]] = []
    inner = _fake_bench(tmp_path, calls)

    def flaky(suite: Suite, dataset: str, out_dir: Path, judge: str, user: str) -> str:
        if judge == "openai/gpt-4.1-mini":
            try:
                raise RuntimeError("Error code: 429 - insufficient_quota")
            except RuntimeError as cause:
                raise RuntimeError("LLM complete failed after 3 attempt(s)") from cause
        return inner(suite, dataset, out_dir, judge, user)

    monkeypatch.setattr(su, "_run_bench", flaky)
    s = _suite(
        playbook=pb,
        dataset=ds,
        judge="openai/gpt-4.1-mini",
        fallback_judge="livekit/openai/gpt-4o-mini",
    )
    assert run_suite(s, tmp_path / "out").status == "passed"
    assert calls == [("livekit/openai/gpt-4o-mini", "livekit/openai/gpt-4o-mini")]


def test_fallback_downgrades_score_floors_to_advisory(tmp_path, monkeypatch) -> None:
    # Under the fallback judge, behavioral checks still gate but score floors
    # (judge-calibrated) become advisory WARNs, and no hash stamp is written.
    pb, ds = _write_inputs(tmp_path)
    calls: list[tuple[str, str]] = []
    inner = _fake_bench(tmp_path, calls)

    def low_score_fallback(
        suite: Suite, dataset: str, out_dir: Path, judge: str, user: str
    ) -> str:
        if judge == "primary":
            raise RuntimeError("insufficient_quota")
        inner(suite, dataset, out_dir, judge, user)
        mdir = su._model_dir(out_dir, suite.models[0])
        (mdir / "report.json").write_text(json.dumps(_report(ts_bye=0.3, comp=0.4)))
        return LOG

    monkeypatch.setattr(su, "_run_bench", low_score_fallback)
    s = _suite(judge="primary", fallback_judge="fb", playbook=pb, dataset=ds)
    r = run_suite(s, tmp_path / "out")
    assert r.status == "passed" and r.detail == "(fallback judge)"
    advisories = [c for c in r.checks if c.advisory]
    assert advisories and all(
        c.check.startswith(("task_success>=", "composite>=")) for c in advisories
    )
    # behavioral checks still gated (and passed)
    assert any(c.check == "goodbye:fired" and c.passed for c in r.checks)
    # no stamp -> next run executes again instead of skipping
    assert run_suite(s, tmp_path / "out").status == "passed"
    assert len(calls) == 2


def test_fallback_behavioral_failure_still_gates(tmp_path, monkeypatch) -> None:
    pb, ds = _write_inputs(tmp_path)

    def fallback_no_goodbye(
        suite: Suite, dataset: str, out_dir: Path, judge: str, user: str
    ) -> str:
        if judge == "primary":
            raise RuntimeError("insufficient_quota")
        mdir = su._model_dir(out_dir, suite.models[0])
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "report.json").write_text(json.dumps(_report()))
        return LOG.replace("interrupt:global_goodbye", "advance=STAY")

    monkeypatch.setattr(su, "_run_bench", fallback_no_goodbye)
    s = _suite(
        judge="primary",
        fallback_judge="fb",
        playbook=pb,
        dataset=ds,
        expect={"bye-case": SuiteExpect(goodbye="fired")},
        min_composite=None,
    )
    assert run_suite(s, tmp_path / "out").status == "failed"


def test_min_turns_floor() -> None:
    # LOG's bye-case section has one turn marker style? Build a section with
    # explicit turn markers to count.
    log = (
        "[eval-progress] mode=playbook case 1/1 rep 1/1 id=bye-case\n"
        "[eval-progress]   turn 0: user said 'hi'\n"
        "[eval-progress]   turn 1: user said 'no'\n"
        "[eval-progress]   turn 2: user said 'bye'\n"
    )
    s = _suite(expect={"bye-case": SuiteExpect(min_turns=3)}, min_composite=None)
    checks = evaluate_expectations(s, _report(), log)
    assert [c.passed for c in checks] == [True]
    s2 = _suite(expect={"bye-case": SuiteExpect(min_turns=5)}, min_composite=None)
    checks = evaluate_expectations(s2, _report(), log)
    assert [c.passed for c in checks] == [False]
    assert "got 3" in checks[0].detail


def test_no_reentry_detects_flow_restart() -> None:
    log = (
        "[eval-progress] mode=playbook case 1/1 rep 1/1 id=bye-case\n"
        "[turn-trace] side=brain version=2 checkpoint=main.greet lang=-\n"
        "[turn-trace] side=brain version=5 checkpoint=main.ask lang=-\n"
        "[turn-trace] side=brain version=9 checkpoint=main.greet lang=-\n"  # restart!
    )
    s = _suite(expect={"bye-case": SuiteExpect(no_reentry=True)}, min_composite=None)
    checks = evaluate_expectations(s, _report(), log)
    assert [c.passed for c in checks] == [False]
    assert "REVISITED" in checks[0].detail and "main.greet" in checks[0].detail
    clean = log.replace(
        "checkpoint=main.greet lang=-\n[turn-trace] side=brain version=5",
        "checkpoint=main.greet lang=-\n[turn-trace] side=brain version=5",
    )
    clean = (
        "[eval-progress] mode=playbook case 1/1 rep 1/1 id=bye-case\n"
        "[turn-trace] side=brain version=2 checkpoint=main.greet lang=-\n"
        "[turn-trace] side=brain version=3 checkpoint=main.greet lang=-\n"  # consecutive = fine
        "[turn-trace] side=brain version=5 checkpoint=main.ask lang=-\n"
    )
    checks = evaluate_expectations(s, _report(), clean)
    assert [c.passed for c in checks] == [True]


def test_silent_afterlife_flags_zombie_replies() -> None:
    noisy = (
        "[eval-progress] mode=playbook case 1/1 rep 1/1 id=bye-case\n"
        "[afterlife] probe='Hello?' reply='Thank you. Goodbye.'\n"
        "[afterlife] probe='I have time now' reply=''\n"
    )
    s = _suite(
        expect={"bye-case": SuiteExpect(silent_afterlife=True)}, min_composite=None
    )
    checks = evaluate_expectations(s, _report(), noisy)
    assert [c.passed for c in checks] == [False]
    assert "1/2 probes drew a reply" in checks[0].detail
    silent = noisy.replace("reply='Thank you. Goodbye.'", "reply=''")
    checks = evaluate_expectations(s, _report(), silent)
    assert [c.passed for c in checks] == [True]
    # no probes at all -> fail loudly (session never ended)
    bare = "[eval-progress] mode=playbook case 1/1 rep 1/1 id=bye-case\n"
    checks = evaluate_expectations(s, _report(), bare)
    assert [c.passed for c in checks] == [False]


def test_no_repeat_replies_counts_markers() -> None:
    log = (
        "[eval-progress] mode=playbook case 1/1 rep 1/1 id=bye-case\n"
        "[repeat-reply] turn 4\n"
        "[repeat-reply] turn 5\n"
    )
    s = _suite(
        expect={"bye-case": SuiteExpect(no_repeat_replies=True)}, min_composite=None
    )
    checks = evaluate_expectations(s, _report(), log)
    assert [c.passed for c in checks] == [False]
    assert "2 repeated replies" in checks[0].detail
