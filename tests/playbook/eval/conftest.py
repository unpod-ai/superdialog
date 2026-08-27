# tests/playbook/eval/conftest.py
"""Session-level hooks that must live in conftest.py to be reliably picked up
by pytest -- pytest_sessionfinish defined inside a test module (as opposed to
pytest_generate_tests, which pytest does collect at module scope) is not
guaranteed to fire."""

from __future__ import annotations

import json

from tests.playbook.eval.test_scenarios import (
    _AGGREGATE_METRICS,
    _CURRENT_CASE_IDS,
    _DUMP_DIR,
)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Print the aggregate playbook-vs-vanilla score once, after every case
    in this session has been scored -- answers "what's the overall score"
    without hand-averaging the per-scenario prints.

    Reads dumped JSON files (_DUMP_DIR/<case_id>.json), NOT the in-memory
    test_scenarios._ALL_RESULTS list. Under pytest-xdist each persona runs
    in its own worker process with its own memory -- _ALL_RESULTS would
    only ever hold one worker's subset there. _DUMP_DIR is a real shared
    filesystem every worker writes into, so it's correct under both plain
    and xdist-parallel runs.

    Under xdist this hook also fires once per WORKER (each worker's own
    session finishing), not just once in the controller -- skip those so
    the aggregate prints exactly once, after ALL workers (and therefore all
    dumps) are done. Only the controller process lacks ``workerinput``.
    """
    if hasattr(session.config, "workerinput"):
        return
    if _DUMP_DIR is None or not _DUMP_DIR.is_dir():
        return
    # _DUMP_DIR persists across runs (rejudge_dumped.py re-scores old dumps
    # without re-driving conversations) -- restrict to case ids THIS session
    # actually collected/parametrized, so a stale dump from an earlier
    # dataset generation (different personas, same playbook) never sneaks
    # into this run's average. _CURRENT_CASE_IDS (populated by
    # pytest_generate_tests, which runs at collection time in every process
    # that needs it) replaces an earlier session.items/callspec-based
    # version that came back empty under pytest-xdist's controller on a
    # real run despite fresh dumps existing on disk.
    dumps = [json.loads(p.read_text(encoding="utf-8")) for p in _DUMP_DIR.glob("*.json")]
    dumps = [
        d for d in dumps if "metrics" in d and d.get("case_id") in _CURRENT_CASE_IDS
    ]
    if not dumps:
        return
    print(
        f"\n{'=' * 74}\n AGGREGATE — {len(dumps)} case(s), playbook vs vanilla"
        f"\n{'=' * 74}"
    )
    for mode in ("playbook", "vanilla"):
        print(f"  [{mode}]")
        for name in _AGGREGATE_METRICS:
            values = [
                d["metrics"][mode][name]["value"]
                for d in dumps
                if name in d["metrics"].get(mode, {})
                and d["metrics"][mode][name]["value"] is not None
            ]
            n_total = len(dumps)
            if not values:
                print(f"    {name:28s} n/a   (0/{n_total} scored)")
                continue
            avg = sum(values) / len(values)
            print(f"    {name:28s} {avg * 100:5.1f}%  ({len(values)}/{n_total} scored)")
