"""Suite runner — execute eval benches as behavioral regression gates.

``superdialog eval bench`` produces scores you eyeball; this layer turns a set
of benches into a one-command, CI-able gate:

- **Suite registry** (YAML): each suite = playbook + dataset + models + the
  BEHAVIORAL expectations that made the run worth doing (this case must fire
  the goodbye interrupt, that control must NOT; task_success floors).
- **Tiers**: ``smoke`` runs each suite's designated smoke cases only (cheap
  pre-merge signal); ``full`` runs everything.
- **Skip-if-unchanged**: a content hash of playbook+dataset+params is stamped
  in the out dir; unchanged suites are skipped unless ``--force``.
- **Quota fallback**: a run that dies on provider quota (429/insufficient
  quota) is retried once with the suite's fallback judge/user models.

Suite YAML shape::

    suites:
      - name: realestate-disconnect
        playbook: examples/playbooks/realestate_site_visit.simple.yaml
        dataset: examples/datasets/realestate_disconnect.evalcases.yaml
        models: [livekit/google/gemma-4-31b-it]
        judge: openai/gpt-4.1-mini
        user_model: openai/gpt-4.1-mini
        fallback_judge: livekit/openai/gpt-4o-mini
        fallback_user: livekit/openai/gpt-4o-mini
        max_turns: 14
        smoke_cases: [explicit-disconnect-mid, objecting-but-staying-pooja]
        min_composite: 0.6
        expect:
          explicit-disconnect-mid: {goodbye: fired, min_task_success: 0.7}
          objecting-but-staying-pooja: {goodbye: absent, min_task_success: 0.5}
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

_GOODBYE_SIGNAL = "interrupt:global_goodbye"
_QUOTA_SIGNALS = ("insufficient_quota", "RateLimitError", "Error code: 429")
_CASE_MARKER = re.compile(r"\[eval-progress\] mode=\S+ case \d+/\d+ .*?id=(\S+)")


class SuiteExpect(BaseModel):
    """Behavioral expectations for one eval case."""

    goodbye: Literal["fired", "absent"] | None = None
    min_task_success: float | None = None
    # Floor on conversation length — encodes "not ended prematurely" for
    # control personas whose business takes N turns but who legitimately say
    # goodbye once finished (a goodbye:absent assertion would flag the honest
    # close; a premature hangup shows up as a short section instead).
    min_turns: int | None = None
    # Route integrity (from [turn-trace] checkpoints): no checkpoint may be
    # RE-entered after the flow moved past it (A..B..A = restart/regression).
    # Caveat: resume=true interrupt detours legitimately revisit — only use
    # on playbooks without resume interrupts.
    no_reentry: bool = False
    # Post-end probing must draw NO reply: an ended session that answers
    # ("Hello?" -> goodbye replay / re-engagement) is a zombie (observed in
    # production). Requires the persona to define afterlife_probes.
    silent_afterlife: bool = False
    # No assistant reply may exactly repeat the previous one (closing loops,
    # parrot loops) — asserted via the runner's [repeat-reply] markers.
    no_repeat_replies: bool = False


class Suite(BaseModel):
    """One bench invocation plus the assertions that gate it."""

    name: str
    playbook: str
    dataset: str
    models: list[str] = Field(min_length=1)
    judge: str = "openai/gpt-4.1-mini"
    user_model: str = "openai/gpt-4.1-mini"
    fallback_judge: str | None = None
    fallback_user: str | None = None
    max_turns: int = 14
    repeats: int = 1
    metrics: str = "task_success,slot_accuracy"
    smoke_cases: list[str] = Field(default_factory=list)
    min_composite: float | None = None
    # Log token that marks the goodbye interrupt firing; playbooks name their
    # interrupt differently (realestate: global_goodbye, kairali:
    # global_caller_goodbye), so the suite pins the exact advance rule string.
    goodbye_signal: str = _GOODBYE_SIGNAL
    expect: dict[str, SuiteExpect] = Field(default_factory=dict)


class CheckResult(BaseModel):
    """One evaluated expectation."""

    case_id: str
    check: str
    passed: bool
    detail: str = ""
    # Advisory checks report but do not gate. Score floors (task_success,
    # composite) are calibrated against the suite's PRIMARY judge; when the
    # quota fallback swaps the judge, those numbers aren't comparable, so
    # they downgrade to advisory. Behavioral checks (goodbye fired/absent)
    # are judge-independent and always gate.
    advisory: bool = False


class SuiteResult(BaseModel):
    """Outcome of one suite run."""

    name: str
    status: Literal["passed", "failed", "skipped", "errored"]
    checks: list[CheckResult] = Field(default_factory=list)
    composite: float | None = None
    detail: str = ""


def load_suites(path: str | Path) -> list[Suite]:
    """Load and validate the suite registry."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [Suite.model_validate(s) for s in doc.get("suites", [])]


def _suite_hash(suite: Suite) -> str:
    """Content hash over everything that changes a suite's outcome."""
    h = hashlib.sha256()
    for f in (suite.playbook, suite.dataset):
        p = Path(f)
        h.update(p.read_bytes() if p.exists() else b"missing")
    h.update(
        json.dumps(
            suite.model_dump(exclude={"expect", "min_composite"}), sort_keys=True
        ).encode()
    )
    return h.hexdigest()


def _exc_chain_text(exc: BaseException) -> str:
    """Concatenated messages of ``exc`` and its cause/context chain.

    Providers wrap the interesting error (`LLMResilienceError ... from
    RateLimitError`), so quota detection must read the whole chain, not just
    the top-level message.
    """
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(parts) < 10:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts)


def _subset_dataset(dataset: str, case_ids: list[str], out: Path) -> str:
    """Write a smoke-tier dataset containing only ``case_ids``."""
    doc = yaml.safe_load(Path(dataset).read_text(encoding="utf-8"))
    doc["cases"] = [c for c in doc.get("cases", []) if c.get("id") in case_ids]
    if not doc["cases"]:
        raise ValueError(f"smoke_cases {case_ids} not found in {dataset}")
    out.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return str(out)


def split_log_by_case(log: str) -> dict[str, str]:
    """Split a bench log into per-case sections keyed by case id."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in log.splitlines():
        m = _CASE_MARKER.search(line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf)
            current, buf = m.group(1), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf)
    return sections


def evaluate_expectations(
    suite: Suite, report: dict[str, Any], log: str
) -> list[CheckResult]:
    """Assert the suite's behavioral expectations against report + log."""
    checks: list[CheckResult] = []
    sections = split_log_by_case(log)
    mode = (report.get("modes") or [{}])[0]
    by_case: dict[str, dict[str, Any]] = {
        c.get("case_id", ""): c for c in mode.get("case_results", [])
    }
    for case_id, exp in suite.expect.items():
        section = sections.get(case_id)
        if section is None and case_id not in by_case:
            checks.append(
                CheckResult(
                    case_id=case_id,
                    check="present",
                    passed=False,
                    detail="case absent from run (smoke tier or dataset drift?)",
                )
            )
            continue
        if exp.goodbye is not None and section is not None:
            fired = suite.goodbye_signal in section
            want = exp.goodbye == "fired"
            checks.append(
                CheckResult(
                    case_id=case_id,
                    check=f"goodbye:{exp.goodbye}",
                    passed=fired == want,
                    detail=f"goodbye interrupt {'fired' if fired else 'absent'}",
                )
            )
        if exp.min_turns is not None and section is not None:
            turn_ids = [int(m) for m in re.findall(r"turn (\d+):", section)]
            turns = (max(turn_ids) + 1) if turn_ids else 0
            checks.append(
                CheckResult(
                    case_id=case_id,
                    check=f"turns>={exp.min_turns}",
                    passed=turns >= exp.min_turns,
                    detail=f"got {turns}",
                )
            )
        if exp.no_reentry and section is not None:
            # A version RESET (later version < earlier) marks a fresh machine —
            # the factual-probe phase rebuilds it, legitimately re-greeting.
            # Reentry is only meaningful WITHIN one session segment, so reset
            # the route there; otherwise the probe greeting reads as a restart.
            traces = re.findall(
                r"\[turn-trace\].*?version=(\d+).*?checkpoint=(\S+)", section
            )
            route: list[str] = []
            prev_ver = -1
            for ver_s, cp in traces:
                ver = int(ver_s)
                if ver < prev_ver:  # new session segment (probe/reset)
                    route = []
                prev_ver = ver
                if not route or route[-1] != cp:
                    route.append(cp)
            revisited = sorted({cp for cp in route if route.count(cp) > 1})
            checks.append(
                CheckResult(
                    case_id=case_id,
                    check="no_reentry",
                    passed=not revisited,
                    detail=(
                        f"route={'>'.join(route)}"
                        + (f" REVISITED={revisited}" if revisited else "")
                    )[:300],
                )
            )
        if exp.silent_afterlife and section is not None:
            replies = re.findall(
                r"\[afterlife\] probe=.*? reply='(.*)'$", section, re.M
            )
            noisy = [r for r in replies if r.strip()]
            checks.append(
                CheckResult(
                    case_id=case_id,
                    check="silent_afterlife",
                    passed=bool(replies) and not noisy,
                    detail=(
                        f"{len(noisy)}/{len(replies)} probes drew a reply"
                        if replies
                        else "no afterlife probes ran (session never ended?)"
                    ),
                )
            )
        if exp.no_repeat_replies and section is not None:
            repeats = len(re.findall(r"\[repeat-reply\]", section))
            checks.append(
                CheckResult(
                    case_id=case_id,
                    check="no_repeat_replies",
                    passed=repeats == 0,
                    detail=f"{repeats} repeated replies",
                )
            )
        if exp.min_task_success is not None:
            results = by_case.get(case_id, {}).get("metric_results", {})
            vals = [
                r.get("value")
                for r in results.get("task_success", [])
                if r.get("value") is not None
            ]
            score = min(vals) if vals else None
            checks.append(
                CheckResult(
                    case_id=case_id,
                    check=f"task_success>={exp.min_task_success}",
                    passed=score is not None and score >= exp.min_task_success,
                    detail=f"got {score}",
                )
            )
    if suite.min_composite is not None:
        comp = mode.get("composite_mean")
        checks.append(
            CheckResult(
                case_id="(suite)",
                check=f"composite>={suite.min_composite}",
                passed=comp is not None and comp >= suite.min_composite,
                detail=f"got {comp:.3f}" if comp is not None else "missing",
            )
        )
    return checks


class _Tee(io.TextIOBase):
    """Mirror writes to the real stdout while capturing them."""

    def __init__(self, mirror: Any) -> None:
        self._mirror = mirror
        self.buffer_ = io.StringIO()

    def write(self, s: str) -> int:
        self._mirror.write(s)
        return self.buffer_.write(s)

    def flush(self) -> None:
        self._mirror.flush()


@contextlib.contextmanager
def _capture_engine_logs(stream: Any):
    """Tee ``superdialog`` engine log records into the bench capture.

    The behavioral checks grep the captured log for logger-emitted markers
    ('[DIRECTOR] verdict advance=…' at INFO, '[turn-trace] …' at DEBUG).
    Nothing configures a logging handler in this process, so without this
    those records are dropped and goodbye:fired / no_reentry are
    unfalsifiable — every suite failed goodbye:fired regardless of engine
    behavior after 4eed0b9 converted the markers from print() to logging.
    Scoped to the bench run and always removed; the logger level is
    restored so the eval process doesn't leak DEBUG elsewhere.
    """
    logger = logging.getLogger("superdialog")
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def _run_bench(suite: Suite, dataset: str, out_dir: Path, judge: str, user: str) -> str:
    """Invoke ``eval bench`` in-process; return the captured log text."""
    from .cli import cmd_bench

    args = argparse.Namespace(
        playbook=suite.playbook,
        dataset=dataset,
        models=",".join(suite.models),
        modes="playbook",
        judge_model=judge,
        user_model=user,
        gen_model=judge,
        director_model=None,
        talker_model=None,
        personas=None,
        regen=False,
        n_probes=1,
        metrics=suite.metrics,
        repeats=suite.repeats,
        max_turns=suite.max_turns,
        out=str(out_dir),
    )
    tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee), _capture_engine_logs(tee):
        rc = cmd_bench(args)
    log = tee.buffer_.getvalue()
    if rc != 0:
        raise RuntimeError(f"bench exited {rc}")
    return log


def _model_dir(out_dir: Path, model: str) -> Path:
    return out_dir / re.sub(r"[/:]", "_", model)


def run_suite(
    suite: Suite,
    out_root: Path,
    *,
    tier: Literal["smoke", "full"] = "full",
    force: bool = False,
) -> SuiteResult:
    """Run one suite (with skip-if-unchanged and quota fallback); gate it."""
    out_dir = out_root / suite.name / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / ".content-hash"
    digest = _suite_hash(suite) + f":{tier}"
    if not force and stamp.exists() and stamp.read_text() == digest:
        return SuiteResult(
            name=suite.name,
            status="skipped",
            detail="inputs unchanged (--force to rerun)",
        )
    try:
        dataset = suite.dataset
        if tier == "smoke" and suite.smoke_cases:
            dataset = _subset_dataset(
                suite.dataset, suite.smoke_cases, out_dir / "smoke-dataset.yaml"
            )
            # Only gate the cases that actually run; the "present" check stays
            # meaningful in full tier (dataset drift), not against the subset.
            suite = suite.model_copy(
                update={
                    "expect": {
                        k: v for k, v in suite.expect.items() if k in suite.smoke_cases
                    }
                }
            )
        fallback_used = False
        try:
            log = _run_bench(suite, dataset, out_dir, suite.judge, suite.user_model)
        except Exception as exc:  # noqa: BLE001 — inspect for quota, else re-raise
            blob = _exc_chain_text(exc)
            if any(s in blob for s in _QUOTA_SIGNALS) and suite.fallback_judge:
                print(f"[suite:{suite.name}] quota hit — retrying via fallback models")
                fallback_used = True
                log = _run_bench(
                    suite,
                    dataset,
                    out_dir,
                    suite.fallback_judge,
                    suite.fallback_user or suite.fallback_judge,
                )
            else:
                raise
    except Exception as exc:  # noqa: BLE001 — a broken suite must not kill the rest
        return SuiteResult(name=suite.name, status="errored", detail=str(exc)[:300])
    (out_dir / "run.log").write_text(log, encoding="utf-8")
    report_path = _model_dir(out_dir, suite.models[0]) / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = evaluate_expectations(suite, report, log)
    if fallback_used:
        # Score floors were calibrated against the primary judge; under the
        # fallback judge they inform but do not gate.
        for c in checks:
            if not c.passed and (
                c.check.startswith("task_success>=")
                or c.check.startswith("composite>=")
            ):
                c.advisory = True
                c.detail += " [advisory: fallback judge]"
    composite = (report.get("modes") or [{}])[0].get("composite_mean")
    passed = all(c.passed or c.advisory for c in checks)
    # A fallback run never stamps: the suite re-verifies under the calibrated
    # judge once the primary provider recovers.
    if passed and not fallback_used:
        stamp.write_text(digest)
    return SuiteResult(
        name=suite.name,
        status="passed" if passed else "failed",
        checks=checks,
        composite=composite,
        detail="(fallback judge)" if fallback_used else "",
    )


def cmd_suite(args: argparse.Namespace) -> int:
    """``superdialog eval suite`` — run the registry, gate on expectations."""
    suites = load_suites(args.config)
    if args.only:
        suites = [s for s in suites if s.name == args.only]
        if not suites:
            print(f"no suite named {args.only!r} in {args.config}")
            return 2
    out_root = Path(args.out)
    results = [run_suite(s, out_root, tier=args.tier, force=args.force) for s in suites]
    print("\n================ SUITE SUMMARY ================")
    failed = False
    for r in results:
        comp = f" composite={r.composite:.3f}" if r.composite is not None else ""
        print(f"[{r.status.upper():7}] {r.name}{comp} {r.detail}")
        for c in r.checks:
            mark = "ok  " if c.passed else ("WARN" if c.advisory else "FAIL")
            print(f"    {mark} {c.case_id}: {c.check} ({c.detail})")
        failed = failed or r.status in ("failed", "errored")
    return 1 if failed else 0


def register(subparsers: Any) -> None:
    """Wire ``eval suite`` into the eval CLI."""
    p = subparsers.add_parser("suite", help="Run eval suites as behavioral gates")
    p.add_argument("--config", required=True, help="Suite registry YAML")
    p.add_argument("--tier", choices=["smoke", "full"], default="full")
    p.add_argument("--only", default=None, help="Run a single suite by name")
    p.add_argument("--force", action="store_true", help="Ignore content-hash skip")
    p.add_argument("--out", default="./eval-out/suites")
    p.set_defaults(fn=cmd_suite)


__all__ = [
    "CheckResult",
    "Suite",
    "SuiteExpect",
    "SuiteResult",
    "cmd_suite",
    "evaluate_expectations",
    "load_suites",
    "register",
    "run_suite",
    "split_log_by_case",
]
