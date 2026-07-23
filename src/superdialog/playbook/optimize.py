"""Reflective prose optimizer: scoring, reflection, paired-round loop."""

from __future__ import annotations

import asyncio
import json
from math import sqrt
from statistics import stdev
from typing import Awaitable, Callable

from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, Field

from .agent import PlaybookAgent
from .director import CompletesLLM
from .editable import Edit, EditableDoc, make_editable
from .eval_bridge import (
    EvalReport,
    PersonaSpec,
    SessionMetrics,
    SpeaksUser,
    run_eval,
)
from .eval.scorer import ObjectiveBreakdown, score_report
from .events import EventLog
from .models import Playbook
from .replay import first_affected_version, replay

AgentFactory = Callable[[Playbook], PlaybookAgent]


_JINJA = Environment()
_EVENT_LOG_CAP = 6000  # chars of event log shown per worst session
_JINJA_CHECKED_SUFFIXES = (".guidance", ".say_verbatim", ".say")
_REGRESSION_FLOOR = (
    0.05  # max allowed drop per individual metric before candidate is rejected
)
_REJECTED_HISTORY_CAP = 3  # most-recent rejected proposals shown to the candidate LLM

_REFLECT_RULES = """\
You improve conversational voice-agent playbook prose.
You will see the current playbook YAML, every editable field address,
evidence from the WORST sessions (failures), and the BEST sessions (successes).

THINK (silently — do not output this reasoning):
1. Why did each failing session fail? (Slot not collected, premature goodbye,
   persona confused by language, repair loop, wrong branch taken?)
2. What are the successful sessions doing right? — protect that behaviour.
3. What is the single highest-impact change? Prioritise it.
4. Would changing guidance wording, a say template, or a repair step fix this?

Then return ONLY a JSON array of edits: [{"address": "...", "new_text": "..."}]

Hard rules:
- Use only addresses from the EDITABLE FIELDS list — verbatim, no invented paths.
- new_text must be a string, or a list of strings for never_say-style fields
  (only append to those lists; never remove existing entries).
- Never alter factual claims, prices, product names, regulatory language, or
  hard decision boundaries.
- Prefer editing guidance/say/instructions over restructuring the flow.
- If slot_accuracy was low → make the extraction cue more explicit in guidance.
- If repair_count was high → strengthen the re-prompt or add clarification language.
- If a session ended prematurely → soften or remove any early-exit condition.
- Propose 1–4 focused, high-quality edits. Quality beats quantity.
- No commentary, no markdown fences, no explanation — only the JSON array.
"""


def _worst_sessions(report: EvalReport, k: int = 3) -> list[SessionMetrics]:
    """The k weakest sessions: incomplete first, then inaccurate, then repair-heavy."""
    ranked = sorted(
        report.sessions,
        key=lambda s: (s.completed, s.slot_accuracy, -s.repair_count),
    )
    return ranked[:k]


def _best_sessions(report: EvalReport, k: int = 2) -> list[SessionMetrics]:
    """The k strongest sessions — show LLM what to protect."""
    ranked = sorted(
        report.sessions,
        key=lambda s: (s.completed, s.slot_accuracy, -s.repair_count),
        reverse=True,
    )
    return ranked[:k]


def _format_session(s: SessionMetrics, cap: int = _EVENT_LOG_CAP) -> str:
    return (
        f"persona={s.persona} completed={s.completed} outcome={s.outcome}\n"
        f"slot_diffs={s.slot_diffs} repair_count={s.repair_count}\n"
        f"turns_per_checkpoint={s.turns_per_checkpoint}\n"
        f"log:\n{s.event_log_jsonl[:cap]}"
    )


def _no_regression(
    inc: "ObjectiveBreakdown", cand: "ObjectiveBreakdown"
) -> tuple[bool, str]:
    """True when no individual metric dropped more than _REGRESSION_FLOOR.

    Returns (ok, reason) — reason is empty when ok=True.
    """
    if cand.completion_rate < inc.completion_rate - _REGRESSION_FLOOR:
        return False, (
            f"completion_rate regressed {inc.completion_rate:.3f}→{cand.completion_rate:.3f}"
        )
    if cand.slot_accuracy < inc.slot_accuracy - _REGRESSION_FLOOR:
        return False, (
            f"slot_accuracy regressed {inc.slot_accuracy:.3f}→{cand.slot_accuracy:.3f}"
        )
    # Higher turns/checkpoint = worse smoothness; allow at most _REGRESSION_FLOOR extra turns.
    # Skip when incumbent had no completed sessions (mean_tpc=0 means no baseline, not "0 turns").
    if (
        inc.mean_turns_per_checkpoint > 0
        and cand.mean_turns_per_checkpoint
        > inc.mean_turns_per_checkpoint + _REGRESSION_FLOOR * 10
    ):
        return False, (
            f"mean_turns regressed {inc.mean_turns_per_checkpoint:.2f}→{cand.mean_turns_per_checkpoint:.2f}"
        )
    return True, ""


def _objective_samples(report: EvalReport) -> list[float]:
    """Per-session objective values — the raw material for noise estimation.

    ``ObjectiveBreakdown`` only carries the aggregate, not a variance term,
    so each session is scored in isolation via ``score_report`` itself
    (rather than duplicating its formula) — the per-session value can never
    drift out of sync with the aggregate.
    """
    return [score_report(EvalReport(sessions=[s])).objective for s in report.sessions]


def _noise_margin(
    inc_report: EvalReport, cand_report: EvalReport, *, z: float = 1.0
) -> float:
    """Minimum objective improvement needed to clear estimated sampling noise.

    Returns 0.0 (falls back to the plain point-estimate compare) when either
    side has fewer than 2 sessions — variance can't be estimated from a
    single sample. With >=2 sessions per side, the standard error of the
    difference is estimated from the per-session objective spread, and the
    candidate must beat the incumbent by more than ``z`` standard errors
    before the win counts as real rather than sampling noise.
    """
    inc_samples = _objective_samples(inc_report)
    cand_samples = _objective_samples(cand_report)
    if len(inc_samples) < 2 or len(cand_samples) < 2:
        return 0.0
    se_inc = stdev(inc_samples) / sqrt(len(inc_samples))
    se_cand = stdev(cand_samples) / sqrt(len(cand_samples))
    return z * sqrt(se_inc**2 + se_cand**2)


def _clears_acceptance_bar(
    inc_b: "ObjectiveBreakdown",
    cand_b: "ObjectiveBreakdown",
    inc_report: EvalReport,
    cand_report: EvalReport,
) -> bool:
    """True when the candidate beats the incumbent by more than the
    estimated sampling-noise margin, with no individual metric regression."""
    margin = _noise_margin(inc_report, cand_report)
    no_reg, _ = _no_regression(inc_b, cand_b)
    return cand_b.objective > inc_b.objective + margin and no_reg


async def cro_guard(
    candidate: EditableDoc,
    edits: list[Edit],
    report: EvalReport,
    director_llm: CompletesLLM,
    *,
    k: int = 2,
) -> tuple[bool, str]:
    """CRO gate (Shepherd §5.2): suffix-replay the candidate over guard logs.

    The guard set is the incumbent's best completed sessions. Each log is
    forked at ``first_affected_version`` and only that suffix is re-judged —
    turns the edits cannot touch are held constant, and a session whose path
    never visits an edited checkpoint costs zero LLM calls. A candidate that
    flips Director decisions on a previously-good session is rejected before
    paying for a full paired eval.
    """
    guards = [
        s for s in _best_sessions(report, k=k) if s.completed and s.event_log_jsonl
    ]
    if not guards:
        return True, ""
    playbook = candidate.compile()
    for s in guards:
        log = EventLog.from_jsonl(s.event_log_jsonl)
        start = first_affected_version(log, playbook, edits)
        if start is None:
            continue  # edits never touch this session's path
        rep = await replay(log, playbook, director_llm, from_version=start)
        if not rep.stable:
            d = rep.diffs[0]
            return False, (
                f"cro-guard: persona={s.persona} destabilized at v{d.at_version} "
                f"({d.kind}: {d.recorded!r} -> {d.replayed!r})"
            )
    return True, ""


def _reflect_messages(
    doc: EditableDoc,
    report: EvalReport,
    k: int = 3,
    golden_transcript: str | None = None,
    real_traces: "list[dict] | None" = None,
    rejected: "list[tuple[list[Edit], str]] | None" = None,
) -> list[dict[str, str]]:
    """Build the candidate-LLM prompt from the doc and session evidence."""
    fields_block = "\n".join(f"- {f.address}: {f.text!r}" for f in doc.fields())
    worst_block = "\n---\n".join(_format_session(s) for s in _worst_sessions(report, k))
    best_sessions = _best_sessions(report, k=2)
    # Only include best-session block when there are genuinely good runs to contrast
    if best_sessions and best_sessions[0].completed:
        best_block = "\n---\n".join(
            _format_session(s, cap=_EVENT_LOG_CAP // 2) for s in best_sessions
        )
        best_section = f"\n\nBEST SESSIONS (protect what works):\n{best_block}"
    else:
        best_section = ""
    golden_section = (
        f"\n\nGOLDEN TRANSCRIPT (a hand-verified successful call — emulate this tone and flow):\n{golden_transcript}"
        if golden_transcript
        else ""
    )
    if rejected:
        rejected_lines = "\n".join(
            f"- {e.address} -> {e.new_text!r} (rejected: {reason})"
            for edits, reason in rejected
            for e in edits
        )
        rejected_section = (
            "\n\nALREADY TRIED AND REJECTED (do not repeat these ideas):\n"
            f"{rejected_lines}"
        )
    else:
        rejected_section = ""
    user = (
        f"PLAYBOOK:\n{doc.emit()}\n\n"
        f"EDITABLE FIELDS:\n{fields_block}\n\n"
        f"WORST SESSIONS (these need to improve):\n{worst_block}"
        f"{best_section}"
        f"{golden_section}"
        f"{rejected_section}"
    )
    if real_traces:
        try:
            # Import lazily — supervoice may not be on the path in pure superdialog tests.
            from playground.harness.langfuse_fetch import summarise_traces  # type: ignore[import]

            real_block = summarise_traces(real_traces)
        except ImportError:
            real_block = f"(real_traces available: {len(real_traces)} calls)"
        user += (
            "\n\nREAL CALL DATA (production calls — these failure patterns take priority "
            "over synthetic session results; ensure your edits address them first):\n"
            + real_block
        )
    return [
        {"role": "system", "content": _REFLECT_RULES},
        {"role": "user", "content": user},
    ]


def _parse_edits(raw: str) -> list[Edit]:
    """Parse the candidate's JSON edit array; raise ValueError when malformed."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    if not isinstance(data, list) or not data:
        raise ValueError("expected a non-empty JSON array of edits")
    return [Edit.model_validate(item) for item in data]


def _check_jinja(edits: list[Edit]) -> None:
    """Syntax-parse template-bearing edits; broken Jinja fails at runtime."""
    for edit in edits:
        if edit.address.endswith(_JINJA_CHECKED_SUFFIXES) and isinstance(
            edit.new_text, str
        ):
            try:
                _JINJA.parse(edit.new_text)
            except TemplateSyntaxError as exc:
                raise ValueError(f"{edit.address}: broken Jinja: {exc}") from exc


class RoundTrace(BaseModel):
    """One optimization round: same-round paired scores plus the edit list."""

    round_no: int
    accepted: bool
    incumbent_breakdown: ObjectiveBreakdown
    candidate_breakdown: ObjectiveBreakdown | None = None
    edits: list[Edit] = Field(default_factory=list)
    detail: str = ""
    # Evaluated candidate's emitted YAML, set whenever candidate_breakdown
    # is — lets a good-but-rejected round be re-validated and promoted
    # later (see _reconcile_with_frontier).
    candidate_yaml: str | None = None


class ParetoFrontier(BaseModel):
    """Non-dominated candidate rounds over completion/slot/smoothness.

    The round loop never picks its output from the frontier directly
    (cross-round scores come from different eval runs, so a stale score
    can't be trusted on its own) — but after the loop ends,
    ``_reconcile_with_frontier`` spends one fresh paired re-eval to check
    whether the frontier's best-scoring member actually beats the final
    incumbent under current, noise-safe scrutiny before promoting it.
    """

    members: list[RoundTrace] = Field(default_factory=list)

    @staticmethod
    def _vector(t: RoundTrace) -> tuple[float, float, float]:
        b = t.candidate_breakdown
        assert b is not None
        return (
            b.completion_rate,
            b.slot_accuracy,
            1.0 / (1.0 + max(0.0, b.mean_turns_per_checkpoint - 1.0)),
        )

    @classmethod
    def _dominates(cls, a: RoundTrace, b: RoundTrace) -> bool:
        va, vb = cls._vector(a), cls._vector(b)
        return all(x >= y for x, y in zip(va, vb)) and va != vb

    def consider(self, t: RoundTrace) -> None:
        """Add `t` unless dominated; evict members it dominates."""
        if t.candidate_breakdown is None:
            return
        if any(self._dominates(m, t) for m in self.members):
            return
        self.members = [m for m in self.members if not self._dominates(t, m)]
        self.members.append(t)


async def _reconcile_with_frontier(
    incumbent: EditableDoc,
    final_b: ObjectiveBreakdown,
    frontier: ParetoFrontier,
    eval_fn: Callable[[EditableDoc], Awaitable[EvalReport]],
) -> tuple[EditableDoc, ObjectiveBreakdown, RoundTrace | None]:
    """Give the Pareto frontier one chance to override the final incumbent.

    A round can be rejected (e.g. by the regression floor) while still
    scoring higher than what the loop ends up keeping — that candidate
    lives on in the frontier but was otherwise gone for good. Historical
    frontier scores come from a different eval run than the final
    incumbent's, so promoting one on that score alone would reintroduce
    the exact cross-round noise problem same-round pairing exists to
    avoid. Instead: only bother when the frontier's best-scoring member
    beats the final incumbent's objective, then spend ONE fresh paired
    eval — both evaluated now, together — before promoting.

    Returns (incumbent, final_b, trace_entry) — unchanged with a `None`
    trace entry when there was nothing worth checking.
    """
    candidates = [t for t in frontier.members if t.candidate_yaml is not None]
    if not candidates:
        return incumbent, final_b, None
    best = max(candidates, key=lambda t: t.candidate_breakdown.objective)
    if best.candidate_breakdown.objective <= final_b.objective:
        return incumbent, final_b, None

    frontier_doc = make_editable(best.candidate_yaml)
    inc_report, cand_report = await asyncio.gather(
        eval_fn(incumbent), eval_fn(frontier_doc)
    )
    inc_b = score_report(inc_report)
    cand_b = score_report(cand_report)
    promoted = _clears_acceptance_bar(inc_b, cand_b, inc_report, cand_report)
    trace_entry = RoundTrace(
        round_no=-1,  # sentinel: post-loop reconciliation, not a regular round
        accepted=promoted,
        incumbent_breakdown=inc_b,
        candidate_breakdown=cand_b,
        edits=best.edits,
        detail=(
            "promoted from Pareto frontier after re-validation"
            if promoted
            else "frontier candidate did not clear re-validation against current incumbent"
        ),
    )
    if promoted:
        return frontier_doc, cand_b, trace_entry
    return incumbent, final_b, trace_entry


async def propose_edits(
    doc: EditableDoc,
    report: EvalReport,
    candidate_llm: CompletesLLM,
    *,
    max_attempts: int = 3,
    golden_transcript: str | None = None,
    real_traces: "list[dict] | None" = None,
    rejected: "list[tuple[list[Edit], str]] | None" = None,
) -> tuple[EditableDoc, list[Edit]] | None:
    """Ask the candidate LLM for prose edits; validate; retry; None on failure.

    The candidate output is untrusted text: it is parsed and validated, never
    executed. ValidationError, MutationError and JSONDecodeError are all
    ValueError subclasses, so one except clause covers every reject path.

    A failed attempt's error is fed back into the conversation before
    retrying, so a retry is a real correction rather than the same prompt
    resent verbatim in hope of a different roll.

    ``real_traces``: structured Langfuse trace dicts from ``langfuse_fetch``.
    When provided they are injected into the reflection prompt so the LLM
    can target real production failure patterns, not just synthetic ones.

    ``rejected``: (edits, reason) pairs from earlier rounds in the same
    optimize() run that were proposed and then rejected — shown to the LLM
    so it doesn't keep re-proposing an idea that already failed.
    """
    messages = _reflect_messages(
        doc,
        report,
        golden_transcript=golden_transcript,
        real_traces=real_traces,
        rejected=rejected,
    )
    for _ in range(max_attempts):
        raw = await candidate_llm.complete(messages)
        try:
            edits = _parse_edits(raw)
            _check_jinja(edits)
            candidate = doc.apply(edits)  # whitelist + recompile validation
        except ValueError as exc:
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"That output was rejected: {exc}. Return ONLY a "
                        "corrected JSON array of edits — same rules as "
                        "before, no commentary."
                    ),
                },
            ]
            continue
        return candidate, edits
    return None


class OptimizeReport(BaseModel):
    """The optimize run's result: final artifact plus the full metric trace."""

    final_yaml: str
    initial_breakdown: ObjectiveBreakdown
    final_breakdown: ObjectiveBreakdown
    trace: list[RoundTrace]
    frontier: list[RoundTrace]


async def optimize(
    doc: EditableDoc,
    *,
    personas: list[PersonaSpec],
    candidate_llm: CompletesLLM,
    user_llm: SpeaksUser,
    agent_factory: AgentFactory,
    rounds: int = 3,
    n: int = 1,
    candidates_per_round: int = 1,
    patience: int = 2,
    reflect_attempts: int = 3,
    golden_transcript: str | None = None,
    real_traces: "list[dict] | None" = None,
    director_llm: CompletesLLM | None = None,
) -> OptimizeReport:
    """Paired-round reflective optimization. Returns the final incumbent.

    Acceptance compares only same-round scores: each round evaluates the
    incumbent AND every surviving candidate fresh, so all face the same
    sampling noise. The Pareto frontier is reported but never picks the
    output.

    ``candidates_per_round`` samples that many independent edit proposals
    each round (default 1: identical to a single-candidate round). Duplicate
    proposals (identical edit sets — e.g. a deterministic candidate_llm) are
    deduped before spending any eval budget. The incumbent is evaluated once
    per round regardless of ``candidates_per_round`` and shared across all
    surviving candidates in that round, rather than re-evaluated per
    candidate. Of the candidates that pass the gate and the accept criteria
    (objective improved, no individual metric regressed), the
    highest-objective one becomes the next incumbent. Diversity across the
    k samples relies on the candidate LLM's own sampling variance — a fully
    deterministic LLM (temperature 0) will produce k identical proposals,
    which then collapse to one after dedup.

    ``director_llm`` enables the CRO guard gate: candidates are suffix-replayed
    over the incumbent's best recorded sessions and rejected (no paired eval
    spent) when they destabilize a previously-good session. Pass the same
    model the agent's Director runs on.
    """

    async def _eval(d: EditableDoc) -> EvalReport:
        playbook = d.compile()
        return await run_eval(lambda: agent_factory(playbook), personas, user_llm, n)

    async def _gate(
        candidate: EditableDoc, edits: list[Edit], report: EvalReport
    ) -> tuple[bool, str]:
        """CRO guard when a director_llm is provided; open gate otherwise."""
        if director_llm is None:
            return True, ""
        return await cro_guard(candidate, edits, report, director_llm)

    incumbent = doc
    last_report = await _eval(incumbent)
    initial_b = score_report(last_report)
    final_b = initial_b
    frontier = ParetoFrontier()
    trace: list[RoundTrace] = []
    rejected_history: list[tuple[list[Edit], str]] = []
    stale = 0
    for round_no in range(1, rounds + 1):
        proposals = await asyncio.gather(
            *[
                propose_edits(
                    incumbent,
                    last_report,
                    candidate_llm,
                    max_attempts=reflect_attempts,
                    golden_transcript=golden_transcript,
                    real_traces=real_traces,
                    rejected=rejected_history[-_REJECTED_HISTORY_CAP:] or None,
                )
                for _ in range(candidates_per_round)
            ]
        )
        seen: set[tuple[tuple[str, str], ...]] = set()
        unique: list[tuple[EditableDoc, list[Edit]]] = []
        for proposal in proposals:
            if proposal is None:
                continue
            cand_doc, edits = proposal
            key = tuple((e.address, str(e.new_text)) for e in edits)
            if key in seen:
                continue
            seen.add(key)
            unique.append((cand_doc, edits))

        if not unique:
            trace.append(
                RoundTrace(
                    round_no=round_no,
                    accepted=False,
                    incumbent_breakdown=final_b,
                    detail="no valid candidate",
                )
            )
            # Not counted toward `stale`: a parse/validation failure is a
            # cheap formatting miss (already retried reflect_attempts times,
            # zero eval cost) — not a substantive "this idea didn't work"
            # signal. `rounds` still caps the loop regardless.
        else:
            gate_results = await asyncio.gather(
                *[_gate(cand, edits, last_report) for cand, edits in unique]
            )
            for (_, edits), (ok, reason) in zip(unique, gate_results):
                if not ok:
                    rejected_history.append((edits, reason))
            gated = [ce for ce, (ok, _) in zip(unique, gate_results) if ok]

            if not gated:
                first_reason = next(r for ok, r in gate_results if not ok)
                trace.append(
                    RoundTrace(
                        round_no=round_no,
                        accepted=False,
                        incumbent_breakdown=final_b,
                        edits=unique[0][1],
                        detail=first_reason,
                    )
                )
                stale += 1
            else:
                inc_report, *cand_reports = await asyncio.gather(
                    _eval(incumbent), *[_eval(cand) for cand, _ in gated]
                )
                inc_b = score_report(inc_report)
                scored = [
                    (cand, edits, score_report(rep), rep)
                    for (cand, edits), rep in zip(gated, cand_reports)
                ]
                passing = [
                    s
                    for s in scored
                    if _clears_acceptance_bar(inc_b, s[2], inc_report, s[3])
                ]
                if passing:
                    candidate, edits, cand_b, cand_report = max(
                        passing, key=lambda s: s[2].objective
                    )
                    t = RoundTrace(
                        round_no=round_no,
                        accepted=True,
                        incumbent_breakdown=inc_b,
                        candidate_breakdown=cand_b,
                        edits=edits,
                        candidate_yaml=candidate.emit(),
                    )
                    trace.append(t)
                    frontier.consider(t)
                    incumbent, last_report, final_b = candidate, cand_report, cand_b
                    stale = 0
                else:
                    best_cand, best_edits, best_b, _ = max(
                        scored, key=lambda s: s[2].objective
                    )
                    _, reg_reason = _no_regression(inc_b, best_b)
                    t = RoundTrace(
                        round_no=round_no,
                        accepted=False,
                        incumbent_breakdown=inc_b,
                        candidate_breakdown=best_b,
                        edits=best_edits,
                        detail=reg_reason,
                        candidate_yaml=best_cand.emit(),
                    )
                    trace.append(t)
                    frontier.consider(t)
                    for _, edits, b, _ in scored:
                        _, reason = _no_regression(inc_b, b)
                        rejected_history.append(
                            (edits, reason or "objective did not improve")
                        )
                    last_report, final_b = inc_report, inc_b
                    stale += 1
        if stale >= patience:
            break
    incumbent, final_b, reconcile_trace = await _reconcile_with_frontier(
        incumbent, final_b, frontier, _eval
    )
    if reconcile_trace is not None:
        trace.append(reconcile_trace)
    return OptimizeReport(
        final_yaml=incumbent.emit(),
        initial_breakdown=initial_b,
        final_breakdown=final_b,
        trace=trace,
        frontier=frontier.members,
    )
