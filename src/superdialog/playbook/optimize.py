"""Reflective prose optimizer: scoring, reflection, paired-round loop."""

from __future__ import annotations

import asyncio
import json
from typing import Callable

from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, Field

from .agent import PlaybookAgent
from .director import CompletesLLM
from .editable import Edit, EditableDoc
from .eval_bridge import (
    EvalReport,
    PersonaSpec,
    SessionMetrics,
    SpeaksUser,
    run_eval,
)
from .eval.scorer import ObjectiveBreakdown, score_report
from .models import Playbook

AgentFactory = Callable[[Playbook], PlaybookAgent]


_JINJA = Environment()
_EVENT_LOG_CAP = 6000  # chars of event log shown per worst session
_JINJA_CHECKED_SUFFIXES = (".guidance", ".say_verbatim", ".say")
_REGRESSION_FLOOR = 0.05  # max allowed drop per individual metric before candidate is rejected

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
>>>>>>> 5ffce79 (feat(playbook): parallel eval runner + robust optimize loop)
        )
    if cand.slot_accuracy < inc.slot_accuracy - _REGRESSION_FLOOR:
        return False, (
            f"slot_accuracy regressed {inc.slot_accuracy:.3f}→{cand.slot_accuracy:.3f}"
        )
    # Higher turns/checkpoint = worse smoothness; allow at most _REGRESSION_FLOOR extra turns.
    # Skip when incumbent had no completed sessions (mean_tpc=0 means no baseline, not "0 turns").
    if (
        inc.mean_turns_per_checkpoint > 0
        and cand.mean_turns_per_checkpoint > inc.mean_turns_per_checkpoint + _REGRESSION_FLOOR * 10
    ):
        return False, (
            f"mean_turns regressed {inc.mean_turns_per_checkpoint:.2f}→{cand.mean_turns_per_checkpoint:.2f}"
        )
    return True, ""


def _reflect_messages(
    doc: EditableDoc,
    report: EvalReport,
    k: int = 3,
    golden_transcript: str | None = None,
    real_traces: "list[dict] | None" = None,
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
    user = (
        f"PLAYBOOK:\n{doc.emit()}\n\n"
        f"EDITABLE FIELDS:\n{fields_block}\n\n"
        f"WORST SESSIONS (these need to improve):\n{worst_block}"
        f"{best_section}"
        f"{golden_section}"
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


class ParetoFrontier(BaseModel):
    """Non-dominated candidate rounds over completion/slot/smoothness.

    Informational only: the loop never picks its output from the frontier
    (cross-round scores come from different eval runs).
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


async def propose_edits(
    doc: EditableDoc,
    report: EvalReport,
    candidate_llm: CompletesLLM,
    *,
    max_attempts: int = 3,
    golden_transcript: str | None = None,
    real_traces: "list[dict] | None" = None,
) -> tuple[EditableDoc, list[Edit]] | None:
    """Ask the candidate LLM for prose edits; validate; retry; None on failure.

    The candidate output is untrusted text: it is parsed and validated, never
    executed. ValidationError, MutationError and JSONDecodeError are all
    ValueError subclasses, so one except clause covers every reject path.

    ``real_traces``: structured Langfuse trace dicts from ``langfuse_fetch``.
    When provided they are injected into the reflection prompt so the LLM
    can target real production failure patterns, not just synthetic ones.
    """
    messages = _reflect_messages(
        doc, report, golden_transcript=golden_transcript, real_traces=real_traces
    )
    for _ in range(max_attempts):
        raw = await candidate_llm.complete(messages)
        try:
            edits = _parse_edits(raw)
            _check_jinja(edits)
            candidate = doc.apply(edits)  # whitelist + recompile validation
        except ValueError:
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
    patience: int = 2,
    reflect_attempts: int = 3,
    golden_transcript: str | None = None,
    real_traces: "list[dict] | None" = None,
) -> OptimizeReport:
    """Paired-round reflective optimization. Returns the final incumbent.

    Acceptance compares only same-round scores: each round evaluates the
    incumbent AND the candidate fresh, so both face the same sampling noise.
    The Pareto frontier is reported but never picks the output.
    """

    async def _eval(d: EditableDoc) -> EvalReport:
        playbook = d.compile()
        return await run_eval(lambda: agent_factory(playbook), personas, user_llm, n)

    incumbent = doc
    last_report = await _eval(incumbent)
    initial_b = score_report(last_report)
    final_b = initial_b
    frontier = ParetoFrontier()
    trace: list[RoundTrace] = []
    stale = 0
    for round_no in range(1, rounds + 1):
        proposal = await propose_edits(
            incumbent,
            last_report,
            candidate_llm,
            max_attempts=reflect_attempts,
            golden_transcript=golden_transcript,
            real_traces=real_traces,
        )
        if proposal is None:
            trace.append(
                RoundTrace(
                    round_no=round_no,
                    accepted=False,
                    incumbent_breakdown=final_b,
                    detail="no valid candidate",
                )
            )
            stale += 1
        else:
            candidate, edits = proposal
            inc_report, cand_report = await asyncio.gather(
                _eval(incumbent), _eval(candidate)
            )
            inc_b = score_report(inc_report)
            cand_b = score_report(cand_report)
            no_reg, reg_reason = _no_regression(inc_b, cand_b)
            accepted = cand_b.objective > inc_b.objective and no_reg
            t = RoundTrace(
                round_no=round_no,
                accepted=accepted,
                detail=reg_reason if not no_reg else "",
                incumbent_breakdown=inc_b,
                candidate_breakdown=cand_b,
                edits=edits,
            )
            trace.append(t)
            frontier.consider(t)
            if accepted:
                incumbent, last_report, final_b = candidate, cand_report, cand_b
                stale = 0
            else:
                last_report, final_b = inc_report, inc_b
                stale += 1
        if stale >= patience:
            break
    return OptimizeReport(
        final_yaml=incumbent.emit(),
        initial_breakdown=initial_b,
        final_breakdown=final_b,
        trace=trace,
        frontier=frontier.members,
    )
