"""Reflective prose optimizer: scoring, reflection, paired-round loop."""

from __future__ import annotations

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
_EVENT_LOG_CAP = 4000  # chars of event log shown per worst session
_JINJA_CHECKED_SUFFIXES = (".guidance", ".say_verbatim", ".say")

_REFLECT_RULES = """\
You improve conversational playbook prose. You will see the current playbook,
the exact list of editable field addresses, and evidence from the worst
self-play sessions.

Return ONLY a JSON array of edits: [{"address": "...", "new_text": "..."}].
Rules:
- Use only addresses from the EDITABLE FIELDS list, verbatim.
- new_text is a string (or a list of strings for never_say-style fields;
  never remove existing entries).
- Do not alter factual claims, prices, or hard boundaries anywhere.
- Propose at least one edit. No commentary, no markdown fences.
"""


def _worst_sessions(report: EvalReport, k: int = 3) -> list[SessionMetrics]:
    """The k weakest sessions: incomplete, then inaccurate, then repair-heavy."""
    ranked = sorted(
        report.sessions,
        key=lambda s: (s.completed, s.slot_accuracy, -s.repair_count),
    )
    return ranked[:k]


def _reflect_messages(
    doc: EditableDoc,
    report: EvalReport,
    k: int = 3,
    real_traces: "list[dict] | None" = None,
) -> list[dict[str, str]]:
    """Build the candidate-LLM prompt from the doc and the failing evidence."""
    fields_block = "\n".join(f"- {f.address}: {f.text!r}" for f in doc.fields())
    sessions: list[str] = []
    for s in _worst_sessions(report, k):
        sessions.append(
            f"persona={s.persona} completed={s.completed} outcome={s.outcome}\n"
            f"slot_diffs={s.slot_diffs} repair_count={s.repair_count}\n"
            f"turns_per_checkpoint={s.turns_per_checkpoint}\n"
            f"log:\n{s.event_log_jsonl[:_EVENT_LOG_CAP]}"
        )
    user = (
        f"PLAYBOOK:\n{doc.emit()}\n\n"
        f"EDITABLE FIELDS:\n{fields_block}\n\n"
        f"WORST SESSIONS:\n" + "\n---\n".join(sessions)
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
    messages = _reflect_messages(doc, report, real_traces=real_traces)
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
            incumbent, last_report, candidate_llm,
            max_attempts=reflect_attempts,
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
            inc_report = await _eval(incumbent)
            cand_report = await _eval(candidate)
            inc_b = score_report(inc_report)
            cand_b = score_report(cand_report)
            accepted = cand_b.objective > inc_b.objective
            t = RoundTrace(
                round_no=round_no,
                accepted=accepted,
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
