"""Reflective prose optimizer: scoring, reflection, paired-round loop."""

from __future__ import annotations

import json
from statistics import mean

from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel

from .director import CompletesLLM
from .editable import Edit, EditableDoc
from .eval_bridge import EvalReport, SessionMetrics

W_COMPLETION = 0.4
W_SLOT = 0.3
W_SMOOTHNESS = 0.2
W_REPAIR = 0.1


class ObjectiveBreakdown(BaseModel):
    """Scalar objective plus its per-dimension breakdown."""

    objective: float
    completion_rate: float
    slot_accuracy: float
    mean_turns_per_checkpoint: float
    repair_rate: float


def _smoothness(mean_turns_per_checkpoint: float) -> float:
    """Map mean turns/checkpoint to [0, 1]; 1 turn -> 1.0, more -> less."""
    return 1.0 / (1.0 + max(0.0, mean_turns_per_checkpoint - 1.0))


def score_report(report: EvalReport) -> ObjectiveBreakdown:
    """Score an eval report. Pure: no LLM, no I/O.

    Smoothness is averaged over completed sessions only, so fail-fast
    incomplete sessions cannot game the mean (they pay via completion).
    """
    if not report.sessions:
        return ObjectiveBreakdown(
            objective=0.0,
            completion_rate=0.0,
            slot_accuracy=0.0,
            mean_turns_per_checkpoint=0.0,
            repair_rate=0.0,
        )
    per_completed = [
        mean(s.turns_per_checkpoint.values())
        for s in report.sessions
        if s.completed and s.turns_per_checkpoint
    ]
    mean_tpc = mean(per_completed) if per_completed else 0.0
    total_turns = sum(s.turns for s in report.sessions)
    total_repairs = sum(s.repair_count for s in report.sessions)
    repair_rate = total_repairs / total_turns if total_turns else 0.0
    smooth = _smoothness(mean_tpc) if per_completed else 0.0
    objective = (
        W_COMPLETION * report.completion_rate
        + W_SLOT * report.mean_slot_accuracy
        + W_SMOOTHNESS * smooth
        + W_REPAIR * (1.0 - min(1.0, repair_rate))
    )
    return ObjectiveBreakdown(
        objective=objective,
        completion_rate=report.completion_rate,
        slot_accuracy=report.mean_slot_accuracy,
        mean_turns_per_checkpoint=mean_tpc,
        repair_rate=repair_rate,
    )


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
    doc: EditableDoc, report: EvalReport, k: int = 3
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


async def propose_edits(
    doc: EditableDoc,
    report: EvalReport,
    candidate_llm: CompletesLLM,
    *,
    max_attempts: int = 3,
) -> tuple[EditableDoc, list[Edit]] | None:
    """Ask the candidate LLM for prose edits; validate; retry; None on failure.

    The candidate output is untrusted text: it is parsed and validated, never
    executed. ValidationError, MutationError and JSONDecodeError are all
    ValueError subclasses, so one except clause covers every reject path.
    """
    messages = _reflect_messages(doc, report)
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
