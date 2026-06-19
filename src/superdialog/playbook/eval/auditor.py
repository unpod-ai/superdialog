# src/superdialog/playbook/eval/auditor.py
"""SessionAuditor: post-session three-layer analysis."""

from __future__ import annotations

import json

from .models import AuditReport
from ..agent import PlaybookAgent
from ..director import CompletesLLM
from ..events import AdvanceEvent, UtteranceEvent
from ..models import Playbook

_QUALITY_SYSTEM = """\
You score a conversational agent's single utterance for quality.
Return ONLY a JSON object: {"score": N} where N is 1 (very poor) to 5 (excellent).
Criteria: natural tone, relevance to conversation, no routing leaks, no confusion.
"""

W_PATH = 0.3
W_SLOT = 0.3
W_QUALITY = 0.4


def _required_slots(playbook: Playbook) -> list[str]:
    slots: list[str] = []
    for journey in playbook.journeys.values():
        for cp in journey.checkpoints:
            for key, spec in cp.slots.items():
                if spec.required and key not in slots:
                    slots.append(key)
    return slots


def _known_checkpoint_ids(playbook: Playbook) -> set[str]:
    ids: set[str] = set()
    for jname, journey in playbook.journeys.items():
        for cp in journey.checkpoints:
            ids.add(f"{jname}.{cp.id}")
    return ids


class SessionAuditor:
    """Three-layer post-session auditor.

    Layer 1 — path validity: all AdvanceEvent targets exist in the playbook.
    Layer 2 — slot completeness: required slots filled by session end.
    Layer 3 — response quality: LLM judge scores each assistant utterance 1-5.
    """

    def __init__(self, playbook: Playbook, judge_llm: CompletesLLM) -> None:
        self._playbook = playbook
        self._judge = judge_llm
        self._known_ids = _known_checkpoint_ids(playbook)
        self._required = _required_slots(playbook)

    async def audit(
        self, agent: PlaybookAgent, session_id: str = ""
    ) -> AuditReport:
        log = agent.runtime.log
        state = agent.runtime.state

        # Layer 1 — path validity
        path: list[str] = []
        violations: list[str] = []
        for e in log.events:
            if isinstance(e, AdvanceEvent):
                path.append(e.to_checkpoint)
                if e.to_checkpoint not in self._known_ids:
                    violations.append(
                        f"unknown checkpoint '{e.to_checkpoint}' in AdvanceEvent"
                    )
        path_valid = len(violations) == 0

        # Layer 2 — slot completeness
        coverage: dict[str, bool] = {}
        for key in self._required:
            coverage[key] = state.slot_value(key) is not None
        filled = sum(1 for v in coverage.values() if v)
        slot_completeness = filled / len(coverage) if coverage else 1.0

        # Layer 3 — response quality
        assistant_utts = [
            e for e in log.events
            if isinstance(e, UtteranceEvent) and e.role == "assistant"
        ]
        quality_scores: list[float] = []
        transcript_so_far: list[dict[str, str]] = []
        for e in log.events:
            if not isinstance(e, UtteranceEvent):
                continue
            if e.role != "system":
                if e.role == "assistant" and e in assistant_utts:
                    score = await self._score_utterance(
                        transcript_so_far, e.text
                    )
                    quality_scores.append(score)
                transcript_so_far.append({"role": e.role, "content": e.text})

        response_quality = (
            sum(quality_scores) / len(quality_scores) / 5.0
            if quality_scores
            else 0.0
        )

        path_score = 1.0 if path_valid else 0.0
        overall_score = (
            W_PATH * path_score
            + W_SLOT * slot_completeness
            + W_QUALITY * response_quality
        )

        critical: list[str] = []
        if not path_valid:
            critical.extend(violations)
        if slot_completeness < 0.5:
            critical.append(
                f"low slot completeness: {slot_completeness:.0%} of required slots captured"
            )

        return AuditReport(
            session_id=session_id,
            checkpoint_path=path,
            path_valid=path_valid,
            path_violations=violations,
            slot_coverage=coverage,
            slot_completeness=slot_completeness,
            response_quality=response_quality,
            overall_score=overall_score,
            critical_issues=critical,
        )

    async def _score_utterance(
        self, context: list[dict[str, str]], utterance: str
    ) -> float:
        messages = [
            {"role": "system", "content": _QUALITY_SYSTEM},
            *context[-6:],
            {"role": "user", "content": f"Score this agent utterance: {utterance!r}"},
        ]
        raw = await self._judge.complete(messages)
        try:
            data = json.loads(raw.strip())
            score = float(data.get("score", 3))
            return max(1.0, min(5.0, score))
        except (ValueError, KeyError):
            return 3.0


__all__ = ["SessionAuditor"]