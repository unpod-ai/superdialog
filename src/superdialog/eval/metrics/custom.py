"""Custom transcript-based LLM-judge metrics (mode-blind, ground-truth aware)."""

from __future__ import annotations

import json
from typing import Any

from superdialog.eval.dataset.models import EvalSample
from superdialog.eval.metrics.base import MetricResult
from superdialog.playbook.eval.runner import SpeaksUser


async def _judge_json(llm: SpeaksUser, prompt: str) -> dict[str, Any]:
    """Ask the judge for a JSON verdict; raise on unparseable output."""
    raw = await llm.complete([{"role": "user", "content": prompt}])
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge output: {raw[:120]!r}")
    return json.loads(raw[start : end + 1])


def _convo_text(sample: EvalSample) -> str:
    if isinstance(sample.user_input, list):
        return "\n".join(f"{m['role']}: {m['content']}" for m in sample.user_input)
    return str(sample.user_input)


class TaskSuccessJudge:
    """Did the conversation achieve the journey goal? Graded 0-1 from transcript."""

    name = "task_success"
    applies_to = ("conversation",)

    def __init__(self, judge_llm: SpeaksUser) -> None:
        self._llm = judge_llm

    async def score(self, sample: EvalSample) -> MetricResult:
        prompt = (
            "You are grading a goal-oriented assistant conversation.\n"
            f"GOAL: {sample.reference}\n"
            f"CONVERSATION:\n{_convo_text(sample)}\n\n"
            "Did the assistant achieve the goal? Reply with JSON: "
            '{"completed": bool, "graded": 0.0-1.0, "reason": str}.'
        )
        try:
            v = await _judge_json(self._llm, prompt)
            return MetricResult(
                name=self.name,
                value=float(v["graded"]),
                passed=bool(v["completed"]),
                reason=str(v.get("reason", "")),
            )
        except Exception as exc:
            return MetricResult(
                name=self.name,
                value=None,
                errored=True,
                reason=f"judge parse error: {exc}",
            )
