import json

import pytest

from superdialog.playbook.eval.scenario_metrics import (
    constraint_adherence,
    pii_violation_rate,
)


class _FakeJudge:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    async def complete(self, messages, **kwargs) -> str:
        self.prompts.append(messages[-1]["content"])
        return self._reply


async def test_constraint_adherence_passes_when_judge_says_obeyed():
    judge = _FakeJudge(json.dumps({"violated": False, "reason": "never mentioned pricing"}))
    utterances = ["I can help you book a slot.", "Your green fee is set once confirmed."]
    result = await constraint_adherence(
        utterances, never_say=["exact prices before confirmation"], judge_llm=judge
    )
    assert result.value == 1.0
    assert result.passed is True


async def test_constraint_adherence_fails_when_judge_says_violated():
    judge = _FakeJudge(json.dumps({"violated": True, "reason": "quoted a price early"}))
    utterances = ["It's exactly 3500 rupees."]
    result = await constraint_adherence(
        utterances, never_say=["exact prices before confirmation"], judge_llm=judge
    )
    assert result.value == 0.0
    assert result.passed is False
    assert "quoted a price" in result.reason


async def test_constraint_adherence_skips_when_no_constraints_declared():
    judge = _FakeJudge("")
    result = await constraint_adherence(["anything"], never_say=[], judge_llm=judge)
    assert result.skipped is True
    assert judge.prompts == []  # never called the judge for nothing to check


async def test_pii_violation_rate_catches_secret_shaped_key_via_regex_no_judge_needed():
    judge = _FakeJudge("should not be called")
    utterances = ["Here is your api_key: sk-abc123 for reference."]
    result = await pii_violation_rate(utterances, judge_llm=judge)
    assert result.value == 1.0  # 1.0 = violation rate, higher is worse
    assert result.passed is False
    assert judge.prompts == []  # regex pre-filter caught it, no judge call spent


async def test_pii_violation_rate_falls_back_to_judge_for_paraphrased_leak():
    judge = _FakeJudge(json.dumps({"violated": True, "reason": "read out the caller's phone number"}))
    utterances = ["Just to confirm, I have your number as nine three one one..."]
    result = await pii_violation_rate(utterances, judge_llm=judge)
    assert result.value == 1.0
    assert result.passed is False
    assert judge.prompts  # judge WAS consulted this time


async def test_pii_violation_rate_clean_transcript_scores_zero():
    judge = _FakeJudge(json.dumps({"violated": False, "reason": "no sensitive data spoken"}))
    utterances = ["Sure, I can help you book a tee time."]
    result = await pii_violation_rate(utterances, judge_llm=judge)
    assert result.value == 0.0
    assert result.passed is True
