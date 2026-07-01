"""The 4 deterministic (non-LLM) metrics, kept alongside the RAGAS ones.

These are the framework's own eval metrics, computed WITHOUT any LLM call:

    completion    reached the expected outcome / closed cleanly
    data_capture  fraction of ground-truth slots actually present in transcript
    smoothness    1 / (1 + excess turns per checkpoint)   — penalises loops
    repairs       fraction of agent turns that are re-asks / corrections

On a golden transcript these read from text (semantic substring match — the fix
for the old exact-match fragility where "John" != "John Smith"). When a live
``SessionMetrics`` is available, prefer :func:`from_session_metrics` which uses
the real checkpoint/slot/repair counts instead of text proxies.

ponytail: substring slot match is the cheap, deterministic replacement for an
LLM judge; upgrade to embedding similarity only if substring measurably misses.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .loader import BenchmarkSample

# outcome keywords that signal a clean close in a golden transcript
_CLOSE_HINTS = (
    "have a wonderful day",
    "have a great day",
    "have a good day",
    "reach out",
    "will contact you",
    "will connect",
    "will call you back",
    "team will",
    "goodbye",
    "see you",
    "take care",
)

_REPAIR_HINTS = (
    "sorry",
    "i didn't catch",
    "didn't get that",
    "could you repeat",
    "say that again",
    "let me confirm",
    "just to confirm",
    "is that correct",
    "you mean",
    "actually",
    "i meant",
)


@dataclass(frozen=True)
class DeterministicScores:
    completion: float
    data_capture: float
    smoothness: float
    repairs: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _agent_turns(conversation: list[dict]) -> list[str]:
    return [
        t.get("content", "")
        for t in conversation
        if t.get("role") in ("agent", "ai", "assistant")
    ]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@]+", " ", s.lower()).strip()


def _slot_present(value: object, haystack: str) -> bool:
    """Semantic-ish match: is the slot value expressed anywhere in the transcript?

    Case-insensitive, punctuation-insensitive. For a multi-word value, count it
    present if any token of length > 2 lands (so "John Smith" matches "John").
    Digits (phone) matched with separators stripped.
    """
    v = str(value).strip()
    if not v:
        return False
    hay = _norm(haystack)
    hay_digits = re.sub(r"\D", "", haystack)

    vnorm = _norm(v)
    if vnorm and vnorm in hay:
        return True
    # digit runs (phone/last4/yob)
    vdigits = re.sub(r"\D", "", v)
    if len(vdigits) >= 3 and vdigits in hay_digits:
        return True
    # token fallback for multi-word values
    tokens = [t for t in vnorm.split() if len(t) > 2]
    return any(t in hay for t in tokens)


def score_deterministic(sample: BenchmarkSample) -> DeterministicScores:
    """Compute the 4 deterministic metrics from a (golden or generated) transcript."""
    convo = sample.conversation
    agent_texts = _agent_turns(convo)
    joined = " ".join(t.get("content", "") for t in convo)
    joined_lower = joined.lower()

    # completion — did the agent close cleanly / signal the expected outcome?
    completion = 0.0
    if sample.expected_outcome and _slot_present(
        sample.expected_outcome.replace("_", " "), joined
    ):
        completion = 1.0
    elif any(h in joined_lower for h in _CLOSE_HINTS):
        completion = 1.0

    # data_capture — fraction of ground-truth slots present in the transcript
    slots = sample.ground_truth_slots
    if slots:
        hit = sum(1 for val in slots.values() if _slot_present(val, joined))
        data_capture = hit / len(slots)
    else:
        # no slots to capture (e.g. wrong_number / not_interested) -> not penalised
        data_capture = 1.0

    # smoothness — excess agent turns beyond one-per-checkpoint means loops/friction
    checkpoints = max(1, len(sample.expected_checkpoint_path))
    tpc = len(agent_texts) / checkpoints
    smoothness = 1.0 / (1.0 + max(0.0, tpc - 1.0))

    # repairs — fraction of agent turns that look like a re-ask / correction
    if agent_texts:
        repair_turns = sum(
            1 for t in agent_texts if any(h in t.lower() for h in _REPAIR_HINTS)
        )
        repairs = repair_turns / len(agent_texts)
    else:
        repairs = 0.0

    return DeterministicScores(
        completion=round(completion, 4),
        data_capture=round(data_capture, 4),
        smoothness=round(smoothness, 4),
        repairs=round(repairs, 4),
    )


def from_session_metrics(sm) -> DeterministicScores:  # type: ignore[no-untyped-def]
    """Build scores from a live ``eval.models.SessionMetrics`` (real run path).

    Prefer this over :func:`score_deterministic` when the runner produced actual
    checkpoint/slot/repair counts — no text heuristics needed.
    """
    tpc_values = list(getattr(sm, "turns_per_checkpoint", {}).values()) or [1]
    mean_tpc = sum(tpc_values) / len(tpc_values)
    turns = max(1, int(getattr(sm, "turns", 1)))
    return DeterministicScores(
        completion=1.0 if getattr(sm, "completed", False) else 0.0,
        data_capture=round(float(getattr(sm, "slot_accuracy", 0.0)), 4),
        smoothness=round(1.0 / (1.0 + max(0.0, mean_tpc - 1.0)), 4),
        repairs=round(int(getattr(sm, "repair_count", 0)) / turns, 4),
    )


def _self_check() -> None:
    """Runnable check — fails loudly if the metric logic breaks."""
    happy = BenchmarkSample(
        id="t1",
        playbook=None,
        scenario_type="happy",
        difficulty="easy",
        persona={"ground_truth_slots": {"name": "Rahul", "city": "Delhi",
                                        "alternate_number": "9876543210"}},
        conversation=[
            {"role": "agent", "content": "Namaste! May I have your name?"},
            {"role": "user", "content": "Rahul"},
            {"role": "agent", "content": "Thanks Rahul. Your number?"},
            {"role": "user", "content": "98765 43210"},
            {"role": "agent", "content": "And city?"},
            {"role": "user", "content": "Delhi"},
            {"role": "agent", "content": "Our team will contact you. Have a wonderful day!"},
        ],
        ground_truth={"expected_outcome": "lead_captured",
                      "expected_checkpoint_path": ["greet", "name", "number", "city", "close"]},
        ragas_sample={},
    )
    s = score_deterministic(happy)
    # all 3 slots present (name substring, phone digit-run, city substring)
    assert s.data_capture == 1.0, s
    # clean close hint present
    assert s.completion == 1.0, s
    # 4 agent turns over 5 checkpoints -> tpc<1 -> smoothness 1.0
    assert s.smoothness == 1.0, s
    # no repair hints
    assert s.repairs == 0.0, s

    missing = BenchmarkSample(
        id="t2", playbook=None, scenario_type="x", difficulty="easy",
        persona={"ground_truth_slots": {"name": "Rahul", "city": "Delhi"}},
        conversation=[
            {"role": "agent", "content": "Sorry, could you repeat? Just to confirm?"},
            {"role": "user", "content": "Rahul"},
        ],
        ground_truth={"expected_outcome": "lead_captured",
                      "expected_checkpoint_path": ["a", "b"]},
        ragas_sample={},
    )
    s2 = score_deterministic(missing)
    assert s2.data_capture == 0.5, s2          # only name present, city missing
    assert s2.repairs == 1.0, s2               # the one agent turn is a repair
    print("deterministic self-check OK:", s.as_dict(), s2.as_dict())


if __name__ == "__main__":
    _self_check()


__all__ = ["DeterministicScores", "score_deterministic", "from_session_metrics"]
