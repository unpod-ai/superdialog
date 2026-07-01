"""The 7 RAGAS metrics, scored by two judges (gpt-4o-mini + claude-haiku).

RAGAS is an optional dependency (``uv add ragas`` / the ``benchmark`` extra). It
is imported lazily so the deterministic path and the loader work without it.

Metric set (user-selected):
    ConversationRelevance, AgentGoalAccuracy (with reference), TopicAdherence,
    ConversationCompleteness, AnswerCorrectness, AspectCritic:coherence,
    AnswerRelevancy

Judges run independently — each judge LLM scores every metric, so the panel can
show whether the two judges agree. All judge calls route through litellm (a base
superdialog dependency), so model strings are litellm-style:
    "gpt-4o-mini", "anthropic/claude-haiku-4-5-20251001"

ponytail: the two version-sensitive spots are _build_metrics() (RAGAS class
names move between 0.2.x releases) and _to_messages() (message class import).
If a RAGAS upgrade breaks scoring, those two functions are the only things to
touch — everything else is plain data plumbing.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from .loader import BenchmarkSample

# Single fixed judge (the "ruler"): one neutral model scores every SUT run so
# the RAGAS columns are directly comparable and the judge never varies. Using
# sonnet (stronger than either SUT) also keeps the judge from grading its own
# output. Judge cost is eval overhead and is NOT counted in the SUT cost row.
DEFAULT_JUDGES = ("anthropic/claude-sonnet-4-6",)

# our 7 metric keys, in panel display order
METRIC_KEYS = (
    "conversation_relevance",
    "agent_goal_accuracy",
    "topic_adherence",
    "conversation_completeness",
    "answer_correctness",
    "coherence",
    "answer_relevancy",
)


@dataclass(frozen=True)
class RagasScores:
    conversation_relevance: float | None = None
    agent_goal_accuracy: float | None = None
    topic_adherence: float | None = None
    conversation_completeness: float | None = None
    answer_correctness: float | None = None
    coherence: float | None = None
    answer_relevancy: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


class RagasNotInstalled(RuntimeError):
    """Raised when scoring is attempted but the ``ragas`` package is missing."""


def _require_ragas():
    try:
        import ragas  # noqa: F401
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RagasNotInstalled(
            "ragas is not installed. Install with `uv add ragas` (or the "
            "`benchmark` extra) inside superdialog/ before running RAGAS scoring."
        ) from e


def _build_judge_llm(model: str):
    """Wrap a litellm model as a RAGAS evaluator LLM.

    ponytail: version-sensitive. Uses ragas' LangchainLLMWrapper over a litellm
    chat model, which is the stable path across ragas 0.2.x.
    """
    from langchain_community.chat_models import ChatLiteLLM
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatLiteLLM(model=model, temperature=0.0))


def _to_messages(user_input: list[dict]):
    """Convert dataset user_input ([{role: human|ai}]) to RAGAS messages."""
    from ragas.messages import AIMessage, HumanMessage

    out = []
    for turn in user_input:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "human":
            out.append(HumanMessage(content=content))
        elif role == "ai":
            out.append(AIMessage(content=content))
    return out


def _build_metrics(llm):
    """Instantiate the 7 metrics bound to a judge LLM.

    ponytail: version-sensitive RAGAS class names live here only. Returns a dict
    of {metric_key: (metric_instance, kind)} where kind is "multi" or "single".
    """
    from ragas.metrics import (
        AgentGoalAccuracyWithReference,
        AspectCritic,
        TopicAdherenceScore,
    )
    # answer_correctness + answer_relevancy are embedding-based and DEFERRED:
    # not built here so they are never attempted (no wasted work, no log spam).
    # Re-add AnswerCorrectness / ResponseRelevancy with an embeddings model when
    # embeddings are wired. RagasScores keeps their fields (default None -> n/a).

    coherence = AspectCritic(
        name="coherence",
        definition=(
            "Does the assistant's side of the conversation flow logically and "
            "stay internally consistent from start to finish?"
        ),
        llm=llm,
    )
    conversation_relevance = AspectCritic(
        name="conversation_relevance",
        definition=(
            "Is every assistant response relevant and responsive to what the "
            "user just said, without ignoring or talking past them?"
        ),
        llm=llm,
    )
    conversation_completeness = AspectCritic(
        name="conversation_completeness",
        definition=(
            "Did the assistant address all of the user's needs and collect all "
            "required information before closing?"
        ),
        llm=llm,
    )
    return {
        "conversation_relevance": (conversation_relevance, "multi"),
        "agent_goal_accuracy": (AgentGoalAccuracyWithReference(llm=llm), "multi"),
        "topic_adherence": (TopicAdherenceScore(llm=llm), "multi"),
        "conversation_completeness": (conversation_completeness, "multi"),
        "coherence": (coherence, "multi"),
    }


async def _score_one_judge(sample: BenchmarkSample, model: str) -> RagasScores:
    from ragas.dataset_schema import MultiTurnSample, SingleTurnSample

    llm = _build_judge_llm(model)
    metrics = _build_metrics(llm)

    user_input = sample.ragas_sample.get("user_input", [])
    messages = _to_messages(user_input)
    reference = sample.reference
    topics = sample.reference_topics

    multi = MultiTurnSample(
        user_input=messages,
        reference=reference,
        reference_topics=topics or None,
    )
    # single-turn view: last assistant response vs reference answer
    last_ai = next(
        (t["content"] for t in reversed(user_input) if t.get("role") == "ai"),
        "",
    )
    first_human = next(
        (t["content"] for t in user_input if t.get("role") == "human"),
        "",
    )
    single = SingleTurnSample(
        user_input=first_human,
        response=last_ai,
        reference=reference,
        retrieved_contexts=sample.ragas_sample.get("retrieved_contexts", []),
    )

    results: dict[str, float | None] = {k: None for k in METRIC_KEYS}
    for key, (metric, kind) in metrics.items():
        try:
            if kind == "single":
                results[key] = float(await metric.single_turn_ascore(single))
            else:
                results[key] = float(await metric.multi_turn_ascore(multi))
        except Exception as e:  # one metric failing must not kill the rest
            results[key] = None
            print(f"[ragas] {model} {key} failed: {e}")
    return RagasScores(**results)


def score_ragas(
    sample: BenchmarkSample, judge_models: tuple[str, ...] = DEFAULT_JUDGES
) -> dict[str, RagasScores]:
    """Score one sample with every judge. Returns {judge_model: RagasScores}."""
    _require_ragas()

    async def _run() -> dict[str, RagasScores]:
        tasks = {m: _score_one_judge(sample, m) for m in judge_models}
        done = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), done))

    return asyncio.run(_run())


__all__ = [
    "RagasScores",
    "RagasNotInstalled",
    "score_ragas",
    "DEFAULT_JUDGES",
    "METRIC_KEYS",
]
