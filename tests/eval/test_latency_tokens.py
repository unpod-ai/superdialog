"""Latency + input-token metrics: capture, threading, scoring, aggregation."""

from __future__ import annotations

from superdialog.eval.dataset.models import EvalCase, EvalSample
from superdialog.eval.endpoints.base import Transcript
from superdialog.eval.metrics.base import MetricResult, MetricSuite
from superdialog.eval.metrics.custom import EfficiencyMetric, TokenCostMetric
from superdialog.eval.metrics.registry import build_suite
from superdialog.eval.results import CaseResult
from superdialog.eval.runner import _aggregate_mode, samples_from_run
from superdialog.eval.scoring import DEFAULT_WEIGHTS, case_framework
from superdialog.playbook.eval.models import PersonaSpec
from superdialog.playbook.providers import provider_adapters
from tests.eval.fakes import FakeProvider


# --- providers: usage events carry the emitting role ------------------------


class _Chunk:
    def __init__(self, text: str, usage: dict | None = None) -> None:
        self.text = text
        self.usage = usage


class _ChunkProvider:
    """LLMProvider double: complete() with usage metadata, stream() of chunks."""

    model = "fake/model"

    async def complete(self, messages, **kw):
        class _Res:
            text = "ok"
            metadata = {"prompt_tokens": 42}

        return _Res()

    async def stream(self, messages, **kw):
        yield _Chunk("hel")
        yield _Chunk("lo", usage={"prompt_tokens": 7})


async def test_provider_adapters_tag_usage_events_with_role():
    events = []

    async def sink(e):
        events.append(e)

    director, talker = provider_adapters(_ChunkProvider(), sink)
    await director.complete([{"role": "user", "content": "hi"}])
    async for _ in talker.stream([{"role": "user", "content": "hi"}]):
        pass
    assert [(e.role, e.tokens_in) for e in events] == [("director", 42), ("talker", 7)]


# --- transcript: metadata threading ------------------------------------------


def _transcript_with_tokens() -> Transcript:
    t = Transcript()
    t.add("assistant", "hi", latency_ms=100.0, metadata={"input_tokens": 500})
    t.add("user", "book")
    t.add(
        "assistant",
        "ok",
        latency_ms=300.0,
        metadata={
            "input_tokens": 1500,
            "director_tokens": 900,
            "talker_tokens": 600,
            "llm_calls": 2,
        },
    )
    return t


def _case() -> EvalCase:
    return EvalCase(
        id="c1",
        playbook="p",
        reference="goal",
        persona=PersonaSpec(name="s", traits="", goal="goal"),
    )


def test_samples_carry_token_arrays():
    sample = samples_from_run(_case(), "playbook", _transcript_with_tokens(), [])[0]
    assert sample.metadata["input_tokens_per_turn"] == [500.0, 1500.0]
    assert sample.metadata["director_tokens_per_turn"] == [900.0]
    assert sample.metadata["latencies_ms"] == [100.0, 300.0]


def test_samples_token_arrays_empty_when_endpoint_reports_nothing():
    t = Transcript()
    t.add("assistant", "hi")
    sample = samples_from_run(_case(), "vanilla", t, [])[0]
    assert sample.metadata["input_tokens_per_turn"] == []


# --- TokenCostMetric ----------------------------------------------------------


async def test_token_cost_scores_mean_and_split():
    sample = samples_from_run(_case(), "playbook", _transcript_with_tokens(), [])[0]
    res = await TokenCostMetric().score(sample)
    assert res.value == 1000.0  # mean(500, 1500)
    assert res.extra["tokens_max"] == 1500.0
    assert res.extra["director_tokens_mean"] == 900.0
    assert res.extra["talker_tokens_mean"] == 600.0


async def test_token_cost_skips_without_data():
    res = await TokenCostMetric().score(
        EvalSample(kind="conversation", user_input=[], metadata={})
    )
    assert res.skipped and res.value is None


# --- registry: free metrics always included -----------------------------------


def test_build_suite_force_includes_free_metrics():
    suite = build_suite(["task_success"], FakeProvider({"*": "{}"}), "m")
    assert isinstance(suite, MetricSuite)
    names = {getattr(m, "name", "") for m in suite._metrics}
    assert {"efficiency", "token_cost"} <= names


# --- aggregation: extras pooled into MetricAggregate ---------------------------


def _case_result(metric_results: dict) -> CaseResult:
    return CaseResult(
        case_id="c1",
        mode="playbook",
        metric_results=metric_results,
        guardrail_failed=False,
        turns=2,
    )


def test_aggregate_pools_numeric_extras():
    cr = _case_result(
        {
            "token_cost": [
                MetricResult(
                    name="token_cost", value=1000.0, extra={"tokens_mean": 1000.0}
                )
            ]
        }
    )
    mode = _aggregate_mode("playbook", [cr], DEFAULT_WEIGHTS)
    assert mode.aggregates["token_cost"].extras["tokens_mean"] == 1000.0


# --- framework score: quality gate then cost -----------------------------------


def _quality_results(task: float, slots: float) -> dict:
    return {
        "task_success": [MetricResult(name="task_success", value=task)],
        "slot_accuracy": [MetricResult(name="slot_accuracy", value=slots)],
        "efficiency": [
            MetricResult(name="efficiency", value=2.0, extra={"latency_p95": 2000.0})
        ],
        "token_cost": [
            MetricResult(name="token_cost", value=1500.0, extra={"tokens_mean": 1500.0})
        ],
    }


def test_framework_zero_when_quality_below_target():
    assert case_framework(_case_result(_quality_results(0.9, 1.0))) == 0.0


def test_framework_zero_on_guardrail_fail():
    cr = _case_result(_quality_results(1.0, 1.0))
    cr.guardrail_failed = True
    assert case_framework(cr) == 0.0


def test_framework_rewards_low_cost_when_quality_perfect():
    # both normalizers hit exactly -> 1 / (2 * 2) = 0.25
    assert case_framework(_case_result(_quality_results(1.0, 1.0))) == 0.25
    # zero-cost run scores 1.0
    free = _quality_results(1.0, 1.0)
    free["efficiency"][0].extra["latency_p95"] = 0.0
    free["token_cost"][0].extra["tokens_mean"] = 0.0
    assert case_framework(_case_result(free)) == 1.0


async def test_efficiency_extras_expose_latency_percentiles():
    sample = samples_from_run(_case(), "playbook", _transcript_with_tokens(), [])[0]
    res = await EfficiencyMetric().score(sample)
    assert res.extra["latency_p50"] == 200.0  # median of [100, 300]
    assert res.extra["latency_p95"] > 200.0


# --- journey stops at the natural call end -------------------------------------


class _EndingEndpoint:
    """Ends the session after the second user turn."""

    def __init__(self) -> None:
        self._turns = 0

    async def start(self) -> str:
        return "hello"

    async def turn(self, text: str) -> str:
        self._turns += 1
        return "bye" if self._turns >= 2 else "ok"

    @property
    def ended(self) -> bool:
        return self._turns >= 2

    def reset(self) -> None:
        self._turns = 0


class _ChattyUser:
    async def complete(self, messages) -> str:
        return "next line"


async def test_drive_journey_stops_when_endpoint_ends():
    from superdialog.eval.runner import drive_journey

    persona = PersonaSpec(name="p", traits="", goal="g", max_turns=10)
    t = await drive_journey(_EndingEndpoint(), persona, _ChattyUser())
    # greeting + 2 exchanges — NOT 10: the journey stopped at the call end.
    assert t.turn_count() == 2
    assert t.records[-1].text == "bye"
