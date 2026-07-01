"""Superdialog benchmark: score playbooks with 4 deterministic + 7 RAGAS metrics.

Scorer-first entry points:

    from superdialog.benchmark import load_named, build_report, render_panel

    samples = load_named("universal")
    report = build_report("Golden", samples)            # deterministic only, free
    print(render_panel([report], dataset="universal"))

    # spend on RAGAS judges (gpt-4o-mini + claude-haiku) when ready:
    report = build_report("Golden", samples, run_ragas=True)
"""

from __future__ import annotations

from .cost import cost_from_response, cost_from_tokens
from .deterministic import DeterministicScores, score_deterministic
from .loader import BenchmarkSample, load_dataset, load_named
from .panel import render_panel
from .ragas_scorer import DEFAULT_JUDGES, RagasNotInstalled, RagasScores, score_ragas
from .report import ModeReport, SampleResult, build_report

__all__ = [
    "BenchmarkSample",
    "load_dataset",
    "load_named",
    "DeterministicScores",
    "score_deterministic",
    "cost_from_response",
    "cost_from_tokens",
    "RagasScores",
    "RagasNotInstalled",
    "score_ragas",
    "DEFAULT_JUDGES",
    "SampleResult",
    "ModeReport",
    "build_report",
    "render_panel",
]
