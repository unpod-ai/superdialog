"""Render an A/B RunResult as report.json + report.md."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from superdialog.eval.results import MetricAggregate, RunResult


def write_report(run: RunResult, out_dir: str) -> tuple[str, str]:
    """Write report.json (full) and report.md (headline). Returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "report.json")
    md_path = os.path.join(out_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(run), fh, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(run))
    return json_path, md_path


def render_combined(entries: list[tuple[str, RunResult]]) -> str:
    """ONE big table across every model+mode: metric rows (incl. composite and
    guardrail-violation) × columns for each model's vanilla/playbook/Δ.

    ``entries`` is ``[(model_uri, RunResult), ...]``. Everything lands in a single
    table — no separate per-mode/per-model sub-reports to stitch together."""
    col_headers: list[str] = ["metric"]
    for model, run in entries:
        for md in run.modes:
            col_headers.append(f"{model} · {md.mode}")
        if len(run.modes) == 2:
            col_headers.append(f"{model} · Δ")

    metrics = sorted(
        {m for _, run in entries for md in run.modes for m in md.aggregates}
    )
    lines = [
        "# A/B Eval — Combined Report",
        "",
        f"Dataset(s): {', '.join(sorted({run.dataset for _, run in entries}))}",
        "",
        "| " + " | ".join(col_headers) + " |",
        "|" + "---|" * len(col_headers),
    ]

    def _metric_row(metric: str) -> list[str]:
        cells = [metric]
        for _model, run in entries:
            for md in run.modes:
                cells.append(_fmt(md.aggregates.get(metric)))
            if len(run.modes) == 2:
                a = run.modes[0].aggregates.get(metric)
                b = run.modes[-1].aggregates.get(metric)
                if a and b and a.mean is not None and b.mean is not None:
                    cells.append(f"{b.mean - a.mean:+.3f}")
                else:
                    cells.append("—")
        return cells

    def _scalar_row(label: str, value_of) -> list[str]:
        cells = [label]
        for _model, run in entries:
            for md in run.modes:
                cells.append(value_of(md))
            if len(run.modes) == 2:
                cells.append("—")  # delta not meaningful for composite/rates
        return cells

    for metric in metrics:
        lines.append("| " + " | ".join(_metric_row(metric)) + " |")
    lines.append(
        "| "
        + " | ".join(
            _scalar_row("composite (gated)", lambda md: f"{md.composite_mean:.3f}")
        )
        + " |"
    )
    lines.append(
        "| "
        + " | ".join(
            _scalar_row(
                "guardrail violation %", lambda md: f"{md.guardrail_violation_rate:.1%}"
            )
        )
        + " |"
    )
    return "\n".join(lines)


def write_combined(entries: list[tuple[str, RunResult]], out_dir: str) -> str:
    """Write the single combined ``report.md`` at ``out_dir``; returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_combined(entries))
    return md_path


_EMPTY_AGG = MetricAggregate(metric="", mean=None, std=0.0, n=0, errored=0, skipped=0)


def _ms(v: float | None) -> str:
    return f"{v:.0f}ms" if isinstance(v, (int, float)) else "—"


def _num(v: float | None, ndigits: int = 0) -> str:
    return f"{v:.{ndigits}f}" if isinstance(v, (int, float)) else "—"


def _fmt(agg: MetricAggregate | None) -> str:
    if agg is None or agg.mean is None:
        return "—"
    s = f"{agg.mean:.3f}"
    if agg.std:
        s += f" ±{agg.std:.3f}"
    flags = []
    if agg.errored:
        flags.append(f"{agg.errored}err")
    if agg.skipped:
        flags.append(f"{agg.skipped}skip")
    if flags:
        s += f" ({','.join(flags)})"
    return s


def render_markdown(run: RunResult) -> str:
    """A headline metric table (mode columns + delta), composite/guardrail, drilldown."""
    modes = run.modes
    names = [m.mode for m in modes]
    lines = [f"# A/B Eval Report — {run.dataset}", ""]

    two = len(modes) >= 2
    header = "| metric | " + " | ".join(names)
    if two:
        header += f" | Δ ({names[-1]}−{names[0]})"
    header += " |"
    lines += [header, "|" + "---|" * (len(names) + (2 if two else 1))]

    all_metrics = sorted({m for md in modes for m in md.aggregates})
    for metric in all_metrics:
        cells = [metric] + [_fmt(md.aggregates.get(metric)) for md in modes]
        if two:
            a, b = modes[0].aggregates.get(metric), modes[-1].aggregates.get(metric)
            if a and b and a.mean is not None and b.mean is not None:
                cells.append(f"{b.mean - a.mean:+.3f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Latency & tokens (per assistant turn)", ""]
    lines += [
        "| mode | latency p50 | latency p95 | input tok/turn | director | talker | LLM calls/turn |",  # noqa: E501
        "|---|---|---|---|---|---|---|",
    ]
    for md in modes:
        eff = (md.aggregates.get("efficiency") or _EMPTY_AGG).extras
        tok = (md.aggregates.get("token_cost") or _EMPTY_AGG).extras
        lines.append(
            f"| {md.mode} "
            f"| {_ms(eff.get('latency_p50'))} | {_ms(eff.get('latency_p95'))} "
            f"| {_num(tok.get('tokens_mean'))} "
            f"| {_num(tok.get('director_tokens_mean'))} "
            f"| {_num(tok.get('talker_tokens_mean'))} "
            f"| {_num(tok.get('llm_calls_mean'), 1)} |"
        )

    lines += ["", "## Composite & guardrails", ""]
    lines += [
        "| mode | composite (gated) | framework (quality-gated cost) | guardrail violation rate |",  # noqa: E501
        "|---|---|---|---|",
    ]
    for md in modes:
        lines.append(
            f"| {md.mode} | {md.composite_mean:.3f} | {md.framework_mean:.3f} | "
            f"{md.guardrail_violation_rate:.1%} |"
        )

    lines += ["", "## Per-case drilldown", ""]
    for md in modes:
        lines += [f"### {md.mode}", ""]
        for cr in md.case_results:
            gv = " ⚠️ GUARDRAIL FAIL" if cr.guardrail_failed else ""
            bits = []
            for name, results in sorted(cr.metric_results.items()):
                scored = [
                    r.value
                    for r in results
                    if r.value is not None and not r.errored and not r.skipped
                ]
                if scored:
                    bits.append(f"{name}={sum(scored) / len(scored):.2f}")
            lines.append(
                f"- **{cr.case_id}** ({cr.turns} turns){gv}: {', '.join(bits)}"
            )
        lines.append("")
    return "\n".join(lines)
