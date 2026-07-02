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

    lines += ["", "## Composite & guardrails", ""]
    lines += [
        "| mode | composite (gated) | guardrail violation rate |",
        "|---|---|---|",
    ]
    for md in modes:
        lines.append(
            f"| {md.mode} | {md.composite_mean:.3f} | "
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
