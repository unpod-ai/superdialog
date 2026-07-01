"""Render the benchmark panel as terminal text.

Layout (recovered design):
    - header: playbook + dataset + N scenarios
    - DETERMINISTIC block: the 4 framework metrics (always shown)
    - RAGAS block: 7 metrics, one sub-table per judge (gpt-4o-mini, claude-haiku)
    - one column per mode (With SuperDialog / Raw LLM) with a Δ column when
      exactly two modes are present

Deterministic metrics are always present; RAGAS rows appear only for modes that
were scored with ``run_ragas=True``.
"""

from __future__ import annotations

from .report import DET_KEYS, ModeReport

_DET_LABELS = {
    "completion": "Completion rate",
    "data_capture": "Data capture",
    "smoothness": "Smoothness",
    "repairs": "Repairs",
}
_RAGAS_LABELS = {
    "conversation_relevance": "Turn relevance",
    "agent_goal_accuracy": "Goal accuracy",
    "topic_adherence": "Topic adherence",
    "conversation_completeness": "Conversation comp.",
    "answer_correctness": "Answer correct.",
    "coherence": "Coherence",
    "answer_relevancy": "Answer relevancy",
}

_METRIC_W = 22
_COL_W = 16


def _fmt(v: float | None) -> str:
    """Score as whole-integer percent (0.33 -> '33%'). None -> em dash."""
    return "—" if v is None else f"{round(v * 100)}%"


def _delta(a: float | None, b: float | None) -> str:
    """Delta in percentage points (integer), with direction arrow."""
    if a is None or b is None:
        return ""
    d = round((a - b) * 100)
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "═")
    return f"{d:+d}pp {arrow}"


def _row(label: str, cols: list[str], delta: str = "") -> str:
    cells = "".join(c.ljust(_COL_W) for c in cols)
    return f"  {label.ljust(_METRIC_W)}{cells}{delta}"


def _header_row(reports: list[ModeReport]) -> str:
    cols = [r.label for r in reports]
    delta = "Δ" if len(reports) == 2 else ""
    return _row("Metric", cols, delta)


def _sep(reports: list[ModeReport]) -> str:
    width = 2 + _METRIC_W + _COL_W * len(reports) + 10
    return "  " + "─" * (width - 2)


def render_panel(
    reports: list[ModeReport],
    *,
    dataset: str = "",
    playbook: str | None = None,
) -> str:
    if not reports:
        return "(no reports)"
    n = reports[0].n
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  SUPERDIALOG BENCHMARK PANEL")
    if playbook:
        lines.append(f"  playbook: {playbook}")
    lines.append(f"  dataset: {dataset or '(mixed)'}  |  scenarios: {n}")
    lines.append("=" * 78)
    lines.append("")

    two = len(reports) == 2
    lines.append(_header_row(reports))
    lines.append(_sep(reports))

    # DETERMINISTIC block (always)
    lines.append("  [ deterministic — framework's own eval ]")
    for k in DET_KEYS:
        cols = [_fmt(r.det_mean.get(k)) for r in reports]
        d = _delta(reports[0].det_mean.get(k), reports[1].det_mean.get(k)) if two else ""
        lines.append(_row(_DET_LABELS[k], cols, d))

    # RAGAS block, per judge
    judges = sorted({j for r in reports for j in r.ragas_mean})
    if judges:
        for judge in judges:
            lines.append(_sep(reports))
            lines.append(f"  [ RAGAS — judge: {judge} ]")
            for mk, label in _RAGAS_LABELS.items():
                cols = [_fmt(r.ragas_mean.get(judge, {}).get(mk)) for r in reports]
                d = (
                    _delta(
                        reports[0].ragas_mean.get(judge, {}).get(mk),
                        reports[1].ragas_mean.get(judge, {}).get(mk),
                    )
                    if two
                    else ""
                )
                lines.append(_row(label, cols, d))
    else:
        lines.append(_sep(reports))
        lines.append("  [ RAGAS — not run (run_ragas=False) ]")

    # COST block — system-under-test LLM cost only (not judge/eval cost)
    lines.append(_sep(reports))
    cost_cols = [
        ("$0.0000" if r.cost_usd == 0 else f"${r.cost_usd:.4f}") for r in reports
    ]
    lines.append(_row("Cost / run USD (SUT)", cost_cols))

    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------- markdown output


def _md_row(label: str, cells: list[str], delta: str | None) -> str:
    body = " | ".join(cells)
    tail = f" | {delta}" if delta is not None else ""
    return f"| {label} | {body}{tail} |"


def _md_table(reports: list[ModeReport], keys, labels: dict, getter) -> list[str]:
    """One markdown table: a row per metric, a column per report, optional Δ."""
    two = len(reports) == 2
    ncol = len(reports) + (1 if two else 0)
    header = _md_row("Metric", [r.label for r in reports], "Δ (pp)" if two else None)
    sep = "| " + " | ".join(["---"] * (ncol + 1)) + " |"
    rows = [header, sep]
    for k in keys:
        cells = [_fmt(getter(r, k)) for r in reports]
        d = _delta(getter(reports[0], k), getter(reports[1], k)) if two else None
        rows.append(_md_row(labels[k], cells, d))
    return rows


def render_markdown(
    reports: list[ModeReport],
    *,
    dataset: str = "",
    playbook: str | None = None,
) -> str:
    """Aligned markdown tables (integer-percent scores) for a report doc."""
    if not reports:
        return "_(no reports)_"
    meta = f"**scenarios:** {reports[0].n}"
    if playbook:
        meta += f"  |  **playbook:** {playbook}"
    lines = [meta, "", "**Deterministic — framework's own eval**", ""]
    lines += _md_table(reports, DET_KEYS, _DET_LABELS, lambda r, k: r.det_mean.get(k))

    judges = sorted({j for r in reports for j in r.ragas_mean})
    for judge in judges:
        lines += ["", f"**RAGAS — judge: `{judge}`**", ""]
        lines += _md_table(
            reports,
            list(_RAGAS_LABELS.keys()),
            _RAGAS_LABELS,
            lambda r, k, _j=judge: r.ragas_mean.get(_j, {}).get(k),
        )

    # cost row (USD, not percent)
    two = len(reports) == 2
    lines += ["", "**Cost — system-under-test only**", ""]
    lines.append(_md_row("Metric", [r.label for r in reports], "Δ (pp)" if two else None))
    lines.append("| " + " | ".join(["---"] * (len(reports) + (2 if two else 1))) + " |")
    cost_cells = [f"${r.cost_usd:.4f}" for r in reports]
    lines.append(_md_row("Cost / run USD", cost_cells, "" if two else None))
    return "\n".join(lines)


def render_big_table(
    reports: list[ModeReport],
    *,
    dataset: str = "",
    playbook: str | None = None,
    judge: str | None = None,
) -> str:
    """ONE consolidated markdown table: every metric is a row, every report a
    column (model x mode). Deterministic + RAGAS + cost in a single table.

    Assumes a single fixed judge across all reports (takes each report's first
    judge). Columns are aligned by padding so it passes strict MD linters.
    """
    if not reports:
        return "_(no reports)_"

    def _ragas(r: ModeReport, mk: str):
        j = next(iter(r.ragas_mean), None)
        return r.ragas_mean.get(j, {}).get(mk) if j else None

    header = ["Metric"] + [r.label for r in reports]
    body: list[list[str]] = []
    for k in DET_KEYS:
        body.append(
            [f"{_DET_LABELS[k]} (det.)"] + [_fmt(r.det_mean.get(k)) for r in reports]
        )
    for mk, label in _RAGAS_LABELS.items():
        body.append([f"{label} (RAGAS)"] + [_fmt(_ragas(r, mk)) for r in reports])
    body.append(
        ["Cost / run USD (SUT)"] + [f"${r.cost_usd:.4f}" for r in reports]
    )

    ncol = len(header)
    w = [max(len(header[i]), *(len(row[i]) for row in body)) for i in range(ncol)]
    mk_row = lambda cells: "| " + " | ".join(cells[i].ljust(w[i]) for i in range(ncol)) + " |"
    sep = "| " + " | ".join("-" * w[i] for i in range(ncol)) + " |"
    lines = []
    meta = f"**scenarios:** {reports[0].n}"
    if playbook:
        meta += f"  |  **playbook:** {playbook}"
    if judge:
        meta += f"  |  **judge:** {judge}"
    lines.append(meta)
    lines.append("")
    lines.append(mk_row(header))
    lines.append(sep)
    lines.extend(mk_row(r) for r in body)
    return "\n".join(lines)


__all__ = ["render_panel", "render_markdown", "render_big_table"]
