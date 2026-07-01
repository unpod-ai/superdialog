"""CLI: python -m superdialog.benchmark <dataset> [--ragas]

    python -m superdialog.benchmark universal            # deterministic only (free)
    python -m superdialog.benchmark universal --ragas     # + RAGAS judges (LLM cost)

Loads examples/datasets/<dataset>_dataset.jsonl, scores it, prints the panel.
Scorer-first: without --ragas it never touches an LLM.
"""

from __future__ import annotations

import sys

from .loader import load_named
from .panel import render_panel
from .report import build_report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_ragas = "--ragas" in argv
    argv = [a for a in argv if a != "--ragas"]
    dataset = argv[0] if argv else "universal"

    samples = load_named(dataset)
    playbook = next((s.playbook for s in samples if s.playbook), None)
    report = build_report(
        "With SuperDialog (golden)", samples, run_ragas=run_ragas
    )
    print(render_panel([report], dataset=dataset, playbook=playbook))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
