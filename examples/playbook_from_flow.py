"""Migrate a legacy ConversationFlow graph to a Playbook.

Existing flows keep working on DialogMachine unchanged; ``compile_flow``
is the on-ramp to the Playbook engine. It converts a flow JSON losslessly:
conversational nodes become checkpoints, tool-bearing computational chains
become pipelines behind synthetic intermediate checkpoints, silence nodes
fold into ``policies.silence``, global edges become interrupts, and every
action compiles 1:1 to a ToolSpec with its templates rewritten into the
``{env, slots, results}`` namespace. ``coverage_report`` then audits the
mapping — anything in an ``unmapped_*`` list is a compiler bug.

Usage (from the repo root)::

    uv run python examples/playbook_from_flow.py              # built-in demo
    uv run python examples/playbook_from_flow.py my_flow.json # your own flow

The coverage summary prints to stderr; the compiled playbook YAML prints
to stdout, so the output pipes cleanly::

    uv run python examples/playbook_from_flow.py my_flow.json > playbook.yaml
"""

from __future__ import annotations

import sys
from typing import Any

import yaml

from superdialog.flow.models import ConversationFlow
from superdialog.playbook import Playbook, compile_flow, coverage_report
from superdialog.playbook.compiler import CoverageReport

# A tiny legacy flow: greet collects details, then an auto_proceed
# (computational) node fires an HTTP action and routes on its result.
# The compiler turns that node into a pipeline owned by a synthetic
# intermediate checkpoint, and the goodbye global edge into an interrupt.
DEMO_FLOW: dict[str, Any] = {
    "system_prompt": "You are a clinic booking assistant.",
    "initial_node": "greet",
    "environment_variables": {"API_BASE_URL": "https://api.example.com"},
    "actions": [
        {
            "id": "create_booking",
            "name": "Create booking",
            "method": "POST",
            "url": "{{API_BASE_URL}}/bookings",
            "body": {"name": "{{name}}", "date": "{{date}}"},
            "store_response_as": "booking_result",
        }
    ],
    "nodes": [
        {
            "id": "greet",
            "name": "Greet and collect details",
            "instruction": "Greet the caller; collect their name and date.",
            "edges": [
                {
                    "id": "e_details",
                    "condition": "caller provided their name and a date",
                    "target_node_id": "book",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["name", "date"],
                    },
                }
            ],
        },
        {
            "id": "book",
            "name": "Create the booking",
            "auto_proceed": True,
            "actions": [{"trigger": "on_enter", "action_id": "create_booking"}],
            "edges": [
                {
                    "id": "e_ok",
                    "condition": "booking_result.success == true",
                    "target_node_id": "close",
                },
                {
                    "id": "e_fail",
                    "condition": "booking_result.success == false — route back",
                    "target_node_id": "greet",
                },
            ],
        },
        {
            "id": "close",
            "name": "Close",
            "static_text": "You are booked. Goodbye!",
            "is_final": True,
        },
    ],
    "global_edges": [
        {
            "id": "g_bye",
            "condition": "caller says goodbye",
            "target_node_id": "close",
        }
    ],
}


def _fmt(items: list[str]) -> str:
    return ", ".join(items) if items else "(none)"


def _summary(flow: ConversationFlow, pb: Playbook, report: CoverageReport) -> str:
    """Render the coverage report as a short, scannable block."""
    edge_count = sum(len(n.edges) for n in flow.nodes) + len(flow.global_edges)
    lines = [
        f"flow: {len(flow.nodes)} nodes, {edge_count} edges, "
        f"{len(flow.actions)} actions",
        f"playbook: {len(pb.checkpoint_ids())} checkpoints, {len(pb.tools)} "
        f"tools, {len(pb.pipelines)} pipelines, {len(pb.interrupts)} interrupts",
        f"unmapped nodes: {_fmt(report.unmapped_nodes)}",
        f"unmapped edges: {_fmt(report.unmapped_edges)}",
        f"unmapped actions: {_fmt(report.unmapped_actions)}",
        f"orphans: {_fmt(report.orphans)}",
    ]
    if report.dropped:
        lines.append("dropped buckets (informational, absorbed by policies/rules):")
        lines += [
            f"  {bucket}: {_fmt(items)}" for bucket, items in report.dropped.items()
        ]
    else:
        lines.append("dropped buckets: (none)")
    if report.notes:
        lines.append("notes:")
        lines += [f"  - {note}" for note in report.notes]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    """Compile a flow (file arg or the built-in demo) and emit YAML."""
    flow = (
        ConversationFlow.load(argv[1])
        if len(argv) > 1
        else ConversationFlow.model_validate(DEMO_FLOW)
    )
    pb = compile_flow(flow)
    report = coverage_report(flow, pb)
    print(_summary(flow, pb, report), file=sys.stderr)
    print(file=sys.stderr)

    text = yaml.safe_dump(
        pb.model_dump(mode="json", exclude_defaults=True),
        sort_keys=False,
        allow_unicode=True,
        width=80,
    )
    Playbook.from_yaml(text)  # round-trip sanity: the emitted YAML revalidates
    sys.stdout.write(text)


if __name__ == "__main__":
    main(sys.argv)
