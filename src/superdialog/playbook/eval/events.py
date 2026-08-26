"""Shared event-log parsing helpers for scenario-eval metrics.

Every helper here operates on the plain-dict event list produced by
``json.loads``-ing ``SessionMetrics.event_log_jsonl`` line by line — the same
JSONL shape ``superdialog.playbook.events.EventLog.to_jsonl()`` emits. Field
names are the event classes' own field names (see
``superdialog/playbook/events.py``); nothing here re-derives or renames them.
"""

from __future__ import annotations

import json
from typing import Any


def parse_events(event_log_jsonl: str) -> list[dict[str, Any]]:
    """One dict per non-blank JSONL line, in original order."""
    return [json.loads(line) for line in event_log_jsonl.splitlines() if line]


def assistant_utterances(events: list[dict[str, Any]]) -> list[str]:
    """Every assistant-spoken line, in order."""
    return [
        e["text"]
        for e in events
        if e.get("type") == "utterance" and e.get("role") == "assistant"
    ]


def assistant_utterances_by_checkpoint(
    events: list[dict[str, Any]], initial_checkpoint: str | None = None
) -> list[tuple[str, str]]:
    """(checkpoint_id, utterance_text) pairs, in order.

    Tracks the active checkpoint via ``advance`` events (same pattern as
    ``runner._turns_per_checkpoint``) so each assistant utterance is paired
    with whichever checkpoint was live when it was spoken -- needed to scope
    a never_say check to the checkpoint that declared it, rather than a
    flattened union across the whole playbook (a rule scoped to one
    checkpoint, e.g. banning a price-ceiling phrase, would otherwise
    false-fail an unrelated checkpoint that reuses the same words for a
    different, legitimate reason).
    """
    current = initial_checkpoint
    out: list[tuple[str, str]] = []
    for e in events:
        if e.get("type") == "advance":
            current = e.get("to_checkpoint")
        elif e.get("type") == "utterance" and e.get("role") == "assistant":
            out.append((current or "", e["text"]))
    return out


def tool_calls_and_results(
    events: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair each ``tool_call`` with the ``tool_result`` immediately following it.

    The event log is append-only and a result always follows its call in the
    same turn (see ``ToolExecutor.execute`` — it appends the call event, then
    the result event, and never interleaves two tool executions), so pairing
    by adjacent (call, next-result) scan is exact.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending_call: dict[str, Any] | None = None
    for e in events:
        if e.get("type") == "tool_call":
            pending_call = e
        elif e.get("type") == "tool_result" and pending_call is not None:
            pairs.append((pending_call, e))
            pending_call = None
    return pairs


def slot_writes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ``slot_write`` event, with the ``type`` key stripped (redundant)."""
    return [
        {k: v for k, v in e.items() if k != "type"}
        for e in events
        if e.get("type") == "slot_write"
    ]


def any_real_success(
    events: list[dict[str, Any]], *, tool_ids: set[str] | None = None
) -> bool:
    """Playbook-agnostic evidence bar: at least one tool call really succeeded.

    Without ``tool_ids`` this is deliberately generic (any successful tool
    result at all) so it works for any playbook with no per-playbook
    configuration. Pass ``tool_ids`` (e.g. ``{"confirm_booking"}``) to require
    success specifically from the playbook's terminal/booking tool(s).
    """
    for e in events:
        if e.get("type") != "tool_result" or e.get("ok") is not True:
            continue
        if tool_ids is None or e.get("tool") in tool_ids:
            return True
    return False
