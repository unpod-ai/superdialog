"""Playbook traversal recorder — derives a session history JSON from an EventLog.

Mirrors the shape of superdialog.traversal.build_traversal so the same
visualisers can consume both dialog-machine and playbook sessions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import (
    AdvanceEvent,
    DegradedEvent,
    EventLog,
    SessionEndEvent,
    SlotWriteEvent,
    ToolCallEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from .models import Playbook
from .state import ConversationState


def build_playbook_traversal(
    log: EventLog,
    playbook: Playbook,
    *,
    source: str = "",
    model: str = "",
    started_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a traversal JSON from a completed playbook EventLog.

    The output mirrors the shape of superdialog.traversal.build_traversal
    so the same visualisers can consume both dialog-machine and playbook
    sessions without schema changes.

    Args:
        log: The EventLog from a completed (or in-progress) session.
        playbook: The Playbook used to run the session.
        source: Display name for the playbook file (e.g. "hotel.yaml").
        model: Model URI used (e.g. "openai/gpt-4o-mini").
        started_at: UTC datetime when the session started; None when unknown.
    """
    ended_at = datetime.now(timezone.utc)
    _ts = (started_at or ended_at).strftime("%Y%m%d_%H%M%S_%f")[:20]
    session_id = f"{_ts}_{os.urandom(3).hex()}"

    events = log.events

    # --- walk events: group into checkpoint windows ---
    # Each window: (advance_event_that_entered_this_cp, [all subsequent events
    # until the next AdvanceEvent])
    windows: list[tuple[AdvanceEvent | None, list[Any]]] = []
    current_advance: AdvanceEvent | None = None
    current_bucket: list[Any] = []

    for e in events:
        if isinstance(e, AdvanceEvent):
            windows.append((current_advance, current_bucket))
            current_advance = e
            current_bucket = []
        else:
            current_bucket.append(e)
    windows.append((current_advance, current_bucket))  # flush last window

    # --- visit counts per checkpoint ---
    visit_count: dict[str, int] = {}
    for adv, _ in windows:
        if adv is not None:
            visit_count[adv.to_checkpoint] = visit_count.get(adv.to_checkpoint, 0) + 1

    # --- build traversal steps ---
    traversal_steps: list[dict[str, Any]] = []
    step_num = 0

    for adv, bucket in windows:
        if adv is None:
            continue  # pre-session env_writes etc.
        step_num += 1

        bot_message: str | None = None
        user_message: str | None = None
        for e in bucket:
            if isinstance(e, UtteranceEvent):
                if e.role == "assistant" and bot_message is None:
                    bot_message = e.text
                elif e.role == "user" and user_message is None:
                    user_message = e.text

        slots_written: dict[str, Any] = {}
        for e in bucket:
            if isinstance(e, SlotWriteEvent):
                slots_written[e.key] = {
                    "value": e.value,
                    "status": e.status,
                    "by": e.by,
                    "version": e.version,
                }

        # Pair ToolCallEvent + ToolResultEvent in FIFO order per tool name.
        # Use a list queue per tool so the same tool called twice doesn't
        # clobber the first call's args with the second.
        tool_calls: list[dict[str, Any]] = []
        pending: dict[str, list[dict[str, Any]]] = {}
        for e in bucket:
            if isinstance(e, ToolCallEvent):
                pending.setdefault(e.tool, []).append({"tool": e.tool, "args": dict(e.args)})
            elif isinstance(e, ToolResultEvent):
                queue = pending.get(e.tool)
                entry = queue.pop(0) if queue else {"tool": e.tool}
                if queue is not None and not queue:
                    del pending[e.tool]
                entry.update({"ok": e.ok, "status": e.status, "error": e.error})
                tool_calls.append(entry)
        for queue in pending.values():  # unpaired (in-progress sessions)
            tool_calls.extend(queue)

        degraded = any(isinstance(e, DegradedEvent) for e in bucket)

        try:
            cp = playbook.checkpoint(adv.to_checkpoint)
            cp_goal, cp_terminal = cp.goal, cp.terminal
        except (KeyError, Exception):
            cp_goal, cp_terminal = "", False

        traversal_steps.append({
            "step": step_num,
            "from_checkpoint": adv.from_checkpoint,
            "to_checkpoint": adv.to_checkpoint,
            "advance_rule": adv.rule,
            "advance_by": adv.by,
            "version": adv.version,
            "goal": cp_goal,
            "bot_message": bot_message,
            "user_message": user_message,
            "slots_written": slots_written,
            "tool_calls": tool_calls,
            "degraded": degraded,
        })

    # --- session outcome ---
    end_event = next((e for e in events if isinstance(e, SessionEndEvent)), None)
    is_complete = end_event is not None
    outcome = end_event.outcome if end_event else None

    # --- final slots ---
    final_state = ConversationState.fold(log, playbook=playbook)
    final_slots = {
        k: {"value": sv.value, "status": sv.status, "by": sv.by}
        for k, sv in final_state.slots.items()
    }

    # --- checkpoint catalogue from playbook ---
    all_cp_ids: list[str] = [
        f"{j_name}.{cp.id}"
        for j_name, journey in playbook.journeys.items()
        for cp in journey.checkpoints
    ]

    checkpoint_nodes: list[dict[str, Any]] = []
    for cp_id in all_cp_ids:
        try:
            cp = playbook.checkpoint(cp_id)
            g, t = cp.goal, cp.terminal
        except (KeyError, Exception):
            g, t = "", False
        checkpoint_nodes.append({
            "id": cp_id,
            "goal": g,
            "is_terminal": t,
            "visited": cp_id in visit_count,
            "visit_count": visit_count.get(cp_id, 0),
        })

    # --- graph edges ---
    # Collect traversed edge keys (from→to:rule) → step they fired at
    traversed_edges: dict[str, int] = {}
    for step in traversal_steps:
        key = f"{step['from_checkpoint']}→{step['to_checkpoint']}:{step['advance_rule']}"
        traversed_edges.setdefault(key, step["step"])

    graph_edges: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    # Edges declared in the playbook's advance_when rules
    for j_name, journey in playbook.journeys.items():
        for cp in journey.checkpoints:
            cp_full = f"{j_name}.{cp.id}"
            for rule in cp.advance_when:
                edge_key = f"{cp_full}→{rule.to}:{rule.rule_id}"
                if edge_key in seen_keys:
                    continue
                seen_keys.add(edge_key)
                graph_edges.append({
                    "id": edge_key,
                    "from_checkpoint": cp_full,
                    "to_checkpoint": rule.to,
                    "rule": rule.rule_id,
                    "condition": rule.when,
                    "judge": rule.judge,
                    "traversed": edge_key in traversed_edges,
                    "traversed_at_step": traversed_edges.get(edge_key),
                })

    # Runtime-synthesised edges (init, auto, pipeline, interrupt, policy)
    # that are not listed in advance_when
    for step in traversal_steps:
        edge_key = f"{step['from_checkpoint']}→{step['to_checkpoint']}:{step['advance_rule']}"
        if edge_key not in seen_keys:
            seen_keys.add(edge_key)
            graph_edges.append({
                "id": edge_key,
                "from_checkpoint": step["from_checkpoint"],
                "to_checkpoint": step["to_checkpoint"],
                "rule": step["advance_rule"],
                "condition": "",
                "judge": "runtime",
                "traversed": True,
                "traversed_at_step": step["step"],
            })

    return {
        "session_id": session_id,
        "playbook_file": source,
        "model": model,
        "started_at": started_at.isoformat() if started_at else None,
        "ended_at": ended_at.isoformat(),
        "is_complete": is_complete,
        "outcome": outcome,
        "checkpoints": checkpoint_nodes,
        "traversal": traversal_steps,
        "final_slots": final_slots,
        "graph": {
            "checkpoints": checkpoint_nodes,
            "advance_edges": graph_edges,
        },
    }


def save_playbook_traversal(traversal: dict[str, Any], out_dir: str | Path) -> Path:
    """Write traversal JSON to out_dir. Creates dir if missing. Returns path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"traversal_{traversal['session_id']}.json"
    path.write_text(json.dumps(traversal, indent=2, ensure_ascii=False))
    return path
