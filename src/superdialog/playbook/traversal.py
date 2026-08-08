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
    SpeechCorrectionEvent,
    ToolCallEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from .models import Playbook
from .state import ConversationState


def _scan_bucket(
    bucket: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Extract slot writes, paired tool calls, and degraded flag from events."""
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
    return slots_written, tool_calls, degraded


def build_playbook_traversal(
    log: EventLog,
    playbook: Playbook,
    *,
    source: str = "",
    model: str = "",
    started_at: datetime | None = None,
    latency: dict[str, Any] | None = None,
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

    # G38 append-only barge-in truncation: exports must carry what the caller
    # HEARD, not what was generated. Dict comprehension keeps the last
    # correction per utterance, matching the fold's last-wins order.
    heard = {
        e.utterance_version: e.heard_text
        for e in events
        if isinstance(e, SpeechCorrectionEvent)
    }

    def _utt_text(e: UtteranceEvent) -> str:
        return heard.get(e.version, e.text)

    # --- per-turn windows ---
    # Window 0 holds everything before the first user utterance (env writes,
    # init/auto advances, greeting). Window k (k >= 1) starts at user turn
    # k's utterance and runs until the next user utterance. Advances keep the
    # same turn mapping as the old _turn_for_advance walk: an advance fired
    # while processing turn k sits inside window k.
    turn_windows: list[list[Any]] = [[]]
    for e in events:
        if isinstance(e, UtteranceEvent) and e.role == "user":
            turn_windows.append([])
        turn_windows[-1].append(e)

    # --- visit counts per checkpoint (advances only; dwell never counts) ---
    visit_count: dict[str, int] = {}
    for e in events:
        if isinstance(e, AdvanceEvent):
            visit_count[e.to_checkpoint] = visit_count.get(e.to_checkpoint, 0) + 1

    # --- build traversal steps ---
    # Every user turn yields at least one step: turns whose window contains
    # AdvanceEvents yield one step per advance (quiescence chains unchanged);
    # turns with no advance yield a DWELL step (from == to, rule="dwell") so
    # multi-turn dwell inside a checkpoint is visible per-turn.
    # In a quiescence chain every step of the turn carries the same
    # bot_message/user_message (the turn's exchange stamped on each of its
    # steps), so a transcript-style consumer must dedupe by turn.
    traversal_steps: list[dict[str, Any]] = []
    step_num = 0
    current_cp: str | None = None

    def _cp_goal(cp_id: str) -> str:
        try:
            return playbook.checkpoint(cp_id).goal
        except Exception:
            return ""

    _dir_turns = (latency or {}).get("director", {}).get("per_turn_ms", [])
    _tlk_turns = (latency or {}).get("talker", {}).get("per_turn_ms", [])

    for turn, window in enumerate(turn_windows):
        # Split the window at its AdvanceEvents: pre_bucket = events before
        # the first advance; each segment = (advance, events after it).
        pre_bucket: list[Any] = []
        segments: list[tuple[AdvanceEvent, list[Any]]] = []
        for e in window:
            if isinstance(e, AdvanceEvent):
                segments.append((e, []))
            elif segments:
                segments[-1][1].append(e)
            else:
                pre_bucket.append(e)

        if turn == 0:
            # Pre-first-user-turn (init/env, greeting): today's behavior —
            # one step per advance, no turn/latency; pre-init events dropped.
            for adv, bucket in segments:
                step_num += 1
                bot = next(
                    (_utt_text(e) for e in bucket
                     if isinstance(e, UtteranceEvent) and e.role == "assistant"),
                    None,
                )
                slots_written, tool_calls, degraded = _scan_bucket(bucket)
                traversal_steps.append({
                    "step": step_num,
                    "from_checkpoint": adv.from_checkpoint,
                    "to_checkpoint": adv.to_checkpoint,
                    "advance_rule": adv.rule,
                    "advance_by": adv.by,
                    "corroborated": None,
                    "version": adv.version,
                    "goal": _cp_goal(adv.to_checkpoint),
                    "bot_message": bot,
                    "user_message": None,
                    "slots_written": slots_written,
                    "tool_calls": tool_calls,
                    "degraded": degraded,
                    "turn": None,
                    "director_ms": None,
                    "talker_ms": None,
                })
                current_cp = adv.to_checkpoint
            continue

        # window[0] is this turn's user utterance; the paired bot_message is
        # the first assistant utterance AFTER it (never one from before).
        user_message = window[0].text
        bot_message = next(
            (_utt_text(e) for e in window[1:]
             if isinstance(e, UtteranceEvent) and e.role == "assistant"),
            None,
        )
        # Per-turn latency (0-indexed arrays, turn 1 = index 0).
        director_ms = _dir_turns[turn - 1] if turn <= len(_dir_turns) else None
        talker_ms = _tlk_turns[turn - 1] if turn <= len(_tlk_turns) else None

        if not segments:
            if current_cp is None:
                continue  # user spoke before any checkpoint was entered
            step_num += 1
            slots_written, tool_calls, degraded = _scan_bucket(window)
            traversal_steps.append({
                "step": step_num,
                "from_checkpoint": current_cp,
                "to_checkpoint": current_cp,
                "advance_rule": "dwell",
                "advance_by": None,
                "corroborated": None,
                "version": window[0].version,
                "goal": _cp_goal(current_cp),
                "bot_message": bot_message,
                "user_message": user_message,
                "slots_written": slots_written,
                "tool_calls": tool_calls,
                "degraded": degraded,
                "turn": turn,
                "director_ms": director_ms,
                "talker_ms": talker_ms,
            })
            continue

        # Advance turn: one step per advance; the first advance also absorbs
        # the pre-advance events so the director's slot writes land on the
        # step they caused. No dwell step for this turn (no double-counting).
        for i, (adv, bucket) in enumerate(segments):
            step_num += 1
            scan = pre_bucket + bucket if i == 0 else bucket
            slots_written, tool_calls, degraded = _scan_bucket(scan)
            traversal_steps.append({
                "step": step_num,
                "from_checkpoint": adv.from_checkpoint,
                "to_checkpoint": adv.to_checkpoint,
                "advance_rule": adv.rule,
                "advance_by": adv.by,
                "corroborated": adv.corroborated,
                "version": adv.version,
                "goal": _cp_goal(adv.to_checkpoint),
                "bot_message": bot_message,
                "user_message": user_message,
                "slots_written": slots_written,
                "tool_calls": tool_calls,
                "degraded": degraded,
                "turn": turn,
                "director_ms": director_ms,
                "talker_ms": talker_ms,
            })
            current_cp = adv.to_checkpoint

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
    # Collect traversed edge keys (from→to:rule) → step they fired at.
    # Dwell steps are stays, not transitions — they never become edges.
    traversed_edges: dict[str, int] = {}
    for step in traversal_steps:
        if step["advance_rule"] == "dwell":
            continue
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
    # that are not listed in advance_when. Dwell steps stay excluded.
    for step in traversal_steps:
        if step["advance_rule"] == "dwell":
            continue
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
        "latency": latency or {},
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
