"""Event-log replay harness: re-run the Director over a recorded log.

Regression layer 3 (design doc §5): the optimizer's inner evaluation
primitive. Each recorded user utterance is re-evaluated by the Director
under a (possibly mutated) playbook/prompts, and the replayed decision is
diffed against what the recorded log shows the Director actually did.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .director import CompletesLLM, Director
from .editable import Edit
from .events import AdvanceEvent, Event, EventLog, SlotWriteEvent, UtteranceEvent
from .models import Playbook
from .state import ConversationState

# Runtime-made advance rules: not Director decisions, never diffed.
_RUNTIME_RULES = ("init", "auto", "pipeline", "on_failure")


def _edit_checkpoint_refs(playbook: Playbook, edits: Sequence[Edit]) -> set[str] | None:
    """Full checkpoint refs (``journey.id``) the edits target; None = global.

    FullDoc addresses (``journeys.<j>.checkpoints.<cp>.*``) name their ref
    directly; SimpleDoc addresses (``steps.<id>.*``) match any compiled
    checkpoint whose short id equals the step id. Anything else (persona,
    opening, closing, ...) affects every turn, so the whole set is global.
    """
    refs: set[str] = set()
    all_refs = playbook.checkpoint_ids()
    for edit in edits:
        parts = edit.address.split(".")
        if parts[0] == "journeys" and len(parts) >= 4 and parts[2] == "checkpoints":
            refs.add(f"{parts[1]}.{parts[3]}")
        elif parts[0] == "steps" and len(parts) >= 2:
            matched = {r for r in all_refs if r.partition(".")[2] == parts[1]}
            if not matched:
                return None  # unknown step: be conservative, replay everything
            refs.update(matched)
        else:
            return None
    return refs


def first_affected_version(
    log: EventLog, playbook: Playbook, edits: Sequence[Edit]
) -> int | None:
    """Version of the first event the edits could have influenced (CRO split).

    Checkpoint-scoped edits only matter once the recorded conversation enters
    an edited checkpoint: the boundary is the ``AdvanceEvent`` into it (0 when
    an edited checkpoint is the initial one). Global edits return 0. Returns
    None when no edited checkpoint is ever visited — the edits provably cannot
    change this log's decisions, so nothing needs replaying.
    """
    refs = _edit_checkpoint_refs(playbook, edits)
    if refs is None:
        return 0
    if playbook.initial_checkpoint_id in refs:
        return 0
    for e in log.events:
        if isinstance(e, AdvanceEvent) and e.to_checkpoint in refs:
            return e.version
    return None


class DecisionDiff(BaseModel):
    """One divergence between recorded and replayed Director decisions."""

    at_version: int  # version of the user utterance evaluated
    kind: Literal["advance", "slot", "missing_advance", "extra_advance"]
    recorded: str | None  # recorded value (target / "key=value") or None
    replayed: str | None


class ReplayReport(BaseModel):
    """Outcome of replaying a recorded log against a playbook."""

    turns: int  # user utterances replayed
    turns_held: int = 0  # prefix turns held constant (before from_version)
    advance_matches: int = 0
    slot_matches: int = 0
    diffs: list[DecisionDiff] = Field(default_factory=list)

    @property
    def stable(self) -> bool:
        """True when every replayed decision matched the recorded one."""
        return not self.diffs


def _is_director_advance(e: Event) -> bool:
    """True for advances the Director decided (llm rule or interrupt).

    Pipeline/policy advances are runtime-made, not Director decisions;
    ``interrupt:*`` IS a Director decision and is included.
    """
    return (
        isinstance(e, AdvanceEvent)
        and e.by == "director"
        and e.rule not in _RUNTIME_RULES
        and not e.rule.startswith("policy:")
    )


def _decisions(events: Sequence[Event]) -> tuple[str | None, dict[str, Any]]:
    """Extract (advance target, slot writes) attributable to the Director.

    TODO: quiescence-time slot writes (pipeline ``error_slot`` exports, expr
    ``set:`` writes) are stamped ``by="director"`` by the runtime, so a
    recorded pipeline-failure session replays as unstable even with an
    identical playbook+LLM. Needs a provenance marker on those events.
    """
    target = next((e.to_checkpoint for e in events if _is_director_advance(e)), None)
    slots = {
        e.key: e.value
        for e in events
        if isinstance(e, SlotWriteEvent) and e.by == "director"
    }
    return target, slots


def _diff_advance(
    at: int, recorded: str | None, replayed: str | None
) -> DecisionDiff | None:
    """Diff one turn's advance targets; None when they agree."""
    if recorded == replayed:
        return None
    if recorded is None:
        kind: Literal["advance", "missing_advance", "extra_advance"] = "extra_advance"
    elif replayed is None:
        kind = "missing_advance"
    else:
        kind = "advance"
    return DecisionDiff(at_version=at, kind=kind, recorded=recorded, replayed=replayed)


def _diff_slots(
    at: int, recorded: dict[str, Any], replayed: dict[str, Any]
) -> tuple[int, list[DecisionDiff]]:
    """Diff one turn's slot writes per key; return (matches, diffs)."""
    matches = 0
    diffs: list[DecisionDiff] = []
    for key in sorted(set(recorded) | set(replayed)):
        if key in recorded and key in replayed and recorded[key] == replayed[key]:
            matches += 1
            continue
        diffs.append(
            DecisionDiff(
                at_version=at,
                kind="slot",
                recorded=f"{key}={recorded[key]}" if key in recorded else None,
                replayed=f"{key}={replayed[key]}" if key in replayed else None,
            )
        )
    return matches, diffs


async def replay(
    log: EventLog,
    playbook: Playbook,
    director_llm: CompletesLLM,
    *,
    from_version: int = 0,
) -> ReplayReport:
    """Re-run the Director over each recorded user utterance and diff decisions.

    For each user utterance at version V the log prefix up to and including V
    is folded into state, ``Director.evaluate`` is called, and its decision is
    compared against the Director-attributable events recorded between V and
    the next user utterance. Degraded decisions count as "no decision". Pure
    over ``log``: reads only, never mutates.

    ``from_version`` is the CRO suffix boundary (``first_affected_version``):
    user utterances recorded before it are held constant (counted in
    ``turns_held``), so a candidate edit pays Director calls only for the
    turns it could actually have changed.
    """
    director = Director(playbook, director_llm)
    events = log.events
    all_user_idx = [
        i
        for i, e in enumerate(events)
        if isinstance(e, UtteranceEvent) and e.role == "user"
    ]
    user_idx = [i for i in all_user_idx if events[i].version >= from_version]
    report = ReplayReport(
        turns=len(user_idx), turns_held=len(all_user_idx) - len(user_idx)
    )
    for n, i in enumerate(user_idx):
        at = events[i].version
        end = user_idx[n + 1] if n + 1 < len(user_idx) else len(events)
        rec_target, rec_slots = _decisions(events[i + 1 : end])

        prefix = EventLog(events=events[: i + 1])
        state = ConversationState.fold(prefix, playbook=playbook)
        decision = await director.evaluate(state)
        rep_target, rep_slots = (
            (None, {}) if decision.degraded else _decisions(decision.events)
        )

        adv_diff = _diff_advance(at, rec_target, rep_target)
        if adv_diff is not None:
            report.diffs.append(adv_diff)
        elif rec_target is not None:
            report.advance_matches += 1
        matches, slot_diffs = _diff_slots(at, rec_slots, rep_slots)
        report.slot_matches += matches
        report.diffs.extend(slot_diffs)
    return report
