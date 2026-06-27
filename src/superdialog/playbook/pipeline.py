"""Director-side pipeline runner: ordered tool steps with typed result branches."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .events import Event, EventLog, ToolResultEvent
from .models import Playbook, RetrySpec
from .state import ConversationState
from .toolexec import ToolExecutor


class PipelineResult(BaseModel):
    ok: bool = True
    advance_to: str | None = None
    events: list[Event] = Field(default_factory=list)
    error_slot: dict[str, str] = Field(default_factory=dict)


def _refold(state: ConversationState, events: list[Event]) -> ConversationState:
    """Overlay pipeline-internal events so later steps see results/env updates.

    Only results and env are overlaid. A tool's ``slot_updates`` may emit a
    SlotWriteEvent, but slots are deliberately NOT overlaid mid-pipeline (slot
    confirmation is the Director's domain); such writes land when the caller
    applies ``result.events`` to the log at pipeline end. A later step needing
    the value within the same pipeline should read it from env, not the slot.
    """
    log = EventLog()
    for e in events:
        log.append(e.model_copy(update={"version": 0}))
    overlay = ConversationState.fold(log)
    merged = state.model_copy(deep=True)
    merged.tool_results.update(overlay.tool_results)
    merged.env.update(overlay.env)
    for tool, n in overlay.tool_call_counts.items():
        merged.tool_call_counts[tool] = merged.tool_call_counts.get(tool, 0) + n
    return merged


class PipelineRunner:
    def __init__(self, playbook: Playbook, executor: ToolExecutor) -> None:
        self._pb = playbook
        self._ex = executor

    async def _execute_with_middleware(
        self, tool_id: str, state: ConversationState
    ) -> list[Event]:
        events = await self._ex.execute(self._pb.tool(tool_id), state)
        mw = self._pb.middleware
        result = next((e for e in events if isinstance(e, ToolResultEvent)), None)
        if mw and result is not None and result.status == mw.on_status:
            refresh_events = await self._ex.execute(
                self._pb.tool(mw.refresh_with), state
            )
            # Replay state overlays only the refresh events; a run_once tool
            # that 401s is intentionally replayed (real-log count converges
            # after the caller appends these events).
            state = _refold(state, refresh_events)
            replay_events = await self._ex.execute(self._pb.tool(tool_id), state)
            return [*events, *refresh_events, *replay_events]
        return events

    async def run(self, pipeline_id: str, state: ConversationState) -> PipelineResult:
        """Run a pipeline's steps in order and report the outcome.

        The returned ``result.events`` are NOT yet in any log: the caller
        must append them to the real event log (including middleware refresh
        events, so a refreshed token outlives the run). Error reporting is
        asymmetric: typed ``http_<code>`` branches are author-handled and
        leave ``error_slot`` empty, while retry-exhaust and no-branch
        failures set ``error_context``.
        """
        spec = self._pb.pipeline(pipeline_id)
        result = PipelineResult()
        for step in spec.steps:
            attempts = 0
            outcome: str | None = None
            last: ToolResultEvent | None = None
            while True:
                events = await self._execute_with_middleware(step.tool, state)
                result.events.extend(events)
                state = _refold(state, events)
                last = next(
                    (e for e in reversed(events) if isinstance(e, ToolResultEvent)),
                    None,
                )
                if last is None:  # skipped (when/run_once): treat as ok-continue
                    outcome = "continue"
                    break
                key = (
                    "ok"
                    if last.ok
                    else f"http_{last.status}"
                    if last.status
                    else "failed"
                )
                target = step.on.get(key)
                if target is None:
                    target = step.on.get("ok" if last.ok else "failed")
                if isinstance(target, RetrySpec):
                    if attempts < target.retry:
                        attempts += 1
                        continue
                    result.ok = False
                    result.advance_to = target.on_exhaust
                    result.error_slot = {"error_context": f"{pipeline_id}:{step.tool}"}
                    return result
                outcome = (
                    target if target is not None else ("continue" if last.ok else None)
                )
                break
            if outcome is None:  # failure with no branch: stop, report
                result.ok = False
                result.error_slot = {"error_context": f"{pipeline_id}:{step.tool}"}
                return result
            if outcome != "continue":
                result.ok = last is None or last.ok
                result.advance_to = outcome
                return result
        return result
