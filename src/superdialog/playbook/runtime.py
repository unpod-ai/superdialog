"""PlaybookRuntime: owns the event log, runs the system to quiescence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .director import CompletesLLM, Director
from .events import (
    AdvanceEvent,
    DegradedEvent,
    EnvWriteEvent,
    Event,
    EventLog,
    ExternalEvent,
    SessionEndEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from .models import Checkpoint, Playbook
from .pipeline import PipelineRunner
from .render import render_template
from .state import ConversationState
from .toolexec import HttpFn, PythonToolFn, ToolExecutor

_WRAP_UP_NOTE = "wrap this step up; offer the essentials and move on"
_TURN_BUDGET_GRACE = 2  # extra turns past budget before on_failure routing


class ExternalResult(BaseModel):
    """Outcome of an external event the host may need to act on."""

    prompt: str | None = None  # speech the host should play (silence policy)


class PlaybookRuntime:
    """Central conductor: appends events, then runs the system to quiescence.

    Owns the EventLog (``rt.log.append(...)`` is a supported public pattern),
    applies policies (silence, turn budget), surfaces pass-through speech from
    say_verbatim checkpoints traversed during quiescence, and records Director
    degradation and stale-speech repair notes as events.
    """

    def __init__(
        self,
        playbook: Playbook,
        director_llm: CompletesLLM,
        http: HttpFn,
        python_tools: dict[str, PythonToolFn] | None = None,
        max_hops: int = 8,
    ) -> None:
        self.log = EventLog()
        self._pb = playbook
        self._director = Director(playbook, director_llm)
        self._executor = ToolExecutor(http=http, python_tools=python_tools)
        self._pipelines = PipelineRunner(playbook, self._executor)
        self._max_hops = max_hops
        self._state_cache: ConversationState | None = None
        self._state_cache_version = -1

    @property
    def state(self) -> ConversationState:
        """Current state, refolded whenever the log has new events.

        Returns a shared cached snapshot; callers must not mutate it.
        """
        if self._state_cache is None or self._state_cache_version != self.log.version:
            self._state_cache = ConversationState.fold(self.log, self._pb)
            self._state_cache_version = self.log.version
        return self._state_cache

    def load_log(self, log: EventLog) -> None:
        """Replace the event log wholesale and invalidate the state cache.

        The cache's version check alone cannot tell two different logs of
        equal length apart, so a wholesale swap must drop the snapshot too.
        """
        self.log = log
        self._state_cache = None
        self._state_cache_version = -1

    async def start(self) -> list[str]:
        """Seed env, enter the initial checkpoint, and run to quiescence."""
        for key, value in self._pb.env.items():
            self.log.append(EnvWriteEvent(key=key, value=value))
        pass_through: list[str] = []
        await self._advance(self._pb.initial_checkpoint_id, "init", pass_through)
        pass_through.extend(await self._quiesce())
        return pass_through

    async def on_user_text(self, text: str, *, record: bool = True) -> list[str]:
        """Process one user utterance: Director verdict, policies, quiescence.

        Returning implies the runtime is quiescent: all hops and policies have
        resolved (Task 11's barrier relies on this as API).

        ``record`` defaults to True so the utterance is appended to the log.
        The streaming agent appends it itself BEFORE snapshotting the Talker's
        state (so the Talker sees the current turn), then calls this with
        ``record=False`` to avoid a double-append. Everything below reads the
        folded ``self.state``, so skipping the append is safe.
        """
        if record:
            self.log.append(UtteranceEvent(role="user", text=text))
        decision = await self._director.evaluate(self.state)
        pass_through: list[str] = []
        if decision.degraded:
            self.log.append(DegradedEvent(component="director", detail=decision.detail))
            # LLM-free policies still apply in degraded mode.
            await self._apply_turn_budget(pass_through)
            pass_through.extend(await self._quiesce())
            return pass_through
        advance = next(
            (e for e in decision.events if isinstance(e, AdvanceEvent)), None
        )
        if advance is not None and not advance.rule.startswith("interrupt:"):
            current = self.state.checkpoint_id
            if current is not None:  # we are leaving: surface its verbatim line
                self._speak_verbatim(self._pb.checkpoint(current), pass_through)
        await self._apply_with_entry(decision.events, pass_through)
        await self._apply_turn_budget(pass_through)
        pass_through.extend(await self._quiesce())
        return pass_through

    async def on_external(self, event: ExternalEvent) -> ExternalResult:
        """Record an external event and apply the matching policy/handler."""
        self.log.append(event)
        if event.kind == "silence":
            return await self._handle_silence()
        return await self._handle_handler(event)

    async def check_repairs(self) -> None:
        """Emit a repair note when the Talker re-asked for an answered slot."""
        last = next(
            (
                e
                for e in reversed(self.log.events)
                if isinstance(e, UtteranceEvent)
                and e.role == "assistant"
                and e.spoke_from_version is not None
            ),
            None,
        )
        if last is None or last.spoke_from_version is None or "?" not in last.text:
            return
        if any(  # idempotent: this stale utterance was already repaired
            isinstance(e, SteeringNoteEvent)
            and e.kind == "repair"
            and e.version > last.version
            for e in self.log.events
        ):
            return
        cp_id = self.state.checkpoint_id
        if cp_id is None:
            return
        cp = self._pb.checkpoint(cp_id)
        for e in self.log.events:
            if (
                isinstance(e, SlotWriteEvent)
                and e.version > last.spoke_from_version
                and e.status == "confirmed"
                and e.key in cp.slots
            ):
                self.log.append(
                    SteeringNoteEvent(
                        kind="repair",
                        text=(
                            f"You already have {e.key}={e.value}; "
                            "acknowledge it instead of re-asking."
                        ),
                    )
                )
                return

    # -- quiescence -----------------------------------------------------------

    async def _quiesce(self) -> list[str]:
        """Hop through pipelines/expr rules/auto advances until stable."""
        pass_through: list[str] = []
        for _ in range(self._max_hops):
            if not await self._hop(pass_through):
                break
        else:  # never went quiescent: audit the runaway hop loop
            self.log.append(
                DegradedEvent(component="director", detail="quiesce_hop_exhaustion")
            )
        return pass_through

    async def _hop(self, pass_through: list[str]) -> bool:
        """One quiescence hop; True when the system moved and should re-hop."""
        state = self.state
        if state.ended or state.checkpoint_id is None:
            return False
        cp = self._pb.checkpoint(state.checkpoint_id)
        if cp.pipeline and not self._pipeline_ran_this_entry(state):
            if await self._run_checkpoint_pipeline(cp, cp.pipeline, pass_through):
                return True
            # pipeline finished without routing: fall through to expr rules
        decision = await self._director.evaluate(self.state, expr_only=True)
        if decision.events:
            advanced = any(isinstance(e, AdvanceEvent) for e in decision.events)
            failure_routed = any(  # error_context write == failure routing
                isinstance(e, SlotWriteEvent) and e.key == "error_context"
                for e in decision.events
            )
            if advanced and not failure_routed:
                self._speak_verbatim(cp, pass_through)
            await self._apply_with_entry(decision.events, pass_through)
            return True
        if cp.auto and cp.advance_when:
            self._speak_verbatim(cp, pass_through)
            await self._advance(cp.advance_when[0].to, "auto", pass_through)
            return True
        return False

    async def _run_checkpoint_pipeline(
        self, cp: Checkpoint, pipeline_id: str, pass_through: list[str]
    ) -> bool:
        """Run the checkpoint's pipeline; True when it routed (hop consumed)."""
        result = await self._pipelines.run(pipeline_id, self.state)
        self._apply(result.events)
        self.log.append(
            ToolResultEvent(tool=pipeline_id, store_as="pipeline", ok=result.ok)
        )
        for key, value in result.error_slot.items():
            self.log.append(
                SlotWriteEvent(key=key, value=value, status="confirmed", by="director")
            )
        if result.advance_to:
            if result.ok:  # success routing surfaces the verbatim line;
                self._speak_verbatim(cp, pass_through)  # retry-exhaust stays silent
            await self._advance(result.advance_to, "pipeline", pass_through)
            return True
        if not result.ok and cp.on_failure:
            await self._advance(cp.on_failure, "on_failure", pass_through)
            return True
        return False

    # -- policies ---------------------------------------------------------------

    async def _handle_silence(self) -> ExternalResult:
        policy = self._pb.policies.silence
        if policy is None:
            return ExternalResult()
        n = self.state.silence_count
        if n <= policy.max_prompts:
            prompt = policy.prompts[n - 1] if n - 1 < len(policy.prompts) else None
            if prompt is not None:  # the user hears it: it belongs in the log
                self.log.append(UtteranceEvent(role="assistant", text=prompt))
            return ExternalResult(prompt=prompt)
        if not policy.then:
            return ExternalResult()
        await self._advance(policy.then, "policy:silence", [])
        await self._quiesce()
        return ExternalResult()

    async def _handle_handler(self, event: ExternalEvent) -> ExternalResult:
        key = f"{event.kind}.{event.name}"
        handler = next((h for h in self._pb.handlers if h.on == key), None)
        if handler is None:
            return ExternalResult()
        result = await self._pipelines.run(handler.pipeline, self.state)
        self._apply(result.events)
        for k, v in result.error_slot.items():
            self.log.append(
                SlotWriteEvent(key=k, value=v, status="confirmed", by="director")
            )
        pass_through: list[str] = []
        if result.advance_to:
            # Handler advances stay silent: ExternalResult cannot carry
            # pass-through speech, and logging unheard speech would make the
            # transcript lie about what the user heard.
            await self._advance(result.advance_to, "pipeline", pass_through)
        await self._quiesce()
        return ExternalResult()

    async def _apply_turn_budget(self, pass_through: list[str]) -> None:
        state = self.state
        if state.ended or state.checkpoint_id is None:
            return
        cp = self._pb.checkpoint(state.checkpoint_id)
        if not cp.turn_budget or state.user_turns_in_checkpoint <= cp.turn_budget:
            return
        self.log.append(SteeringNoteEvent(text=_WRAP_UP_NOTE, kind="steer"))
        over_grace = (
            state.user_turns_in_checkpoint > cp.turn_budget + _TURN_BUDGET_GRACE
        )
        if over_grace and cp.on_failure:
            await self._advance(cp.on_failure, "policy:turn_budget", pass_through)

    # -- transitions --------------------------------------------------------------

    async def _advance(self, to: str, rule: str, pass_through: list[str]) -> None:
        """Leave the current checkpoint for ``to`` and enter the target."""
        by: Literal["director", "policy"] = (
            "policy" if rule.startswith("policy:") else "director"
        )
        self.log.append(
            AdvanceEvent(
                from_checkpoint=self.state.checkpoint_id,
                to_checkpoint=to,
                rule=rule,
                by=by,
            )
        )
        await self._enter(to, pass_through)

    async def _enter(self, cp_ref: str, pass_through: list[str]) -> None:
        """Run on_enter tools (failures are data) and handle terminal ends."""
        cp = self._pb.checkpoint(cp_ref)
        for tool_id in cp.on_enter:
            events = await self._executor.execute(self._pb.tool(tool_id), self.state)
            self._apply(events)
        if cp.terminal:
            self._speak_verbatim(cp, pass_through)
            if not self.state.ended:
                self.log.append(SessionEndEvent(outcome=cp.outcome))

    def _speak_verbatim(self, cp: Checkpoint, pass_through: list[str]) -> None:
        """Speak cp.say_verbatim at most once per checkpoint entry."""
        if not cp.say_verbatim:
            return
        state = self.state
        text = render_template(cp.say_verbatim, self._pb, state)
        for e in self.log.events:
            if (
                isinstance(e, UtteranceEvent)
                and e.role == "assistant"
                and e.version > state.checkpoint_entered_version
                and e.text == text
            ):
                return  # already spoken since this entry
        # No spoke_from_version: author verbatim can't be stale, and stamping
        # one would draw spurious repair notes on question-shaped verbatims.
        self.log.append(UtteranceEvent(role="assistant", text=text))
        pass_through.append(text)

    def _pipeline_ran_this_entry(self, state: ConversationState) -> bool:
        result = state.tool_results.get("pipeline")
        return result is not None and result.version > state.checkpoint_entered_version

    def _apply(self, events: list[Event]) -> None:
        for e in events:  # events arrive unstamped from director/pipeline
            self.log.append(e)

    async def _apply_with_entry(
        self, events: list[Event], pass_through: list[str]
    ) -> None:
        """Apply events; when they changed the checkpoint, enter the target."""
        before = self.state.checkpoint_id
        self._apply(events)
        after = self.state.checkpoint_id
        if after is not None and after != before:
            await self._enter(after, pass_through)
