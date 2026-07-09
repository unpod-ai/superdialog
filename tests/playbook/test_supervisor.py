"""Supervisor (loop 2): trigger detectors, trajectory verdict, apply verbs."""

import json
import textwrap

from superdialog.playbook.agent import PlaybookAgent
from superdialog.playbook.events import (
    AdvanceEvent,
    DegradedEvent,
    RevertEvent,
    ScratchpadEvent,
    SlotWriteEvent,
    SteeringNoteEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import COMPENSATE_MARKER, PlaybookRuntime
from superdialog.playbook.supervisor import (
    Supervisor,
    SupervisorDecision,
    detect_triggers,
)
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_replay import SequencedLLM
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE = {"slots": {}, "advance": None, "note": None}

SUP_YAML = textwrap.dedent("""
    journeys:
      flow:
        checkpoints:
          - id: ask
            goal: "Route the caller"
            turn_budget: 2
            advance_when:
              - {when: "wants to book", judge: llm, to: flow.book}
          - id: book
            goal: "Confirm the booking"
            advance_when:
              - {when: "changes mind", judge: llm, to: flow.ask}
          - id: close
            terminal: true
    tools:
      - id: hold_slot
        method: POST
        url: "https://api.test/hold"
        store_response_as: hold_result
        tier: compensable
        compensate: release_hold
      - id: release_hold
        method: POST
        url: "https://api.test/release"
        tier: reversible
""")


def _pb() -> Playbook:
    return Playbook.from_yaml(SUP_YAML)


def _runtime(
    verdicts: list[dict] | None = None, http: FakeHttp | None = None
) -> PlaybookRuntime:
    rt = PlaybookRuntime(
        _pb(),
        director_llm=SequencedLLM(verdicts or [_IDLE]),
        http=http or FakeHttp([]),
    )
    rt.log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="flow.ask", rule="init")
    )
    return rt


def _user_turn(rt: PlaybookRuntime, text: str = "hello") -> None:
    rt.log.append(UtteranceEvent(role="user", text=text))


# -- trigger detectors ----------------------------------------------------------


def test_no_triggers_on_a_healthy_conversation() -> None:
    rt = _runtime()
    _user_turn(rt)
    assert detect_triggers(rt.state, rt.log, _pb()) == []


def test_repair_streak_trigger() -> None:
    rt = _runtime()
    rt.log.append(SteeringNoteEvent(text="fix a", kind="repair"))
    rt.log.append(SteeringNoteEvent(text="fix b", kind="repair"))
    assert "repair_streak" in detect_triggers(rt.state, rt.log, _pb())


def test_oscillation_trigger() -> None:
    rt = _runtime()
    rt.log.append(
        AdvanceEvent(from_checkpoint="flow.ask", to_checkpoint="flow.book", rule="r")
    )
    rt.log.append(
        AdvanceEvent(from_checkpoint="flow.book", to_checkpoint="flow.ask", rule="r")
    )
    # One return trip is a legitimate change of mind, not derailment.
    assert "oscillation" not in detect_triggers(rt.state, rt.log, _pb())
    rt.log.append(
        AdvanceEvent(from_checkpoint="flow.ask", to_checkpoint="flow.book", rule="r")
    )
    # ...but a full round trip (ask,book,ask,book) is ping-ponging.
    assert "oscillation" in detect_triggers(rt.state, rt.log, _pb())


def test_turn_budget_trigger() -> None:
    rt = _runtime()
    for _ in range(3):  # budget on flow.ask is 2
        _user_turn(rt)
    assert "turn_budget" in detect_triggers(rt.state, rt.log, _pb())


def test_degraded_trigger() -> None:
    rt = _runtime()
    rt.log.append(DegradedEvent(component="director", detail="llm_error"))
    assert "degraded" in detect_triggers(rt.state, rt.log, _pb())


def test_slot_churn_trigger() -> None:
    rt = _runtime()
    for v in ("a", "b", "c"):
        rt.log.append(
            SlotWriteEvent(key="city", value=v, status="provisional", by="director")
        )
    assert "slot_churn:city" in detect_triggers(rt.state, rt.log, _pb())


def test_repeated_interrupt_trigger() -> None:
    rt = _runtime()
    for _ in range(2):
        rt.log.append(
            AdvanceEvent(
                from_checkpoint="flow.ask",
                to_checkpoint="flow.book",
                rule="interrupt:goodbye",
            )
        )
        rt.log.append(
            AdvanceEvent(
                from_checkpoint="flow.book", to_checkpoint="flow.ask", rule="resume"
            )
        )
    assert "repeated_interrupt:goodbye" in detect_triggers(rt.state, rt.log, _pb())


def test_compensation_pending_trigger() -> None:
    rt = _runtime()
    rt.log.append(
        SteeringNoteEvent(text=f"{COMPENSATE_MARKER} hold_slot: ...", kind="repair")
    )
    assert "compensation_pending" in detect_triggers(rt.state, rt.log, _pb())


# -- review ---------------------------------------------------------------------


class _VerdictLLM:
    """Supervisor LLM scripted to one JSON verdict; counts calls."""

    def __init__(self, verdict: dict | str) -> None:
        self.verdict = verdict
        self.calls = 0

    async def complete(self, messages: list[dict], **kwargs: object) -> str:
        self.calls += 1
        return (
            self.verdict if isinstance(self.verdict, str) else json.dumps(self.verdict)
        )


async def test_review_is_free_when_healthy() -> None:
    llm = _VerdictLLM({"action": "none"})
    sup = Supervisor(llm, _pb())
    rt = _runtime()
    _user_turn(rt)
    assert await sup.review(rt) is None
    assert llm.calls == 0  # no trigger -> no LLM spend


async def test_review_fires_on_trigger_and_respects_cooldown() -> None:
    llm = _VerdictLLM({"action": "inject", "note": "steady on"})
    sup = Supervisor(llm, _pb(), cooldown_turns=2)
    rt = _runtime()
    rt.log.append(SteeringNoteEvent(text="fix a", kind="repair"))
    rt.log.append(SteeringNoteEvent(text="fix b", kind="repair"))
    _user_turn(rt)
    decision = await sup.review(rt)
    assert decision is not None and decision.action == "inject"
    assert llm.calls == 1
    _user_turn(rt)  # only one new user turn: still cooling down
    assert await sup.review(rt) is None
    assert llm.calls == 1
    _user_turn(rt)
    assert await sup.review(rt) is not None  # cooldown elapsed
    assert llm.calls == 2


async def test_turn_budget_camp_forces_forward_inject_when_verdict_none() -> None:
    # A caller camping in an early step (turn_budget) must not be judged benign:
    # even when the verdict passively returns none, the supervisor floors it to a
    # forward-steering inject so the camp is broken before the hard backstop.
    llm = _VerdictLLM({"action": "none"})
    sup = Supervisor(llm, _pb())
    rt = _runtime()
    for _ in range(3):  # budget on flow.ask is 2 -> turn_budget fires
        _user_turn(rt)
    decision = await sup.review(rt)
    assert decision is not None
    assert decision.action == "inject"
    assert decision.reason == "turn_budget_camp"
    assert decision.note.strip()  # a real forward steer, not empty
    assert "Route the caller" in decision.note  # references the step's goal


async def test_turn_budget_camp_keeps_a_real_verdict() -> None:
    # The floor only rescues passive verdicts; a decisive redirect is untouched.
    llm = _VerdictLLM({"action": "redirect", "to_checkpoint": "flow.book"})
    sup = Supervisor(llm, _pb())
    rt = _runtime()
    for _ in range(3):
        _user_turn(rt)
    decision = await sup.review(rt)
    assert decision is not None and decision.action == "redirect"


async def test_malformed_verdict_degrades_loudly_but_safely() -> None:
    sup = Supervisor(_VerdictLLM("not json at all"), _pb())
    rt = _runtime()
    rt.log.append(SteeringNoteEvent(text="fix a", kind="repair"))
    rt.log.append(SteeringNoteEvent(text="fix b", kind="repair"))
    _user_turn(rt)
    assert await sup.review(rt) is None
    assert any(
        isinstance(e, DegradedEvent) and e.component == "supervisor"
        for e in rt.log.events
    )


# -- apply verbs ------------------------------------------------------------------


async def test_apply_inject_lands_a_repair_brief() -> None:
    rt = _runtime()
    sup = Supervisor(_VerdictLLM({}), _pb())
    await sup.apply(
        rt,
        SupervisorDecision(action="inject", note="Caller wants Tuesday, not Monday."),
    )
    assert rt.state.steering_note == "Caller wants Tuesday, not Monday."
    assert rt.state.steering_kind == "repair"


async def test_apply_redirect_advances_and_keeps_the_brief() -> None:
    rt = _runtime()
    sup = Supervisor(_VerdictLLM({}), _pb())
    await sup.apply(
        rt,
        SupervisorDecision(
            action="redirect",
            to_checkpoint="flow.book",
            note="Acknowledge the mix-up.",
            reason="belongs at booking",
        ),
    )
    assert rt.state.checkpoint_id == "flow.book"
    adv = [e for e in rt.log.events if isinstance(e, AdvanceEvent)][-1]
    assert adv.by == "supervisor" and adv.rule.startswith("supervisor:")
    # the brief must survive the advance (fold clears steering on advance)
    assert rt.state.steering_note == "Acknowledge the mix-up."


async def test_apply_redirect_rejects_unknown_target() -> None:
    rt = _runtime()
    sup = Supervisor(_VerdictLLM({}), _pb())
    await sup.apply(
        rt, SupervisorDecision(action="redirect", to_checkpoint="flow.ghost")
    )
    assert rt.state.checkpoint_id == "flow.ask"
    assert any(
        isinstance(e, DegradedEvent) and e.component == "supervisor"
        for e in rt.log.events
    )


async def test_apply_rewind_confirmation_roundtrip() -> None:
    http = FakeHttp([(200, {"released": True})])
    rt = _runtime(http=http)
    _user_turn(rt, "book it")
    anchor = rt.log.version
    rt.log.append(
        ToolResultEvent(
            tool="hold_slot", store_as="hold_result", ok=True, data={"hold_id": "h1"}
        )
    )
    sup = Supervisor(_VerdictLLM({}), _pb())
    # 1) unconfirmed rewind across the hold -> confirmation steer, no undo yet
    await sup.apply(
        rt,
        SupervisorDecision(action="rewind", to_version=anchor, reason="wrong slot"),
    )
    assert http.calls == []
    note = rt.state.steering_note or ""
    assert note.startswith(COMPENSATE_MARKER) and "hold_slot" in note
    assert "compensation_pending" in detect_triggers(rt.state, rt.log, _pb())
    # 2) caller approves -> confirmed rewind compensates and reverts
    await sup.apply(
        rt,
        SupervisorDecision(
            action="rewind",
            to_version=anchor,
            confirmed=True,
            note="Rebooked; confirm the new time.",
            reason="approved",
        ),
    )
    assert len(http.calls) == 1 and "release" in http.calls[0]["url"]
    assert "hold_result" not in rt.state.tool_results
    assert any(isinstance(e, RevertEvent) for e in rt.log.events)
    assert rt.state.steering_note == "Rebooked; confirm the new time."


async def test_apply_redirect_to_terminal_is_blocked() -> None:
    """A transcript-derived verdict must not end the call via a terminal jump."""
    rt = _runtime()
    sup = Supervisor(_VerdictLLM({}), _pb())
    await sup.apply(
        rt,
        SupervisorDecision(
            action="redirect", to_checkpoint="flow.close", reason="injected hangup"
        ),
    )
    assert rt.state.checkpoint_id == "flow.ask"  # did not advance
    assert not rt.state.ended
    assert any(
        isinstance(e, DegradedEvent) and "redirect_terminal_blocked" in e.detail
        for e in rt.log.events
    )


async def test_apply_rewind_confirmed_without_pending_marker_still_asks() -> None:
    """A verdict cannot fire compensation in one shot without a surfaced ask."""
    http = FakeHttp([(200, {"released": True})])
    rt = _runtime(http=http)
    _user_turn(rt, "book it")
    anchor = rt.log.version
    rt.log.append(
        ToolResultEvent(tool="hold_slot", store_as="hold_result", ok=True, data={})
    )
    sup = Supervisor(_VerdictLLM({}), _pb())
    # Hostile: confirmed=True on the FIRST attempt, before any confirmation was
    # ever surfaced to the caller. Must degrade to needs_confirmation, not undo.
    await sup.apply(
        rt,
        SupervisorDecision(
            action="rewind", to_version=anchor, confirmed=True, reason="undo now"
        ),
    )
    assert http.calls == []  # compensation did NOT fire
    assert (rt.state.steering_note or "").startswith(COMPENSATE_MARKER)


async def test_apply_rewind_bad_version_degrades() -> None:
    rt = _runtime()
    sup = Supervisor(_VerdictLLM({}), _pb())
    await sup.apply(rt, SupervisorDecision(action="rewind", to_version=99))
    assert any(
        isinstance(e, DegradedEvent) and e.component == "supervisor"
        for e in rt.log.events
    )


async def test_apply_handover_marks_scratchpad_and_steers() -> None:
    rt = _runtime()
    sup = Supervisor(_VerdictLLM({}), _pb())
    await sup.apply(
        rt, SupervisorDecision(action="handover", reason="caller demands a human")
    )
    assert any(
        isinstance(e, ScratchpadEvent) and "handover_requested" in e.text
        for e in rt.log.events
    )
    assert rt.state.steering_kind == "repair"


# -- agent integration -------------------------------------------------------------


async def test_agent_runs_supervisor_after_a_derailed_turn() -> None:
    supervisor_llm = _VerdictLLM(
        {"action": "inject", "note": "Slow down; re-confirm the city."}
    )
    agent = PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=StreamLLM(["Which", " city?"]),
        director_llm=SequencedLLM([_IDLE]),
        http=FakeHttp([]),
        supervisor_llm=supervisor_llm,
    )
    await agent.runtime.start()
    # Derail: two repair steers since entry -> repair_streak trigger.
    agent.runtime.log.append(SteeringNoteEvent(text="fix a", kind="repair"))
    agent.runtime.log.append(SteeringNoteEvent(text="fix b", kind="repair"))
    async for _chunk in agent.stream_turn("uh, hmm"):
        pass
    assert supervisor_llm.calls == 1
    assert agent.runtime.state.steering_note == "Slow down; re-confirm the city."


def test_supervisor_is_off_by_default() -> None:
    """No flag, no explicit llm -> no loop 2 (legacy, byte-compatible)."""
    agent = PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=StreamLLM(["hi"]),
        director_llm=SequencedLLM([_IDLE]),
        http=FakeHttp([]),
    )
    assert agent._supervisor is None


async def test_supervisor_fires_through_dialogmachine_eval_path() -> None:
    """End-to-end through DialogMachine -> PlaybookAgent — the SAME path the
    eval harness (InProcessPlaybook) and production LiteV2 handler take.

    guidelines.supervisor=true activates loop 2 with no explicit supervisor_llm;
    a director that churns 'city' three times trips slot_churn on turn 3, so the
    supervisor reviews the trajectory and injects a repair brief. This is the
    proof the loop is linked to the real communication system, not just to the
    agent constructor in isolation.
    """
    import yaml as _yaml

    from superdialog import DialogMachine

    doc = _yaml.safe_load(MINIMAL_YAML)
    doc["guidelines"] = {"supervisor": True}  # the live wire
    dm = DialogMachine(source=doc, llm="openai/gpt-4o-mini", engine="playbook")
    # One override doubles as Director (each turn) and Supervisor (on trigger):
    # three distinct 'city' values -> slot_churn on turn 3 -> the inject verdict.
    dm._talker_override = StreamLLM(["ok"])
    dm._director_override = SequencedLLM(
        [
            {"slots": {"city": "Pune"}, "advance": None, "note": None},
            {"slots": {"city": "Mumbai"}, "advance": None, "note": None},
            {"slots": {"city": "Delhi"}, "advance": None, "note": None},
            {"action": "inject", "note": "Caller keeps changing the city — pin it."},
        ]
    )
    await dm.turn("Pune")
    await dm.turn("no, Mumbai")
    await dm.turn("actually Delhi")
    agent = dm._pb
    assert agent._supervisor is not None  # the flag wired it, no explicit llm
    assert (
        agent.runtime.state.steering_note == "Caller keeps changing the city — pin it."
    )


async def test_guidelines_flag_wires_supervisor_without_explicit_llm() -> None:
    """guidelines.supervisor=true is the live wire: loop 2 runs with no
    supervisor_llm passed, reusing the Director model. This is the path the
    production handler / DialogMachine / CLI take — they construct a
    PlaybookAgent from the playbook and never pass supervisor_llm."""
    pb = Playbook.from_yaml(MINIMAL_YAML)
    pb.guidelines.supervisor = True
    # One model doubles as Director (turn verdict) and Supervisor (trajectory
    # review): idle for the turn, then an inject decision for the review.
    llm = SequencedLLM([_IDLE, {"action": "inject", "note": "Re-confirm the city."}])
    agent = PlaybookAgent(
        playbook=pb,
        talker_llm=StreamLLM(["Which", " city?"]),
        director_llm=llm,
        http=FakeHttp([]),
        # NB: no supervisor_llm — the guidelines flag alone must wire it.
    )
    assert agent._supervisor is not None
    await agent.runtime.start()
    agent.runtime.log.append(SteeringNoteEvent(text="fix a", kind="repair"))
    agent.runtime.log.append(SteeringNoteEvent(text="fix b", kind="repair"))
    async for _chunk in agent.stream_turn("uh, hmm"):
        pass
    assert agent.runtime.state.steering_note == "Re-confirm the city."
