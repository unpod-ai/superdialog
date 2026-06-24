import textwrap

from superdialog.playbook.events import (
    ExternalEvent,
    SessionEndEvent,
    SlotWriteEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_toolexec import FakeHttp


def _runtime(llm_payload: dict, http_responses=()) -> PlaybookRuntime:
    pb = Playbook.from_yaml(MINIMAL_YAML)
    return PlaybookRuntime(
        pb,
        director_llm=CannedLLM(llm_payload),
        http=FakeHttp(list(http_responses)),
    )


async def test_session_start_enters_initial_checkpoint_and_seeds_env() -> None:
    pb = Playbook.from_yaml(MINIMAL_YAML).model_copy(
        update={"env": {"API_BASE_URL": "https://api.test"}}
    )
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=FakeHttp([]),
    )
    await rt.start()
    assert rt.state.checkpoint_id == "booking.collect"
    assert rt.state.env["API_BASE_URL"] == "https://api.test"


async def test_user_event_advances_through_pipeline_to_terminal() -> None:
    rt = _runtime(
        {
            "slots": {"city": "Pune", "date": "2026-06-11"},
            "advance": "booking.confirm",
            "note": None,
        },
        http_responses=[(200, {"data": {"hold_id": "h1"}})],
    )
    await rt.start()
    speech = await rt.on_user_text("Pune tomorrow")
    # collect -> confirm (llm) -> pipeline (hold ok -> continue) ->
    # expr rule pipeline.ok -> close (terminal)
    assert rt.state.ended and rt.state.outcome == "confirmed"
    assert any(isinstance(e, SessionEndEvent) for e in rt.log.events)
    # confirm's say_verbatim surfaced as pass-through speech + logged utterance
    assert any("held" in s for s in speech)
    assert any(
        e.type == "utterance" and e.role == "assistant" and "held" in e.text
        for e in rt.log.events
    )


async def test_silence_policy_prompts_then_routes() -> None:
    rt = _runtime({"slots": {}, "advance": None, "note": None})
    await rt.start()
    r1 = await rt.on_external(ExternalEvent(kind="silence", name="user_silence"))
    assert r1.prompt == "Can you hear me?"
    r2 = await rt.on_external(ExternalEvent(kind="silence", name="user_silence"))
    assert r2.prompt == "Are you there?"
    r3 = await rt.on_external(ExternalEvent(kind="silence", name="user_silence"))
    assert r3.prompt is None
    assert rt.state.checkpoint_id == "booking.close"


async def test_degraded_director_is_logged_not_fatal() -> None:
    class BadLLM:
        async def complete(self, messages, **kwargs) -> str:
            return "not json {"

    pb = Playbook.from_yaml(MINIMAL_YAML)
    rt = PlaybookRuntime(pb, director_llm=BadLLM(), http=FakeHttp([]))
    await rt.start()
    await rt.on_user_text("hello?")
    assert any(e.type == "degraded" for e in rt.log.events)
    assert not rt.state.ended


async def test_stale_talker_speech_triggers_repair_note() -> None:
    rt = _runtime({"slots": {}, "advance": None, "note": None})
    await rt.start()
    stale_version = rt.state.version
    rt.log.append(UtteranceEvent(role="user", text="my city is Pune"))
    rt.log.append(
        UtteranceEvent(
            role="assistant",
            text="Which city would you like?",
            spoke_from_version=stale_version,
        )
    )
    rt.log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    await rt.check_repairs()
    notes = [e for e in rt.log.events if e.type == "steering_note"]
    assert any(n.kind == "repair" for n in notes)


TURN_BUDGET_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: a
            goal: "chat"
            turn_budget: 1
            slots:
              x: {type: str}
            advance_when:
              - {when: "user is done", judge: llm, to: j.b}
          - id: b
            terminal: true
            outcome: done
""")


async def test_turn_budget_steers() -> None:
    pb = Playbook.from_yaml(TURN_BUDGET_YAML)
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=FakeHttp([]),
    )
    await rt.start()
    await rt.on_user_text("hello")
    assert not [e for e in rt.log.events if e.type == "steering_note"]
    await rt.on_user_text("still chatting")
    notes = [e for e in rt.log.events if e.type == "steering_note"]
    assert any("wrap" in n.text for n in notes)
    # budget exceeded but within grace and no on_failure: stay put
    assert rt.state.checkpoint_id == "j.a"


INTERRUPT_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: a
            goal: "chat"
            say_verbatim: "Your booking is held."
            advance_when:
              - {when: "user is done", judge: llm, to: j.done}
          - id: done
            terminal: true
            outcome: done
    interrupts:
      - {id: goodbye, when: "caller says goodbye", judge: llm,
         to: j.done, resume: false}
""")


async def test_interrupt_exit_does_not_speak_verbatim() -> None:
    pb = Playbook.from_yaml(INTERRUPT_YAML)
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM(
            {"slots": {}, "advance": None, "note": None, "interrupt": "goodbye"}
        ),
        http=FakeHttp([]),
    )
    await rt.start()
    speech = await rt.on_user_text("goodbye now")
    # interrupt bail-out must not speak the checkpoint's success line
    assert rt.state.ended
    assert not any("held" in s for s in speech)
    assert not any(
        e.type == "utterance" and e.role == "assistant" and "held" in e.text
        for e in rt.log.events
    )


PIPELINE_ROUTE_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: collect
            goal: "get started"
            advance_when:
              - {when: "ready to hold", judge: llm, to: j.hold}
          - id: hold
            say_verbatim: "Your spot is held."
            pipeline: hold_pipe
          - id: done
            terminal: true
            outcome: done
    tools:
      - id: hold_slot
        method: POST
        url: "https://api.test/hold"
    pipelines:
      - id: hold_pipe
        steps:
          - tool: hold_slot
            on: {ok: j.done}
""")


async def test_pipeline_success_routing_speaks_verbatim() -> None:
    pb = Playbook.from_yaml(PIPELINE_ROUTE_YAML)
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": "j.hold", "note": None}),
        http=FakeHttp([(200, {"data": {}})]),
    )
    await rt.start()
    speech = await rt.on_user_text("book it")
    assert rt.state.ended and rt.state.outcome == "done"
    # ok-routing surfaces the routed checkpoint's verbatim line
    assert any("held" in s for s in speech)


async def test_silence_prompts_logged() -> None:
    rt = _runtime({"slots": {}, "advance": None, "note": None})
    await rt.start()
    r = await rt.on_external(ExternalEvent(kind="silence", name="user_silence"))
    assert r.prompt == "Can you hear me?"
    assert any(
        e.type == "utterance" and e.role == "assistant" and e.text == "Can you hear me?"
        for e in rt.log.events
    )


async def test_check_repairs_idempotent() -> None:
    rt = _runtime({"slots": {}, "advance": None, "note": None})
    await rt.start()
    stale_version = rt.state.version
    rt.log.append(UtteranceEvent(role="user", text="my city is Pune"))
    rt.log.append(
        UtteranceEvent(
            role="assistant",
            text="Which city would you like?",
            spoke_from_version=stale_version,
        )
    )
    rt.log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    await rt.check_repairs()
    await rt.check_repairs()
    repairs = [
        e for e in rt.log.events if e.type == "steering_note" and e.kind == "repair"
    ]
    assert len(repairs) == 1


PING_PONG_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: a
            auto: true
            advance_when:
              - {when: "always", judge: llm, to: j.b}
          - id: b
            auto: true
            advance_when:
              - {when: "always", judge: llm, to: j.a}
""")


async def test_hop_exhaustion_logged() -> None:
    pb = Playbook.from_yaml(PING_PONG_YAML)
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=FakeHttp([]),
    )
    await rt.start()  # a <-> b auto ping-pong never settles
    assert any(
        e.type == "degraded" and e.detail == "quiesce_hop_exhaustion"
        for e in rt.log.events
    )


async def test_start_seeds_session_start_anchor() -> None:
    from superdialog.playbook.events import SessionStartEvent
    from superdialog.playbook.models import GuidelineConfig

    base = Playbook.from_yaml(MINIMAL_YAML)
    pb = base.model_copy(update={"guidelines": GuidelineConfig(timezone="Asia/Kolkata")})
    rt = PlaybookRuntime(
        pb,
        director_llm=CannedLLM({"slots": {}, "advance": None, "note": None}),
        http=FakeHttp([]),
    )
    await rt.start()
    starts = [e for e in rt.log.events if isinstance(e, SessionStartEvent)]
    assert len(starts) == 1
    assert starts[0].timezone == "Asia/Kolkata"
    assert rt.log.events[0].type == "session_start"
    assert rt.state.now is not None and rt.state.now.tzinfo is not None


async def test_degraded_path_still_applies_policies() -> None:
    class BadLLM:
        async def complete(self, messages, **kwargs) -> str:
            return "not json {"

    pb = Playbook.from_yaml(TURN_BUDGET_YAML)
    rt = PlaybookRuntime(pb, director_llm=BadLLM(), http=FakeHttp([]))
    await rt.start()
    await rt.on_user_text("hello")
    await rt.on_user_text("still here")
    assert any(e.type == "degraded" for e in rt.log.events)
    # turn budget (LLM-free) still steers despite Director degradation
    notes = [e for e in rt.log.events if e.type == "steering_note"]
    assert any("wrap" in n.text for n in notes)
