"""Intent interception: tier-gated guards before tools materialize."""

import json
import textwrap

from superdialog.playbook.events import SlotWriteEvent, SteeringNoteEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from tests.playbook.test_replay import SequencedLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE = {"slots": {}, "advance": None, "note": None}
_TO_ACT = {"slots": {"city": "Pune"}, "advance": "flow.act", "note": None}

INTERCEPT_YAML = textwrap.dedent("""
    journeys:
      flow:
        checkpoints:
          - id: gather
            goal: "Get the city"
            gate: soft
            slots:
              city: {type: str, required: true}
            advance_when:
              - {when: "ready", judge: llm, to: flow.act, requires: [city]}
          - id: act
            goal: "Place the hold"
            gate: soft
            slots:
              date: {type: str, required: true}
            on_enter: [hold_slot]
            advance_when:
              - {when: "held", judge: llm, to: flow.done}
          - id: done
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
      - id: legacy_post
        method: POST
        url: "https://api.test/legacy"
        store_response_as: legacy_result
""")


def _pb(yaml: str = INTERCEPT_YAML) -> Playbook:
    return Playbook.from_yaml(yaml)


async def test_compensable_on_enter_denied_until_required_met() -> None:
    http = FakeHttp([(200, {"hold_id": "h1"})])
    rt = PlaybookRuntime(
        _pb(), director_llm=SequencedLLM([_TO_ACT, _IDLE, _IDLE]), http=http
    )
    await rt.start()
    # Turn 1: advance into flow.act; date is missing -> hold_slot is denied.
    await rt.on_user_text("book me in Pune")
    assert rt.state.checkpoint_id == "flow.act"
    assert http.calls == []  # the POST never fired
    assert rt.state.steering_note is not None
    assert "hold_slot" in rt.state.steering_note
    # Turn 2: the missing detail arrives -> the denied tool finally runs.
    rt.log.append(
        SlotWriteEvent(key="date", value="2026-07-10", status="confirmed", by="tool")
    )
    await rt.on_user_text("the 10th")
    assert len(http.calls) == 1
    assert "hold" in http.calls[0]["url"]
    assert rt.state.tool_results["hold_result"].ok


async def test_denied_on_enter_does_not_spam_steers() -> None:
    rt = PlaybookRuntime(
        _pb(), director_llm=SequencedLLM([_TO_ACT, _IDLE, _IDLE]), http=FakeHttp([])
    )
    await rt.start()
    await rt.on_user_text("book me in Pune")
    await rt.on_user_text("hmm")  # still no date: denial repeats, steer doesn't
    steers = [
        e
        for e in rt.log.events
        if isinstance(e, SteeringNoteEvent) and "hold_slot" in e.text
    ]
    assert len(steers) == 1


async def test_unannotated_tool_keeps_legacy_behavior() -> None:
    yaml = INTERCEPT_YAML.replace("on_enter: [hold_slot]", "on_enter: [legacy_post]")
    http = FakeHttp([(200, {"ok": True})])
    rt = PlaybookRuntime(_pb(yaml), director_llm=SequencedLLM([_TO_ACT]), http=http)
    await rt.start()
    await rt.on_user_text("book me in Pune")  # date still missing...
    assert len(http.calls) == 1  # ...but unannotated tools are never guarded


class _AllowLLM:
    """Intercept classifier scripted to a fixed allow verdict."""

    def __init__(self, allow: bool) -> None:
        self.allow = allow
        self.calls = 0

    async def complete(self, messages: list[dict], **kwargs: object) -> str:
        self.calls += 1
        return json.dumps({"allow": self.allow})


def _irreversible_yaml() -> str:
    yaml = INTERCEPT_YAML.replace("tier: compensable", "tier: irreversible")
    return "\n".join(ln for ln in yaml.splitlines() if "compensate:" not in ln)


async def test_irreversible_classifier_fails_closed() -> None:
    http = FakeHttp([(200, {"ok": True})])
    guard = _AllowLLM(allow=False)
    rt = PlaybookRuntime(
        _pb(_irreversible_yaml()),
        director_llm=SequencedLLM([_TO_ACT, _IDLE]),
        http=http,
        intercept_llm=guard,
    )
    await rt.start()
    rt.log.append(
        SlotWriteEvent(key="date", value="2026-07-10", status="confirmed", by="tool")
    )
    await rt.on_user_text("book me in Pune")  # slots fine, classifier says no
    assert http.calls == []
    assert guard.calls >= 1


async def test_irreversible_classifier_allows() -> None:
    http = FakeHttp([(200, {"ok": True})])
    rt = PlaybookRuntime(
        _pb(_irreversible_yaml()),
        director_llm=SequencedLLM([_TO_ACT, _IDLE]),
        http=http,
        intercept_llm=_AllowLLM(allow=True),
    )
    await rt.start()
    rt.log.append(
        SlotWriteEvent(key="date", value="2026-07-10", status="confirmed", by="tool")
    )
    await rt.on_user_text("book me in Pune")
    assert len(http.calls) == 1


async def test_irreversible_without_classifier_uses_expr_guard_only() -> None:
    http = FakeHttp([(200, {"ok": True})])
    rt = PlaybookRuntime(
        _pb(_irreversible_yaml()),
        director_llm=SequencedLLM([_TO_ACT, _IDLE]),
        http=http,
    )
    await rt.start()
    rt.log.append(
        SlotWriteEvent(key="date", value="2026-07-10", status="confirmed", by="tool")
    )
    await rt.on_user_text("book me in Pune")
    assert len(http.calls) == 1  # confirmed slots satisfy the strict expr guard


PIPELINE_YAML = textwrap.dedent("""
    journeys:
      flow:
        checkpoints:
          - id: gather
            goal: "Get the city"
            gate: soft
            slots:
              city: {type: str, required: true}
            advance_when:
              - {when: "ready", judge: llm, to: flow.act, requires: [city]}
          - id: act
            goal: "Place the hold"
            gate: soft
            slots:
              date: {type: str, required: true}
            pipeline: hold_pipe
            advance_when:
              - {when: "pipeline.ok", judge: expr, to: flow.done}
          - id: done
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
    pipelines:
      - id: hold_pipe
        steps:
          - tool: hold_slot
""")


async def test_pipeline_preflight_denies_before_first_side_effect() -> None:
    http = FakeHttp([(200, {"hold_id": "h1"})])
    rt = PlaybookRuntime(
        Playbook.from_yaml(PIPELINE_YAML),
        director_llm=SequencedLLM([_TO_ACT, _IDLE]),
        http=http,
    )
    await rt.start()
    await rt.on_user_text("book me in Pune")  # date missing -> pipeline gated
    assert rt.state.checkpoint_id == "flow.act"
    assert http.calls == []
    assert "pipeline" not in rt.state.tool_results  # never started
    # Detail arrives -> pipeline runs on the next quiesce and routes onward.
    rt.log.append(
        SlotWriteEvent(key="date", value="2026-07-10", status="confirmed", by="tool")
    )
    await rt.on_user_text("the 10th")
    assert len(http.calls) == 1
    assert rt.state.checkpoint_id == "flow.done"
