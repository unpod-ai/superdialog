"""Reversible trace: RevertEvent fold semantics, runtime.rewind, tier guards."""

import textwrap

import pytest

from superdialog.playbook.events import (
    AdvanceEvent,
    EventLog,
    RevertEvent,
    SlotWriteEvent,
    ToolCallEvent,
    ToolResultEvent,
    UtteranceEvent,
)
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime, RewindOutcome
from superdialog.playbook.state import ConversationState
from tests.playbook.test_replay import SequencedLLM
from tests.playbook.test_toolexec import FakeHttp

_IDLE = {"slots": {}, "advance": None, "note": None}

REWIND_YAML = textwrap.dedent("""
    journeys:
      booking:
        checkpoints:
          - id: collect
            goal: "Have city and date"
            gate: soft
            slots:
              city: {type: str, required: true}
            advance_when:
              - {when: "details complete", judge: llm, to: booking.confirm,
                 requires: [city]}
          - id: confirm
            goal: "Hold the slot"
            advance_when:
              - {when: "confirmed", judge: llm, to: booking.close}
          - id: close
            terminal: true
            outcome: confirmed
    tools:
      - id: lookup
        method: GET
        url: "https://api.test/lookup"
        store_response_as: lookup_result
      - id: legacy_post
        method: POST
        url: "https://api.test/legacy"
        store_response_as: legacy_result
      - id: hold_slot
        method: POST
        url: "https://api.test/hold"
        store_response_as: hold_result
        tier: compensable
        compensate: release_hold
      - id: release_hold
        method: POST
        url: "https://api.test/release"
        store_response_as: release_result
        tier: reversible
      - id: send_sms
        method: POST
        url: "https://api.test/sms"
        store_response_as: sms_result
        tier: irreversible
""")


def _pb() -> Playbook:
    return Playbook.from_yaml(REWIND_YAML)


def _runtime(http: FakeHttp | None = None) -> PlaybookRuntime:
    return PlaybookRuntime(
        _pb(), director_llm=SequencedLLM([_IDLE]), http=http or FakeHttp([])
    )


def _seed(rt: PlaybookRuntime) -> None:
    """v1 advance→collect, v2 user, v3 slot city=Pune."""
    rt.log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    rt.log.append(UtteranceEvent(role="user", text="Pune please"))
    rt.log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )


# -- fold semantics -----------------------------------------------------------


def test_fold_supersedes_state_but_keeps_transcript() -> None:
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="Pune please"))
    log.append(
        SlotWriteEvent(key="city", value="Pune", status="confirmed", by="director")
    )
    log.append(
        AdvanceEvent(
            from_checkpoint="booking.collect",
            to_checkpoint="booking.confirm",
            rule="llm:booking.confirm",
        )
    )
    log.append(UtteranceEvent(role="assistant", text="Holding it."))
    log.append(RevertEvent(superseded_from=3, superseded_to=5, reason="wrong city"))
    s = ConversationState.fold(log)
    assert s.checkpoint_id == "booking.collect"  # the advance was rewound
    assert "city" not in s.slots  # the slot write was rewound
    assert [t.text for t in s.transcript] == ["Pune please", "Holding it."]
    assert s.completed == []  # the superseded advance never completed collect


def test_revert_of_revert_reactivates_the_earlier_range() -> None:
    log = EventLog()
    log.append(SlotWriteEvent(key="x", value=1, status="confirmed", by="director"))
    log.append(SlotWriteEvent(key="x", value=2, status="confirmed", by="director"))
    log.append(RevertEvent(superseded_from=2, superseded_to=2, reason="undo v2"))
    assert ConversationState.fold(log).slots["x"].value == 1
    log.append(RevertEvent(superseded_from=3, superseded_to=3, reason="undo undo"))
    assert ConversationState.fold(log).slots["x"].value == 2


def test_superseded_user_turns_do_not_count_but_keep_language() -> None:
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="booking.collect", rule="init")
    )
    log.append(UtteranceEvent(role="user", text="नमस्ते", language="hi"))
    log.append(RevertEvent(superseded_from=2, superseded_to=2, reason="oops"))
    s = ConversationState.fold(log)
    assert s.user_turns_in_checkpoint == 0  # rewound turn doesn't count
    assert s.language == "hi"  # the caller's language doesn't un-happen
    assert len(s.transcript) == 1


def test_event_log_subscribe_and_unsubscribe() -> None:
    log = EventLog()
    q = log.subscribe()
    stamped = log.append(UtteranceEvent(role="user", text="hi"))
    assert q.get_nowait() == stamped
    log.unsubscribe(q)
    log.append(UtteranceEvent(role="user", text="again"))
    assert q.empty()


# -- runtime.rewind -----------------------------------------------------------


async def test_rewind_pure_state_is_done_with_repair_steer() -> None:
    rt = _runtime()
    _seed(rt)
    outcome = await rt.rewind(2, "caller corrected city", repair_note="Ack the fix.")
    assert outcome == RewindOutcome(status="done")
    assert "city" not in rt.state.slots
    assert rt.state.steering_note == "Ack the fix."
    assert rt.state.steering_kind == "repair"
    assert any(isinstance(e, RevertEvent) for e in rt.log.events)


async def test_rewind_across_reversible_get_is_done() -> None:
    rt = _runtime()
    _seed(rt)
    rt.log.append(ToolCallEvent(tool="lookup"))
    rt.log.append(
        ToolResultEvent(tool="lookup", store_as="lookup_result", ok=True, data={})
    )
    outcome = await rt.rewind(2, "undo")
    assert outcome.status == "done"
    assert "lookup_result" not in rt.state.tool_results


async def test_rewind_refuses_unannotated_post() -> None:
    rt = _runtime()
    _seed(rt)
    rt.log.append(
        ToolResultEvent(tool="legacy_post", store_as="legacy_result", ok=True)
    )
    outcome = await rt.rewind(2, "undo")
    assert outcome.status == "refused"
    assert outcome.pending == ["legacy_post"]
    assert not any(isinstance(e, RevertEvent) for e in rt.log.events)


async def test_rewind_refuses_irreversible() -> None:
    rt = _runtime()
    _seed(rt)
    rt.log.append(ToolResultEvent(tool="send_sms", store_as="sms_result", ok=True))
    outcome = await rt.rewind(2, "undo")
    assert outcome.status == "refused"
    assert "send_sms" in outcome.pending


async def test_rewind_compensable_confirm_flow() -> None:
    http = FakeHttp([(200, {"released": True})])
    rt = _runtime(http)
    _seed(rt)
    rt.log.append(
        ToolResultEvent(
            tool="hold_slot", store_as="hold_result", ok=True, data={"hold_id": "h1"}
        )
    )
    first = await rt.rewind(2, "change date")
    assert first.status == "needs_confirmation"
    assert first.pending == ["hold_slot"]
    assert http.calls == []  # nothing undone without the caller's approval

    second = await rt.rewind(2, "change date", confirmed=True)
    assert second.status == "done"
    assert http.calls and "release" in http.calls[0]["url"]
    # the compensation result lands OUTSIDE the superseded range
    assert "release_result" in rt.state.tool_results
    assert "hold_result" not in rt.state.tool_results


async def test_rewind_compensation_failure_aborts() -> None:
    http = FakeHttp([(500, {"error": "backend down"})])
    rt = _runtime(http)
    _seed(rt)
    rt.log.append(
        ToolResultEvent(tool="hold_slot", store_as="hold_result", ok=True, data={})
    )
    outcome = await rt.rewind(2, "undo", confirmed=True)
    assert outcome.status == "refused"
    assert "compensation failed" in outcome.detail
    assert not any(isinstance(e, RevertEvent) for e in rt.log.events)
    assert "hold_result" in rt.state.tool_results  # state untouched


async def test_rewind_ignores_already_superseded_effects() -> None:
    """A second rewind must not re-compensate an already-rewound tool call."""
    http = FakeHttp([(200, {"released": True})])
    rt = _runtime(http)
    _seed(rt)
    rt.log.append(
        ToolResultEvent(tool="hold_slot", store_as="hold_result", ok=True, data={})
    )
    assert (await rt.rewind(2, "undo", confirmed=True)).status == "done"
    assert len(http.calls) == 1
    outcome = await rt.rewind(1, "undo more")  # range covers the dead hold too
    assert outcome.status == "done"  # no needs_confirmation: it's already undone
    assert len(http.calls) == 1  # and no second release call


async def test_rewind_bad_version_raises() -> None:
    rt = _runtime()
    _seed(rt)
    with pytest.raises(ValueError):
        await rt.rewind(rt.log.version, "no-op range")
    with pytest.raises(ValueError):
        await rt.rewind(-1, "negative")


# -- tier model validation ------------------------------------------------------


def _tool_doc(**tool: object) -> dict:
    return {
        "journeys": {"j": {"checkpoints": [{"id": "a", "goal": "g"}]}},
        "tools": [{"id": "t", "method": "POST", "url": "https://x", **tool}],
    }


def test_compensate_requires_compensable_tier() -> None:
    with pytest.raises(ValueError, match="requires tier"):
        Playbook.model_validate(_tool_doc(compensate="t2"))


def test_compensate_must_reference_a_known_tool() -> None:
    with pytest.raises(ValueError, match="unknown compensate tool"):
        Playbook.model_validate(_tool_doc(tier="compensable", compensate="ghost"))


def test_compensate_cannot_be_itself() -> None:
    with pytest.raises(ValueError, match="cannot be itself"):
        Playbook.model_validate(_tool_doc(tier="compensable", compensate="t"))


def test_effective_tier_defaults() -> None:
    pb = _pb()
    assert pb.tool("lookup").effective_tier == "reversible"  # GET
    assert pb.tool("legacy_post").effective_tier == "compensable"  # write, unmarked
    assert pb.tool("send_sms").effective_tier == "irreversible"  # explicit
