"""A resume=True detour must not be skipped, and must never orphan its stack entry.

Before the fix, _hop evaluated the expr fast-path (step 3) BEFORE the resume
return (step 5), and the resume return is additionally gated on
user_turns_in_checkpoint >= 1 so it cannot fire on the entry hop at all. A
detour target whose own expr rule was already satisfied therefore advanced
straight off it: the caller's off-flow question went unanswered, and because
that advance is logged with rule != "resume", ConversationState.fold never
popped the stacked entry — leaving a stale checkpoint that a later detour would
"resume" the caller to.

The LLM-verdict path (PlaybookRuntime.on_turn) always gave resume priority; the
quiescence path now matches it.
"""

import textwrap

from superdialog.playbook.events import AdvanceEvent, SlotWriteEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime
from superdialog.playbook.state import ConversationState
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_toolexec import FakeHttp

# j.kb is a resume=True detour target that ALSO carries its own expr rule, and
# `preset` is seeded so that rule is already true the moment we arrive.
DETOUR_YAML = textwrap.dedent("""
    journeys:
      j:
        checkpoints:
          - id: ask
            goal: "collect the thing"
            slots:
              thing: {type: str, required: true}
            advance_when:
              - {when: "thing captured", judge: llm, to: j.done, requires: [thing]}
          - id: kb
            goal: "answer the off-flow question"
            advance_when:
              - {judge: expr, when: "slots.preset is not None", to: j.done}
          - id: done
            terminal: true
            outcome: done
    interrupts:
      - {id: aside, when: "caller asks something off-flow", judge: llm,
         to: j.kb, resume: true}
""")


def _runtime() -> PlaybookRuntime:
    return PlaybookRuntime(
        Playbook.from_yaml(DETOUR_YAML),
        director_llm=CannedLLM(
            {"slots": {}, "advance": None, "note": None, "interrupt": "aside"}
        ),
        http=FakeHttp([]),
    )


async def test_satisfied_expr_rule_does_not_skip_a_resume_detour() -> None:
    rt = _runtime()
    await rt.start()
    assert rt.state.checkpoint_id == "j.ask"
    # seed the slot the detour target's own expr rule reads, the way a host
    # seeds a known value (by="compiler"), so the rule is true on arrival
    rt.log.append(
        SlotWriteEvent(key="preset", value="x", status="confirmed", by="compiler")
    )

    await rt.on_user_text("wait, unrelated question")

    # we must still be ON the detour target, owing the caller an answer
    assert rt.state.checkpoint_id == "j.kb", "expr rule skipped the detour"
    assert not rt.state.ended, "detour fell through to the terminal checkpoint"
    assert rt.state.resume_stack == ["j.ask"], "the step we left must be recorded"
    assert rt.state.entered_via_resume


async def test_leaving_a_detour_by_another_route_pops_the_stack() -> None:
    pb = Playbook.from_yaml(DETOUR_YAML)
    log = _runtime().log
    log.append(
        AdvanceEvent(
            from_checkpoint="j.ask", to_checkpoint="j.kb", rule="interrupt:aside"
        )
    )
    assert ConversationState.fold(log, pb).resume_stack == ["j.ask"]

    # an advance that is NOT the synthesized "resume" — expr/auto/llm, a
    # turn_budget force-advance, a supervisor redirect, on_failure
    log.append(
        AdvanceEvent(from_checkpoint="j.kb", to_checkpoint="j.done", rule="expr")
    )
    state = ConversationState.fold(log, pb)
    assert state.resume_stack == [], "stale entry orphaned on the resume stack"
    assert not state.entered_via_resume


async def test_resume_return_still_pops_exactly_once() -> None:
    pb = Playbook.from_yaml(DETOUR_YAML)
    log = _runtime().log
    log.append(
        AdvanceEvent(
            from_checkpoint="j.ask", to_checkpoint="j.kb", rule="interrupt:aside"
        )
    )
    log.append(
        AdvanceEvent(from_checkpoint="j.kb", to_checkpoint="j.ask", rule="resume")
    )
    state = ConversationState.fold(log, pb)
    assert state.checkpoint_id == "j.ask"
    assert state.resume_stack == []
    assert not state.entered_via_resume
