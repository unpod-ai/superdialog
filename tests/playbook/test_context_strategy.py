"""Per-checkpoint context strategy: append / reset / reset_with_summary.

Pipecat Flows exposes this per node; our checkpoints are the same concept at a
finer granularity (a topic is 4-7 checkpoints here, not one), so the default is
``append`` -- today's behavior -- and an author opts a checkpoint in at a real
topic boundary. See the field docstring for the "is the caller coming back?"
test that decides it.

Two rules carry most of the weight:

* ``append`` must render byte-identically to the pre-feature engine, because
  two production playbooks run on it untouched.
* the FINAL transcript entry always survives -- both the budget packer and a
  reset. It is the caller's current utterance (logged before the Talker
  renders), so dropping it leaves the Talker answering a question it cannot
  see. This is Pipecat's ``min_messages_after_summary`` floor, and under reset
  it is load-bearing rather than merely nice.
"""

import textwrap

import anyio

from superdialog.playbook.compact import _MAX_SUMMARY_TOKENS, compact
from superdialog.playbook.director import _verdict_prompt
from superdialog.playbook.events import AdvanceEvent, EventLog, UtteranceEvent
from superdialog.playbook.models import Playbook
from superdialog.playbook.render import render_view
from superdialog.playbook.state import ConversationState, TranscriptEntry

_YAML = textwrap.dedent("""
    persona: "You are a booking assistant."
    journeys:
      main:
        checkpoints:
          - id: collect
            goal: "Collect the city"
            guidance: "Ask for the city."
            advance_when:
              - {when: "city given", judge: llm, to: main.appended}
          - id: appended
            goal: "Keep talking"
            guidance: "Continue."
            advance_when:
              - {when: "done", judge: llm, to: main.fresh}
          - id: fresh
            goal: "A brand new topic"
            context: reset
            guidance: "Start clean."
            advance_when:
              - {when: "done", judge: llm, to: main.digested}
          - id: digested
            goal: "New topic, keep the gist"
            context: reset_with_summary
            summary_prompt: "Summarize the caller's order details only."
            guidance: "Start clean but informed."
            terminal: true
            outcome: closed
""")


def _without_context_fields(text: str) -> str:
    """The same playbook with every context/summary_prompt line removed.

    Filtering by content rather than exact-match replace: textwrap.dedent has
    already shifted the indentation, so a hard-coded prefix silently matches
    nothing and the "identical" assertion passes vacuously.
    """
    return "\n".join(
        ln
        for ln in text.split("\n")
        if not ln.strip().startswith(("context:", "summary_prompt:"))
    )


def _pb() -> Playbook:
    return Playbook.from_yaml(_YAML)


def _state(pb: Playbook, land_on: str, before: int = 6, after: int = 0):
    """`before` turns, then advance to `land_on`, then `after` turns."""
    log = EventLog()
    log.append(
        AdvanceEvent(from_checkpoint=None, to_checkpoint="main.collect", rule="i")
    )
    for i in range(before):
        log.append(UtteranceEvent(role="user", text=f"early user {i}"))
        log.append(UtteranceEvent(role="assistant", text=f"early agent {i}"))
    if land_on != "main.collect":
        log.append(
            AdvanceEvent(
                from_checkpoint="main.collect", to_checkpoint=land_on, rule="r"
            )
        )
    for i in range(after):
        log.append(UtteranceEvent(role="user", text=f"late user {i}"))
        log.append(UtteranceEvent(role="assistant", text=f"late agent {i}"))
    return ConversationState.fold(log, playbook=pb)


def _chat(view) -> list[str]:
    return [m["content"] for m in view.messages if m["role"] != "system"]


# --- append: the regression guard for two live playbooks ----------------------


def test_append_is_the_default_when_nothing_is_declared() -> None:
    pb = _pb()
    assert pb.checkpoint("main.appended").context is None
    assert pb.policies.context == "append"


def test_append_renders_the_same_chat_as_the_unannotated_engine() -> None:
    """No `context:` anywhere must produce exactly what it produced before."""
    plain = Playbook.from_yaml(_without_context_fields(_YAML))
    assert "context:" not in _without_context_fields(_YAML)
    annotated = _pb()
    a = render_view(plain, _state(plain, "main.fresh", before=6), token_budget=40_000)
    b = render_view(
        annotated, _state(annotated, "main.appended", before=6), token_budget=40_000
    )
    assert _chat(a) == _chat(b)


def test_append_keeps_pre_entry_turns() -> None:
    pb = _pb()
    view = render_view(pb, _state(pb, "main.appended", before=6), token_budget=40_000)
    assert any("early user 0" in c for c in _chat(view))


# --- reset --------------------------------------------------------------------


def test_reset_drops_pre_entry_turns() -> None:
    pb = _pb()
    view = render_view(
        pb, _state(pb, "main.fresh", before=6, after=2), token_budget=40_000
    )
    chat = _chat(view)
    assert not any("early user" in c for c in chat), "pre-entry history leaked"
    assert any("late user 0" in c for c in chat), "post-entry history lost"


def test_reset_always_keeps_the_triggering_turn() -> None:
    """The utterance that CAUSED the advance predates it, so a strict reset
    would hide the very question this checkpoint exists to answer."""
    pb = _pb()
    view = render_view(pb, _state(pb, "main.fresh", before=6), token_budget=40_000)
    chat = _chat(view)
    assert chat and chat != ["[start]"], "reset produced an empty view"
    assert "early agent 5" in chat[-1], f"trigger turn dropped: {chat}"


def test_reset_with_summary_behaves_like_reset_for_the_transcript() -> None:
    pb = _pb()
    view = render_view(
        pb, _state(pb, "main.digested", before=6, after=2), token_budget=40_000
    )
    assert not any("early user" in c for c in _chat(view))


# --- playbook-level default + per-checkpoint override -------------------------

_WITH_DEFAULT = _YAML.replace("\njourneys:", "\npolicies:\n  context: reset\njourneys:")


def test_playbook_default_applies_to_unannotated_checkpoints() -> None:
    pb = Playbook.from_yaml(_WITH_DEFAULT)
    assert pb.policies.context == "reset"
    view = render_view(
        pb, _state(pb, "main.appended", before=6, after=1), token_budget=40_000
    )
    assert not any("early user 0" in c for c in _chat(view))


def test_checkpoint_override_beats_the_playbook_default() -> None:
    pb = Playbook.from_yaml(
        _WITH_DEFAULT.replace(
            "      - id: appended\n",
            "      - id: appended\n        context: append\n",
        )
    )
    view = render_view(
        pb, _state(pb, "main.appended", before=6, after=1), token_budget=40_000
    )
    assert any("early user 0" in c for c in _chat(view))


# --- the recent-turn floor (Pipecat's min_messages_after_summary) -------------


def test_newest_turn_survives_an_absurd_budget() -> None:
    """A system block bigger than the whole budget used to drop every turn,
    leaving the Talker blind to what the caller just said."""
    pb = _pb()
    view = render_view(pb, _state(pb, "main.appended", before=6), token_budget=1)
    chat = _chat(view)
    assert chat != ["[start]"], "newest turn dropped under budget pressure"
    assert "early agent 5" in chat[-1]


# --- Director honors the strategy too ----------------------------------------


def _director_transcript(pb: Playbook, state: ConversationState) -> str:
    msgs = _verdict_prompt(pb, pb.checkpoint(state.checkpoint_id), state)
    return msgs[-1]["content"]


def test_director_window_honors_reset() -> None:
    pb = _pb()
    text = _director_transcript(pb, _state(pb, "main.fresh", before=6, after=2))
    assert "early user" not in text
    assert "late user 0" in text


def test_director_window_appends_by_default() -> None:
    pb = _pb()
    text = _director_transcript(pb, _state(pb, "main.appended", before=6, after=2))
    assert "early user" in text


def test_director_still_sees_known_slots_after_a_reset() -> None:
    """Change detection compares the utterance to `Already known`, not to
    transcript history -- so a reset must not cost the Director that basis."""
    pb = _pb()
    msgs = _verdict_prompt(
        pb, pb.checkpoint("main.fresh"), _state(pb, "main.fresh", before=6, after=1)
    )
    assert "Already known:" in msgs[0]["content"]


# --- compact(): per-checkpoint prompt + a real output cap --------------------


class _Echo:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self.text


def _entries() -> list[TranscriptEntry]:
    return [TranscriptEntry(role="user", text="I want tee time at nine", version=1)]


def test_compact_uses_a_custom_prompt_when_given() -> None:
    llm = _Echo("ok")

    async def go() -> None:
        await compact(llm, "prior", _entries(), prompt="ONLY order details.")

    anyio.run(go)
    assert "ONLY order details." in llm.calls[0][0]["content"]


def test_compact_falls_back_to_the_default_prompt() -> None:
    llm = _Echo("ok")

    async def go() -> None:
        await compact(llm, "prior", _entries())

    anyio.run(go)
    assert "running memory of a live phone call" in llm.calls[0][0]["content"]


def test_compact_clamps_a_runaway_summary() -> None:
    """The summary sits in the PROTECTED block -- the budget packer can never
    trim it, and the prompt tells the next pass to preserve it, so an
    over-long summary ratchets and permanently starves the transcript."""
    llm = _Echo("This is a sentence about the booking. " * 2000)

    async def go() -> str:
        return await compact(llm, "prior", _entries())

    out = anyio.run(go)
    assert len(out.encode()) // 4 <= _MAX_SUMMARY_TOKENS
    assert out, "clamping must not empty the summary"


def test_compact_leaves_a_normal_summary_untouched() -> None:
    normal = "Caller is Rohit. Wants a Sept tee time for four."
    llm = _Echo(normal)

    async def go() -> str:
        return await compact(llm, "prior", _entries())

    assert anyio.run(go) == normal
