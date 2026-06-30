"""PlaybookAgent.draft_turn — A2a speculative draft-and-hold.

A draft is the reply the agent *would* give for a not-yet-final utterance. It
must (1) stream a reply, (2) leave ZERO trace in the event log, and — the
load-bearing safety property — (3) NEVER fire a tool, even when the verdict
would advance into a checkpoint whose pipeline calls a tool. Tools are
irreversible side effects; speculating them on a half-sentence is the one thing
that must be impossible.
"""

from superdialog.playbook import PlaybookAgent
from superdialog.playbook.events import UtteranceEvent
from superdialog.playbook.models import Playbook
from tests.playbook.test_agent import SlowStreamLLM, _agent
from tests.playbook.test_director import CannedLLM
from tests.playbook.test_models import MINIMAL_YAML
from tests.playbook.test_talker import StreamLLM
from tests.playbook.test_toolexec import FakeHttp

# A verdict that advances into booking.confirm, whose pipeline calls HTTP in a
# real turn (see test_turn_includes_pass_through). On a DRAFT the pipeline must
# be skipped entirely.
_ADVANCING_VERDICT = {
    "slots": {"city": "Pune", "date": "2026-06-12"},
    "advance": "booking.confirm",
    "note": None,
}


async def test_draft_streams_a_reply() -> None:
    agent = _agent()
    chunks = [c async for c in agent.draft_turn("hello")]
    assert [c.text for c in chunks if c.text] == ["Which", " city?"]
    assert chunks[-1].done


async def test_draft_leaves_no_trace_in_log() -> None:
    agent = _agent()
    await agent.runtime.start()  # session init is not draft trace; baseline after
    before = agent.runtime.log.version
    async for _ in agent.draft_turn("hello"):
        pass
    # No user utterance, no assistant speech, no advance — nothing committed.
    assert agent.runtime.log.version == before
    assert not any(
        isinstance(e, UtteranceEvent) and e.text == "hello"
        for e in agent.runtime.log.events
    )


async def test_draft_fires_no_tools_even_when_advancing() -> None:
    """The safety invariant: an advancing draft must not run the checkpoint
    pipeline (HTTP tool). http_responses is empty — a real turn would consume
    one; the draft must consume none."""
    http = FakeHttp([])  # no responses queued
    agent = PlaybookAgent(
        playbook=Playbook.from_yaml(MINIMAL_YAML),
        talker_llm=StreamLLM(["Confirmed", "."]),
        director_llm=CannedLLM(_ADVANCING_VERDICT),
        http=http,
    )
    async for _ in agent.draft_turn("Pune tomorrow"):
        pass
    assert http.calls == []  # the tool pipeline never ran on a draft


async def test_draft_then_real_turn_unaffected() -> None:
    """A draft leaves no trace, so a following real turn behaves as if the
    draft never happened."""
    agent = _agent()
    async for _ in agent.draft_turn("hello"):
        pass
    result = await agent.turn("hello")
    assert result.text == "Which city?"
    users = [
        e
        for e in agent.runtime.log.events
        if isinstance(e, UtteranceEvent) and e.role == "user" and e.text == "hello"
    ]
    assert len(users) == 1  # only the real turn's utterance, not the draft's


async def test_draft_aborts_cleanly_for_revise() -> None:
    """aclose() mid-draft (a continuation arrived) returns cleanly and leaves
    no trace — the host will re-draft on the merged text."""
    agent = _agent(talker_llm=SlowStreamLLM(["Which", " city", " today?"]))
    await agent.runtime.start()  # baseline after session init
    before = agent.runtime.log.version
    gen = agent.draft_turn("hel")
    first = await gen.__anext__()
    assert first.text == "Which"
    await gen.aclose()  # revise: must not raise
    assert agent.runtime.log.version == before  # still zero trace
