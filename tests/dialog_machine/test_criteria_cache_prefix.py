"""Prompt-cache prefix behavior for ``CriteriaJudge.build_evaluation_messages``.

The flow-llmadapter engine originally led its system content with a volatile
date, so it could not be cached. It now hoists the flow's fixed ``system_prompt``
to the FRONT (a stable, node-independent prefix) and annotates it with
``_cache_prefix`` — while staying byte-identical to the original output when no
system prompt is supplied.
"""

from __future__ import annotations

from superdialog.flow.models import Edge, FlowNode
from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
from superdialog.machine.criteria import CriteriaJudge

PERSONA = "You are Arjun, a meticulous golf-booking agent. Always be concise."


def _make_node() -> FlowNode:
    return FlowNode(
        id="n1",
        name="Greet",
        instruction="Say hi.",
        edges=[Edge(id="e1", condition="done", target_node_id="n2")],
    )


def _content(system_prompt: str) -> dict:
    judge = CriteriaJudge()
    return judge.build_evaluation_messages(
        _make_node(), history=[], userdata={}, system_prompt=system_prompt
    )[0]


def test_system_content_is_str() -> None:
    assert isinstance(_content(PERSONA)["content"], str)


def test_persona_leads_and_is_annotated() -> None:
    """A non-empty system_prompt is hoisted to the front and cached."""
    msg = _content(PERSONA)
    content: str = msg["content"]
    assert content.startswith(PERSONA)  # stable persona leads
    assert msg[CACHE_PREFIX_KEY] == PERSONA
    assert content.startswith(msg[CACHE_PREFIX_KEY])  # seam contract holds


def test_volatile_content_still_present_after_reorder() -> None:
    """Reorder preserves every piece of information, just moved after the prefix."""
    content: str = _content(PERSONA)["content"]
    for marker in (
        "Today's date is ",
        "Current node: n1",
        "Available transitions:",
        "recommended_edge_id MUST",
        '"criteria_met"',
    ):
        assert marker in content, marker
    # the date no longer LEADS — it now sits after the persona prefix
    assert not content.startswith("Today's date is ")


def test_empty_system_prompt_not_annotated_and_date_leads() -> None:
    """No persona → original layout preserved (date leads, no annotation)."""
    msg = _content("")
    assert CACHE_PREFIX_KEY not in msg
    assert msg["content"].startswith("Today's date is ")


def test_whitespace_only_system_prompt_not_annotated() -> None:
    msg = _content("   ")
    assert CACHE_PREFIX_KEY not in msg  # nothing meaningful to cache
