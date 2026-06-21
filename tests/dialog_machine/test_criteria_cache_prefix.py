"""Prompt-cache prefix safety check for ``CriteriaJudge``.

The flow-llmadapter engine builds its system message in
``CriteriaJudge.build_evaluation_messages``. The leading bytes of that content
are a *volatile* timestamp ("Today's date is <weekday, day month year> ..."),
not a fixed persona/preamble, and the fixed ``system_prompt`` is concatenated
near the END of the string.

Because nothing stable leads the content, this engine is in the
``reorder_needed`` state: it MUST NOT be annotated with ``_cache_prefix`` (doing
so would violate ``content.startswith(stable)`` or require touching the content
bytes). These tests pin that decision so a future edit cannot silently add an
incorrect marker.
"""

from __future__ import annotations

from superdialog.flow.models import Edge, FlowNode
from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
from superdialog.machine.criteria import CriteriaJudge


def _make_node() -> FlowNode:
    return FlowNode(
        id="n1",
        name="Greet",
        instruction="Say hi.",
        edges=[Edge(id="e1", condition="done", target_node_id="n2")],
    )


def test_system_content_is_str() -> None:
    """The leading message content must be a plain string."""
    judge = CriteriaJudge()
    messages = judge.build_evaluation_messages(
        _make_node(),
        history=[],
        userdata={},
        system_prompt="You are a careful evaluator.",
    )
    system = messages[0]
    assert system["role"] == "system"
    assert isinstance(system["content"], str)


def test_volatile_date_leads_content() -> None:
    """Content begins with a volatile timestamp, not a fixed preamble."""
    judge = CriteriaJudge()
    messages = judge.build_evaluation_messages(
        _make_node(),
        history=[],
        userdata={},
        system_prompt="You are a careful evaluator.",
    )
    content: str = messages[0]["content"]
    # The first line is the date — proof that volatile text leads the string.
    assert content.startswith("Today's date is ")
    # The fixed system_prompt is appended, so it does NOT lead the content.
    assert not content.startswith("You are a careful evaluator.")


def test_no_cache_prefix_annotation() -> None:
    """reorder_needed engine: the marker must NOT be added.

    Annotating here is unsafe because no stable substring leads ``content``;
    ``mark_cache_prefix`` requires ``content.startswith(_cache_prefix)``.
    """
    judge = CriteriaJudge()
    messages = judge.build_evaluation_messages(
        _make_node(),
        history=[],
        userdata={},
        system_prompt="You are a careful evaluator.",
    )
    assert CACHE_PREFIX_KEY not in messages[0]
