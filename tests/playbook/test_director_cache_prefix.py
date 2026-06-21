"""Hermetic test: verdict system message carries a valid cache-prefix marker."""

from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
from superdialog.playbook.director import _VERDICT_PREAMBLE, _verdict_prompt
from tests.playbook.test_director import _state


def test_verdict_system_prefix_is_true_prefix() -> None:
    pb, state = _state()
    cp = pb.checkpoint(state.checkpoint_id)
    messages = _verdict_prompt(pb, cp, state)
    system = messages[0]

    assert system["role"] == "system"
    # (a) content is a plain string
    assert isinstance(system["content"], str)
    # (b) annotated prefix is a true leading substring of content
    assert system["content"].startswith(system[CACHE_PREFIX_KEY])
    # (c) the prefix equals the fixed preamble source
    assert system[CACHE_PREFIX_KEY] == _VERDICT_PREAMBLE


def test_prefix_holds_with_confidence_field() -> None:
    # request_confidence injects volatile text right after the preamble; the
    # stable preamble must still be a true prefix of the assembled content.
    pb, state = _state()
    cp = pb.checkpoint(state.checkpoint_id)
    messages = _verdict_prompt(pb, cp, state, request_confidence=True)
    system = messages[0]
    assert system[CACHE_PREFIX_KEY] == _VERDICT_PREAMBLE
    assert system["content"].startswith(_VERDICT_PREAMBLE)
    assert "confidence" in system["content"]
