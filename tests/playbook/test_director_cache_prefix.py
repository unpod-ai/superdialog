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
    # (c) the prefix covers the whole session-constant head: preamble, rules
    # of engagement, and the playbook-wide interrupt block — big enough to
    # clear provider cache minimums (the bare preamble was ~40 tokens).
    prefix = system[CACHE_PREFIX_KEY]
    assert prefix.startswith(_VERDICT_PREAMBLE)
    assert "SLOT RULE" in prefix
    assert "Interrupts:" in prefix
    # (d) volatile content stays OUT of the prefix
    for volatile in ("Already known:", "Tool results:", "Current step:"):
        assert volatile not in prefix
        assert volatile in system["content"]


def test_prefix_holds_with_confidence_field() -> None:
    # request_confidence extends the constant head; the annotated prefix must
    # still be a true leading substring of the assembled content.
    pb, state = _state()
    cp = pb.checkpoint(state.checkpoint_id)
    messages = _verdict_prompt(pb, cp, state, request_confidence=True)
    system = messages[0]
    assert system["content"].startswith(system[CACHE_PREFIX_KEY])
    assert "confidence" in system[CACHE_PREFIX_KEY]
