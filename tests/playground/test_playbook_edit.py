"""propose_edit — the AI builder core (LLM injected via the conftest fake).

Run with: ``uv run --extra playground pytest tests/playground/test_playbook_edit.py``
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("loguru")
pytest.importorskip("unpod")

from superdialog.llm.provider import CompletionResult

from playground.agents.playbooks import canonical_path, playbook_registry
from playground.harness.playbook_edit import EditProposal, propose_edit


def _valid_text() -> str:
    infos = playbook_registry()
    return canonical_path(infos[0].id).read_text(encoding="utf-8")


async def test_propose_edit_parses_summary_and_yaml(fake_llm_provider):
    new_yaml = _valid_text()
    fake_llm_provider.scripted = [
        CompletionResult(
            text=f"Added an SMS confirmation step.\n```yaml\n{new_yaml}\n```",
            tool_calls=[],
            metadata={},
        )
    ]
    proposal = await propose_edit(
        _valid_text(), "add an sms confirmation step", fake_llm_provider.complete
    )
    assert isinstance(proposal, EditProposal)
    assert proposal.summary == "Added an SMS confirmation step."
    assert proposal.yaml.strip() == new_yaml.strip()
    assert proposal.valid is True
    assert proposal.errors == []
    # The instruction reached the model as the user turn.
    user_msg = fake_llm_provider.calls[-1]["messages"][-1]["content"]
    assert "add an sms confirmation step" in user_msg


async def test_propose_edit_flags_invalid_yaml_without_raising(fake_llm_provider):
    fake_llm_provider.scripted = [
        CompletionResult(
            text="Broke it.\n```yaml\nplaybook: []\n```", tool_calls=[], metadata={}
        )
    ]
    proposal = await propose_edit(_valid_text(), "break it", fake_llm_provider.complete)
    assert proposal.valid is False
    assert proposal.errors
    assert proposal.yaml.strip() == "playbook: []"  # still returned for inspection
