"""Tests for natural-language -> simple-playbook generation."""

import textwrap

import pytest

from superdialog.playbook import Playbook
from superdialog.playbook.generate import generate_simple_playbook
from tests.playbook.test_optimize import CannedEditsLLM

_GOOD_YAML = textwrap.dedent("""
    goal: "Book a demo call."
    persona:
      name: Ava
      identity: "You are Ava, a scheduling assistant."
    playbook:
      - id: greet
        purpose: "Open the call."
        say: "Greet and ask how you can help."
        done_when: "Caller responds."
      - id: book
        purpose: "Capture a slot."
        say: "Ask for a day and time."
        collect: [day, time]
        done_when: "Day and time captured."
    interrupts:
      - {when: "Caller says goodbye.", to: main.book}
""")


async def test_generates_validated_simple_yaml() -> None:
    llm = CannedEditsLLM([_GOOD_YAML])
    text = await generate_simple_playbook("a demo-call booking agent", llm)
    pb = Playbook.from_yaml(text)  # round-trips through the unified loader
    assert "main.greet" in pb.checkpoint_ids()
    # the description and the simple-format schema reached the LLM
    prompt = " ".join(m["content"] for m in llm.calls[0])
    assert "demo-call booking agent" in prompt
    assert "done_when" in prompt and "interrupts" in prompt


async def test_fenced_output_is_accepted() -> None:
    llm = CannedEditsLLM(["```yaml\n" + _GOOD_YAML + "\n```"])
    text = await generate_simple_playbook("an agent", llm)
    assert Playbook.from_yaml(text)


async def test_invalid_output_retries_then_raises() -> None:
    llm = CannedEditsLLM(["journeys: {}", "not yaml: at: all: ["])
    with pytest.raises(ValueError):
        await generate_simple_playbook("an agent", llm, max_attempts=2)
    assert len(llm.calls) == 2
