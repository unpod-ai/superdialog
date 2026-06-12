"""AI playbook builder: rewrite a playbook YAML from a natural-language edit.

Pure + LLM-injected: :func:`propose_edit` takes an async ``complete`` callable
(``LitellmProvider.complete``), so tests run without keys or network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from playground.harness.playbook_store import validate_yaml


class _Result(Protocol):
    text: str


Complete = Callable[..., Awaitable[_Result]]

_SYSTEM_PROMPT = (
    "You edit voice-agent playbooks written in superdialog YAML. You are given "
    "the current playbook and an instruction. Return the COMPLETE updated "
    "playbook, preserving the existing format (simple format uses top-level "
    "`goal`, `persona`, `playbook` (a list of steps with id/purpose/say/collect/"
    "done_when), `facts`, `interrupts`; full format uses `journeys` with "
    "`checkpoints`). Keep ids stable where possible and keep it valid.\n\n"
    "Respond with EXACTLY: a one-line summary of what you changed, then the full "
    "updated playbook in a single ```yaml fenced block. Output nothing else."
)

_FENCE = re.compile(r"```(?:ya?ml)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class EditProposal:
    """An LLM-proposed playbook rewrite plus its validation verdict."""

    yaml: str
    summary: str
    valid: bool
    errors: list[str]
    steps: int
    journey: str


def _parse(reply: str) -> tuple[str, str]:
    """Return ``(summary, yaml)`` from the model reply."""
    match = _FENCE.search(reply)
    if not match:
        # No fence — treat the whole reply as YAML, no summary.
        return "Updated the playbook.", reply.strip()
    yaml_text = match.group(1).strip()
    summary = reply[: match.start()].strip().splitlines()
    return (summary[0].strip() if summary else "Updated the playbook."), yaml_text


def _user_prompt(current_yaml: str, instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n\nCurrent playbook:\n```yaml\n{current_yaml}\n```"
    )


async def propose_edit(
    current_yaml: str,
    instruction: str,
    complete: Complete,
) -> EditProposal:
    """Ask the LLM to rewrite ``current_yaml`` per ``instruction`` and validate."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(current_yaml, instruction)},
    ]
    result = await complete(messages)
    summary, new_yaml = _parse(result.text or "")
    vr = validate_yaml(new_yaml)
    return EditProposal(
        yaml=new_yaml,
        summary=summary,
        valid=vr.valid,
        errors=vr.errors,
        steps=vr.steps,
        journey=vr.journey,
    )
