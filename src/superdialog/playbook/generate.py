"""Natural-language description -> simple-format playbook YAML.

The default creation path: ``superdialog generate`` (and library callers)
produce a validated simple-format playbook. The legacy flow generator
(``superdialog flow generate`` / ``create_dialog_flow``) remains available.
"""

from __future__ import annotations

import yaml

from .director import CompletesLLM
from .simple import simple_to_playbook

_GEN_SYSTEM = """\
You author conversational-agent playbooks in the SIMPLE YAML format. Given
a description, return ONLY the YAML document (no commentary, no fences).

Schema:
- goal: one sentence — what makes the call a success, including fallbacks.
- persona: {name, language, voice_style, identity}. identity is "You are
  <name>, ..." prose; language may be a list like ["en", "hi"].
- opening / closing: optional one-line instructions.
- playbook: ordered steps, each {id, purpose, say, collect?, done_when}.
  * say: what to say and how — the agent's playbook for the step.
  * collect: slot keys to capture. AT MOST 2 per step (more causes stalls).
  * done_when: an observable condition, not an intention.
  Steps chain linearly; the last step closes the call.
- facts: grounding data as YAML (pricing, policies). Never inventable.
- objections: [{trigger, handle}] prose steering.
- boundaries: ["NEVER ..."] compliance rules.
- fallback_actions: {name: instruction} for failed happy paths.
- interrupts: [{when, to}] global jumps; to is "main.<step id>". ALWAYS
  include a goodbye/busy interrupt routing to the closing step — without
  an early exit, satisfied or busy callers loop forever.
"""


async def generate_simple_playbook(
    prompt: str,
    llm: CompletesLLM,
    *,
    max_attempts: int = 3,
) -> str:
    """Generate validated simple-format playbook YAML from a description.

    The output is parsed and compiled (``simple_to_playbook``) before being
    returned, so a successful return is always a loadable playbook. Raises
    ``ValueError`` when every attempt produced an invalid document.
    """
    messages = [
        {"role": "system", "content": _GEN_SYSTEM},
        {"role": "user", "content": f"AGENT DESCRIPTION:\n{prompt.strip()}"},
    ]
    last_error = "no attempts made"
    for _ in range(max_attempts):
        raw = await llm.complete(messages)
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith(("yaml", "yml")):
                text = text.split("\n", 1)[1] if "\n" in text else ""
        try:
            doc = yaml.safe_load(text)
            simple_to_playbook(doc)  # full validation, incl. interrupt refs
        except Exception as exc:  # yaml errors are not ValueError subclasses
            last_error = str(exc)
            continue
        return text.strip() + "\n"
    raise ValueError(f"playbook generation failed: {last_error}")
