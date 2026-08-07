"""Shared fixtures for continuity-v2 tests."""

import json
import textwrap

CONTINUITY_YAML = textwrap.dedent("""
    persona: "Test assistant."
    journeys:
      main:
        checkpoints:
          - id: ask_location
            goal: "Capture location"
            gate: soft
            slots:
              location: {type: str}
            guidance: "Ask for location."
            advance_when:
              - {when: "location given", judge: llm, to: main.pitch,
                 requires: [location]}
          - id: pitch
            goal: "Pitch the product"
            gate: soft
            guidance: "Pitch."
            advance_when:
              - {when: "caller responded in any way", judge: llm,
                 to: main.ask_budget}
          - id: ask_budget
            goal: "Capture budget"
            gate: soft
            slots:
              budget: {type: str}
            guidance: "Ask for budget."
            advance_when:
              - {when: "budget given", judge: llm, to: main.close,
                 requires: [budget]}
          - id: pricing_faq
            goal: "Answer pricing questions"
            gate: soft
            guidance: "Answer pricing."
          - id: close
            terminal: true
            outcome: done
    interrupts:
      - {id: price_guardrail, when: "caller asks about price", judge: llm,
         to: main.pricing_faq, resume: true}
""")


class SeqLLM:
    """Director stub returning a DIFFERENT canned verdict per call.

    Unlike CannedLLM (same verdict forever), this pops from a sequence —
    needed for multi-turn scenarios (interrupt, then self-interrupt, then
    plain turn). Falls back to an empty verdict when exhausted.
    """

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)

    async def complete(self, messages, **kwargs) -> str:
        if self._payloads:
            return json.dumps(self._payloads.pop(0))
        return json.dumps({"slots": {}, "advance": None, "note": None})
