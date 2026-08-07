"""Shared fixtures for continuity-v2 tests."""

import json
import textwrap

from superdialog.playbook.models import Playbook
from superdialog.playbook.runtime import PlaybookRuntime

# Imported from a sibling test module — acceptable smell, beats three
# drifting copies of FakeHttp (reviewer-sanctioned).
from tests.playbook.test_toolexec import FakeHttp

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
            slots:
              location: {type: str}
            guidance: "Answer pricing."
            advance_when:
              - {when: "caller names location", judge: llm, to: main.pitch,
                 requires: [location]}
          - id: availability_faq
            goal: "Answer availability questions"
            gate: soft
            slots:
              callback_time: {type: str}
            guidance: "Answer availability."
          - id: close
            terminal: true
            outcome: done
    interrupts:
      - {id: price_guardrail, when: "caller asks about price", judge: llm,
         to: main.pricing_faq, resume: true}
      - {id: global_goodbye, when: "caller says goodbye", judge: llm,
         to: main.close, resume: false}
      - {id: availability_guardrail, when: "caller asks about availability",
         judge: llm, to: main.availability_faq, resume: true}
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


def make_runtime(
    payloads: list[dict], pb: Playbook | None = None
) -> PlaybookRuntime:
    """Runtime wired for continuity tests: SeqLLM director, no-op http."""
    return PlaybookRuntime(
        pb or Playbook.from_yaml(CONTINUITY_YAML),
        director_llm=SeqLLM(payloads),
        http=FakeHttp([]),
    )
