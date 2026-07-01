"""In-process ConversationEndpoint implementations (default transport)."""

from __future__ import annotations

from typing import Any

from superdialog.agent import TurnResult
from superdialog.agents.llm_agent import LLMAgent
from superdialog.llm.resolver import resolve_llm

# The primer that asks the vanilla LLM to greet first, mirroring the playbook
# engine speaking a say_verbatim greeting before the user's first turn.
_VANILLA_BEGIN = "<BEGIN>"
_VANILLA_SUFFIX = (
    "\n\nWhen you receive the token <BEGIN>, greet the caller and begin the "
    "conversation. Otherwise respond as the assistant described above."
)


class InProcessVanilla:
    """Runs the SAME playbook as a flat system prompt through a single LLM.

    ``playbook_text`` is the raw playbook file contents (no orchestration).
    ``llm`` may be a model-URI string or an already-constructed LLMProvider.
    """

    def __init__(self, playbook_text: str, llm: Any) -> None:
        self._system = playbook_text + _VANILLA_SUFFIX
        self._llm = resolve_llm(llm) if isinstance(llm, str) else llm
        self._agent = self._new_agent()

    def _new_agent(self) -> LLMAgent:
        return LLMAgent(self._llm, system_prompt=self._system)

    async def start(self) -> str:
        """Greet the caller by feeding the <BEGIN> primer token."""
        res = await self._agent.turn(_VANILLA_BEGIN)
        return _text(res)

    async def turn(self, text: str) -> str:
        """Feed one user utterance; return the assistant reply text."""
        res = await self._agent.turn(text)
        return _text(res)

    def reset(self) -> None:
        """Discard conversation state by rebuilding the underlying agent."""
        self._agent = self._new_agent()


def _text(res: TurnResult | Any) -> str:
    return res.text if hasattr(res, "text") else str(res)
