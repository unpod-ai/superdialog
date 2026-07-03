"""In-process ConversationEndpoint implementations (default transport)."""

from __future__ import annotations

from typing import Any

from superdialog.agent import TurnResult
from superdialog.agents.llm_agent import LLMAgent
from superdialog.dialog_machine import DialogMachine
from superdialog.llm.resolver import resolve_llm

# The primer that asks the vanilla LLM to greet first, mirroring the playbook
# engine speaking a say_verbatim greeting before the user's first turn.
_VANILLA_BEGIN = "<BEGIN>"
_VANILLA_SUFFIX = (
    "\n\nWhen you receive the token <BEGIN>, greet the caller and begin the "
    "conversation. Otherwise respond as the assistant described above."
)

# The Talker's real-time barrier (default 0.4s) exists to keep a live voice
# call from hanging while the Director settles; past it the Talker gives up on
# the real reply and speaks a filler/hold line. Offline eval has no real-time
# pressure and every Director call is a multi-second LLM round-trip, so the
# production barrier would make the playbook emit filler EVERY turn and score
# ~0. Widen it so the barrier always waits for the Director's real answer.
_OFFLINE_BARRIER_S = 120.0
_OFFLINE_HOLD_S = 120.0


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


class InProcessPlaybook:
    """Wraps DialogMachine running the playbook engine (Director + Talker).

    ``agent_model`` sets both roles; ``director_model``/``talker_model`` override
    each role individually (per-role LLM control into the state machine).
    """

    def __init__(
        self,
        playbook_path: str,
        *,
        agent_model: str,
        director_model: str | None = None,
        talker_model: str | None = None,
    ) -> None:
        self._path = playbook_path
        self._talker = talker_model or agent_model
        self._director = director_model or agent_model
        self._machine = self._new_machine()

    def _new_machine(self) -> DialogMachine:
        return DialogMachine(
            source=self._path,
            llm=self._talker,
            director_llm=self._director,
            engine="playbook",
            barrier_timeout=_OFFLINE_BARRIER_S,
            hold_timeout=_OFFLINE_HOLD_S,
            settle_before_speak=True,
        )

    async def start(self) -> str:
        """Speak the playbook's opening greeting."""
        return (await self._machine.start()).text

    async def turn(self, text: str) -> str:
        """Feed one user utterance; return the assistant reply text."""
        return (await self._machine.turn(text)).text

    def reset(self) -> None:
        """Discard conversation state by rebuilding the DialogMachine."""
        self._machine = self._new_machine()


def _text(res: TurnResult | Any) -> str:
    return res.text if hasattr(res, "text") else str(res)
