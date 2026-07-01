"""Deterministic LLM/user test doubles for the eval harness."""

from __future__ import annotations

from typing import Any, AsyncIterator


class _Reply(str):
    """A canned reply that doubles as an ``LLMProvider`` ``CompletionResult``.

    It is a ``str`` — so the playbook-side ``CompletesLLM.complete`` surface
    (returns ``str``) and the Task 1.1 tests (``"x" in reply``) are satisfied —
    that also exposes ``.text``/``.tool_calls``/``.metadata``, which the vanilla
    path's ``LLMAgent`` reads off ``LLMProvider.complete``. One value serves both
    consumers of ``FakeProvider``.
    """

    __slots__ = ()

    @property
    def text(self) -> str:
        return str(self)

    @property
    def tool_calls(self) -> list[Any]:
        return []

    @property
    def metadata(self) -> dict[str, Any]:
        return {}


class FakeProvider:
    """Implements the LLMProvider surface (complete + stream) deterministically.

    ``script`` maps a substring marker found in the last user message to a canned
    reply; ``"*"`` is the fallback. This mirrors the ``complete``/``stream``
    signatures the Director (CompletesLLM) and Talker (StreamsLLM) depend on.
    """

    def __init__(self, script: dict[str, str]) -> None:
        self._script = script

    def _pick(self, messages: list[dict[str, Any]]) -> str:
        last = messages[-1]["content"] if messages else ""
        for marker, reply in self._script.items():
            if marker != "*" and marker in str(last):
                return reply
        return self._script.get("*", "")

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> _Reply:
        return _Reply(self._pick(messages))

    async def stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        words = self._pick(messages).split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"


class FakeSpeaksUser:
    """A SpeaksUser that replays fixed lines then returns '' (like ScriptedUser)."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._i = 0

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        if self._i >= len(self._lines):
            return ""
        line = self._lines[self._i]
        self._i += 1
        return line
