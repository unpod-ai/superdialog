"""Playbook quickstart: drive examples/playbooks/booking.yaml in a text REPL.

Run from the repo root::

    export OPENAI_API_KEY=sk-...
    uv run python examples/playbook_quickstart.py

Any litellm model string works — override the default (``openai/gpt-5.1``)::

    export SUPERDIALOG_MODEL="anthropic/claude-sonnet-4-5"

What it shows:

- ``Playbook.load`` on the generously commented booking.yaml
- litellm-backed adapters: the Talker streams raw tokens, the Director
  returns one plain-text completion per verdict
- ``PlaybookAgent`` streaming chunks live, with the booking pipeline
  hitting httpbin over real HTTP (``httpx_http``)

Without a usable API key the script exits with a pointer instead of
degrading into recovery lines mid-conversation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator, cast

import anyio

from superdialog.llm.litellm_provider import LitellmProvider
from superdialog.playbook import Playbook, PlaybookAgent, httpx_http
from superdialog.stream import StreamChunk

PLAYBOOK_PATH = Path(__file__).parent / "playbooks" / "booking.yaml"
DEFAULT_MODEL = "openai/gpt-5.1"

_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


class DirectorLLM:
    """``CompletesLLM`` adapter: one plain-text completion per verdict."""

    def __init__(self, provider: LitellmProvider) -> None:
        self._provider = provider

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return (await self._provider.complete(list(messages))).text


class TalkerLLM:
    """``StreamsLLM`` adapter: yield raw text tokens from the provider."""

    def __init__(self, provider: LitellmProvider) -> None:
        self._provider = provider

    async def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        async for chunk in self._provider.stream(list(messages)):
            if chunk.text:
                yield chunk.text


def _missing_key_hint(model: str) -> str | None:
    """Name the missing API key env var for known providers, else None."""
    provider = model.partition("/")[0].lower()
    env_var = _KEY_ENV.get(provider)
    if env_var is None or os.environ.get(env_var):
        return None
    return (
        f"No {env_var} set for model {model!r}.\n"
        f'  export {env_var}="sk-..."        # key for {provider}\n'
        '  export SUPERDIALOG_MODEL="..."    # or any other litellm model\n'
        "Then re-run: uv run python examples/playbook_quickstart.py"
    )


def _build_agent(model: str) -> PlaybookAgent:
    provider = LitellmProvider(model)
    return PlaybookAgent(
        playbook=Playbook.load(str(PLAYBOOK_PATH)),
        talker_llm=TalkerLLM(provider),
        director_llm=DirectorLLM(provider),
        http=httpx_http,  # the booking pipeline makes real HTTP calls
    )


async def main() -> None:
    model = os.environ.get("SUPERDIALOG_MODEL", DEFAULT_MODEL)
    hint = _missing_key_hint(model)
    if hint is not None:
        print(hint)
        return
    agent = _build_agent(model)
    print(f"Glow Studio booking demo on {model} — type 'quit' to exit.")
    print("Try: hi, I'm Sam — can I get a haircut next Friday?")

    while True:
        try:
            text = await anyio.to_thread.run_sync(input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        if not text.strip():
            continue

        print("mira> ", end="", flush=True)
        metadata: dict[str, Any] = {}
        try:
            chunks = cast(
                AsyncIterator[StreamChunk], await agent.turn(text, stream=True)
            )
            async for chunk in chunks:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                if chunk.done and chunk.turn is not None:
                    metadata = chunk.turn.metadata
            print()
        except Exception as exc:  # noqa: BLE001 — REPL surface, not a library
            print(f"\n[error] LLM call failed: {exc}")
            print(
                "Check your provider API key (e.g. OPENAI_API_KEY) and "
                "SUPERDIALOG_MODEL, then try again."
            )
            return
        if metadata.get("ended"):
            print(f"[session ended — outcome: {metadata.get('outcome')}]")
            break


if __name__ == "__main__":
    anyio.run(main)
