"""A minimal LangChain BaseChatModel backed by superdialog's own LLM resolver.

Lets RAGAS (which speaks LangChain) run on superdialog's provider stack instead
of a hardcoded OpenAI key.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from superdialog.llm.resolver import resolve_llm

_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


def _to_provider_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    return [
        {"role": _ROLE.get(m.type, "user"), "content": str(m.content)} for m in messages
    ]


def _wrap(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class ResolverChatModel(BaseChatModel):
    """LangChain chat model delegating to resolve_llm(model_uri).complete."""

    model_uri: str

    @property
    def _llm_type(self) -> str:
        return "superdialog-resolver"

    async def _acomplete(self, messages: list[BaseMessage]) -> str:
        provider = resolve_llm(self.model_uri)
        res = await provider.complete(_to_provider_messages(messages))
        return res if isinstance(res, str) else getattr(res, "text", str(res))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return _wrap(await self._acomplete(messages))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # ponytail: sync path is a fallback; RAGAS uses the async path.
        import asyncio

        return _wrap(asyncio.run(self._acomplete(messages)))
