"""Unit test for the ResolverChatModel LangChain shim (no real provider)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from superdialog.eval.metrics import _langchain_shim as shim
from tests.eval.fakes import FakeProvider


async def test_resolver_chat_model_uses_superdialog_provider(monkeypatch):
    monkeypatch.setattr(
        shim, "resolve_llm", lambda uri: FakeProvider({"*": "the reply"})
    )
    model = shim.ResolverChatModel(model_uri="x")
    result = await model._agenerate([HumanMessage(content="hi")])
    assert result.generations[0].message.content == "the reply"
