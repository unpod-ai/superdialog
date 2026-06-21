"""Hermetic tests for the prompt-cache prefix annotation in ToolCallAdapter.

The routing call (``evaluate_criteria``) builds a leading ``system`` message
whose content begins with a FIXED routing-rule preamble; the volatile time line
and ``[CURRENT DATA]`` block follow it. These tests confirm the adapter tags
that preamble as the cacheable prefix (``_cache_prefix``) on the system dict it
hands to the provider, and that the prefix is a true byte prefix of the content.

Offline only — the LLM provider is stubbed; no network.
"""
from __future__ import annotations

from typing import Any

import pytest

from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY
from superdialog.llm.provider import CompletionResult
from superdialog.machine.adapters.toolcall_adapter import (
    LLMCallData,
    ToolCallAdapter,
)
from superdialog.machine.models import ToolDescriptor


class _StubFlow:
    """Minimal flow exposing the attrs _build_instructions reads."""

    system_prompt = "You are Asha, a warm clinic assistant. Always be concise."
    agent_language = "en"


class _StubMachine:
    """Minimal machine stub for _build_instructions / evaluate_criteria."""

    context = None  # resolve_language tolerates a missing context

    def __init__(self) -> None:
        self._flow = _StubFlow()

    def get_tools_for_node(self, node: Any) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(
                id="book_appointment",
                description="Caller wants to book an appointment",
                is_data_collection=False,
                input_schema=None,
                target_node_id="book",
            )
        ]


class _StubNode:
    """Minimal FlowNode-like object."""

    id = "greet"
    instruction = "Greet the caller and ask how you can help."
    static_text = None
    is_final = False
    edges: list[Any] = []


class _StubProvider:
    """Records the messages it is handed; returns a fixed tool call."""

    def __init__(self) -> None:
        self.seen_messages: list[dict[str, Any]] | None = None

    async def complete(
        self, messages: list[dict[str, Any]], tools=None, **opts: Any
    ) -> CompletionResult:
        self.seen_messages = messages
        return CompletionResult(
            text="",
            tool_calls=[
                {"function": {"name": "__stay_on_node__", "arguments": "{}"}}
            ],
            metadata={"prompt_tokens": 10, "completion_tokens": 1},
        )


async def _run_eval() -> tuple[ToolCallAdapter, _StubProvider, list[LLMCallData]]:
    adapter = ToolCallAdapter(model_id="anthropic/claude-3-5-sonnet")
    adapter._machine = _StubMachine()
    provider = _StubProvider()
    adapter._provider = provider  # bypass live resolution

    received: list[LLMCallData] = []

    async def _cb(d: LLMCallData) -> None:
        received.append(d)

    adapter._on_llm_complete = _cb

    await adapter.evaluate_criteria(
        node=_StubNode(),
        history=[{"role": "user", "content": "hi there"}],
        userdata={},
        silent=False,
    )
    return adapter, provider, received


def _expected_routing_rule() -> str:
    """The exact non-silent routing-rule preamble built by the adapter."""
    return (
        "ROUTING RULE: Call __stay_on_node__ unless the caller's message "
        "CLEARLY and UNAMBIGUOUSLY satisfies one specific edge condition. "
        "MANDATORY __stay_on_node__ cases:\n"
        "- CONTEXT MISMATCH: caller says something that has nothing to do with "
        "the agent's current question (e.g. agent asked 'can we talk?' and "
        "caller says a year/number; agent asked yes/no and caller gave unrelated "
        "data; the input makes no sense as an answer to the current question)\n"
        "- Asking agent to repeat/re-read ('address बताइए', 'फिर से बोलिए', "
        "'didn't hear', 'what was that', 'tell me again')\n"
        "- Compliments, tangents, frustration outbursts, filler, testing phrases\n"
        "- Partial or cut-off sentences\n"
        "- 'no' as a correction not a goodbye\n"
        "- 'thank you for confirming' — NOT a card receipt confirmation\n"
        "- Any response that doesn't directly answer the agent's current question\n"
        "OBJECTIVE RULE: Always complete the current node's objective before "
        "transitioning. If the caller's response doesn't fulfill the objective, stay.\n"
        "Never force-fit an off-topic response to the closest-sounding edge."
    )


@pytest.mark.anyio
async def test_system_message_carries_cache_prefix():
    _adapter, provider, _received = await _run_eval()
    assert provider.seen_messages is not None
    system_msg = provider.seen_messages[0]
    assert system_msg["role"] == "system"
    # (a) content is a plain str — nothing upstream sees a non-string.
    assert isinstance(system_msg["content"], str)
    # The cache prefix annotation rides with the dict to the provider.
    assert CACHE_PREFIX_KEY in system_msg


@pytest.mark.anyio
async def test_cache_prefix_is_true_leading_substring():
    _adapter, provider, _received = await _run_eval()
    assert provider.seen_messages is not None
    system_msg = provider.seen_messages[0]
    content = system_msg["content"]
    prefix = system_msg[CACHE_PREFIX_KEY]
    # (b) content begins byte-for-byte with the annotated prefix.
    assert content.startswith(prefix)
    # The prefix is non-empty and strictly shorter (volatile tail follows).
    assert prefix
    assert len(prefix) < len(content)


@pytest.mark.anyio
async def test_cache_prefix_is_routing_rule_plus_persona():
    """The cacheable prefix is routing rule + flow persona (both fixed every
    turn), so the large persona is cached, not just the small routing rule."""
    _adapter, provider, _received = await _run_eval()
    assert provider.seen_messages is not None
    system_msg = provider.seen_messages[0]
    prefix = system_msg[CACHE_PREFIX_KEY]
    expected = f"{_expected_routing_rule()}\n\n{_StubFlow.system_prompt}"
    assert prefix == expected
    # the persona is genuinely inside the cached prefix (the point of the reorder)
    assert _StubFlow.system_prompt in prefix
    # the volatile time line is NOT in the cached prefix
    assert "[TODAY]" not in prefix


@pytest.mark.anyio
async def test_cache_prefix_is_routing_rule_only_without_persona():
    """No flow system_prompt -> prefix is just the routing rule (nothing else
    stable leads), and the time line stays first in the body."""
    adapter = ToolCallAdapter(model_id="anthropic/claude-3-5-sonnet")
    machine = _StubMachine()
    machine._flow.system_prompt = ""  # flow without a persona
    adapter._machine = machine
    provider = _StubProvider()
    adapter._provider = provider

    await adapter.evaluate_criteria(
        node=_StubNode(),
        history=[{"role": "user", "content": "hi"}],
        userdata={},
        silent=False,
    )
    assert provider.seen_messages is not None
    system_msg = provider.seen_messages[0]
    assert system_msg[CACHE_PREFIX_KEY] == _expected_routing_rule()
    # with no persona, the volatile time line leads the body (original layout)
    content = system_msg["content"]
    assert content.startswith(_expected_routing_rule() + "\n\n[TODAY]")
