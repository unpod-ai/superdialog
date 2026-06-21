"""Tests for prompt-cache prefix marking (llm/prompt_cache.py).

The seam must be byte-identical to the legacy prompt when caching is off, on an
automatic-cache provider, or on any error — and must mark only the stable block
(+ last tool) for explicit-cache-control providers when enabled.
"""

from __future__ import annotations

from superdialog.llm.prompt_cache import (
    CACHE_PREFIX_KEY,
    PromptCacheConfig,
    mark_cache_prefix,
    uses_explicit_cache_control,
)

STABLE = "You are Saanvi, a calm wellness assistant. Always speak numbers as words."
DYNAMIC = "\nCurrent date and time: 2026-06-21. Known slots: {service: detox}."
ANTHROPIC = "anthropic/claude-haiku-4-5"
OPENAI = "openai/gpt-4.1-mini"


def _annotated() -> list[dict]:
    return [
        {"role": "system", "content": STABLE + DYNAMIC, CACHE_PREFIX_KEY: STABLE},
        {"role": "user", "content": "hi"},
    ]


def _tools() -> list[dict]:
    return [
        {"type": "function", "function": {"name": "a"}},
        {"type": "function", "function": {"name": "b"}},
    ]


# ── provider classification ──────────────────────────────────────────────────


def test_explicit_cache_providers():
    assert uses_explicit_cache_control(ANTHROPIC)
    assert uses_explicit_cache_control("bedrock/anthropic.claude-3")
    assert uses_explicit_cache_control("vertex_ai/claude")
    assert not uses_explicit_cache_control(OPENAI)
    assert not uses_explicit_cache_control("deepseek/deepseek-chat")
    assert not uses_explicit_cache_control("gpt-4.1-mini")


# ── disabled / automatic / error → byte-identical legacy prompt ──────────────


def test_disabled_strips_key_and_keeps_string():
    msgs, tools = mark_cache_prefix(_annotated(), None, ANTHROPIC, enabled=False)
    assert all(CACHE_PREFIX_KEY not in m for m in msgs)
    assert msgs[0]["content"] == STABLE + DYNAMIC  # still a bare string, intact
    assert msgs[0]["role"] == "system"


def test_automatic_provider_not_marked_just_stripped():
    msgs, _ = mark_cache_prefix(_annotated(), None, OPENAI, enabled=True)
    assert all(CACHE_PREFIX_KEY not in m for m in msgs)
    assert msgs[0]["content"] == STABLE + DYNAMIC  # OpenAI keeps bare string


def test_unannotated_messages_returned_identically():
    plain = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    msgs, _ = mark_cache_prefix(plain, None, ANTHROPIC, enabled=True)
    assert msgs == plain  # content unchanged when nothing is annotated
    assert all(m1 is m2 for m1, m2 in zip(msgs, plain))  # dicts never copied


def test_disabled_unannotated_is_same_object():
    plain = [{"role": "system", "content": "x"}]
    msgs, _ = mark_cache_prefix(plain, None, ANTHROPIC, enabled=False)
    assert msgs is plain  # off + no key -> zero allocation, true no-op


def test_prefix_mismatch_fails_open():
    bad = [{"role": "system", "content": "DIFFERENT", CACHE_PREFIX_KEY: STABLE}]
    msgs, _ = mark_cache_prefix(bad, None, ANTHROPIC, enabled=True)
    assert msgs[0]["content"] == "DIFFERENT"  # unchanged string
    assert CACHE_PREFIX_KEY not in msgs[0]  # key always stripped


# ── explicit provider + enabled → marked correctly ──────────────────────────


def test_anthropic_marks_stable_block_only():
    msgs, _ = mark_cache_prefix(_annotated(), None, ANTHROPIC, enabled=True)
    content = msgs[0]["content"]
    assert isinstance(content, list) and len(content) == 2
    assert content[0]["text"] == STABLE
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[1]["text"] == DYNAMIC
    assert "cache_control" not in content[1]  # dynamic block NOT cached
    assert CACHE_PREFIX_KEY not in msgs[0]


def test_ttl_threaded_into_cache_control():
    msgs, _ = mark_cache_prefix(_annotated(), None, ANTHROPIC, enabled=True, ttl="1h")
    assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_last_tool_marked_only():
    _, tools = mark_cache_prefix(_annotated(), _tools(), ANTHROPIC, enabled=True)
    assert "cache_control" not in tools[0]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


def test_no_dynamic_means_single_block():
    msgs = [{"role": "system", "content": STABLE, CACHE_PREFIX_KEY: STABLE}]
    out, _ = mark_cache_prefix(msgs, None, ANTHROPIC, enabled=True)
    assert len(out[0]["content"]) == 1
    assert out[0]["content"][0]["text"] == STABLE


def test_does_not_mutate_input():
    src = _annotated()
    mark_cache_prefix(src, None, ANTHROPIC, enabled=True)
    assert src[0][CACHE_PREFIX_KEY] == STABLE  # original untouched
    assert isinstance(src[0]["content"], str)


# ── config ───────────────────────────────────────────────────────────────────


def test_config_from_env_default_off(monkeypatch):
    monkeypatch.delenv("SUPERDIALOG_PROMPT_CACHING", raising=False)
    assert PromptCacheConfig.from_env().enabled is False


def test_config_from_env_on(monkeypatch):
    monkeypatch.setenv("SUPERDIALOG_PROMPT_CACHING", "true")
    monkeypatch.setenv("SUPERDIALOG_PROMPT_CACHE_TTL", "1h")
    cfg = PromptCacheConfig.from_env()
    assert cfg.enabled is True and cfg.ttl == "1h"


# ── cache-token extraction (telemetry) ───────────────────────────────────────

from types import SimpleNamespace  # noqa: E402

from superdialog.llm.anyllm_provider import _extract_usage  # noqa: E402


def test_extract_usage_anthropic_cache_fields():
    u = SimpleNamespace(
        prompt_tokens=500,
        completion_tokens=20,
        cache_read_input_tokens=480,
        cache_creation_input_tokens=0,
    )
    out = _extract_usage(u)
    assert out["cache_read_tokens"] == 480
    assert out["cache_write_tokens"] == 0
    assert out["prompt_tokens"] == 500


def test_extract_usage_openai_cached_tokens():
    u = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=15,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
    )
    out = _extract_usage(u)
    assert out["cache_read_tokens"] == 1024
    assert "cache_write_tokens" not in out


def test_extract_usage_deepseek_hit_tokens():
    u = SimpleNamespace(
        prompt_tokens=800, completion_tokens=10, prompt_cache_hit_tokens=768
    )
    assert _extract_usage(u)["cache_read_tokens"] == 768


def test_extract_usage_no_cache_fields_omitted():
    u = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    out = _extract_usage(u)
    assert "cache_read_tokens" not in out and "cache_write_tokens" not in out
    assert out == {"prompt_tokens": 10, "completion_tokens": 5}


# ── end-to-end through the ResilientProvider seam ────────────────────────────

import anyio  # noqa: E402

from superdialog.llm.provider import CompletionResult  # noqa: E402
from superdialog.llm.resilience import (  # noqa: E402
    ResilienceConfig,
    ResilientProvider,
)


class _CapturingBackend:
    """Minimal LLMProvider that records what the seam hands it."""

    def __init__(self, model: str):
        self.model = model
        self.seen: dict = {}

    async def complete(self, messages, tools=None, **opts):
        self.seen["messages"] = messages
        self.seen["tools"] = tools
        return CompletionResult(text="ok", tool_calls=[], metadata={})


def _run_complete(model: str, enabled: bool):
    backend = _CapturingBackend(model)
    rp = ResilientProvider(
        backend,
        ResilienceConfig(timeout_s=None),
        prompt_cache=PromptCacheConfig(enabled=enabled),
    )
    msgs = [
        {"role": "system", "content": STABLE + DYNAMIC, CACHE_PREFIX_KEY: STABLE},
        {"role": "user", "content": "hi"},
    ]
    anyio.run(lambda: rp.complete(msgs))
    return backend.seen["messages"]


def test_seam_marks_annotated_prefix_when_enabled():
    sys_msg = _run_complete(ANTHROPIC, enabled=True)[0]
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert sys_msg["content"][0]["text"] == STABLE
    assert CACHE_PREFIX_KEY not in sys_msg  # private key never reaches backend


def test_seam_noop_when_disabled():
    sys_msg = _run_complete(ANTHROPIC, enabled=False)[0]
    assert sys_msg["content"] == STABLE + DYNAMIC  # bare string, byte-identical
    assert CACHE_PREFIX_KEY not in sys_msg


# ── live: real Anthropic cache read (gated on ANTHROPIC_API_KEY) ─────────────
#
# Validated 2026-06-21 against anthropic/claude-haiku-4-5 through the seam:
#   litellm  ON  -> turn1 cache_write=6021, turn2 cache_read=6021  (full telemetry)
#   any-llm  ON  -> turn2 cache_read=6022 (caching works); write count NOT
#                   surfaced — any-llm normalizes to OpenAI usage shape, which
#                   has cached_tokens (read) but no cache_creation (write) field.
#   both     OFF -> no cache interaction at all.

import os  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # parent super/.env
_HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Large fixed prefix to clear the per-model cache minimum (Haiku ~2048 tokens).
_BIG_STABLE = "You are a precise assistant.\n" + (
    "Policy clause: be accurate, concise, and never invent facts. " * 400
)


def _big_msgs(suffix: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": _BIG_STABLE + "\n\nDynamic: " + suffix,
            CACHE_PREFIX_KEY: _BIG_STABLE,
        },
        {"role": "user", "content": suffix},
    ]


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
def test_live_anthropic_cache_read_on_second_turn():
    """Two real turns: the second reads the stable prefix from Anthropic's cache."""
    from superdialog.llm.litellm_provider import LitellmProvider

    rp = ResilientProvider(
        LitellmProvider(ANTHROPIC),
        ResilienceConfig(timeout_s=60),
        prompt_cache=PromptCacheConfig(enabled=True),
    )

    async def _two_turns():
        await rp.complete(_big_msgs("Reply with one word: alpha."))
        return await rp.complete(_big_msgs("Reply with one word: omega."))

    r2 = anyio.run(_two_turns)
    assert (r2.metadata.get("cache_read_tokens") or 0) > 0


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
def test_live_anthropic_no_cache_when_disabled():
    """Caching off → Anthropic reports no cache read even for a repeated prefix."""
    from superdialog.llm.litellm_provider import LitellmProvider

    rp = ResilientProvider(
        LitellmProvider(ANTHROPIC),
        ResilienceConfig(timeout_s=60),
        prompt_cache=PromptCacheConfig(enabled=False),
    )

    async def _two_turns():
        await rp.complete(_big_msgs("Reply with one word: one."))
        return await rp.complete(_big_msgs("Reply with one word: two."))

    r2 = anyio.run(_two_turns)
    assert not r2.metadata.get("cache_read_tokens")


def test_seam_strips_key_for_automatic_provider():
    sys_msg = _run_complete(OPENAI, enabled=True)[0]
    assert sys_msg["content"] == STABLE + DYNAMIC  # OpenAI stays bare string
    assert CACHE_PREFIX_KEY not in sys_msg
