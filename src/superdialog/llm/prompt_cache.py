"""Provider-side prompt caching: mark the stable prompt prefix as cacheable.

Assemblers annotate the leading ``system`` message with a private
``_cache_prefix`` key holding the *stable* persona/preamble substring (the bytes
that repeat every turn). ``content`` itself stays a bare string, so nothing
upstream that treats system content as text is affected.

:func:`mark_cache_prefix` runs once at the :class:`ResilientProvider` seam — the
last step before the backend call — and does one of:

* caching off, an automatic-cache provider (OpenAI/Deepseek/xAI), or any error
  → **strip** the private key and return byte-identical legacy messages;
* an explicit-cache-control provider (Anthropic/Bedrock/Vertex/Gemini) with
  caching on → split the system content at the stable boundary into two text
  blocks and tag the stable block (and the last tool) with ``cache_control``.

Automatic-cache providers still benefit: the assembler now guarantees a stable
prompt prefix, which is all their server-side cache needs — no marker required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Private message key an assembler sets to the stable prefix substring.
CACHE_PREFIX_KEY = "_cache_prefix"

#: Providers whose prompt caching needs an explicit ``cache_control`` marker.
#: Everything else (openai, deepseek, xai, azure-openai, …) caches
#: automatically from a stable prefix and must NOT receive a marker.
_EXPLICIT_CACHE_PREFIXES = (
    "anthropic",
    "bedrock",
    "vertex_ai",
    "vertex_ai_beta",
    "gemini",
)


@dataclass(frozen=True)
class PromptCacheConfig:
    """Prompt-caching knobs. Disabled by default — flip on per deployment."""

    enabled: bool = False
    ttl: str | None = None  # None -> provider default (5m); some support "1h".

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "PromptCacheConfig":
        e = env if env is not None else os.environ
        raw = e.get("SUPERDIALOG_PROMPT_CACHING")
        enabled = bool(raw) and raw.strip().lower() in ("1", "true", "yes", "on")
        return cls(enabled=enabled, ttl=e.get("SUPERDIALOG_PROMPT_CACHE_TTL") or None)


def _provider_prefix(model: str) -> str:
    return model.split("/", 1)[0].lower() if model else ""


def uses_explicit_cache_control(model: str) -> bool:
    """True when the model's provider needs an explicit ``cache_control`` marker."""
    return _provider_prefix(model) in _EXPLICIT_CACHE_PREFIXES


def _strip_prefix_keys(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop the private ``_cache_prefix`` key wherever present.

    Returns the *same* list object when no key is found, so the common
    (caching-off, un-annotated) path is a true no-op.
    """
    changed = False
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and CACHE_PREFIX_KEY in m:
            out.append({k: v for k, v in m.items() if k != CACHE_PREFIX_KEY})
            changed = True
        else:
            out.append(m)
    return out if changed else messages


def _cache_control(ttl: str | None) -> dict[str, Any]:
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cc["ttl"] = ttl
    return cc


def _mark_last_tool(
    tools: list[dict[str, Any]], ttl: str | None
) -> list[dict[str, Any]]:
    """Tag the final tool so the whole tools prefix caches as one segment."""
    out = list(tools)
    out[-1] = {**out[-1], "cache_control": _cache_control(ttl)}
    return out


def _apply_markers(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    ttl: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Split the first annotated system message and tag stable block + tools."""
    new_msgs: list[dict[str, Any]] = []
    marked = False
    for m in messages:
        if not (isinstance(m, dict) and CACHE_PREFIX_KEY in m):
            new_msgs.append(m)
            continue
        base = {k: v for k, v in m.items() if k != CACHE_PREFIX_KEY}
        stable = m[CACHE_PREFIX_KEY]
        content = m.get("content")
        if (
            not marked
            and m.get("role") == "system"
            and isinstance(content, str)
            and stable
            and content.startswith(stable)
        ):
            dynamic = content[len(stable) :]
            blocks: list[dict[str, Any]] = [
                {"type": "text", "text": stable, "cache_control": _cache_control(ttl)}
            ]
            if dynamic:
                blocks.append({"type": "text", "text": dynamic})
            base["content"] = blocks
            marked = True
        new_msgs.append(base)
    if tools:
        tools = _mark_last_tool(tools, ttl)
    return new_msgs, tools


def mark_cache_prefix(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    *,
    enabled: bool,
    ttl: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return ``(messages, tools)`` with the stable prefix marked cacheable.

    Always strips the private annotation; fail-open to byte-identical legacy
    messages when caching is off, the provider auto-caches, or any error occurs.
    """
    clean = _strip_prefix_keys(messages)
    if not enabled or not uses_explicit_cache_control(model):
        return clean, tools
    try:
        return _apply_markers(messages, tools, ttl)
    except Exception as exc:  # caching must never fail a turn
        logger.debug("prompt cache marking skipped: %s", exc)
        return clean, tools


__all__ = [
    "CACHE_PREFIX_KEY",
    "PromptCacheConfig",
    "mark_cache_prefix",
    "uses_explicit_cache_control",
]
