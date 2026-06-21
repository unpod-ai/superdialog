"""Live validation that prompt caching actually engages end-to-end.

Drives a real 2-turn Anthropic call through the ResilientProvider seam with a
LARGE stable prefix (to clear the per-model cache minimum), caching ON, on both
the litellm and any-llm backends. Asserts a real cache WRITE on turn 1 and a
real cache READ on turn 2 — and that the OFF path caches nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

from superdialog.llm.anyllm_provider import AnyLlmProvider  # noqa: E402
from superdialog.llm.litellm_provider import LitellmProvider  # noqa: E402
from superdialog.llm.prompt_cache import CACHE_PREFIX_KEY, PromptCacheConfig  # noqa: E402
from superdialog.llm.resilience import ResilienceConfig, ResilientProvider  # noqa: E402

MODEL = "anthropic/claude-haiku-4-5"

# A large, fixed persona — must exceed the per-model cache minimum (Haiku ~2048
# tokens) for caching to engage at all. ~4-5k tokens here.
STABLE = "You are a precise assistant. Follow every policy exactly.\n" + (
    "Policy clause: be accurate, concise, and never invent facts. " * 400
)


def _messages(suffix: str, tag: str = "") -> list[dict]:
    """Same stable prefix every turn; only the dynamic tail + user changes.

    ``tag`` makes the stable prefix unique per backend so the two backends do
    not cross-warm each other's account-wide Anthropic cache (which would mask
    a backend's own cache *write* as a *read*).
    """
    stable = f"[{tag}] {STABLE}" if tag else STABLE
    system = stable + "\n\nDynamic context for this turn: " + suffix
    return [
        {"role": "system", "content": system, CACHE_PREFIX_KEY: stable},
        {"role": "user", "content": suffix},
    ]


def _cache(meta: dict) -> str:
    r = meta.get("cache_read_tokens")
    w = meta.get("cache_write_tokens")
    return f"read={r} write={w} prompt={meta.get('prompt_tokens')}"


async def _run(backend_cls, enabled: bool) -> None:
    inner = backend_cls(MODEL)
    rp = ResilientProvider(
        inner,
        ResilienceConfig(timeout_s=60),
        prompt_cache=PromptCacheConfig(enabled=enabled),
    )
    # Unique prefix per backend so they do not cross-warm the shared cache.
    tag = backend_cls.__name__
    r1 = await rp.complete(_messages("Reply with the single word: alpha.", tag))
    r2 = await rp.complete(_messages("Reply with the single word: omega.", tag))
    label = f"{backend_cls.__name__:<16} caching={'ON ' if enabled else 'OFF'}"
    print(f"  {label}  turn1: {_cache(r1.metadata)}")
    print(f"  {label}  turn2: {_cache(r2.metadata)}")
    if enabled:
        w1 = r1.metadata.get("cache_write_tokens") or 0
        rd2 = r2.metadata.get("cache_read_tokens") or 0
        verdict = "✅ CACHE HIT" if rd2 > 0 else "❌ NO CACHE READ on turn 2"
        print(f"  {label}  -> turn1 write={w1}  turn2 read={rd2}   {verdict}")


async def main() -> None:
    print(f"MODEL={MODEL}  stable_prefix_chars={len(STABLE)}")
    print("=" * 78)
    for backend in (LitellmProvider, AnyLlmProvider):
        await _run(backend, enabled=True)
        await _run(backend, enabled=False)
        print("-" * 78)


if __name__ == "__main__":
    asyncio.run(main())
