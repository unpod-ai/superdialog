"""Canonical JSON: one serializer for anything embedded in an LLM prompt.

Same logical value ⇒ same bytes (sorted keys, compact separators, ASCII-safe),
so prompt prefixes stay byte-identical across turns, forks, and replays — the
property provider prompt caches key on. A dict that happens to iterate in a
different order must never silently invalidate the whole KV cache.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to canonical, byte-stable JSON."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
