"""Fold transcript turns dropped by budget pressure into ``state.summary``.

``render.render_view`` packs the transcript newest-first and drops the oldest
entries once the budget is spent, so on a long call the facts established early
(the caller's name, what they agreed to) leave the Talker's view entirely. The
system block's summary is protected from that pressure -- this module rewrites
it to carry the dropped turns, so the facts survive as prose after the verbatim
turns are gone.

Off the speech path: called from the post-turn block, never before speaking.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from .render import estimate_tokens
from .state import TranscriptEntry

_log = logging.getLogger(__name__)

# The summary competes with guidance and slots for the protected system block,
# so it must stay short enough to never be the reason the block is oversized.
_MAX_SENTENCES = 5

# ...and the sentence limit above is only a REQUEST to the model. This is the
# enforcement, and it matters more here than in a general chat app: the summary
# lives in the protected block, so the budget packer can never trim it, and the
# instruction below tells the next pass to preserve what it is given. An
# over-long summary therefore ratchets -- it cannot shrink and cannot be
# dropped, permanently starving the transcript it exists to protect.
_MAX_SUMMARY_TOKENS = 220
# Sentence terminators incl. the Devanagari danda: clamping mid-word would put
# a truncated fragment into the Talker's prompt as if it were a whole fact.
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")

_INSTRUCTION = f"""You maintain the running memory of a live phone call.

Rewrite the memory below so it also covers the dropped turns that follow it.

Rules:
- PRESERVE every fact already in the existing memory. It may include a digest
  of an EARLIER call with this caller; that digest must survive your rewrite.
- ADD what the dropped turns established: names, numbers, dates, commitments,
  questions already answered, anything the caller would be annoyed to repeat.
- Drop pleasantries, filler and small talk.
- Never invent, infer or guess. Only what was actually said.
- At most {_MAX_SENTENCES} sentences, plain prose, no headings, no bullets.

Reply with the rewritten memory and nothing else."""


def _render_entries(entries: Sequence[TranscriptEntry]) -> str:
    return "\n".join(f"{e.role}: {e.text}" for e in entries if e.text.strip())


def _clamp(text: str) -> str:
    """Trim to _MAX_SUMMARY_TOKENS on a sentence boundary; never return "".

    A single sentence longer than the whole budget still gets through -- half a
    sentence is worse than a slightly long one, and "" would read as failure
    and silently keep the previous summary.
    """
    if estimate_tokens(text) <= _MAX_SUMMARY_TOKENS:
        return text
    kept: list[str] = []
    used = 0
    for sentence in _SENTENCE_END.split(text):
        cost = estimate_tokens(sentence)
        if kept and used + cost > _MAX_SUMMARY_TOKENS:
            break
        kept.append(sentence)
        used += cost + 1  # the joining space is real bytes in the final string
    out = " ".join(kept).strip()
    _log.warning(
        "[compact] summary clamped %d -> ~%d estimated tokens; tighten the "
        "checkpoint's summary_prompt if this recurs",
        estimate_tokens(text),
        estimate_tokens(out),
    )
    return out


async def compact(
    llm: Any,
    prior: str,
    entries: Sequence[TranscriptEntry],
    prompt: str = "",
) -> str:
    """Rewrite ``prior`` to also cover ``entries``. Returns "" on any failure.

    ``prompt`` overrides the default instruction -- a checkpoint running
    ``reset_with_summary`` passes its own ``summary_prompt`` so a booking step
    can ask for order details where an objection detour asks for concerns.

    "" means "leave the existing summary alone" -- a broken or empty completion
    must never blank out memory the call already depends on. Errors are logged
    and swallowed here so the caller's post-turn block cannot be broken by a
    slow or down compactor model.
    """
    body = _render_entries(entries)
    if not body:
        return ""
    messages = [
        {"role": "system", "content": prompt.strip() or _INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"Existing memory:\n{prior.strip() or '(none yet)'}\n\n"
                f"Dropped turns:\n{body}"
            ),
        },
    ]
    try:
        raw = await llm.complete(messages)
    except Exception as exc:  # noqa: BLE001 -- loud, never fatal
        _log.error("[compact] FAILED %s", type(exc).__name__, exc_info=True)
        return ""
    text = str(getattr(raw, "text", raw) or "").strip()
    if not text:
        _log.warning("[compact] empty completion; keeping the existing summary")
        return ""
    return _clamp(text)
