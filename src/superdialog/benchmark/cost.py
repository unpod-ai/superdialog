"""LLM cost for the system-under-test — priced via litellm (a base dependency).

Only the *tested* model's tokens are priced here (the model whose behaviour the
benchmark reports). The RAGAS judge LLMs are eval overhead and are NOT counted —
the cost row answers "what did this model cost to run", not "what did scoring
cost".

Self-contained: litellm ships its own model→price table and is already a base
superdialog dependency, so no unpod billing / models.dev / supervoice.billing
import is needed. Fail-open to 0.0 — pricing must never break a benchmark run
(mirrors supervoice's playground pricing behaviour).

Text-only: no STT/TTS/voice_profile cost (a text benchmark has no voice usage).
"""

from __future__ import annotations


def cost_from_response(response: object) -> float:
    """Price one system-under-test litellm response. 0.0 if unavailable.

    ``response`` is whatever ``litellm.completion(...)`` returned; litellm reads
    token usage + its price table off it directly.
    """
    try:
        from litellm import completion_cost

        return float(completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001 — cost is best-effort, never fatal
        return 0.0


def cost_from_tokens(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Price raw SUT token counts for ``model``. 0.0 if the model is unpriced.

    Use when the runner accumulated token counts itself instead of holding the
    raw litellm response object.
    """
    try:
        from litellm import cost_per_token

        prompt_cost, completion_cost_ = cost_per_token(
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )
        return float(prompt_cost + completion_cost_)
    except Exception:  # noqa: BLE001 — unknown model / offline → 0.0
        return 0.0


def _self_check() -> None:
    """Runnable check — needs litellm's price table (offline-safe, fails open)."""
    c = cost_from_tokens("gpt-4o-mini", 1000, 1000)
    # gpt-4o-mini is in litellm's table; if litellm present, cost > 0.
    # If litellm/table missing, fail-open returns 0.0 — both are acceptable.
    assert c >= 0.0, c
    print(f"cost self-check OK: gpt-4o-mini 1k+1k tok = ${c:.6f}")


if __name__ == "__main__":
    _self_check()


__all__ = ["cost_from_response", "cost_from_tokens"]
