"""Backend A/B latency benchmark: any-llm vs litellm, for flow & playbook engines.

Holds the model fixed and swaps ONLY the backend via URI scheme prefix
(``anyllm/...`` vs ``litellm/...``), so the measured delta is pure SDK/transport
overhead. Same wellness scenario in both engines (apples-to-apples):

    flow     -> examples/flows/health_wellness_3node.json
    playbook -> examples/playbooks/health_wellness_3node.yaml

Matrix: backend{anyllm,litellm} x engine{flow,playbook} x provider{openai,anthropic}.

Usage:
    python bench_backend_ab.py smoke        # N=1, validate the path cheaply
    python bench_backend_ab.py full 5       # N=5 (default), write bench_backend_ab.json

Note: the graph (flow) engine streams *locally* (chunks an already-complete
response) in v0.x, so its TTFT ~= total. The playbook engine streams the Talker
live, so its TTFT < total. Both totals are real end-to-end LLM round-trips.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
# Keys live in .env / ../.env (both providers). Load both; first wins per key.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

# Quiet the very chatty litellm logger so progress stays readable.
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

FLOW_JSON = ROOT / "examples/flows/health_wellness_3node.json"
PLAYBOOK_YAML = ROOT / "examples/playbooks/health_wellness_3node.yaml"

# Fixed user script that walks greeting -> collect_details -> confirm(terminal).
SCRIPT = [
    "Hi, I'd like to book a Panchakarma detox session.",
    "Next Monday at ten in the morning works for me.",
    "Yes, that's perfect. Thank you!",
]

BACKENDS = ["anyllm", "litellm"]
PROVIDERS = {
    "openai": "openai/gpt-4.1-mini",
    "anthropic": "anthropic/claude-haiku-4-5",
}
ENGINES = ["flow", "playbook"]


def _build_machine(engine: str, uri: str):
    """Construct a DialogMachine for the given engine + backend-prefixed URI."""
    from superdialog import DialogMachine, Flow

    if engine == "flow":
        return DialogMachine(flow=Flow.load(str(FLOW_JSON)), llm=uri)
    from superdialog.playbook import Playbook

    return DialogMachine(
        source=Playbook.load(str(PLAYBOOK_YAML)), llm=uri, engine="playbook"
    )


async def _stream_turn_metrics(machine, text: str) -> tuple[float, float]:
    """Drive one streaming turn; return (ttft_s, total_s)."""
    t0 = time.perf_counter()
    ttft: float | None = None
    stream = await machine.turn(text, stream=True)
    async for chunk in stream:
        if ttft is None and getattr(chunk, "text", None):
            ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    if ttft is None:  # no text streamed (rare) -> ttft == total
        ttft = total
    return ttft, total


async def _run_convo(engine: str, uri: str) -> dict:
    """Run start() + the scripted turns; collect per-turn TTFT/total."""
    machine = _build_machine(engine, uri)

    t0 = time.perf_counter()
    await machine.start()
    start_lat = time.perf_counter() - t0

    ttfts: list[float] = []
    totals: list[float] = []
    wall0 = time.perf_counter()
    for text in SCRIPT:
        ttft, total = await _stream_turn_metrics(machine, text)
        ttfts.append(round(ttft, 4))
        totals.append(round(total, 4))
    wall = time.perf_counter() - wall0

    try:
        completed = bool(machine.is_complete)
    except Exception:
        completed = None  # engine has no completion signal exposed

    return {
        "start_lat": round(start_lat, 4),
        "turn_ttfts": ttfts,
        "turn_totals": totals,
        "wall": round(wall, 2),
        "completed": completed,
        "turns_used": len(SCRIPT),
    }


async def main(mode: str, n: int) -> None:
    results: dict[str, list[dict]] = {}
    providers = PROVIDERS if mode == "full" else {"openai": PROVIDERS["openai"]}
    backends = BACKENDS if mode == "full" else BACKENDS
    n = 1 if mode == "smoke" else n

    print(
        f"MODE={mode}  N={n}  backends={backends}  "
        f"providers={list(providers)}  engines={ENGINES}  "
        f"script_turns={len(SCRIPT)}"
    )
    print("=" * 92)

    for prov_name, model in providers.items():
        for engine in ENGINES:
            for backend in backends:
                uri = f"{backend}/{model}"
                cell = f"{prov_name}|{engine}|{backend}"
                results[cell] = []
                for i in range(1, n + 1):
                    try:
                        m = await _run_convo(engine, uri)
                        results[cell].append(m)
                        print(
                            f"  ok  {cell:<34} run {i}/{n}  "
                            f"start={m['start_lat']:.2f}s  "
                            f"ttft_p50~{sorted(m['turn_ttfts'])[len(m['turn_ttfts']) // 2]:.2f}s  "
                            f"wall={m['wall']:.1f}s  completed={m['completed']}"
                        )
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        print(f"  ERR {cell:<34} run {i}: {msg[:160]}")

    out = ROOT / (
        f"bench_backend_ab_n{n}.json"
        if mode == "full"
        else f"bench_backend_ab_{mode}.json"
    )
    out.write_text(json.dumps(results, indent=2))
    print("=" * 92)
    print(f"wrote {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(main(mode, n))
