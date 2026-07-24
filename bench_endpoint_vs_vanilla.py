#!/usr/bin/env python3
"""Vanilla LLM vs Deploy-as-Endpoint: A/B benchmark over the eval harness.

Same playbook, two transports -- this is NOT a 3rd mode, it's the existing
`vanilla` vs `playbook` A/B (see eval/README.md), just with `playbook` served
over HTTP instead of in-process:

    vanilla  -> InProcessVanilla     playbook text as one flat system prompt,
                                      no framework, direct LLM call.
    endpoint -> OpenAICompatEndpoint the SAME playbook's Director+Talker
                                      engine, but running for real behind the
                                      playground's "Deploy as Endpoint" pool
                                      (https://inference.unpod.ai).

Dataset: auto-generated personas+probes from the SAME playbook (mirrors
`superdialog eval gen-dataset`), cached next to --playbook unless --dataset
points at an existing one.

Metrics: every metric this repo defines is printed at startup regardless of
selection (see print_metric_catalog / metrics/registry.py), so a report never
silently uses an undocumented subset:
    custom : task_success, slot_accuracy, guardrail, efficiency, token_cost
    ragas  : faithfulness, answer_correctness, topic_adherence, goal_accuracy
             (needs `uv sync --extra ragas`; pass --ragas to include them)

Usage:
    python bench_endpoint_vs_vanilla.py \
        --playbook path/to/deployed_playbook.yaml \
        --endpoint-model brightsmile_dental_customer_support \
        --endpoint-api-key sk_... \
        --vanilla-model openai/gpt-4.1-mini \
        --judge-model anthropic/claude-sonnet-4-6

Env: OPENAI_API_KEY / ANTHROPIC_API_KEY for vanilla+judge+persona-sim calls;
the endpoint key can also come from UNPOD_ENDPOINT_API_KEY.

Known simplifications (ponytail):
- No streaming -> no TTFT, only whole-turn latency (already captured by the
  `efficiency` metric's p50/p95). Add a streaming ConversationEndpoint if
  first-token latency specifically becomes the question.
- `token_cost` will show "endpoint reported no token usage" for the endpoint
  mode -- the deployed pool is a black box, no litellm response object to
  price. Vanilla gets a real $/token number; endpoint doesn't. Wire
  `platform_client.BalanceClient` (wallet delta) if a real billed-cost
  comparison is needed later.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

os.environ.setdefault("LITELLM_LOG", "ERROR")

from superdialog.eval.dataset.generate import build_dataset  # noqa: E402
from superdialog.eval.dataset.models import EvalCase, EvalDataset  # noqa: E402
from superdialog.eval.endpoints.in_process import InProcessVanilla  # noqa: E402
from superdialog.eval.endpoints.openai_client import OpenAICompatEndpoint  # noqa: E402
from superdialog.eval.metrics import registry as metrics_registry  # noqa: E402
from superdialog.eval.metrics.registry import build_suite  # noqa: E402
from superdialog.eval.report import write_report  # noqa: E402
from superdialog.eval.runner import run_ab  # noqa: E402
from superdialog.llm.resolver import resolve_llm  # noqa: E402
from superdialog.playbook.eval.personas import (  # noqa: E402
    generate_personas,
    load_personas,
)
from superdialog.playbook.models import Playbook  # noqa: E402
from superdialog.playbook.providers import provider_adapters  # noqa: E402


def _completer(uri: str):
    """Str-returning `.complete` completer (Director adapter) for a model URI."""
    director, _talker = provider_adapters(resolve_llm(uri))
    return director


def print_metric_catalog() -> None:
    """Every benchmark metric defined in superdialog.eval.metrics.registry,
    regardless of which ones this run actually selects."""
    print("=" * 78)
    print("METRICS DEFINED IN superdialog.eval.metrics.registry")
    print("=" * 78)
    print("custom (always available, no ragas install needed):")
    for name in metrics_registry._CUSTOM:
        print(f"  - {name}")
    print("ragas (needs `uv sync --extra ragas`; pass --ragas to include):")
    for name, (cls_name, applies_to) in metrics_registry._RAGAS.items():
        print(f"  - {name}  (ragas.metrics.{cls_name}, scores {applies_to})")
    print("=" * 78 + "\n")


async def _load_or_gen_dataset(args: argparse.Namespace) -> EvalDataset:
    if args.dataset:
        return EvalDataset.load(args.dataset)
    dataset_path = f"{os.path.splitext(args.playbook)[0]}.evalcases.yaml"
    if os.path.exists(dataset_path) and not args.regen:
        print(f"[dataset] reusing {dataset_path} (pass --regen to rebuild)")
        return EvalDataset.load(dataset_path)
    pb = Playbook.load(args.playbook)
    gen = _completer(args.gen_model)
    personas = (
        load_personas(args.personas)
        if args.personas
        else await generate_personas(pb, gen)
    )
    ds = await build_dataset(pb, personas, gen, n_probes=args.n_probes)
    ds.save(dataset_path)
    print(f"[dataset] wrote {dataset_path} ({len(ds.cases)} cases)")
    return ds


def _endpoint_client(args: argparse.Namespace) -> httpx.AsyncClient:
    api_key = args.endpoint_api_key or os.getenv("UNPOD_ENDPOINT_API_KEY", "")
    assert api_key, "pass --endpoint-api-key or set UNPOD_ENDPOINT_API_KEY"
    # NOTE base_url has NO "/v1" suffix: OpenAICompatEndpoint posts to the
    # literal path "/v1/chat/completions" itself (it's not the OpenAI SDK,
    # which appends "/chat/completions" to a base_url that already ends in
    # "/v1" -- see test_endpoint.py for that convention). Mixing the two up
    # silently 404s against "/v1/v1/chat/completions".
    return httpx.AsyncClient(
        base_url=args.endpoint_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )


async def main(args: argparse.Namespace) -> int:
    print_metric_catalog()

    vanilla_source = args.vanilla_prompt or args.playbook
    with open(vanilla_source, encoding="utf-8") as fh:
        playbook_text = fh.read()

    dataset = await _load_or_gen_dataset(args)
    if args.max_turns:
        dataset = dataset.model_copy(
            update={
                "cases": [
                    c.model_copy(
                        update={
                            "persona": c.persona.model_copy(
                                update={"max_turns": args.max_turns}
                            )
                        }
                    )
                    for c in dataset.cases
                ]
            }
        )

    def _vanilla(case: EvalCase) -> InProcessVanilla:
        return InProcessVanilla(playbook_text, resolve_llm(args.vanilla_model))

    def _endpoint(case: EvalCase) -> OpenAICompatEndpoint:
        client = _endpoint_client(args)
        session_id = f"bench-{case.id}-{uuid.uuid4().hex[:8]}"
        return OpenAICompatEndpoint(
            args.endpoint_url, args.endpoint_model, session_id, client=client
        )

    metric_names = [
        "task_success",
        "slot_accuracy",
        "guardrail",
        "efficiency",
        "token_cost",
    ]
    if args.ragas:
        metric_names += [
            "faithfulness",
            "answer_correctness",
            "topic_adherence",
            "goal_accuracy",
        ]
    print(f"[metrics] scoring this run with: {', '.join(metric_names)}\n")

    suite = build_suite(metric_names, _completer(args.judge_model), args.judge_model)
    user_llm = _completer(args.user_model or args.vanilla_model)

    result = await run_ab(
        dataset,
        modes=["vanilla", "endpoint"],
        endpoint_factories={"vanilla": _vanilla, "endpoint": _endpoint},
        suite=suite,
        user_llm=user_llm,
        metric_names=metric_names,
        repeats=args.repeats,
    )

    json_path, md_path = write_report(result, args.out)
    print(f"\nwrote {json_path} and {md_path}\n")
    print(Path(md_path).read_text(encoding="utf-8"))
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Vanilla LLM vs Deploy-as-Endpoint A/B benchmark"
    )
    ap.add_argument(
        "--playbook",
        required=True,
        help="playbook YAML (source of truth for the vanilla prompt + dataset gen)",
    )
    ap.add_argument(
        "--dataset", default=None, help="existing .evalcases.yaml (else auto-gen)"
    )
    ap.add_argument(
        "--vanilla-prompt",
        default=None,
        help="raw .txt system prompt for the vanilla mode (else --playbook's YAML "
        "text verbatim). --playbook is still required for dataset gen either way.",
    )
    ap.add_argument(
        "--regen", action="store_true", help="regenerate the dataset even if cached"
    )
    ap.add_argument(
        "--personas", default=None, help="personas YAML/JSON (else auto-generate)"
    )
    ap.add_argument("--n-probes", type=int, default=8)
    ap.add_argument(
        "--max-turns", type=int, default=None, help="override persona turn budget"
    )

    ap.add_argument(
        "--endpoint-url",
        default="https://inference.unpod.ai",
        help="bare host, no /v1 suffix (see _endpoint_client note)",
    )
    ap.add_argument(
        "--endpoint-model",
        required=True,
        help="deployed playbook slug -- the OpenAI `model` field",
    )
    ap.add_argument(
        "--endpoint-api-key", default=None, help="sk_... else UNPOD_ENDPOINT_API_KEY"
    )

    ap.add_argument("--vanilla-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4-6")
    ap.add_argument(
        "--user-model", default=None, help="persona-simulator model (else vanilla)"
    )
    ap.add_argument(
        "--gen-model",
        default="openai/gpt-4.1-mini",
        help="model used to author the dataset/personas",
    )

    ap.add_argument(
        "--ragas",
        action="store_true",
        help="also score the 4 wired RAGAS metrics (needs `uv sync --extra ragas`)",
    )
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="./eval-out/endpoint-vs-vanilla")
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
