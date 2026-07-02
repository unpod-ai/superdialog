# `superdialog.eval` — playbook-vs-vanilla A/B evaluation

A/B-evaluate **playbook-mode** superdialog (the Director + Talker state machine)
against a **vanilla** LLM handed the *same* playbook as one flat system prompt.
Everything is scored from the conversation **transcript** — never from engine
internals — so both modes are judged by an identical, framework-agnostic rubric.

Design doc: `docs/plans/2026-07-01-playbook-vs-vanilla-eval-design.md`.

## Two phases

1. **`gen-dataset`** (offline) builds `<playbook>.evalcases.yaml`: personas
   (auto-generated or supplied) plus auto-injected probes. Deterministic input
   you can commit and review.
2. **`run`** drives both modes over that dataset, scores every transcript, and
   writes `report.json` (full) + `report.md` (headline table + per-case
   drilldown).

## Two seams

- **Transport** — `ConversationEndpoint` (`start` / `turn` / `reset`). Ships with
  in-process (`InProcessPlaybook`, `InProcessVanilla`), `HttpEndpoint`, and
  `OpenAICompatEndpoint`. The runner only sees `str` in / `str` out, so a mode is
  just a factory that returns an endpoint.
- **Framework** — `Metric` (`async score(sample) -> MetricResult`). Custom
  LLM-judge metrics plus a `RagasMetric` adapter, assembled by name in
  `registry.build_suite`. RAGAS is **optional** (the `ragas` extra) and currently
  needs a compatible install; the custom judges cover every headline metric
  without it, so RAGAS names raise a clear error only when explicitly requested.

## Headline metrics

| metric | kind | reads |
|---|---|---|
| `task_success` | LLM judge (0–1) | full transcript vs the goal |
| `slot_accuracy` | LLM judge (0–1) | transcript vs `ground_truth_slots` |
| `guardrail` | LLM judge (hard gate) | each guardrail probe reply |
| `efficiency` | pure code | user turns + assistant latency p50/p95 |

`guardrail` is a **hard gate**: any complied-with attack zeroes that case's
composite (`scoring.case_composite`) and counts toward
`guardrail_violation_rate`, regardless of the other scores. Composite weights
live in `scoring.DEFAULT_WEIGHTS`.

## CLI

```bash
# 1. build the dataset once (offline)
superdialog eval gen-dataset --playbook spa.yaml --n-probes 8

# 2. A/B both modes and write a report
superdialog eval run \
    --playbook spa.yaml --dataset spa.evalcases.yaml \
    --modes vanilla,playbook \
    --agent-model openai/gpt-4.1-mini \
    --judge-model openai/gpt-4.1-mini \
    --metrics task_success,slot_accuracy,guardrail,efficiency \
    --out ./eval-out

# 3. expose the playbook as an OpenAI-compatible endpoint for any benchmark
superdialog eval serve --playbook spa.yaml --port 8000
```

## Extending

- **Add a metric**: implement the `Metric` protocol (`custom.py` for a style
  reference), then register its name in `metrics/registry.py`.
- **Add a transport**: implement `ConversationEndpoint` and hand a factory to
  `run_ab(endpoint_factories=...)`.
- **Benchmark integration**: point any OpenAI-speaking eval harness at
  `superdialog eval serve` and grade the playbook like any other model.
