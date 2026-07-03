# SuperDialog - Playbook-vs-Vanilla Eval

**Status:** Canonical
**Parent:** [README.md](README.md)

---

## 1. What it answers

`superdialog.eval` answers one question: **does running a playbook on the
Playbook engine beat handing the same playbook to a raw LLM as a flat system
prompt?** It A/B-evaluates two modes over one dataset and scores both from the
conversation **transcript only** — never from engine internals — so the
playbook and the vanilla baseline are judged by an identical rubric.

This is distinct from the in-framework quality loop in
[04-playbook-guide.md §9](04-playbook-guide.md) (`run_eval`, persona evals,
`superdialog optimize`). That loop asks "is this playbook good enough, and how
do I improve its prose?"; this framework asks "is the playbook *machinery*
earning its keep over a plain prompt?" — the audit you run before adopting the
engine, and the harness you point external benchmarks at.

It also sits beside the team's `superdialog benchmark` command
(`src/superdialog/benchmark/`), a separate RAGAS + deterministic harness that
scores raw-LLM vs with-SuperDialog over a dataset. The two share the spirit but
not the code, and — importantly — pin **different RAGAS versions** (see §6).

---

## 2. The two modes

| Mode | What runs | Endpoint |
|---|---|---|
| `playbook` | The full Director + Talker checkpoint runtime loading the playbook | `InProcessPlaybook` |
| `vanilla` | One raw LLM handed the playbook file as a single flat system prompt | `InProcessVanilla` |

The runner only ever sees `str` in / `str` out, so a "mode" is nothing more
than a factory that returns a `ConversationEndpoint`. That is the first of two
pluggable seams.

---

## 3. Two seams

**Transport — `ConversationEndpoint`** (`start` / `turn` / `reset`). Ships with
the two in-process endpoints above plus `HttpEndpoint` and
`OpenAICompatEndpoint`, so the same dataset can grade an in-process playbook, a
remote SuperDialog server, or any OpenAI-speaking model without touching the
runner.

**Framework — `Metric`** (`async score(sample) -> MetricResult`). Metrics are
assembled by name in `metrics/registry.build_suite`. Custom LLM-judge metrics
cover every headline number; an optional `RagasMetric` adapter maps
RAGAS metrics in when the `ragas` extra is installed. Because the framework is
a seam, swapping RAGAS for another eval library is a registry entry, not a
rewrite.

---

## 4. Headline metrics

| Metric | Kind | Reads |
|---|---|---|
| `task_success` | LLM judge (0–1) | full transcript vs the case goal |
| `slot_accuracy` | LLM judge (0–1) | transcript vs `ground_truth_slots` |
| `guardrail` | LLM judge (hard gate) | each guardrail-probe reply |
| `efficiency` | pure code | user turns + assistant latency p50/p95 |

`guardrail` is a **hard gate**: any complied-with attack zeroes that case's
composite score (`scoring.case_composite`) and counts toward
`guardrail_violation_rate`, regardless of the other metrics. Composite weights
live in `scoring.DEFAULT_WEIGHTS`.

---

## 5. Running it

Two phases — an offline dataset build you can commit and review, then the A/B
run that scores it.

```bash
# 1. build <playbook>.evalcases.yaml once (personas auto-generated + probes injected)
superdialog eval gen-dataset --playbook spa.yaml --n-probes 8

# 2. A/B both modes and write report.json (full) + report.md (headline + drilldown)
superdialog eval run \
    --playbook spa.yaml --dataset spa.evalcases.yaml \
    --modes vanilla,playbook \
    --agent-model openai/gpt-4.1-mini \
    --judge-model openai/gpt-4.1-mini \
    --metrics task_success,slot_accuracy,guardrail,efficiency \
    --out ./eval-out

# 3. (optional) expose the playbook as an OpenAI-compatible endpoint for any external benchmark
superdialog eval serve --playbook spa.yaml --port 8000

# or one shot: build the dataset if missing, then A/B every --models entry
superdialog eval bench --playbook spa.yaml \
    --models openai/gpt-4o-mini --max-turns 20 --out ./eval-out
```

`eval run` also takes `--director-model` / `--talker-model` (per-role LLMs for
playbook mode), `--user-model` (the persona/user-simulator LLM), and
`--repeats`. The dataset format mirrors the RAGAS single-turn/multi-turn shape:
each case carries a persona, `ground_truth_slots`, and a list of probes.

`eval bench` wraps both phases: it reuses `<playbook>.evalcases.yaml` when
present (`--regen` rebuilds it, `--personas` seeds it), accepts every `run`
flag, adds `--max-turns` to override each persona's turn budget, and writes one
report directory per `--models` entry plus a combined `report.md`.

The legacy session audit still lives under this group as
`superdialog eval flow --flow kyc.json --traversal session.json`.

---

## 6. RAGAS: two pins, mutually exclusive

The repo carries **two** RAGAS-based harnesses that pin **incompatible** RAGAS
lines:

| Extra | Package | RAGAS | API |
|---|---|---|---|
| `ragas` | `src/superdialog/eval/` (this framework) | `ragas==0.4.3` + `langchain-community==0.4.1` | 0.4 |
| `benchmark` | `src/superdialog/benchmark/` (team harness) | `ragas>=0.2,<0.3` + `langchain-community>=0.3,<0.4` | 0.2 |

They cannot co-install. `pyproject.toml` declares them conflicting:

```toml
[tool.uv]
conflicts = [[{ extra = "benchmark" }, { extra = "ragas" }]]
```

Without that declaration `uv` cannot resolve the project at all
("`superdialog[benchmark]` and `superdialog[ragas]` are incompatible").
**Install one extra per environment** — `uv sync --extra ragas ...` for this
framework. RAGAS is optional either way: the custom LLM judges produce every
headline metric with no RAGAS installed, so `ragas`-named metrics raise a clear
error only when explicitly requested and unavailable.

> The RAGAS 0.4.3 pin exists because 0.4.3 hard-imports
> `langchain_community.chat_models.vertexai`, a shim deleted in
> langchain-community 0.4.2 — so the last release carrying it (0.4.1) is pinned.
> Consolidating the two harnesses onto a single RAGAS line is a known follow-up.

---

## 7. Extending

- **Add a metric** — implement the `Metric` protocol (`metrics/custom.py` is the
  style reference) and register its name in `metrics/registry.py`.
- **Add a transport** — implement `ConversationEndpoint` and hand a factory to
  `run_ab(endpoint_factories=...)`.
- **Grade with an external benchmark** — point any OpenAI-speaking eval harness
  at `superdialog eval serve` and score the playbook like any other model.

Module reference (terse): `src/superdialog/eval/README.md`. Design doc:
`docs/plans/2026-07-01-playbook-vs-vanilla-eval-design.md`.
