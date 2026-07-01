# SuperDialog Benchmark Report - Kairali

- **dataset:** kairali (12 scenarios)
- **playbook:** health_wellness_booking_flow.yaml
- **mode:** With SuperDialog (framework)
- **models under test (SUT):** gpt-4o-mini, claude-haiku-4-5
- **judges:** each RAGAS metric scored by both claude-haiku (judge:haiku) and gpt-4o-mini (judge:gpt)
- **scores:** whole-integer percent. cost: system-under-test tokens only (judge cost excluded)

## Results

| Metric                            | gpt-4o-mini / judge:haiku | gpt-4o-mini / judge:gpt | claude-haiku / judge:haiku | claude-haiku / judge:gpt |
| --------------------------------- | ------------------------- | ----------------------- | -------------------------- | ------------------------ |
| Completion rate (det.)            | 33%                       | 33%                     | 33%                        | 33%                      |
| Data capture (det.)               | 84%                       | 84%                     | 90%                        | 90%                      |
| Smoothness (det.)                 | 100%                      | 100%                    | 100%                       | 100%                     |
| Repairs (det., lower=better)      | 11%                       | 11%                     | 54%                        | 54%                      |
| Turn relevance (RAGAS)            | 33%                       | 58%                     | 67%                        | 67%                      |
| Goal accuracy (RAGAS)             | 33%                       | 0%                      | 17%                        | 0%                       |
| Topic adherence (RAGAS)           | 54%                       | 33%                     | 65%                        | 49%                      |
| Conversation completeness (RAGAS) | 0%                        | 17%                     | 17%                        | 33%                      |
| Answer correctness (RAGAS)        | n/a                       | n/a                     | n/a                        | n/a                      |
| Coherence (RAGAS)                 | 42%                       | 67%                     | 75%                        | 67%                      |
| Answer relevancy (RAGAS)          | n/a                       | n/a                     | n/a                        | n/a                      |
| Cost / run USD (SUT)              | $0.0111                   | $0.0111                 | $0.2759                    | $0.2759                  |

Deterministic metrics and cost do not depend on the judge, so they repeat across a model's two judge columns.

## Notes

- `n/a` = metric not scored this run (deferred). `answer_correctness` and `answer_relevancy` are embedding-based; the embeddings model was not wired for this run, so they were skipped. All other RAGAS metrics and all 4 deterministic metrics scored normally.
- `topic_adherence` failed on one scenario during the claude-haiku run (a ragas-internal numpy quirk); its score is the mean over the remaining scenarios.
- Cost counts only the benchmarked model's own tokens, not the RAGAS judge LLM calls.
- Read of results: both models land ~33% completion on this run - the deterministic completion proxy is strict (clean-close / expected-outcome detection); RAGAS turn-relevance and coherence are the stronger signals so far. claude-haiku captures more lead data (90% vs 84%) but shows far more repairs (54% vs 11%) and costs ~25x more per run.

> **Note:** this is the earlier **2-judge** run (haiku + gpt-4o-mini judges). Superseded by the single-judge design (`anthropic/claude-sonnet-4-6`). Re-run `python examples/run_benchmark.py kairali --both --sd-only --out` for the current sonnet-judged numbers.