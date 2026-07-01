# SuperDialog Benchmark Report

- **dataset:** kairali (12 scenarios)
- **playbook:** health_wellness_booking_flow.yaml
- **mode:** Raw vs With-SuperDialog
- **models:** gpt-4o-mini, gpt-4.1-mini, anthropic/claude-haiku-4-5-20251001
- **metrics:** 4 deterministic + 7 RAGAS (RAGAS on)
- **judge:** anthropic/claude-sonnet-4-6 (single fixed judge)
- **scores:** whole-integer percent
- **cost:** system-under-test tokens only (judge/eval LLM cost excluded)

## Results

**scenarios:** 12  |  **playbook:** health_wellness_booking_flow.yaml  |  **judge:** anthropic/claude-sonnet-4-6

| Metric                     | Raw LLM (gpt-4o-mini) | With SuperDialog (gpt-4o-mini) | Raw LLM (gpt-4.1-mini) | With SuperDialog (gpt-4.1-mini) | Raw LLM (anthropic/claude-haiku-4-5-20251001) | With SuperDialog (anthropic/claude-haiku-4-5-20251001) |
| -------------------------- | --------------------- | ------------------------------ | ---------------------- | ------------------------------- | --------------------------------------------- | ------------------------------------------------------ |
| Completion rate (det.)     | 42%                   | 33%                            | 33%                    | 58%                             | 33%                                           | 50%                                                    |
| Data capture (det.)        | 92%                   | 85%                            | 95%                    | 93%                             | 92%                                           | 92%                                                    |
| Smoothness (det.)          | 100%                  | 100%                           | 100%                   | 100%                            | 100%                                          | 100%                                                   |
| Repairs (det.)             | 11%                   | 8%                             | 4%                     | 4%                              | 34%                                           | 12%                                                    |
| Turn relevance (RAGAS)     | 67%                   | 67%                            | 33%                    | 58%                             | 92%                                           | 75%                                                    |
| Goal accuracy (RAGAS)      | 25%                   | 25%                            | 17%                    | 25%                             | 17%                                           | 33%                                                    |
| Topic adherence (RAGAS)    | 52%                   | 50%                            | 67%                    | 57%                             | 63%                                           | 58%                                                    |
| Conversation comp. (RAGAS) | 0%                    | 25%                            | 8%                     | 17%                             | 17%                                           | 25%                                                    |
| Answer correct. (RAGAS)    | —                     | —                              | —                      | —                               | —                                             | —                                                      |
| Coherence (RAGAS)          | 67%                   | 58%                            | 55%                    | 50%                             | 75%                                           | 75%                                                    |
| Answer relevancy (RAGAS)   | —                     | —                              | —                      | —                               | —                                             | —                                                      |
| Cost / run USD (SUT)       | $0.0771               | $0.0121                        | $0.1507                | $0.0337                         | $1.1227                                       | $0.2685                                                |

## Notes

- `—` = metric not scored this run.
- Deferred metrics (need an embeddings model): answer_correctness, answer_relevancy. The other RAGAS metrics + all 4 deterministic metrics scored normally.
- Cost counts only the benchmarked model's tokens, not the RAGAS judge LLM calls.
