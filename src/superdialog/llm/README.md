# LLM provider layer

One `LLMProvider` protocol (`complete(messages, tools) -> CompletionResult` and
`stream(...) -> AsyncIterator[StreamChunk]`) with three interchangeable
backends, selected by `resolve_llm(uri)`. Every engine path (flow adapters,
playbook Director/Talker, edge-evaluation) obtains its provider through
`resolve_llm`, so backend choice and resilience apply uniformly.

## Backends

| Backend   | Class             | When                                                            |
|-----------|-------------------|----------------------------------------------------------------|
| `anyllm`  | `AnyLlmProvider`  | **Default.** any-llm-sdk delegates to each provider's official SDK (native tool-calling + multi-provider). |
| `litellm` | `LitellmProvider` | Broadest coverage; fallback when the `anyllm` extra is absent.  |
| `openai`  | `OpenAIProvider`  | Naked `openai` SDK; the LiveKit inference-gateway path; zero extra deps. |

`any-llm-sdk` is an optional extra (`pip install 'superdialog[anyllm]'`, Py ≥
3.11). When it is not installed, `anyllm` URIs fall back to LiteLLM.

## Selecting a backend

Resolution order (first match wins):

1. **URI scheme prefix** — `anyllm/…`, `litellm/…`, `oai/…`.
2. **`SUPERDIALOG_LLM_BACKEND`** env var — `anyllm` | `litellm` | `openai`.
3. **Default** — `anyllm`.

```
openai/gpt-4.1-mini                 # default backend (any-llm)
anyllm/anthropic/claude-haiku-4-5   # AnyLlmProvider
litellm/anthropic/claude-haiku-4-5  # LitellmProvider
oai/gpt-4.1-mini                    # OpenAIProvider (naked openai SDK)
custom/<name>/<model>               # LiteLLM via a registered base_url + key
vllm/<model>@<host>                 # LiteLLM (hosted_vllm via api_base)
ollama/<model>@<host>               # LiteLLM (ollama via api_base)
```

`custom/…` and `…@host` forms are LiteLLM features and always route through
LiteLLM regardless of the selected backend. Custom providers are registered with
`register_llm_provider(name, base_url, api_key, api_style="openai")`; `api_style`
is recorded on the provider config (currently openai-compatible base URLs).

## Resilience (timeout / retry / hedge)

`resolve_llm` wraps the chosen backend in a `ResilientProvider`, so a configurable
per-request timeout, bounded retry-with-backoff, and an optional cross-provider
hedge apply once for every backend and both engines. The raw backend is available
as `.inner`. The wrapper is transparent on the happy path.

| Env var                          | Default | Meaning                                            |
|----------------------------------|---------|----------------------------------------------------|
| `SUPERDIALOG_LLM_TIMEOUT_S`      | `60`    | Per-request timeout seconds. `0`/`none`/`off` disables. |
| `SUPERDIALOG_LLM_MAX_RETRIES`    | `2`     | Extra attempts after the first, on timeout/transient errors. |
| `SUPERDIALOG_LLM_BACKOFF_BASE_S` | `0.5`   | Exponential backoff base.                          |
| `SUPERDIALOG_LLM_BACKOFF_MAX_S`  | `8`     | Backoff cap.                                        |
| `SUPERDIALOG_LLM_HEDGE`          | `false` | Enable a hedge/fallback request.                    |
| `SUPERDIALOG_LLM_HEDGE_MODEL`    | —       | Alternate model URI for the hedge leg.              |
| `SUPERDIALOG_LLM_HEDGE_DELAY_S`  | `2`     | Delay before the hedge leg starts.                  |

The default 60s timeout is a safety net against an infinite hang. **Latency-
sensitive voice** deployments should tune the timeout *down* (e.g. 3–5s) and/or
enable a hedge so a single stalled provider call cannot dominate the p95 tail.
Retries fire only on timeouts and transient failures (transient HTTP status,
rate-limit / connection / service-unavailable markers); non-transient errors
fail fast. On exhaustion a controlled `LLMResilienceError` is raised. Streaming
retries only before the first token (re-issuing a partially-spoken turn is
unsafe); a mid-stream stall is surfaced.
