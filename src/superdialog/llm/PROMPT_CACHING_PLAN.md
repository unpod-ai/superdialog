The findings are confirmed against the actual code: `render.py:95` interleaves `datetime.now()` right after `pb.persona` (the prefix-stability problem), and `LitellmProvider.complete/.stream` are the shared chokepoint where `messages` and `tools` arrive pre-assembled. Note the finding cited a `criteria.py` for the LLMAdapter routing path — the actual files are `llm_adapter.py` + `text_adapter.py` (no `criteria.py`); I'll flag that discrepancy in the plan rather than invent the path. Now writing the plan.

# Implementation Plan: Provider-Side Prompt Caching for superdialog

> ## 0. Corrections applied (adversarial verify pass)
> This plan was synthesized from multi-agent research, then fact-checked. Three corrections were folded in:
> 1. **Injection seam is `ResilientProvider`, not `LitellmProvider`.** The default backend is `AnyLlmProvider` (`resolver.py:29,100`); `LitellmProvider` is only the fallback and `OpenAIProvider` is a third path. The cache-marker helper must run at the common wrapper `ResilientProvider` (`resolver.py:103-118`) or in all three backends — see §3.2 / §4.
> 2. **`machine/criteria.py` exists** — the earlier draft's "missing file" warning was wrong and has been removed (§4 Engine D).
> 3. **`_extract_usage` does NOT yet carry cache fields** — it returns only prompt/completion tokens and must be *extended* (§7).
> The litellm syntax, provider matrix, and prompt-hygiene thesis were verified sound.

## 1. Goal & Expected Impact

**Goal.** Enable provider-side prompt caching across **all** LiteLLM-supported providers so the stable request prefix (system/persona prompt + tool definitions) is re-read from cache on every turn instead of re-billed and re-prefilled at full cost. The work funnels through one seam (`LitellmProvider`) and one prompt-assembly refactor; it does **not** introduce any new SDK or transport.

**What gets cached.** The byte-stable prefix that repeats every turn: the persona/system block and the (per-node-stable) tool array. The growing transcript and per-turn dynamic context stay *after* the cache breakpoint and are processed at full price.

**Expected impact.**
- **TTFT (time-to-first-token).** On Anthropic/Bedrock/Vertex/Gemini, a cache read skips prefill of the cached prefix — the dominant TTFT component for large personas. This is the highest-value win for a real-time voice engine, where every turn re-sends the same large persona.
- **Input cost.** Cache reads cost ~0.1× base input on Anthropic (90% off the cached span); OpenAI cached input is 50% off; Deepseek cache hits are dramatically cheaper than misses (~90%+). The savings scale with the *size of the stable prefix* relative to the per-turn delta.
- **Helps big personas most.** A 200-token persona below the per-model minimum caches nothing (silent no-op). A multi-hundred-to-thousand-token persona + tool schemas clears the threshold on Sonnet/Opus-class models and yields the largest per-turn read discount. The bigger and more stable the persona, the bigger the win — so the engines with the heaviest personas (playbook Talker, flow toolcall routing) benefit most.

**Honest caveat.** The findings are unambiguous that *as the prompt is assembled today, a provider-level cache marker yields ~0% hit rate* — because volatile content (current date/time, slots, transcript-derived data) is fused into the same system string as the persona, ahead of where any breakpoint would land. **The caching win is gated on the prompt-assembly refactor in §4.** Marker plumbing without prefix hygiene is wasted effort.

---

## 2. Provider Support Matrix

Two mechanisms, per the findings: **automatic** (provider caches server-side, no markers) and **explicit `cache_control`** (you mark the cacheable span; LiteLLM translates to each provider's native API). LiteLLM silently skips when a marker lands on a non-supporting provider or the prefix is below the per-model minimum — *no error is raised*.

| Provider (LiteLLM prefix) | Mechanism | Marker needed? | Notes / minimum |
|---|---|---|---|
| OpenAI (`openai/`) | Automatic | No | Auto on prefixes ≥ 1,024 tokens, 128-token increments. Optional `prompt_cache_key`, `prompt_cache_retention`. Reported via `prompt_tokens_details.cached_tokens`. |
| Deepseek (`deepseek/`) | Automatic | No | OpenAI-style auto. Distinct usage fields: `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` (NOT `cached_tokens`). Strict full-prefix match. |
| xAI (`xai/`) | Automatic | No | Listed as supported; OpenAI-style. |
| Anthropic (`anthropic/`) | Explicit `cache_control` | **Yes** | `{"type":"ephemeral"}`. Max 4 breakpoints. Returns `cache_creation_input_tokens` + `cache_read_input_tokens`. Per-model minimums vary (see below). |
| AWS Bedrock (`bedrock/`, `bedrock/converse`, `bedrock/invoke/`) | Explicit `cache_control` | **Yes** | LiteLLM translates `cache_control` → native `cachePoint`. Effectively the Anthropic-on-Bedrock models. |
| Google Gemini (`gemini/`) | Explicit `cache_control` (context caching) | **Yes** | Routed to Google `cachedContents`; ~1,024+ token threshold. Streaming historically did not return cache usage — verify on pinned version (see §9). |
| Vertex AI (`vertex_ai/`, `vertex_ai_beta/`) | Explicit `cache_control` (context caching) | **Yes** | Same as Gemini; supports optional `ttl` (e.g. `"7200s"`). With multiple `cache_control` blocks, the **first** TTL wins. |

**Per-model minimum cacheable prefix (Anthropic-family — caches silently skip below this; verify at integration time, the fetched table was internally inconsistent):** ~1,024 tokens for Claude 3.x / larger Sonnet-Opus; ~2,048–4,096 for Haiku-class and certain newer models. Dependable rule for this engine: a system prompt + tool defs of a few hundred tokens or more typically clears the threshold for Sonnet/Opus-class.

> **Model-ID note (grounded against the canonical Anthropic catalog).** The Anthropic research findings contained some hallucinated/distorted model names from the summarizer (e.g. odd minimums). For our own examples and any default config, use real, current model IDs only — e.g. `anthropic/claude-opus-4-8`, `anthropic/claude-sonnet-4-6`, `anthropic/claude-haiku-4-5`. Do not hardcode model-specific minimums; rely on the silent-skip behavior + telemetry (§7) to confirm a hit, which is provider-version-proof.

**Auto-inject coverage.** LiteLLM's `cache_control_injection_points` is documented as covering Anthropic, Bedrock, Vertex AI, Google AI Studio (Gemini), plus Azure AI, OpenRouter, Databricks, DashScope, MiniMax, Z.ai. OpenAI/Deepseek/xAI don't need it (automatic).

**Sources:** LiteLLM Prompt Caching (`docs.litellm.ai/docs/completion/prompt_caching`), auto-inject tutorial (`/docs/tutorials/prompt_caching`), Vertex (`/docs/providers/vertex`), Bedrock (`/docs/providers/bedrock`), streaming usage (`/docs/completion/usage`); OpenAI prompt-caching guide + API pricing; Deepseek KV-cache + pricing docs; LiteLLM source via DeepWiki (`deepwiki.com/BerriAI/litellm/8.2-prompt-caching`).

---

## 3. Design: Unified `cache_control` Approach

### 3.1 The one invariant everything follows from

Caching is a **prefix match**. The cache key is the exact bytes up to each breakpoint, rendered in order `tools → system → messages`. A single byte change at position N invalidates every breakpoint at position ≥ N. Therefore the design is **90% prompt hygiene, 10% marker plumbing**: guarantee `[stable system/persona][stable serialized tools][dynamic history/user/timestamps]` ordering for every turn, on every provider; then mark the boundary.

### 3.2 Single shared message-prep helper

Add one helper, applied once inside `LitellmProvider` immediately before each `litellm.acompletion` call:

```
mark_cache_prefix(messages, tools, model, *, ttl=None) -> (messages, tools)
```

Behavior:
1. **Provider guard first.** If `not litellm.utils.supports_prompt_caching(model)` → return `messages, tools` **untouched** (preserves the legacy byte-for-byte path for OpenAI/Deepseek/xAI/unsupported providers per the "Direct LiteLLM — byte-for-byte legacy path" constraint). OpenAI-family caches automatically; adding `cache_control` is at best a no-op and, per findings, list-of-parts content shape is the *only* thing that isn't literally a no-op — so we skip them entirely.
2. **Normalize the leading system message to structured-content form.** Caching requires `content: [{"type":"text", ...}]`, not a bare string. The helper converts the first `role:"system"` message's string content to a single text block **only when** caching is supported and the upstream assembler has marked the block as stable (see §3.3).
3. **Mark the last stable system block** with `{"type":"ephemeral"}` (or `{"type":"ephemeral","ttl":ttl}`).
4. **Mark the last tool definition** (when `tools` is present) on its `function` object — caches the whole tools prefix as one segment.
5. Respect the **4-breakpoint max** and the `tools → system → messages` ordering (1h-TTL block, if used, must precede 5m blocks).

The helper is invoked once at the **common provider seam** so it covers all four engine paths *and* all three backends. ⚠️ **Correction (verify pass):** the seam is **`ResilientProvider`** (`resolver.py:103-118`), the wrapper `resolve_llm` returns around *every* backend — **not** `LitellmProvider`. The default backend is `AnyLlmProvider` (`resolver.py:29,100`); `LitellmProvider` is only the fallback and `OpenAIProvider` (`oai/`) is a third path, so a helper placed only in `LitellmProvider` is a no-op in the default config. Apply `mark_cache_prefix` in `ResilientProvider.complete`/`.stream` before delegating to `inner` (or in all three backends). Each backend must forward the marker: LiteLLM translates `cache_control` natively; for `AnyLlmProvider` confirm the official-SDK path forwards `cache_control` content blocks (the Anthropic SDK accepts them natively); OpenAI/`OpenAIProvider` auto-cache and need no marker. It must be **fail-open**: any exception inside `mark_cache_prefix` logs and returns the inputs untouched — a caching helper must never fail a turn (mirrors the graceful-degradation house style in `livekit_services.py`, §5).

### 3.3 Why the helper alone is insufficient — the assembler contract

A provider-level "mark the last system block" rule is only correct if that block is genuinely stable. Today it isn't (date/slots/transcript fused into the system string — §4). So the helper relies on an **upstream contract**: each engine's assembler must emit the system message as **two parts** — a leading **stable** part (persona / flow system prompt / fixed schema preamble) and a trailing **dynamic** part (date, slots, step, reference data, summary, KB) — and tag the boundary so the helper marks the *stable* part, not the fused whole.

Concretely, the assembler emits structured system content already split:
```
{"role":"system","content":[
   {"type":"text","text": STABLE_PERSONA},          # ← helper marks THIS
   {"type":"text","text": DYNAMIC_CONTEXT}           # date/slots/step/transcript-derived
]}
```
The helper marks the **last block whose text is stable**. We mark by **position contract** (assembler guarantees index 0 is stable) rather than guessing — the assembler is the source of truth for what is stable, the helper is the source of truth for the marker syntax.

### 3.4 How OpenAI/Deepseek/xAI benefit automatically

They get the *same prefix-hygiene refactor* but **no markers** (the guard skips them). Because caching is automatic and prefix-based, keeping `[stable persona][stable serialized tools][dynamic]` first-and-byte-stable is exactly what their auto-cache needs. Two extra requirements for them specifically (from findings):
- **Deterministic tool serialization.** Tools must be byte-identical turn-to-turn: same set, same order, same JSON key order. Serialize once per node and reuse; sort keys.
- **No per-turn data before the breakpoint.** Same constraint as Anthropic — the refactor satisfies both at once.

Optionally pass OpenAI `prompt_cache_key` to bias routing for higher hit rates on long-lived sessions (extends the existing convention, §5).

---

## 4. Exact Injection Points in superdialog

All four engine paths funnel through the **common provider seam**. ⚠️ **Correction (verify pass):** that seam is **`ResilientProvider`** (`resolver.py:103-118`) — the wrapper `resolve_llm` returns around *every* backend — **not** `LitellmProvider` (only the fallback; the default is `AnyLlmProvider`, `resolver.py:29,100`). Invoke `mark_cache_prefix` in `ResilientProvider.complete`/`.stream` (which delegate to `inner`), or add it to all three backends (`litellm_provider.py:19-47`/`49-103`, `anyllm_provider.py`, `openai_provider.py`). Either way it runs once before the backend call. But each upstream assembler must first satisfy the §3.3 contract — listed here with the volatile content that must move *after* the boundary.

> **Verified against source.** `render.py:95` literally interleaves `datetime.now(...)` into `parts` right after `pb.persona.strip()`, confirming the findings: as assembled today there is *no stable byte-prefix beyond the persona* and the date sits inside the system block. `LitellmProvider` receives `messages`/`tools` already fused.

### Engine A — Playbook Talker (streaming, NO tools)
- **Assembler:** `render.py::_system_block` (**L91-151**); list built in `render_view` (**L170**: `[{"role":"system","content":system}, *chat]`).
- **Stable:** `pb.persona` (L95) — the only fixed part.
- **Volatile to move after boundary:** `Current date and time` (L94-95), steering notes (L101-102/113-115), current-step guidance (L103-105), still-needed/never-say (L106-112), known slots (L116-118), reference data (L119-123), summary (L124-125), KB block (L131-150).
- **Refactor:** split `_system_block` to return persona as block 0 and the volatile remainder as block 1; `render_view` emits structured system content. No tools on this path.

### Engine B — Playbook Director (structured verdict, NO tools)
- **Assembler:** `director.py::_verdict_prompt` (**L57-114**), returns `[system, user]` at **L111-114**; consumed at `Director.evaluate` (**L278-282**).
- **Stable:** the fixed "You supervise…" JSON-schema instruction preamble (**L93-103**).
- **Volatile to move after boundary:** step id/goal (L104), slot lines (L105-107), known slots (L106), tool results (L107), advance rules/interrupts (L108-109), transcript[-12:] (L113).
- **Refactor:** emit preamble as stable block 0, the per-turn remainder as block 1.

### Engine C — Flow toolcall adapter (tools EVERY turn — the highest-value path)
`src/superdialog/machine/adapters/toolcall_adapter.py`
- **Routing call (hot, tool-bearing):** `ToolCallAdapter.evaluate_criteria` (**L696-956**). System built by `_build_instructions` (**L529-559**), prefixed with routing rules (**L790-829**) and `[CURRENT DATA]` (**L860-862**); messages at **L864-866**; sent at **L870-872** with `tools=tools, tool_choice="required"`.
- **Tools:** built by `_descriptors_to_openai_tools` (**L314-331**, called L730) + constant `__stay_on_node__` hatch (**L768-788**). **Stable within a node**, change at node transitions — so cacheable across consecutive same-node turns; expect a cold write at each node transition.
- **Stable:** `flow_system_prompt` (L533-534) and the routing-rule preamble (L800-828) — but the preamble is currently *prepended* (L829), so it must be reordered to lead.
- **Volatile to move after boundary:** time line (L552-559), node instruction/transitions (L548-549), `[CURRENT DATA]` slot dump (L860-862), `history[-10:]` (L865).
- **Reply call (no tools):** `_generate_via_llm` (**L650-694**, sent L673) — same persona/system split, no tools.

### Engine D — Flow LLMAdapter / CriteriaJudge (default flow adapter, NO tools)
`src/superdialog/machine/adapters/llm_adapter.py`
- **Routing call:** `LLMAdapter.evaluate_criteria` (**L165-231**).
- **Message assembly lives in `src/superdialog/machine/criteria.py`** (verified to exist; the earlier draft wrongly flagged it as missing). `CriteriaJudge.build_evaluation_messages` (**L145-293**, final list **L291-292**) and `CriteriaJudge._ask` → `self._llm.complete` (**L138-143**); `llm_adapter.py` imports it from `superdialog.machine.criteria` (**L27**) and calls `build_evaluation_messages` (**L187**). No grep/action item needed.
- **Stable:** the `system_prompt` arg + trailing fixed JSON-schema block.
- **Volatile to move after boundary:** `Today's date` (currently the **first** line of the system string — the worst case, volatile text at the very front), node id/instruction/edges, userdata dump, reentry/turns notes, then `history`.
- **Reply call:** `LLMAdapter.generate_reply` (**L116-163**, sent L146).

**Why the provider seam still needs the assembler refactor.** By the time messages reach *any* backend the stable and dynamic text are already fused into one string. A provider-level "cache the last system block" rule would cache a block that changes every turn (zero hit rate, plus a wasted write premium each turn). The chokepoint applies the *marker*; the assemblers must first guarantee the *stable prefix*.

---

## 5. Alignment with prompt_manager.py / livekit_services.py Conventions

The superdialog production path (`litellm.acompletion` behind `LLMProvider`) is a different layer from the LiveKit-plugin path in `super/`. We carry over the **conventions**, not the call sites.

- **Stable/dynamic split already exists in production.** `prompt_manager.py::build_static_layers` (`:359-417`, docstring `:365` "Cached by config hash") vs per-call `build_customer_context` (`:420-514`) vs per-turn `build_turn_state_message` (`:517-536`). `PromptLayers` (`:544-577`) makes the boundary explicit (`static: str` "Cacheable" `:553`). **Mirror this typed split** in each superdialog assembler — it is the same stable-prefix / dynamic-suffix line we need.
- **Caveat the production code shares our exact bug.** `_create_assistant_prompt` appends a per-call time line right after `static` (`prompt_manager.py:894-899`) and string-joins everything (`:902-905`). Our refactor must keep `static` as a distinct first segment, not string-join — the same fix prompt_manager would need.
- **Existing provider-side cache convention to extend.** The only cache plumbing in production today is OpenAI `prompt_cache_key` (`livekit_services.py:2385-2391`); surfaced via `LLMConfig.prompt_cache_key` property reading from `extra` (`service_common.py:780-781`), populated in `parse_llm_config` (`:1015`). **Extend this convention** to all providers (don't reinvent a parallel one).
- **Config knobs live in `extra`, not dataclass fields.** `LLMConfig` is `frozen=True` (`service_common.py:751`); all optional knobs are `@property` over `extra` (`reasoning_effort` `:768`, `parallel_tool_calls` `:772`, `top_p` `:776`). **Add caching flags the same way** (§6).
- **Param-injection pattern.** Plugin path builds a `llm_kwargs` dict and conditionally adds keys (`:2380-2391`); inference path accumulates `extra_kwargs` with an explicit None-guard (`:1649-1691`). Our helper follows the same "add only when applicable" shape and the None-guard discipline.
- **Graceful degradation is house style.** Every provider branch falls back on error (`:2359-2376`, `:2519`, `:2582-2585`); TTS wraps failover (`:1316-1346`). `mark_cache_prefix` must **fail open** the same way — never hard-fail a turn.
- **Cache-segment identity convention.** `_static_layers_cache` keys on `hashlib.sha256(json.dumps({...}, sort_keys=True)).hexdigest()[:16]` (`prompt_manager.py:374-416`). Reuse this exact style if we add an in-process cache-key for tool serialization or a `prompt_cache_key` derivation; `sort_keys=True` is also the deterministic-serialization the auto-cache providers need.
- **Telemetry sink already reserved.** `super/core/voice/workflows/post_call/workflow.py:1903` hardcodes `"llm_cached_tokens": 0`. **Wire real `cached_tokens` into this existing key** rather than inventing a metric (§7). Also extend the existing prompt-size INFO line (`prompt_manager.py:1001-1008`) to log cache fields.

---

## 6. Config & Flagging

Add caching knobs following the `extra`-backed `@property` convention (do **not** add frozen-dataclass fields). In superdialog, mirror this on whatever config object backs `LitellmProvider.default_opts` and the per-agent config dict.

| Flag | Type / default | Purpose |
|---|---|---|
| `enable_prompt_caching` | bool, default **off** initially → **on** after rollout | Master switch. Env override `SUPERDIALOG_PROMPT_CACHING` using the house `os.getenv(...) or config(...)` precedence (`livekit_services.py:1174-1179`, `:1326`). |
| `prompt_cache_ttl` | `"5m"` (default) / `"1h"` | Maps to `cache_control` `ttl`. |
| `prompt_cache_key` | str (optional) | Extend existing OpenAI convention; per-session value from a `room:session`-style key (`livekit_services.py:828-843`). |

**Per-provider enable.** No hard per-provider branching is required — `mark_cache_prefix` gates on `litellm.utils.supports_prompt_caching(model)` and LiteLLM silently skips unsupported/under-minimum. The master flag is the kill-switch; the provider guard is automatic.

**Min-token guard.** Do **not** hardcode per-model minimums (the findings' table was inconsistent and they shift by provider version). Rely on: (a) the provider's silent below-threshold no-op, and (b) telemetry (§7) confirming an actual hit. Optionally log a one-time warning when `estimate_tokens(stable_block)` is suspiciously small (e.g. < 512) so authors know a tiny persona won't cache — using the existing `estimate_tokens` (`prompt_manager.py:1001-1008`).

**TTL choice (from findings' recommendation).** Persona/tools are long-lived and low-churn → favor **`1h`** TTL for them (2× write premium, but re-read all hour). History, if ever cached, → default **`5m`**. For steady voice traffic the 5m default is refreshed free on each hit and stays warm; only bursty/idle-gap traffic needs 1h or pre-warming. **Mixing TTLs in one request:** the `1h` block must appear before the `5m` block in prefix order (findings §3); given `tools → system`, mark tools `1h` and any history `5m`, never the reverse.

---

## 7. Telemetry

Surface cache activity per turn and tie it into the existing Observer/TracingProvider metadata path and the reserved `llm_cached_tokens` sink. ⚠️ **Correction (verify pass):** `_extract_usage` (`anyllm_provider.py:33-49`) currently returns only `{prompt_tokens, completion_tokens}` and **drops** all cache fields — they do **not** flow through today. The work is to **extend** `_extract_usage` (and both backends' usage capture) to pull the provider-specific cache fields below, then surface them downstream. (Call sites `litellm_provider.py:38`/`:73` are correct.)

**Provider-aware field reads (must branch — findings are explicit):**
- **Anthropic/Bedrock/Vertex/Gemini:** `usage.cache_creation_input_tokens` (written, billed 1.25×/2×) and `usage.cache_read_input_tokens` (read, billed 0.1×).
- **OpenAI-family/xAI:** `usage.prompt_tokens_details.cached_tokens`.
- **Deepseek:** `usage.prompt_cache_hit_tokens` / `usage.prompt_cache_miss_tokens` — **not** `cached_tokens`.

**Normalize into the `metadata` dict** that `CompletionResult`/`StreamChunk` already carry (`provider.py:11-16` `metadata`; `litellm_provider.py:42-46` builds it). Emit a unified shape, e.g.:
```
cache_read_tokens, cache_write_tokens, cached_tokens (provider-agnostic), cache_hit (bool)
```
Compute `cache_hit = (cache_read > 0) or (cached_tokens > 0) or (prompt_cache_hit_tokens > 0)`. **Verification rule from findings:** if *both* `cache_creation_input_tokens` and `cache_read_input_tokens` are 0 (and `cached_tokens`/`prompt_cache_hit_tokens` are 0), nothing cached — surface this as a warning so a silent invalidator (e.g. a stray timestamp before the breakpoint) is caught early.

**Wire-up points:**
- Feed `cached_tokens` into the existing `super/core/voice/workflows/post_call/workflow.py:1903` `llm_cached_tokens` key (replace the hardcoded `0`).
- Extend the prompt-creation INFO line (`prompt_manager.py:1001-1008`) and the LLM-creation log (`:1683-1686`) to echo per-turn `cache_read` / `cache_write` in the same shape.
- Add to the Observer/TracingProvider span metadata so per-turn TTFT vs cache-hit can be correlated on dashboards. (`observability/observer.py` currently has an unresolved merge conflict — `UU` in git status — resolve that before adding fields.)

**Streaming caveat (load-bearing for this engine).** Cache fields arrive on the **final/usage chunk**, not per token. The existing `LitellmProvider.stream` already captures usage from the trailing chunk into `usage_meta` (`litellm_provider.py:64-103`) and deliberately does **not** pass `stream_options` (comment at `:56-59` explains litellm v1.88 quirk: with `stream_options` set, litellm swallows the usage chunk). **Confirm cache fields survive that path on the pinned litellm version** (see §9 known bug) — the current code reads usage from "any chunk that carries it" (`:71-74`), which should capture cache fields if litellm populates them there.

---

## 8. Testing Strategy

### Hermetic (no network) — assert marker placement, mock the provider
1. **Marker on the stable block only.** Given a refactored assembler output, call `mark_cache_prefix(messages, tools, "anthropic/claude-opus-4-8")` and assert `cache_control == {"type":"ephemeral"}` is on **system content block 0** (the stable persona) and **not** on block 1 (dynamic). Negative case: a system message whose only block contains a timestamp must **not** be marked (or the assembler must have split it first).
2. **Marker on the last tool only.** With a tools array, assert the marker lands on the final tool's `function` object and on no earlier tool.
3. **Provider guard.** `mark_cache_prefix(..., "openai/gpt-4o")` (or whatever `supports_prompt_caching` returns False/auto for) returns `messages`/`tools` **byte-identical** to input (legacy path intact). Use a monkeypatched `litellm.utils.supports_prompt_caching`.
4. **Breakpoint budget & ordering.** Never exceed 4 markers; if both tools(`1h`) and a history block(`5m`) are marked, assert `1h` precedes `5m` in prefix order.
5. **Fail-open.** Inject an exception inside the helper → returns inputs untouched, logs, turn proceeds.
6. **Prefix-stability regression (the bug that motivated this).** Render the same playbook state twice and assert the marked stable block is **byte-identical** across renders; assert the dynamic block differs (date moves). This is the test that catches a future author re-fusing a timestamp into the persona.
7. **Per-engine assembler contract.** For Talker (`render.py`), Director, toolcall, and the LLMAdapter/CriteriaJudge path: assert the emitted system content is a 2-block list with a stable block 0. (For the LLMAdapter path, **first resolve the `criteria.py` discrepancy** from §4.)

### Live (network, gated) — assert real cache reads
8. **Two-turn hit.** Same persona/tools, second turn with one more user message. Assert turn-2 `cache_read_input_tokens > 0` (Anthropic/Bedrock/Vertex) or `prompt_tokens_details.cached_tokens > 0` (OpenAI) or `prompt_cache_hit_tokens > 0` (Deepseek). Turn-1 should show `cache_creation_input_tokens > 0` (a write).
9. **Streaming hit.** Same as (8) via `.stream`, reading cache fields off the terminal chunk's usage — guards against the streaming-usage capture bug (§9).
10. **Below-threshold no-op.** Tiny persona → both cache fields 0, no error (confirms silent skip).

**Test registration (per CLAUDE.md).** Register new test modules in the superdialog test runner. The root repo uses `scripts/run_tests.sh` module registry — superdialog has its own `tests/` and is special-cased for plugin autoload; add a `playbook-caching` (or similar) module to `module_test_paths()` and `ALL_MODULES`, and to `scripts/run_tests.sh` per the global instructions. Live tests should be marked (e.g. `@pytest.mark.integration`) and skipped without provider keys.

---

## 9. Rollout Sequence & Risks

### Sequence
1. **Resolve the `observer.py` merge conflict** (`UU` in git status) before touching telemetry.
2. **Land the prompt-assembly refactor first, caching OFF.** Split each assembler into stable/dynamic blocks (§4) behind no behavior change for non-caching paths — structured system content normalizes identically via LiteLLM. Ship and verify *no regression* on existing turns (byte-equivalence of the joined prompt where caching is off). This is the gating change; do it as its own reviewable step.
3. **Land `mark_cache_prefix` + telemetry, flag default OFF.** Hermetic tests green.
4. **Enable on one provider/agent in shadow.** Turn `enable_prompt_caching` on for a single low-risk agent on a caching-supported provider (Anthropic or a Bedrock/Vertex Claude). Watch `cache_read_input_tokens` / `cached_tokens` and TTFT on the dashboard for a few days.
5. **Expand by provider.** OpenAI/Deepseek/xAI get the win from the refactor automatically (no marker), so they're low-risk to enable broadly once the prefix-hygiene tests pass. Anthropic-family widen after the shadow confirms hits land.
6. **Tune TTL** (1h for persona/tools) and optionally add **pre-warming** for bursty traffic (a `max_tokens`-minimal warmup request before traffic; re-warm ~every 5m or ~60m for 1h TTL).

### Risks
- **Prefix-stability pitfalls (the dominant risk).** Any per-turn datum (timestamp, call-id, session-id, random greeting, RAG block, slot value) rendered *into* the stable block silently drops hit rate to 0 with no error. The hermetic byte-stability test (#6) and the telemetry "both cache fields 0" warning are the guardrails. The findings list `render.py:94-95` and CriteriaJudge's leading `Today's date` as the live offenders.
- **Tool churn.** Tools must be byte-identical: same set, **same order, same JSON key order**. Non-deterministic dict iteration or per-turn-regenerated descriptions break the tools+system+messages prefix. Serialize tools deterministically once per node (`sort_keys`), reuse. In the toolcall engine, tools change at node transitions — that's an expected cold write, not a bug.
- **Director vs Talker cache independently.** Different prompts → separate prefixes; fine as long as each is individually stable. Don't try to share.
- **Streaming usage capture.** Known LiteLLM bug (GitHub #7790) where Anthropic streaming cache fields can be missing from the callback/logging `Usage` even when the response object has them; Gemini streaming historically returned no cache usage (#10667). The current `LitellmProvider.stream` reads usage off the response chunk (not the callback path), which is the safer side of this bug — but **validate on the pinned litellm version** with live test #9 before trusting streaming telemetry. Do not switch to `stream_options`-based usage (the code comment at `:56-59` documents why it breaks).
- **Breakpoint limits.** Max 4 per request; 20-block lookback. For long calls, if we ever cache history, a single turn adding >20 blocks (tool_use/tool_result pairs) can make the next breakpoint miss — place an intermediate breakpoint, but for the persona+tools-only scope here this is not yet a concern.
- **Concurrency / cold start.** A cache entry is readable only after the first response *begins*. Parallel requests fired before the first completes each miss and each pay a write. Pre-warm or rely on the natural first turn before fan-out.
- **Content-shape compatibility.** The only thing that isn't literally a no-op for non-caching providers is turning system content into list-of-parts form. The guard avoids this for OpenAI/Deepseek/xAI (we skip them entirely), so they keep bare-string system content. Smoke-test that any *caching* provider tolerates the structured form (LiteLLM normalizes it, but confirm on the pinned version).
- **Model-ID hygiene.** Use only real, current Anthropic IDs in config/examples (`claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`) — the research summarizer hallucinated some names; do not propagate them.

---

## 10. Task Checklist

**Pre-work**
- [ ] Resolve the `src/superdialog/observability/observer.py` merge conflict (`UU`).
- [ ] Grep for `build_evaluation_messages` / `CriteriaJudge` and confirm the real file:line for Engine D (findings cite a non-existent `criteria.py`).
- [ ] Confirm pinned litellm version exposes `litellm.utils.supports_prompt_caching` and the cache `usage` fields on streamed terminal chunks.

**Prompt-assembly refactor (caching OFF — gating step)**
- [ ] Talker: split `render.py::_system_block` (`:91-151`) into stable (persona, `:95`) + dynamic blocks; update `render_view` (`:170`) to emit structured 2-block system content.
- [ ] Director: split `director.py::_verdict_prompt` (`:57-114`) preamble (`:93-103`) from per-turn remainder.
- [ ] Flow toolcall: in `toolcall_adapter.py`, reorder so `flow_system_prompt` (`:533`) + routing preamble (`:800-828`) lead; push time line (`:552-559`) + `[CURRENT DATA]` (`:860-862`) after the boundary; serialize tools (`_descriptors_to_openai_tools` `:314-331` + `__stay_on_node__` `:768-788`) deterministically once per node (`sort_keys`).
- [ ] Flow LLMAdapter/CriteriaJudge: split the resolved assembler so the fixed `system_prompt`/schema preamble leads and `Today's date` + per-turn data trail.
- [ ] Verify no behavior regression with caching off (joined-prompt byte-equivalence where applicable).

**Helper + telemetry**
- [ ] Add `mark_cache_prefix(messages, tools, model, *, ttl=None)` (provider guard via `supports_prompt_caching`, structured-content normalize, mark last stable system block + last tool, 4-breakpoint/ordering rules, **fail-open**).
- [ ] Invoke it once in `LitellmProvider.complete` (before `:27`) and `.stream` (before `:61`).
- [ ] Extend `_extract_usage` (`anyllm_provider.py`, used at `litellm_provider.py:38`/`:73`) to capture provider-specific cache fields and normalize into `metadata`.
- [ ] Wire normalized `cached_tokens` into `super/core/voice/workflows/post_call/workflow.py:1903` (`llm_cached_tokens`).
- [ ] Add cache fields to Observer/TracingProvider span metadata and the prompt-size INFO line (`prompt_manager.py:1001-1008`).

**Config & flags (extra-backed properties, frozen-dataclass-safe)**
- [ ] `enable_prompt_caching` (default off; env `SUPERDIALOG_PROMPT_CACHING` with `os.getenv(...) or config(...)` precedence).
- [ ] `prompt_cache_ttl` (`5m`/`1h`; 1h for persona/tools).
- [ ] Extend `prompt_cache_key` convention to feed OpenAI routing.

**Tests (register a `playbook-caching` module in the superdialog runner)**
- [ ] Hermetic: marker-on-stable-block-only; marker-on-last-tool-only; provider-guard byte-identical passthrough; 4-breakpoint/ordering; fail-open; byte-stability regression across two renders; per-engine 2-block contract.
- [ ] Live (gated/marked): two-turn `cache_read>0`; streaming hit off terminal chunk; below-threshold silent no-op.

**Rollout**
- [ ] Ship refactor → ship helper (flag off) → enable one agent on a caching provider in shadow → watch cache fields + TTFT → widen by provider → tune TTL / add pre-warming for bursty traffic.