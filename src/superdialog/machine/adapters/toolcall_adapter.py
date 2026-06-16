"""ToolCallAdapter -- simulates SimpleFlowAgent's tool-call routing.

Uses OpenAI function_calling to pick edges instead of CriteriaJudge.
Same decision path as production (SimpleFlowAgent) without LiveKit.

Usage::

    adapter = ToolCallAdapter(model_id="gpt-4o-mini", system_prompt=prompt)
    machine = await DialogStateMachine.from_flow(flow, adapter)
    result = await machine.process_turn("I want to book an appointment")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

@dataclass
class LLMCallData:
    node_id: str
    model: str
    call_type: Literal["routing", "generate_reply"]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    prompt_messages: list[dict]
    response_json: dict
    edge_id: str | None


# ── Language extraction patterns ─────────────────────────────────────────────
# Handles all common formats so generate_reply() avoids a second LLM call:
#   [EN] / [HI]   — uppercase block markers (classic format)
#   [en] / [hi]   — lowercase
#   [En] / [Hi]   — mixed case
#   {EN} / {HI} / {en} / {hi}  — curly-brace variants
#   EN line: '...' / HI line: '...'  — line format used in new BOB flows

_LANG_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?P<lang>[A-Za-z]{2})\s+line\s*:\s*['\"](.+)['\"][ \t]*$",
    re.MULTILINE,
)

_LANG_BLOCK_CI_RE = re.compile(
    r"(?:\[|\{)([A-Za-z]{2})(?:\]|\})\s*(.+?)(?=\n?(?:\[|\{)[A-Za-z]{2}(?:\]|\})|\Z)",
    re.DOTALL,
)

# Matches "--- ENGLISH SCRIPT ... ---" / "--- HINGLISH SCRIPT ... ---" sections.
# Captures (1) section label (ENGLISH|HINGLISH|HINDI) and (2) text until next ---
_SCRIPT_SECTION_RE = re.compile(
    r"---\s+(ENGLISH|HINDI|HINGLISH)\s+SCRIPT[^-\n]*---\s*\n(.*?)(?=\n\s*---|$)",
    re.DOTALL | re.IGNORECASE,
)
# Maps 2-char language code → accepted section labels (priority order)
_SCRIPT_LANG_MAP: dict[str, list[str]] = {
    "EN": ["ENGLISH"],
    "HI": ["HINGLISH", "HINDI"],
}

from jinja2 import BaseLoader, ChainableUndefined, Environment

from superdialog.machine.composer import _LANG_MARKER_RE
from superdialog.machine.composer import extract_speech_text as _extract_speech_text
from superdialog.machine.composer import get_time_context as _get_time_context
from superdialog.machine.composer import process_text as _process_text
from superdialog.machine.composer import resolve_language as _resolve_lang
from superdialog.machine.models import CriteriaResult, ToolDescriptor
from superdialog.llm.provider import LLMProvider
from superdialog.llm.resolver import resolve_llm

if TYPE_CHECKING:
    from superdialog.flow.models import CustomAction, FlowNode

logger = logging.getLogger(__name__)

# Deterministic router-condition grammar. Only the safe, data-decidable shapes
# below are evaluated without the LLM; anything else (prose / judged conditions)
# is left to the LLM router. Conservative by design — a miss just falls back.
_GUARD_CMP_RE = re.compile(r"^([\w][\w.]*)\s*(==|!=)\s*(.+)$")
# Trailing prose after an em/en-dash, " - ", or comma is advisory, not logic.
_GUARD_PROSE_SPLIT_RE = re.compile(r"\s+[—–]\s+|\s+-\s+|,")


def _livekit_inference_url() -> str:
    """Convert LIVEKIT_URL (wss://) to HTTPS base URL for inference API."""
    url = os.environ.get("LIVEKIT_INFERENCE_URL") or os.environ.get("LIVEKIT_URL", "")
    return url.replace("wss://", "https://").replace("ws://", "http://")


def _make_openai_client():
    """Return an AsyncOpenAI client.

    LLM_BACKEND=livekit  → true LiveKit inference (JWT from LIVEKIT_API_KEY+SECRET,
                            base_url=LIVEKIT_URL). Mirrors how inference.LLM works
                            internally in livekit_services.py.
    LLM_BACKEND=openai   → OPENAI_API_KEY + default OpenAI endpoint (default)

    Auto-selects livekit when LIVEKIT_API_KEY + LIVEKIT_API_SECRET are set and
    LLM_BACKEND is not explicitly configured.
    """
    from openai import AsyncOpenAI

    lk_api_key = os.environ.get("LIVEKIT_API_KEY") or os.environ.get("LIVEKIT_INFERENCE_API_KEY")
    lk_api_secret = os.environ.get("LIVEKIT_API_SECRET") or os.environ.get("LIVEKIT_INFERENCE_API_SECRET")

    backend = os.environ.get(
        "LLM_BACKEND",
        "livekit" if (lk_api_key and lk_api_secret) else "openai",
    )

    if backend == "livekit" and lk_api_key and lk_api_secret:
        try:
            from livekit.agents.inference.llm import create_access_token, get_default_inference_url
            token = create_access_token(lk_api_key, lk_api_secret)
            # Use the same gateway URL inference.LLM uses — agent-gateway.livekit.cloud/v1
            # NOT the project LIVEKIT_URL (that's for rooms/WebSocket, not inference HTTP)
            base_url = get_default_inference_url()
            logger.debug("[ToolCallAdapter] LLM via LiveKit inference gateway: %s", base_url)
            return AsyncOpenAI(api_key=token, base_url=base_url)
        except ImportError:
            logger.warning("[ToolCallAdapter] livekit-agents not installed — falling back to OpenAI")

    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _ensure_non_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Guarantee at least one non-system message.

    OpenAI tolerates system-only message lists; Anthropic (and other strict
    providers) reject them ("requires at least one non-system message"). At
    silent router nodes the routing state lives in the system prompt, so append
    a minimal user turn when no user/assistant message is present.
    """
    if not any(m.get("role") != "system" for m in messages):
        messages.append({"role": "user", "content": "(continue)"})
    return messages


def _is_zero_speech_step(rendered_instruction: str) -> bool:
    """Return True if the rendered instruction mandates zero speech at the current step.

    Conditions (both must hold):
    1. The instruction contains the explicit zero-speech directive.
    2. All ``current_*=VALUE`` Jinja2 template variables have non-empty values
       (indicates STEP 4 / auto-proceed conditions are met — all required fields set).

    This is a deterministic code-level gate that never relies on the LLM to
    voluntarily output an empty string (which is unreliable).
    """
    _ZERO_SPEECH_DIRECTIVE = "ZERO speech — no words before the tool call"
    if _ZERO_SPEECH_DIRECTIVE not in rendered_instruction:
        return False
    # Find every current_*=VALUE pattern; if any value is empty, STEP 4 does not apply.
    found_any = False
    for m in re.finditer(r"current_\w+=(\S*)", rendered_instruction):
        found_any = True
        if not m.group(1):  # empty value → required field missing
            return False
    return found_any


_SPEECH_STOP_RE = re.compile(
    r"<wait for response>|\nROUTING\s*:|\nCapture only",
    re.IGNORECASE,
)


def _strip_routing_metadata(text: str) -> str:
    """Strip routing/capture instructions that follow the speech text.

    Nodes embed metadata (ROUTING:, Capture only, <wait for response>)
    after the speech text inside the same language block. These must not
    be spoken aloud by TTS.
    """
    m = _SPEECH_STOP_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    return text


def _extract_first_quoted_speech(text: str) -> str | None:
    """Extract the first 'speech' or "speech" string at the start of text.

    Universal extractor for flow instructions that embed LLM routing context
    after the speech text, e.g.:

        'Sure. क्या आप treatment के लिए देख रहे हैं?'

        Per [Non-Negotiable Rules]: do NOT convey therapy duration.

        ROUTING: ...

    The closing quote MUST be at end-of-line (followed by \\n or end-of-string).
    This handles apostrophes correctly: 'I'll arrange...' does NOT stop at the
    apostrophe in 'I'll' because that apostrophe is not at end-of-line.

    Returns the inner text WITHOUT outer quotes. Returns None if the text does
    not start with a quote character.
    """
    t = text.strip()
    if not t or t[0] not in ("'", '"'):
        return None
    q = t[0]
    i = 1
    while i < len(t):
        if t[i] == q and (i + 1 >= len(t) or t[i + 1] in ("\n", "\r")):
            inner = t[1:i].strip()
            return inner or None
        i += 1
    # Fallback: use last occurrence of the quote character
    last = t.rfind(q, 1)
    if last > 0:
        inner = t[1:last].strip()
        return inner or None
    return None


def _extract_for_language(instruction: str, lang: str) -> str | None:
    """Extract speech text for *lang* from instruction without calling an LLM.

    Tried in order:
      1. ``XX line: '...'``  — new BOB-style line format
      2. ``[XX]`` / ``{XX}`` block markers (case-insensitive)
      3. ``--- LANGUAGE SCRIPT --- ... ---`` sections (script-reader format)

    Returns the stripped text or None when no match.
    Routing/metadata after <wait for response> or ROUTING: is stripped.
    """
    if not instruction:
        return None
    target = lang.upper()[:2]  # "EN" or "HI"

    # 1. "XX line: '...'" format
    for m in _LANG_LINE_RE.finditer(instruction):
        if m.group("lang").upper() == target:
            text = _strip_routing_metadata(m.group(2).strip())
            # Unescape doubled single-quotes (flow JSON escapes ' as '' inside '...')
            text = text.replace("''", "'")
            if text:
                return text

    # 2. Block markers [XX] / {XX} (case-insensitive)
    for m in _LANG_BLOCK_CI_RE.finditer(instruction):
        if m.group(1).upper() == target:
            block = m.group(2).strip()
            # Try quoted-speech extraction first (universal: handles any text
            # after the closing quote that is not speech — Per [Non-Negotiable
            # Rules]:, ATTEMPT LIMITS:, PRONUNCIATION:, etc.).
            # Fall back to strip-based approach for unquoted content.
            text = _extract_first_quoted_speech(block) or _strip_routing_metadata(block)
            if text:
                return text

    # 3. "--- LANGUAGE SCRIPT --- ... --- END --- " sections (script-reader format)
    accepted = _SCRIPT_LANG_MAP.get(target, [target])
    for label in accepted:
        for m in _SCRIPT_SECTION_RE.finditer(instruction):
            if m.group(1).upper() == label.upper():
                text = m.group(2).strip()
                # Strip trailing "--- END ... ---" line if captured
                text = re.sub(r"\s*---\s+END\b.*", "", text, flags=re.DOTALL).strip()
                text = _strip_routing_metadata(text)
                if text:
                    return text

    return None


def _extract_agent_says(instruction: str) -> str | None:
    """Extract 'Agent says: <text>', stripping parenthetical stage directions."""
    if not instruction or not instruction.lstrip().startswith("Agent says:"):
        return None
    first_line = instruction.split("\n")[0]
    text = first_line[len("Agent says:"):].strip()
    # Strip (stage directions like this) — not meant for caller
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    return text if text else None



def _strip_provider_prefix(model_id: str) -> str:
    """Strip 'openai/' prefix so OpenAI client gets a bare model name."""
    if "/" in model_id:
        return model_id.split("/", 1)[1]
    return model_id


def _coerce_numeric_strings(d: dict, skip_fields: set[str]) -> dict:
    """Coerce string values that look like ints/floats — Jinja2 renders everything as str."""
    for k, v in d.items():
        if k in skip_fields:
            continue
        if isinstance(v, str):
            try:
                d[k] = int(v)
                continue
            except ValueError:
                pass
            try:
                d[k] = float(v)
            except ValueError:
                pass
        elif isinstance(v, dict):
            _coerce_numeric_strings(v, skip_fields)
    return d


def _descriptors_to_openai_tools(
    descriptors: list[ToolDescriptor],
) -> list[dict[str, Any]]:
    """Convert ToolDescriptors to OpenAI function-calling tool schemas."""
    tools: list[dict[str, Any]] = []
    for desc in descriptors:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": desc.id,
                    "description": desc.description,
                    "parameters": desc.input_schema
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


class ToolCallAdapter:
    """Runtime adapter that uses LLM tool-calling to route edges.

    Mirrors SimpleFlowAgent's instruction construction and presents
    edges as OpenAI function tools. The LLM picks a tool_call instead
    of returning structured JSON via CriteriaJudge.

    Also executes HTTP actions (on_enter/on_exit webhooks) identically
    to LLMAdapter so API-driven flows work correctly.
    """

    supports_criteria: bool = True
    speech_passthrough: bool = False

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        system_prompt: str = "",
        environment_variables: dict[str, str] | None = None,
    ) -> None:
        self._model_id = model_id
        self._system_prompt = system_prompt
        self.responses: list[str] = []
        self.session_ended: bool = False
        self._machine: Any = None  # Set by DialogMachine after from_flow()
        # HTTP action execution state (same as LLMAdapter)
        self._env_vars: dict[str, str] = dict(environment_variables or {})
        self._jinja_env = Environment(loader=BaseLoader(), undefined=ChainableUndefined)
        # Streaming: callback fired with pre-generated text when edge_id is known,
        # allowing TTS to start before criteria evaluation finishes.
        self._edge_id_callback: Any | None = None
        # Observability: callback fired after each LLM call with token/latency data.
        self._on_llm_complete: Callable[[LLMCallData], Awaitable[None]] | None = None
        self._pre_generated_response: str | None = None
        # GET-only URL cache: keyed by "METHOD:rendered_url".
        # Prevents duplicate GET calls with identical parameters (e.g. courses-by-city
        # firing twice when chain visits list_courses_in_city then ask_course_preference).
        # Cache key includes the full URL so a city change (different URL) bypasses cache.
        self._get_cache: dict[str, dict] = {}
        # Provider-agnostic LLM access (resolved lazily from model_id).
        self._provider: LLMProvider | None = None

    def _resolve_provider(self) -> LLMProvider:
        """Resolve (and cache) the LLM provider from the model URI.

        Routes through ``resolve_llm`` so the flow engine is provider-agnostic
        (OpenAI / any-llm / LiteLLM) instead of bound to ``AsyncOpenAI``. An
        injected provider (``set_provider``) takes precedence over self-resolution.
        """
        if self._provider is None:
            self._provider = resolve_llm(self._model_id)
        return self._provider

    def set_provider(self, provider: LLMProvider) -> None:
        """Inject the LLM provider (symmetric with ``LLMAdapter``).

        Lets the owning ``DialogMachine`` share its single resolved provider —
        and hot-swap it via ``set_llm`` — instead of the adapter independently
        re-resolving from ``model_id`` (which left a stale cached provider).
        """
        self._provider = provider

    # ------------------------------------------------------------------
    # Template helpers (mirrors LLMAdapter)
    # ------------------------------------------------------------------

    def _render(self, template_str: str, context: dict[str, Any]) -> str:
        try:
            return self._jinja_env.from_string(template_str).render(**context)
        except Exception as exc:
            logger.warning("ToolCallAdapter: template render failed for %r: %s", template_str[:60], exc)
            return template_str

    def _build_context(self, userdata: dict[str, Any]) -> dict[str, Any]:
        """Merge env_vars + userdata into a flat Jinja2 context."""
        ctx: dict[str, Any] = {}
        ctx.update(self._env_vars)
        ctx.update(userdata)
        return ctx

    # ------------------------------------------------------------------
    # Deterministic routing (data-decidable edges, zero LLM)
    # ------------------------------------------------------------------

    def _parse_clause(self, clause: str) -> str | None:
        """Translate ONE simple predicate to a Jinja boolean, or None.

        Bool/empty predicates are anchored at the start so trailing advisory
        prose ("... is false after retry") is ignored. ``==``/``!=`` require a
        full-string literal comparison (no trailing prose, no variable RHS)."""
        c = clause.strip()
        m = re.match(r"([\w.]+)\s+is\s+not\s+empty\b", c, re.IGNORECASE)
        if m:
            return f"({m.group(1)})"
        m = re.match(r"([\w.]+)\s+is\s+non-?empty\b", c, re.IGNORECASE)
        if m:
            return f"({m.group(1)})"
        m = re.match(r"([\w.]+)\s+is\s+empty\b", c, re.IGNORECASE)
        if m:
            return f"(not ({m.group(1)}))"
        m = re.match(r"([\w.]+)\s+is\s+(true|false)\b", c, re.IGNORECASE)
        if m:
            return f"({m.group(1)}) == {m.group(2).lower()}"
        m = _GUARD_CMP_RE.fullmatch(c)
        if m:
            rhs = m.group(3).strip()
            # Literal RHS only — never another variable, which could hide logic.
            if re.fullmatch(r"(['\"]).*\1|-?\d+(\.\d+)?|true|false", rhs, re.IGNORECASE):
                return f"({m.group(1)}) {m.group(2)} {rhs}"
        return None

    def _condition_to_jinja(self, condition: str) -> str | None:
        """Translate a safe edge condition into a Jinja boolean expression, or
        None for anything outside the grammar (prose / LLM-judged) so the
        caller falls back to the LLM router.

        Supports single predicates and ``and``/``or`` chains of them — every
        clause must parse, otherwise the whole condition is treated as prose
        (defer). This keeps prose routers from ever misfiring."""
        if not condition:
            return None
        # Drop trailing advisory prose ("... — route to retry attempt").
        head = _GUARD_PROSE_SPLIT_RE.split(condition.strip(), maxsplit=1)[0].strip()
        # Compound: try `or` first, then `and`. Mixed/unparsable clauses → defer.
        for sep, joiner in ((r"\s+\bor\b\s+", " or "), (r"\s+\band\b\s+", " and ")):
            parts = re.split(sep, head, flags=re.IGNORECASE)
            if len(parts) > 1:
                subs = [self._parse_clause(p) for p in parts]
                if all(subs):
                    return "(" + joiner.join(subs) + ")"
                return None
        return self._parse_clause(head)

    def _eval_jinja_bool(self, expr: str, ctx: dict[str, Any]) -> bool | None:
        """Render a Jinja boolean expression to True/False, or None on error."""
        try:
            out = (
                self._jinja_env.from_string(
                    "{% if " + expr + " %}1{% else %}0{% endif %}"
                )
                .render(**ctx)
                .strip()
            )
        except Exception as exc:
            logger.debug("guard eval failed for %r: %s", expr[:60], exc)
            return None
        if out == "1":
            return True
        if out == "0":
            return False
        return None

    def _eval_condition_bool(
        self, condition: str, ctx: dict[str, Any]
    ) -> bool | None:
        """Evaluate an edge condition to True/False, or None if indeterminate."""
        expr = self._condition_to_jinja(condition)
        if expr is None:
            return None
        return self._eval_jinja_bool(expr, ctx)

    def deterministic_route(
        self, node: "FlowNode", userdata: dict[str, Any]
    ) -> str | None:
        """Return the single edge id that data unambiguously satisfies, or
        None to defer to the LLM router.

        Fires ONLY when every outgoing edge is decidable from data AND exactly
        one is true — so an LLM/judged edge mixed in always defers. This makes
        data-settled transitions (e.g. ``success is true``) immune to the
        non-deterministic LLM routing that otherwise stalls silent routers.

        Each edge is decided by its explicit ``guard`` (a raw Jinja boolean) if
        present, else by parsing its prose ``condition``."""
        edges = getattr(node, "edges", None) or []
        if len(edges) < 2:
            return None
        ctx = self._build_context(userdata)
        results: dict[str, bool | None] = {}
        for e in edges:
            guard = getattr(e, "guard", None)
            results[e.id] = (
                self._eval_jinja_bool(guard, ctx)
                if guard
                else self._eval_condition_bool(e.condition, ctx)
            )
        if any(v is None for v in results.values()):
            return None  # some edge is prose/indeterminate — defer to LLM
        trues = [eid for eid, v in results.items() if v]
        return trues[0] if len(trues) == 1 else None

    # ------------------------------------------------------------------
    # Instruction builder
    # ------------------------------------------------------------------

    def _build_instructions(self, node: FlowNode, machine: Any) -> str:
        """Build instructions identical to SimpleFlowAgent.__init__."""
        lang = _resolve_lang(machine)

        flow_system_prompt = getattr(machine._flow, "system_prompt", "") or ""
        flow_system_prompt = _process_text(flow_system_prompt, machine, lang)

        if node.static_text:
            node_instruction = (
                "A message has been spoken to the user. "
                "Wait for their response, then use the appropriate tool "
                "to transition to the next step."
            )
        elif node.instruction:
            node_instruction = _process_text(node.instruction, machine, lang)
        else:
            node_instruction = ""

        if node.edges and not node.is_final:
            edge_lines = [f'  - "{e.id}": {e.condition}' for e in node.edges]
            node_instruction += "\n\nAvailable transitions:\n" + "\n".join(edge_lines)

        # Inject current date/time so LLM resolves partial dates (e.g. "28 May") correctly.
        time_ctx = _get_time_context()
        time_line = (
            f"[TODAY] {time_ctx['current_date']}  |  "
            f"IST {time_ctx['current_time_Asia_Kolkata']}"
        )

        base = f"{flow_system_prompt}\n\n{node_instruction}".strip() if flow_system_prompt else node_instruction
        return f"{time_line}\n\n{base}"

    def set_edge_id_callback(self, callback: Any | None) -> None:
        """Set async callback(text: str) fired with pre-generated response when edge_id known."""
        self._edge_id_callback = callback

    # ------------------------------------------------------------------
    # Adapter surface
    # ------------------------------------------------------------------

    async def speak(self, text: str, node: FlowNode) -> None:
        """Record static text as a response."""
        self.responses.append(text)

    async def generate_reply(
        self,
        instruction: str,
        node: FlowNode,
        history: list[dict] | None = None,
        userdata: dict | None = None,
    ) -> str:
        """Generate speech for a node entry.

        Priority:
        0. Pre-generated response from early streaming (0 LLM calls, already streamed to TTS)
        1. [EN]/[HI] language markers → extract relevant language line (0 LLM calls)
        2. "Agent says: <text>" prefix → extract speech line (0 LLM calls)
        3. Call LLM with rendered instruction (1 LLM call — complex/template flows)
        """
        # Return early response pre-generated during criteria streaming (avoid 2nd LLM call)
        if self._pre_generated_response is not None:
            pre = self._pre_generated_response
            self._pre_generated_response = None
            self.responses.append(pre)
            return pre

        lang = _resolve_lang(self._machine) if self._machine else "en"

        # Render Jinja2 templates with current userdata before extraction/LLM
        # Always render templates — _build_context merges env_vars (name, expected_yob, etc.)
        # with userdata. Skipping when userdata is empty would leave {{name}} unresolved.
        rendered_instruction = instruction
        if "{{" in instruction:
            ctx = self._build_context(userdata or {})
            rendered_instruction = self._render(instruction, ctx)

        # Code-level zero-speech gate: if the rendered instruction explicitly
        # mandates silence (e.g. "ZERO speech — no words before the tool call")
        # AND all current_* template variables have non-empty values (meaning
        # the STEP 4 / auto-proceed condition is satisfied), skip the LLM entirely.
        # Relying on the LLM to output an empty string is unreliable.
        if _is_zero_speech_step(rendered_instruction):
            logger.debug("[ToolCallAdapter] zero-speech gate triggered — skipping entry speech")
            return ""

        # 1. Try language-specific extraction (EN line: / [EN] / {EN} — all variants).
        #    Single [EN]+[HI] pair or "XX line:" nodes → zero extra LLM call.
        #    Multi-block nodes (>2 blocks for the same lang) still go to LLM so
        #    it can pick the right step based on conversation progress.
        lang_extracted = _extract_for_language(rendered_instruction, lang)
        if lang_extracted:
            self.responses.append(lang_extracted)
            return lang_extracted

        # 2. Fallback: classic [EN]/[HI] uppercase block markers via composer.
        _lang_match_count = sum(
            1 for m in _LANG_MARKER_RE.finditer(rendered_instruction)
            if m.group(1).lower() == lang or (lang != "en" and m.group(1).lower() == "en")
        )
        extracted = _extract_speech_text(rendered_instruction, self._machine, lang)
        if extracted and _lang_match_count <= 2:  # allow [EN]+[HI] pair — that's 1 step
            self.responses.append(extracted)
            return extracted

        # Try "Agent says: <text>" prefix (BOB-style flows)
        agent_says = _extract_agent_says(rendered_instruction)
        if agent_says:
            self.responses.append(agent_says)
            return agent_says

        # Single-language quoted speech without [EN]/[HI] markers.
        # Handles: 'speech text'\nROUTING: / Per [Non-Negotiable]:  etc.
        # This is the last zero-LLM-call path before falling back to the LLM.
        quoted_speech = _extract_first_quoted_speech(rendered_instruction)
        if quoted_speech:
            self.responses.append(quoted_speech)
            return quoted_speech

        # Complex instruction — call LLM to generate natural reply
        return await self._generate_via_llm(rendered_instruction, history or [], node_id=getattr(node, 'id', ''))

    async def _generate_via_llm(self, instruction: str, history: list[dict], node_id: str = "") -> str:
        """Call LLM to generate entry speech for complex/template instructions."""
        speech_directive = (
            "SPEECH GENERATION MODE: Generate ONLY the agent's natural spoken response. "
            "Do NOT output tool call syntax, JSON objects, function names, routing "
            "decisions, or edge IDs. Output only what the agent would SAY to the caller. "
            "CRITICAL SILENCE RULE: If the instruction contains 'ZERO speech', "
            "'zero words', 'no words before the tool call', 'call the tool immediately', "
            "'Do NOT speak', 'Silent routing node', or any directive to produce NO speech "
            "— output an EMPTY string. Never ask for information when instructed to be silent."
        )
        base = f"{self._system_prompt}\n\n{instruction}" if self._system_prompt else instruction
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": f"{speech_directive}\n\n{base}",
            }
        ]
        messages.extend(history[-6:])
        _ensure_non_system(messages)

        _t0 = time.perf_counter()
        try:
            result = await self._resolve_provider().complete(messages, temperature=0.3)
        except Exception as exc:
            logger.error("[ToolCallAdapter] generate_reply LLM call failed: %s", exc)
            return instruction

        latency_ms = (time.perf_counter() - _t0) * 1000
        meta = result.metadata or {}
        text = result.text or ""
        if self._on_llm_complete is not None:
            await self._on_llm_complete(LLMCallData(
                node_id=node_id,
                model=_strip_provider_prefix(self._model_id),
                call_type="generate_reply",
                latency_ms=latency_ms,
                tokens_in=meta.get("prompt_tokens", 0) or 0,
                tokens_out=meta.get("completion_tokens", 0) or 0,
                prompt_messages=messages,
                response_json={"text": text},
                edge_id=None,
            ))
        self.responses.append(text)
        return text

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
        silent: bool = False,
    ) -> CriteriaResult:
        """Evaluate via LLM tool-calling (streaming for early edge_id detection).

        ``silent=True`` marks a silent auto-chain routing pass (no new user
        turn). In that mode the stay-hatch + ROUTING RULE are reframed around
        DATA/state instead of "did the caller answer my question" — otherwise
        the LLM false-stays when the trailing history looks unrelated to this
        (silent) node, stalling the router chain.
        """
        machine = self._machine

        # Build tool schemas from descriptors
        if machine:
            descriptors = machine.get_tools_for_node(node)
        else:
            descriptors = [
                ToolDescriptor(
                    id=e.id,
                    description=e.condition,
                    is_data_collection=e.input_schema is not None,
                    input_schema=(
                        e.input_schema if isinstance(e.input_schema, dict) else None
                    ),
                    target_node_id=e.target_node_id,
                )
                for e in node.edges
            ]

        tools = _descriptors_to_openai_tools(descriptors)
        if not tools:
            return CriteriaResult(node_id=node.id)

        # Always add a stay-on-node escape hatch so the LLM never force-fits
        # ambiguous input to a real edge. tool_choice="required" means the LLM
        # MUST call something — without this it picks the closest-sounding edge
        # even when none are satisfied (compliments, off-topic, partial sentences).
        # In silent auto-chain mode there is NO new caller turn, so the hatch is
        # reframed around DATA/state: stay only when the data/instruction does
        # not yet decide an edge — NOT because "the caller didn't answer".
        if silent:
            _stay_description = (
                "Call this ONLY when the current DATA and node instruction do "
                "NOT yet satisfy ANY listed edge condition — i.e. the node is "
                "genuinely waiting for information that has not arrived yet "
                "(e.g. no slot chosen, a required field still empty). "
                "There is NO new caller message in this pass, so do NOT stay "
                "merely because 'the caller hasn't answered' — evaluate the "
                "DATA. If the data already satisfies an edge, CALL THAT EDGE, "
                "not this. Leave brief_response empty."
            )
        else:
            _stay_description = (
                "Call this when the caller's response does NOT clearly and "
                "unambiguously satisfy ANY of the listed edge conditions. "
                "ALWAYS call this for: (1) asking agent to repeat/re-read "
                "information ('address बताइए', 'फिर से बोलिए', 'what did you say', "
                "'tell me again', 'didn't hear', 'repeat'); "
                "(2) ambiguous/off-topic input, compliment, filler, partial sentence; "
                "(3) 'no' as a correction not a goodbye; "
                "(4) any question about what was just said. "
                "Staying is ALWAYS safer than a wrong transition. When in doubt — stay. "
                "IMPORTANT: set brief_response to a SHORT contextual reply "
                "(1-2 sentences max). If caller asked to repeat specific info, "
                "repeat ONLY that info. If caller asked a clarifying question, "
                "answer briefly. Do NOT repeat the full agent turn."
            )
        tools.append({
            "type": "function",
            "function": {
                "name": "__stay_on_node__",
                "description": _stay_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brief_response": {
                            "type": "string",
                            "description": (
                                "Short contextual reply (1-2 sentences). "
                                "Repeat ONLY what was asked for — e.g. if caller "
                                "asked for address, say only the address. "
                                "Leave empty if no specific reply needed."
                            ),
                        }
                    },
                },
            },
        })

        instructions = (
            self._build_instructions(node, machine) if machine else self._system_prompt
        )

        # Prepend routing discipline so LLM prefers __stay_on_node__ over false matches.
        # Silent auto-chain pass: the caller-message framing is meaningless (no new
        # caller turn). Route on DATA + node instruction; stay only if data is
        # genuinely not ready. The "doesn't answer my question -> stay" rule is what
        # false-stalls silent routers, so it is dropped here.
        if silent:
            _routing_rule = (
                "ROUTING RULE (silent routing pass — NO new caller message): "
                "Decide the edge purely from the [CURRENT DATA] and the node "
                "instruction. If the data satisfies one edge condition, CALL THAT "
                "EDGE. Call __stay_on_node__ ONLY when the data genuinely does not "
                "yet satisfy any edge (the node is waiting for info not yet present). "
                "Do NOT stay just because the recent history looks unrelated to this "
                "node — there is no caller turn to evaluate here."
            )
        else:
            _routing_rule = (
                "ROUTING RULE: Call __stay_on_node__ unless the caller's message "
                "CLEARLY and UNAMBIGUOUSLY satisfies one specific edge condition. "
                "MANDATORY __stay_on_node__ cases:\n"
                "- CONTEXT MISMATCH: caller says something that has nothing to do with "
                "the agent's current question (e.g. agent asked 'can we talk?' and "
                "caller says a year/number; agent asked yes/no and caller gave unrelated "
                "data; the input makes no sense as an answer to the current question)\n"
                "- Asking agent to repeat/re-read ('address बताइए', 'फिर से बोलिए', "
                "'didn't hear', 'what was that', 'tell me again')\n"
                "- Compliments, tangents, frustration outbursts, filler, testing phrases\n"
                "- Partial or cut-off sentences\n"
                "- 'no' as a correction not a goodbye\n"
                "- 'thank you for confirming' — NOT a card receipt confirmation\n"
                "- Any response that doesn't directly answer the agent's current question\n"
                "OBJECTIVE RULE: Always complete the current node's objective before "
                "transitioning. If the caller's response doesn't fulfill the objective, stay.\n"
                "Never force-fit an off-topic response to the closest-sounding edge."
            )
        instructions = f"{_routing_rule}\n\n{instructions}"

        import re as _re

        # In a silent routing pass there is no user turn to anchor on, so dumping
        # the ENTIRE userdata into [CURRENT DATA] (15 courses + slots + every API
        # result ≈ 8k tok) drowns the router model and it false-stays. Scope the
        # data to the variables this node actually names (instruction + edge
        # conditions). User-driven turns keep the full dump — unchanged.
        _slot_ctx = {k: v for k, v in userdata.items() if k != "_flow_meta" and v not in (None, "")}
        if silent:
            _ref_text = " ".join(
                [node.instruction or ""] + [(e.condition or "") for e in (node.edges or [])]
            )
            _ref_tokens = set(_re.findall(r"[A-Za-z_]\w*", _ref_text))
            _slot_ctx = {k: v for k, v in _slot_ctx.items() if k in _ref_tokens}

        # Also expose template variables referenced in the raw node instruction that
        # are null/unset — so the LLM can reason about them for silent routing nodes
        # (e.g. "name_check" routes on {{name}} which may not be in userdata).
        _TVAR_RE = _re.compile(r"\{\{\s*(\w+)\s*\}\}")
        _raw_instruction = node.instruction or ""
        _referenced_vars = {m for m in _TVAR_RE.findall(_raw_instruction)}
        _null_vars = {
            v: "null"
            for v in _referenced_vars
            if v not in _slot_ctx and v not in ("_flow_meta",)
            and userdata.get(v) in (None, "", [], {})
        }

        _all_data = {**_slot_ctx, **_null_vars}
        if _all_data:
            _slot_lines = "\n".join(f"  {k}: {v}" for k, v in _all_data.items())
            instructions = f"{instructions}\n\n[CURRENT DATA]\n{_slot_lines}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
        messages.extend(history[-10:])
        _ensure_non_system(messages)

        _t0 = time.perf_counter()
        try:
            result = await self._resolve_provider().complete(
                messages, tools=tools, tool_choice="required", temperature=0
            )
        except Exception as exc:
            logger.error("[ToolCallAdapter] LLM call failed: %s", exc)
            return CriteriaResult(node_id=node.id)

        latency_ms = (time.perf_counter() - _t0) * 1000
        meta = result.metadata or {}
        prompt_tokens = meta.get("prompt_tokens", 0) or 0
        completion_tokens = meta.get("completion_tokens", 0) or 0

        function_name = ""
        function_args = ""
        if result.tool_calls:
            fn = result.tool_calls[0].get("function", {}) or {}
            function_name = fn.get("name", "") or ""
            function_args = fn.get("arguments", "") or ""

        # Start the target node's TTS now that the edge is known (early response).
        valid_edge_ids = {e.id for e in (node.edges or [])}
        if (
            function_name
            and function_name in valid_edge_ids
            and self._edge_id_callback
        ):
            await self._fire_early_response(function_name, node)

        if self._on_llm_complete is not None:
            await self._on_llm_complete(LLMCallData(
                node_id=node.id,
                model=_strip_provider_prefix(self._model_id),
                call_type="routing",
                latency_ms=latency_ms,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                prompt_messages=messages,
                response_json={"tool_call": function_name, "args": function_args},
                edge_id=function_name if function_name != "__stay_on_node__" else None,
            ))

        if not function_name:
            logger.info("[ToolCallAdapter] no tool_call returned for node=%s", node.id)
            return CriteriaResult(node_id=node.id)

        if function_name == "__stay_on_node__":
            brief = ""
            if function_args:
                try:
                    brief = json.loads(function_args).get("brief_response", "") or ""
                except (json.JSONDecodeError, TypeError):
                    pass
            logger.info("[ToolCallAdapter] stay_on_node brief=%r node=%s", brief[:60], node.id)
            return CriteriaResult(node_id=node.id, response=brief)

        edge_id = function_name
        extracted_slots: dict[str, Any] = {}
        if function_args:
            try:
                extracted_slots = json.loads(function_args)
            except (json.JSONDecodeError, TypeError):
                pass

        matched_descriptor = next((d for d in descriptors if d.id == edge_id), None)
        if matched_descriptor and matched_descriptor.input_schema:
            schema = matched_descriptor.input_schema
            allowed_keys = set(schema.get("properties", {}).keys())
            if allowed_keys:
                extracted_slots = {k: v for k, v in extracted_slots.items() if k in allowed_keys}
            # Auto-fill required fields with single-value enums — these are constants,
            # not LLM-extracted. LLM often omits them; we set them deterministically.
            props = schema.get("properties", {})
            for req_key in schema.get("required", []):
                if req_key not in extracted_slots and req_key in props:
                    enum_vals = props[req_key].get("enum", [])
                    if len(enum_vals) == 1:
                        extracted_slots[req_key] = enum_vals[0]

        logger.info("[ToolCallAdapter] tool_call=%s slots=%s node=%s", edge_id, extracted_slots, node.id)

        return CriteriaResult(
            node_id=node.id,
            recommended_edge_id=edge_id,
            all_required_met=True,
            extracted_slots=extracted_slots,
        )

    async def _fire_early_response(self, edge_id: str, current_node: "FlowNode") -> None:
        """Pre-generate and stream response for target node when edge_id is known.

        Only fires for nodes with extractable speech text (0 LLM calls).
        Complex nodes fall through to generate_reply() as normal.
        """
        if not self._edge_id_callback or not self._machine:
            return

        edge = next((e for e in (current_node.edges or []) if e.id == edge_id), None)
        if not edge:
            return
        target_node = self._machine._node_map.get(edge.target_node_id)
        if not target_node:
            return

        # Don't fire early response if target node would be smart-skipped by
        # _follow_router_chain. Early TTS would play then get orphaned.
        ctx = self._machine.context
        _node_had_slots = bool(ctx.node_slots.get(target_node.id))
        _is_single_edge = len(target_node.edges) == 1
        if (
            target_node.id in ctx.completed_nodes
            and (_is_single_edge or _node_had_slots)
        ):
            return

        instruction = target_node.instruction or ""
        if "{{" in instruction:
            # Skip early response if any template variable is not yet in userdata.
            # Slots are stored after _do_transition, not during streaming eval.
            import re as _re_check
            needed = set(_re_check.findall(r'\{\{(\w+)\}\}', instruction))
            current_ud = dict(ctx.userdata) if ctx.userdata else {}
            if needed - set(current_ud.keys()):
                return  # Variables not yet available — generate_reply will have them
            ctx_dict = self._build_context(current_ud)
            instruction = self._render(instruction, ctx_dict)

        # Extract speech without LLM — try all scripted formats in priority order.
        # Fires early so TTS starts streaming before generate_reply() is called.
        pre_text = _extract_agent_says(instruction)

        if not pre_text:
            lang = _resolve_lang(self._machine) if self._machine else "en"
            pre_text = _extract_for_language(instruction, lang)

        if not pre_text:
            return

        # Store so generate_reply() returns it without a second LLM call
        self._pre_generated_response = pre_text

        # Push tokens through the callback → token_queue → TTS starts immediately
        await self._edge_id_callback(pre_text)

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute an HTTP action — identical logic to LLMAdapter.execute_action."""

        import httpx

        print(f"[TRACK] ToolCallAdapter.execute_action START - action_id: {action.id}, userdata keys: {list(userdata.keys()) if userdata else 'None'}")

        # run_once: return cached result if already succeeded this session
        if action.run_once and action.store_response_as:
            cached = userdata.get(action.store_response_as, {})
            if isinstance(cached, dict) and cached.get("success"):
                print(f"[TRACK] ToolCallAdapter - run_once HIT for action={action.id}, returning cached result")
                return cached

        ctx = self._build_context(userdata)

        # condition guard
        if action.condition:
            condition_result = self._render(action.condition, ctx)
            if not condition_result.strip():
                print(f"[TRACK] ToolCallAdapter - action={action.id} SKIPPED (condition empty after render)")
                return None

        url = self._render(action.url, ctx)
        headers = {k: self._render(v, ctx) for k, v in action.headers.items()}
        method = action.method.value if hasattr(action.method, "value") else str(action.method)

        print(f"[TRACK] ToolCallAdapter - rendered URL: {url}  (template: {action.url[:80]})")
        print(f"[TRACK] ToolCallAdapter - method: {method}")

        # GET cache: same URL in same session → return cached successful result.
        # Key includes the full rendered URL so a city/param change produces a
        # different key and bypasses the cache (fires the API again as needed).
        if method.upper() == "GET":
            _cache_key = f"GET:{url}"
            if _cache_key in self._get_cache:
                print(f"[TRACK] ToolCallAdapter - GET cache HIT for action={action.id} url={url}")
                return self._get_cache[_cache_key]

        body: Any = None
        if action.body_template:
            body_str = self._render(action.body_template, ctx)
            print(f"[TRACK] ToolCallAdapter - rendered body: {body_str[:200]}")
            try:
                body = json.loads(body_str)
                if isinstance(body, dict):
                    body = _coerce_numeric_strings(body, set(action.string_fields))
            except json.JSONDecodeError:
                body = body_str

        try:
            async with httpx.AsyncClient(timeout=float(action.timeout)) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, dict) else None,
                    content=body.encode() if isinstance(body, str) else None,
                )
                result: dict[str, Any] = {
                    "status": response.status_code,
                    "success": response.status_code < 400,
                    "headers": dict(response.headers),
                    "_rendered_url": url,
                    "_method": method,
                }
                try:
                    result["data"] = response.json()
                except Exception:
                    result["data"] = response.text

                status_tag = "OK" if response.status_code < 400 else "FAILED"
                print(f"[TRACK] ToolCallAdapter - action={action.id} {status_tag} status={response.status_code}")
                print(f"[TRACK] ToolCallAdapter - response data: {json.dumps(result.get('data'), default=str)[:500]}")

                # Apply env_updates (e.g. store ACCESS_TOKEN for subsequent actions)
                for update in action.env_updates:
                    value: Any = result
                    try:
                        for key in update.result_path.split("."):
                            value = value[key]
                        self._env_vars[update.env_key] = str(value)
                        print(f"[TRACK] ToolCallAdapter - env_update: {update.env_key} = {str(value)[:80]}")
                    except (KeyError, TypeError, IndexError):
                        print(f"[TRACK] ToolCallAdapter - env_update FAILED: could not resolve {update.result_path} for {update.env_key}")

                # Populate GET cache for successful responses.
                # Key is "GET:<rendered_url>" — city change → different URL → different key.
                if method.upper() == "GET" and result["success"]:
                    self._get_cache[f"GET:{url}"] = result

                return result

        except Exception as exc:
            logger.error("[ToolCallAdapter] action=%s HTTP error: %s", action.id, exc)
            return {
                "success": False,
                "error": str(exc),
                "status": 0,
                "_rendered_url": url,
                "_method": method,
            }

    async def end_session(self) -> None:
        """Mark session as ended."""
        self.session_ended = True
