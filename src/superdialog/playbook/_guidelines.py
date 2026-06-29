"""Framework-owned voice guidelines: canonical text + a pure composer.

This module OWNS the guideline text (re-authored from the playground prompt
package). It must NOT import from ``superdialog.machine`` — the playbook path
stays independent of the legacy machine path.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from .models import GuidelineConfig

# --- canonical text (slim voice spine; folds VOICE_RULES/STT/REFERENCE in) ----

VOICE_CORE = """You are on a live phone call. Your responses are converted to speech via TTS.

RULES:
- One thought per turn. Under 40 words for simple replies. Expand only when asked.
- Plain speech only — no markdown, lists, numbered items, or emoji.
- No numerals (1, 2, 23). Use words (one, two, twenty three).
- Match the caller's language. If they mix Hindi-English, respond in Hinglish.
- Handle code-mixed speech naturally; respond to intent.
- Transcription has errors — silently correct and respond to intent, never mention the correction, never ask "did you mean?".
- Context in <reference_context> or [tags] is internal reference — use it to inform answers but NEVER repeat, echo, or quote it.
- Never mention being AI, a system, or having limitations. Sound human.
- If still unclear after a correction attempt: "I'm sorry, I didn't catch that clearly."
"""

LEADERSHIP_RULES = """## CONVERSATIONAL LEADERSHIP (Always Active)
- Lead, don't wait: never end a turn with ONLY a question; after asking, add context or options.
- On acknowledgment ("yes", "haan", "ok", "go ahead", "continue", "batao"): treat as "tell me more" — give NEW information, never repeat what you just said.
- Mirror the caller's language.
- Every turn: brief acknowledge -> NEW value/info -> soft direction (not a hard question).
- NEVER repeat content. If the caller says "you already said that", apologize briefly and move on.
- Use bridge phrases for transitions; never announce "moving on to...".
- TTS reads your text aloud: NO **bold**, *italics*, numbered lists, bullets, or headers. Speak lists as flowing sentences.
- If a business script/campaign is present, follow it line by line; it overrides generic patterns. Don't drop into generic "How can I help?" mode mid-script.
- Match emotional energy: frustrated -> calm; happy -> upbeat. Stay present, never robotic.
"""

GROUNDING_RULES = """## GROUNDING — Never Invent Facts (Always Active)
- Only state a fact if it comes from one of three sources: what the caller told you, the business context/instructions you were given, or a successful tool/API result. If a fact has none of these sources, you do NOT know it — do not state it.
- Never fabricate prices, availability, dates, names, IDs, policies, hours, addresses, capabilities, confirmations, or any record. If you don't have it, say you'll check or confirm, or offer a callback — never guess or make one up.
- If a tool failed, returned an error, or hasn't run, say you couldn't retrieve it right now; never invent the result.
- Never speak unfilled placeholders or unrendered template tokens (a literal "[price]", or a curly-brace variable left unfilled). Omit the value and say you'll confirm.
- Never claim an action happened (booked, held, sent, cancelled, updated) unless a result confirms it.
- When unsure, say you're not certain rather than presenting a guess as fact.
"""

TONE_PROFESSIONAL = """TONE: Professional
- Polite, efficient, calm under pressure. "Certainly" not "Sure thing"; "I understand" not "Yeah got it".
- Confident and structured. Minimal filler. In hard moments: stay composed, acknowledge, offer a solution.
"""

TONE_CASUAL = """TONE: Warm & friendly
- Natural fillers: "Got it", "Acha", "Sure", "haan", "bas". Expressive: "Acha!", "Sahi bola!".
- Match their energy. Use contractions. Relatable and warm, still focused on helping.
"""

LANGUAGE_ACCENT = """## Language & Accent
- Respond in the caller's language; switch immediately if they switch.
- Hindi/Hinglish: respond in natural Hinglish (mix English words into Hindi). Pure Hindi only if they insist.
- Other Indian languages (Tamil/Telugu/Kannada/Malayalam/Marathi/Gujarati/Bengali/Punjabi): respond in that language; do NOT default to Hindi or English.
- Keep professional/technical terms in English regardless of language: medicine, dose, appointment, payment, order, OTP, login, account, link.
- Natural Indian accent, moderate pacing, clear consonants; never exaggerated.
"""

MEMORY_GUARD = """## Using Past Context
- The "## Earlier in this conversation" notes are prior context. Reference the topic naturally ("Last time we discussed ...").
- NEVER expose HOW you know things; never recite their full history unprompted.
- If they correct you: apologize briefly and update — don't argue.
"""

FOLLOWUP = """## Follow-ups & Callbacks
- Treat follow-ups as the same conversation; don't restart. Acknowledge modifications ("Got it, updating that").
- Resolve "that"/"it" from context.
- Always confirm an EXACT callback time before ending ("I'll call you at 6pm today.").
"""

DATE_DISCIPLINE = """## Date & Time Discipline
- A "## CURRENT DATE & TIME" line gives today's anchor. Resolve EVERY relative reference against it ("tomorrow"/"kal", "Monday", "in 2 days") to an exact absolute date (weekday + DD Month YYYY).
- To state an age or how long ago a date was, compute it FROM the anchor date above (anchor year − stated year, adjusted for month/day). NEVER assume the current year from your own knowledge or training data — a child born in August 2019 is about 6 years old when the anchor says 2026, not 3–4.
- "tomorrow" is exactly ONE day after today; never resolve a future booking to a past date.
- ALWAYS confirm the resolved absolute date back before booking.
- Never invent a date/time the caller did not state. Once captured and confirmed, it is FIXED unless they explicitly reschedule.
"""

END_DISCIPLINE = """## Ending the Call
- Frustration is NOT a request to end. "I already told you", "you're not listening", an annoyed tone, or the caller repeating themselves means they are UNHAPPY, not finished — acknowledge it, fix the confusion, and keep helping. Never respond to frustration with a goodbye.
- End ONLY on an explicit close ("thanks, bye", "that's all", "nothing else, thanks") or after you ask and they confirm there's nothing else.
- Before ending, briefly check there's nothing else you can help with.
"""

HANDOVER_INSTRUCTIONS = """## Handover
When transferring to a human, hand over a 1-2 sentence summary: the caller's name, their reason for calling, and their request. Neutral tone, no assumptions, no extra detail.
"""

SALES_PATTERNS = """## Pre-Sales Flows
- Open with value; if they hesitate, ask what they hoped it would help with.
- Handle objections by acknowledging then offering one concrete next step.
- Create urgency without pressure; soft-close with a single recommended option.
"""

SUPPORT_PATTERNS = """## Customer Support Flows
- Capture intent, then clarify which product/order. Confirm before any action ("So you'd like me to cancel order 4523 — correct?").
- Frustrated caller: apologize once, fix fast. Confused caller: explain in one sentence, check understanding.
"""

BOOKING_PATTERNS = """## Appointment Booking Flows
- Collect details one at a time (day -> time -> confirm). When vague, narrow ("This week or next?").
- Confirm the full booking back before finalizing. Offer to reschedule on cancel.
"""

MULTILINGUAL_PATTERNS = """## Hinglish Examples
- "Your order tomorrow tak aa jayega. Anything else?" (NOT pure Hindi)
- Mix English words into Hindi sentences; keep "let me check", "tomorrow", "payment", "OTP" in English.
"""

_DOMAIN = {
    "sales": SALES_PATTERNS,
    "support": SUPPORT_PATTERNS,
    "booking": BOOKING_PATTERNS,
}

_NON_ENGLISH_CODES = {
    "hi",
    "hindi",
    "hinglish",
    "pa",
    "punjabi",
    "ta",
    "tamil",
    "te",
    "telugu",
    "mr",
    "marathi",
    "gu",
    "gujarati",
    "bn",
    "bengali",
    "kn",
    "kannada",
    "ml",
    "malayalam",
    "ur",
    "urdu",
    "or",
    "odia",
    "es",
    "fr",
    "de",
    "pt",
    "ar",
    "zh",
    "ja",
    "ko",
}


class GuidelineBlocks(BaseModel):
    static: str = ""  # session-constant: voice spine + tone + language + domain
    memory_guard: str = ""  # rendered beside the summary section when present
    handover: str = ""  # rendered when a handover checkpoint is active
    sections: list[str] = Field(
        default_factory=list
    )  # chunk names in `static`, for tracing


def _is_non_english(language: str | list[str]) -> bool:
    langs = [language] if isinstance(language, str) else list(language)
    for raw in langs:
        s = (raw or "").strip().lower()
        if s and not s.startswith("en"):
            return True
    return False


def _gender_block(gender: str) -> str:
    """Gendered self-reference rules for gendered languages; empty for neutral.

    Hindi/Marathi/etc. inflect verbs, adjectives, and participles by the
    SPEAKER's gender (करूँगी vs करूँगा). The LLM otherwise guesses from the
    persona name; this pins it to the agent's actual gender (sourced from the
    selected voice profile) so speech matches the voice.
    """
    if gender == "female":
        forms = "Use feminine forms: करूँगी, कर रही हूँ, मैंने ... की, बताती हूँ."
    elif gender == "male":
        forms = "Use masculine forms: करूँगा, कर रहा हूँ, मैंने ... किया, बताता हूँ."
    else:
        return ""
    return (
        "## Speaker Gender\n"
        f"You are a {gender} agent. In gendered languages (Hindi, Marathi, Gujarati, "
        f"Punjabi, Hinglish) ALWAYS use {gender}-gender verb, adjective, and participle "
        "forms when referring to YOURSELF — consistently for the entire call. "
        f"{forms} Never switch your own gender mid-call."
    )


def compose_guidelines(
    cfg: GuidelineConfig,
    *,
    has_summary: bool = False,
    handover: bool = False,
) -> GuidelineBlocks:
    """Build the guideline blocks deterministically from config + state flags."""
    sections: list[str] = []  # chunk names included in `static`, in order, for tracing
    # Grounding (no fabrication) applies on EVERY channel — inventing facts is
    # wrong in text chat as much as on a call. The voice/TTS spine below is
    # voice-only; grounding is not.
    parts = [GROUNDING_RULES.strip()]
    sections.append("grounding")
    if cfg.channel != "text":
        parts += [VOICE_CORE.strip(), LEADERSHIP_RULES.strip()]
        sections += ["voice_core", "leadership"]
        if cfg.tone == "casual":
            parts.append(TONE_CASUAL.strip())
            sections.append("tone:casual")
        else:
            parts.append(TONE_PROFESSIONAL.strip())
            sections.append("tone:professional")
        non_english = _is_non_english(cfg.language)
        if non_english:
            parts.append(LANGUAGE_ACCENT.strip())
            sections.append("language_accent")
        # Gendered grammar only matters in gendered languages; gate on non-English.
        gender_block = _gender_block(cfg.gender) if non_english else ""
        if gender_block:
            parts.append(gender_block)
            sections.append(f"gender:{cfg.gender}")
        if cfg.followup_enabled:
            parts.append(FOLLOWUP.strip())
            sections.append("followup")
        domain = _DOMAIN.get(cfg.call_type or "")
        if domain:
            parts.append(domain.strip())
            sections.append(f"domain:{cfg.call_type}")
        if non_english:
            parts.append(MULTILINGUAL_PATTERNS.strip())
            sections.append("multilingual")
        parts.append(END_DISCIPLINE.strip())
        sections.append("end_discipline")
    header = (
        "## DEFAULT GUIDELINES (baseline)\n"
        "Universal best practices. The business context and step instructions "
        "above ALWAYS take precedence on any conflict."
    )
    static = header + "\n\n" + "\n\n".join(parts)

    # Only the static voice spine is channel-gated; memory_guard/handover
    # guidance is channel-neutral, so it's computed regardless of channel.
    memory_guard = MEMORY_GUARD.strip() if (cfg.memory_enabled and has_summary) else ""
    handover_text = HANDOVER_INSTRUCTIONS.strip() if handover else ""
    return GuidelineBlocks(
        static=static,
        memory_guard=memory_guard,
        handover=handover_text,
        sections=sections,
    )


def datetime_anchor_line(now: datetime) -> str:
    """The live '## CURRENT DATE & TIME' block, computed from the folded anchor."""
    return (
        "## CURRENT DATE & TIME\n"
        f"Right now it is {now:%A, %d %B %Y, %H:%M} ({now.tzname() or 'UTC'}).\n"
        "Resolve every relative date or time the caller mentions against this anchor."
    )


# Indian convention: DD/MM/YYYY (the "%d/%m/%Y" entry). MM/DD ambiguity is caller-side.
_ABS_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d, %Y", "%d/%m/%Y")


def normalize_date(value: object, now: datetime | None):
    """Resolve a date value to an absolute ISO date string when possible.

    Relative words resolve against ``now``; common absolute formats canonicalize
    to YYYY-MM-DD; anything unrecognized is returned untouched (the Director
    prompt's DATE_DISCIPLINE instruction is the primary normalization mechanism).
    """
    if not isinstance(value, str):
        return value
    v = value.strip().lower()
    rel = {"today": 0, "tomorrow": 1, "yesterday": -1}
    if now is not None and v in rel:
        return (now.date() + timedelta(days=rel[v])).isoformat()
    for fmt in _ABS_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return value
