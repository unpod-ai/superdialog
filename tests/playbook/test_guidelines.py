from datetime import datetime, timezone

from superdialog.playbook.models import GuidelineConfig
from superdialog.playbook._guidelines import (
    compose_guidelines,
    datetime_anchor_line,
    normalize_date,
)


def test_voice_default_spine_present() -> None:
    b = compose_guidelines(GuidelineConfig())
    assert "live phone call" in b.static.lower()  # VOICE_CORE
    assert "CONVERSATIONAL LEADERSHIP" in b.static  # LEADERSHIP_RULES
    assert "Professional" in b.static  # tone
    assert b.memory_guard == ""  # off by default
    assert b.handover == ""


def test_text_channel_suppresses_voice_spine() -> None:
    b = compose_guidelines(GuidelineConfig(channel="text"))
    # voice/TTS spine suppressed on text, but grounding still applies everywhere
    assert "voice_core" not in b.sections and "On a live phone call" not in b.static
    assert b.sections == ["grounding"]
    assert "Never Invent Facts" in b.static


def test_sections_breakdown_tracks_active_chunks() -> None:
    # default voice spine (grounding leads, fed on every channel)
    assert compose_guidelines(GuidelineConfig()).sections == [
        "grounding",
        "voice_core",
        "leadership",
        "tone:professional",
        "end_discipline",
    ]
    # text channel: grounding still applies, voice spine omitted
    assert compose_guidelines(GuidelineConfig(channel="text")).sections == ["grounding"]
    # full conditional set: casual tone, non-English, domain, followup
    s = compose_guidelines(
        GuidelineConfig(
            tone="casual", language="hi", call_type="support", followup_enabled=True
        )
    ).sections
    assert s == [
        "grounding",
        "voice_core",
        "leadership",
        "tone:casual",
        "language_accent",
        "followup",
        "domain:support",
        "multilingual",
        "end_discipline",
    ]


def test_end_discipline_present_on_voice() -> None:
    blocks = compose_guidelines(GuidelineConfig(channel="voice"))
    assert (
        "not a request to end" in blocks.static.lower()
        or "frustration" in blocks.static.lower()
    )
    assert "end_discipline" in blocks.sections


def test_end_discipline_absent_on_text() -> None:
    blocks = compose_guidelines(GuidelineConfig(channel="text"))
    assert "end_discipline" not in blocks.sections


def test_casual_tone_and_language_and_domain() -> None:
    b = compose_guidelines(
        GuidelineConfig(tone="casual", language="hi", call_type="support")
    )
    assert "Warm" in b.static  # casual tone
    assert "Language & Accent" in b.static
    assert "Customer Support Flows" in b.static  # domain pattern


def test_gender_block_matches_voice_in_gendered_language() -> None:
    # female + Hindi -> feminine grammar block + section
    bf = compose_guidelines(GuidelineConfig(gender="female", language="hi"))
    assert (
        "Speaker Gender" in bf.static
        and "feminine" in bf.static
        and "करूँगी" in bf.static
    )
    assert "gender:female" in bf.sections
    # male + Hindi -> masculine
    bm = compose_guidelines(GuidelineConfig(gender="male", language="hi"))
    assert "masculine" in bm.static and "करूँगा" in bm.static
    assert "gender:male" in bm.sections
    # neutral -> no gender block
    bn = compose_guidelines(GuidelineConfig(gender="neutral", language="hi"))
    assert "Speaker Gender" not in bn.static
    assert not any(s.startswith("gender:") for s in bn.sections)
    # English -> no gendered grammar needed, so no block even with gender set
    be = compose_guidelines(GuidelineConfig(gender="female", language="en"))
    assert "Speaker Gender" not in be.static


def test_followup_block_only_when_enabled() -> None:
    assert (
        "Follow-ups"
        in compose_guidelines(GuidelineConfig(followup_enabled=True)).static
    )
    assert "Follow-ups" not in compose_guidelines(GuidelineConfig()).static


def test_memory_guard_only_when_enabled_and_summary() -> None:
    cfg = GuidelineConfig(memory_enabled=True)
    assert compose_guidelines(cfg, has_summary=True).memory_guard != ""
    assert compose_guidelines(cfg, has_summary=False).memory_guard == ""
    assert compose_guidelines(GuidelineConfig(), has_summary=True).memory_guard == ""


def test_handover_block_only_when_flagged() -> None:
    assert compose_guidelines(GuidelineConfig(), handover=True).handover != ""
    assert compose_guidelines(GuidelineConfig(), handover=False).handover == ""


def test_anchor_line_formats_absolute_date() -> None:
    now = datetime(2026, 6, 24, 14, 30, tzinfo=timezone.utc)
    line = datetime_anchor_line(now)
    assert "CURRENT DATE & TIME" in line
    assert "2026" in line and "June" in line


def test_normalize_date_relative_and_absolute() -> None:
    now = datetime(2026, 6, 24, tzinfo=timezone.utc)
    assert normalize_date("tomorrow", now) == "2026-06-25"
    assert normalize_date("today", now) == "2026-06-24"
    assert normalize_date("yesterday", now) == "2026-06-23"
    assert normalize_date("2026-07-01", now) == "2026-07-01"
    assert normalize_date("1 July 2026", now) == "2026-07-01"
    assert normalize_date("sometime next quarter", now) == "sometime next quarter"
    assert normalize_date(None, now) is None
