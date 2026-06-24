from datetime import datetime, timezone

from superdialog.playbook.models import GuidelineConfig
from superdialog.playbook._guidelines import (
    DATE_DISCIPLINE,
    compose_guidelines,
    datetime_anchor_line,
    normalize_date,
)


def test_voice_default_spine_present() -> None:
    b = compose_guidelines(GuidelineConfig())
    assert "live phone call" in b.static.lower()           # VOICE_CORE
    assert "CONVERSATIONAL LEADERSHIP" in b.static          # LEADERSHIP_RULES
    assert "Professional" in b.static                       # tone
    assert b.memory_guard == ""                             # off by default
    assert b.handover == ""


def test_text_channel_suppresses_voice_spine() -> None:
    b = compose_guidelines(GuidelineConfig(channel="text"))
    assert b.static == ""


def test_casual_tone_and_language_and_domain() -> None:
    b = compose_guidelines(GuidelineConfig(
        tone="casual", language="hi", call_type="support"))
    assert "Warm" in b.static                               # casual tone
    assert "Language & Accent" in b.static
    assert "Customer Support Flows" in b.static             # domain pattern


def test_followup_block_only_when_enabled() -> None:
    assert "Follow-ups" in compose_guidelines(GuidelineConfig(followup_enabled=True)).static
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
