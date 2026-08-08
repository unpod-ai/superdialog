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


def test_normalize_date_weekday_names() -> None:
    # Real production case: caller says bare "Monday?" on a Friday and the
    # Director LLM's own weekday arithmetic resolved it to Wednesday (three
    # days off). now is Friday 2026-08-07; nearest Monday is 2026-08-10.
    now = datetime(2026, 8, 7, tzinfo=timezone.utc)
    assert normalize_date("Monday?", now) == "2026-08-10"
    assert normalize_date("monday", now) == "2026-08-10"
    assert normalize_date("this monday", now) == "2026-08-10"
    assert normalize_date("next monday", now) == "2026-08-10"
    # "this <today's weekday>" means today, not next week.
    assert normalize_date("this friday", now) == "2026-08-07"
    # "next <today's weekday>" skips today, lands the week after.
    assert normalize_date("next friday", now) == "2026-08-14"
    assert normalize_date("coming sunday", now) == "2026-08-09"


def test_normalize_date_relative_counts() -> None:
    now = datetime(2026, 8, 7, tzinfo=timezone.utc)
    assert normalize_date("in 3 days", now) == "2026-08-10"
    assert normalize_date("3 days from now", now) == "2026-08-10"
    assert normalize_date("in 2 weeks", now) == "2026-08-21"
    assert normalize_date("next week", now) == "2026-08-14"


def test_normalize_date_ignores_weekday_names_without_anchor() -> None:
    assert normalize_date("monday", None) == "monday"


def test_normalize_date_bare_day_of_month() -> None:
    # Real production case: caller answered "Just 11" (meaning the 11th of
    # the current month) to "which date would you like to play?". Unresolved
    # this fell through to the literal string "11", which then flowed
    # straight into an availability API URL as `date=11` and 500'd.
    now = datetime(2026, 8, 8, tzinfo=timezone.utc)  # today is the 8th
    assert normalize_date("11", now) == "2026-08-11"
    assert normalize_date("Just 11", now) == "2026-08-11"
    assert normalize_date("the 11th", now) == "2026-08-11"
    assert normalize_date("11th", now) == "2026-08-11"
    # A day already passed this month rolls to next month.
    now_late = datetime(2026, 8, 20, tzinfo=timezone.utc)
    assert normalize_date("11", now_late) == "2026-09-11"
    # A day that doesn't exist in the immediate next month (Jan 31 -> no Feb
    # 31) skips ahead to the next month that has it.
    now_month_end = datetime(2026, 1, 31, tzinfo=timezone.utc)
    assert normalize_date("31", now_month_end) == "2026-01-31"


def test_normalize_date_month_and_day_without_year() -> None:
    # Same failure shape as bare day-of-month: "August 12" has no pattern to
    # match against (_ABS_FORMATS all require a year), so it fell through
    # unresolved straight into the availability API URL.
    now = datetime(2026, 8, 8, tzinfo=timezone.utc)
    assert normalize_date("August 12", now) == "2026-08-12"
    assert normalize_date("august 12", now) == "2026-08-12"
    assert normalize_date("12 August", now) == "2026-08-12"
    assert normalize_date("15th August", now) == "2026-08-15"
    assert normalize_date("the 15th of August", now) == "2026-08-15"
    assert normalize_date("Aug 15", now) == "2026-08-15"
    assert normalize_date("15 Aug", now) == "2026-08-15"
    # A month+day already passed this year rolls to next year.
    now_late = datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert normalize_date("January 5", now_late) == "2027-01-05"


def test_normalize_date_strips_leading_filler_word() -> None:
    # A caller frames a date with "on"/"for"/"at" more often than bare
    # ("on August 15", "for the 11th") -- every existing pattern must match
    # either form the same way.
    now = datetime(2026, 8, 8, tzinfo=timezone.utc)
    assert normalize_date("on August 15", now) == "2026-08-15"
    assert normalize_date("on the 15th of August", now) == "2026-08-15"
    assert normalize_date("for the 11th", now) == "2026-08-11"
    assert normalize_date("on monday", now) == normalize_date("monday", now)
    assert normalize_date("for tomorrow", now) == normalize_date("tomorrow", now)


def test_normalize_date_month_name_with_comma_year() -> None:
    assert normalize_date("August 15, 2026", None) == "2026-08-15"
