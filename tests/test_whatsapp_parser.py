"""Parser tests.

Bias of this suite: it asserts hard on STYLE PRESERVATION. Silent corruption of
slang, casing, emoji or Devanagari is the failure mode that would quietly ruin
the fine-tune three months from now, and it is invisible in aggregate stats.
"""

from datetime import datetime
from pathlib import Path

import pytest

from mehullm.pipeline.whatsapp_parser import (
    Message,
    normalize_line,
    parse_export,
    parse_text,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def android():
    return parse_export(FIXTURES / "android_hinglish.txt")


@pytest.fixture(scope="module")
def ios():
    return parse_export(FIXTURES / "ios_mixed.txt")


def _by_sender(chat, sender: str) -> list[Message]:
    return [m for m in chat.messages if m.sender == sender]


# ---------------------------------------------------------------- formats ---


def test_parses_android_unbracketed(android):
    m = next(m for m in android.messages if m.text.startswith("yaaar"))
    assert m.sender == "Mehul"
    assert m.ts == datetime(2025, 2, 26, 9, 16)


def test_parses_ios_bracketed_with_seconds_and_ampm(ios):
    m = ios.messages[0]
    assert m.sender == "Mehul"
    assert m.text == "subah subah uth gaya"
    assert m.ts == datetime(2025, 2, 26, 9, 15, 2)


def test_pm_converts_to_24h(ios):
    m = next(m for m in ios.messages if m.text.startswith("haan yaar"))
    assert m.ts.hour == 21, "9 PM must become 21:00, not stay 09:00"


def test_midnight_and_noon_ampm_edge_cases():
    chat = parse_text(
        "[01/01/2025, 12:00 AM] A: midnight\n[01/01/2025, 12:30 PM] A: noon\n"
    )
    assert chat.messages[0].ts.hour == 0, "12 AM is hour 0"
    assert chat.messages[1].ts.hour == 12, "12 PM is hour 12"


def test_tilde_prefix_stripped_from_group_push_names(ios):
    assert any(m.sender == "Rohan Sharma" for m in ios.messages)
    assert not any(m.sender and m.sender.startswith("~") for m in ios.messages)


def test_mixed_formats_in_one_file(ios):
    """iOS exports can interleave bracketed and unbracketed lines."""
    assert len({m.sender for m in ios.messages if m.sender}) == 2


# ------------------------------------------------------------ multi-line ---


def test_multiline_message_is_joined_not_split(android):
    m = next(m for m in android.messages if m.text.startswith("ye dekh"))
    assert m.text == "ye dekh\nमैं कल आ रहा हूँ\nपक्का"
    assert m.has_devanagari


def test_multiline_does_not_create_phantom_messages(ios):
    m = next(m for m in ios.messages if m.text.startswith("haan yaar"))
    assert m.text.count("\n") == 2
    assert not any(x.text == "bas thoda aur" for x in ios.messages), (
        "continuation lines must not become standalone messages"
    )


# --------------------------------------------------------------- system ---


def test_encryption_notice_is_system(android):
    m = android.messages[0]
    assert m.is_system and m.sender is None


def test_system_message_containing_a_colon_is_not_mis_split(android):
    """'X changed the subject to: Y' would otherwise yield a bogus sender."""
    m = next(m for m in android.messages if "changed the subject" in m.text)
    assert m.is_system
    assert m.sender is None, "must not become sender='Mehul changed the subject to'"


def test_group_membership_events_are_system(android):
    m = next(m for m in android.messages if "added Priya" in m.text)
    assert m.is_system


def test_missed_call_is_system(ios):
    m = next(m for m in ios.messages if "Missed voice call" in m.text)
    assert m.is_system


# ---------------------------------------------------------------- media ---


def test_android_media_placeholder(android):
    m = next(m for m in android.messages if m.is_media)
    assert m.sender == "Mehul"
    assert not m.is_content


def test_ios_media_variants_including_lrm_prefix(ios):
    media = [m for m in ios.messages if m.is_media]
    assert len(media) == 2, "both 'image omitted' and '<attached: ...>' must be caught"


def test_tombstone_detected(android):
    m = next(m for m in android.messages if m.is_tombstone)
    assert m.sender == "Mehul"
    assert not m.is_content


def test_edited_suffix_stripped_but_message_kept(android):
    m = next(m for m in android.messages if m.was_edited)
    assert m.text == "lol that was fast", "suffix stripped, content preserved"
    assert m.is_content, "an edited message is still authored text"


def test_empty_message_body_is_not_content(ios):
    empty = [m for m in ios.messages if m.sender and not m.text.strip()]
    assert empty and not empty[0].is_content


# ------------------------------------------------- STYLE PRESERVATION ---
# These are the tests that protect the fine-tune.


def test_lowercase_and_slang_preserved_verbatim(android):
    m = next(m for m in android.messages if "fr fr" in m.text)
    assert m.text == "fr fr no cap, chalo phir", "no case folding, no slang rewriting"


def test_elongation_preserved(android):
    """'yaaar' vs 'yaar' is signal, not noise."""
    assert any("yaaar" in m.text for m in android.messages)


def test_emoji_preserved(android):
    assert any("😎" in m.text for m in android.messages)
    assert any("🙏" in m.text for m in android.messages)


def test_devanagari_conjuncts_not_decomposed():
    """NFKC/NFKD would break matras. We use NFC."""
    raw = "01/01/2025, 10:00 - A: क्षमा करें\n"
    m = parse_text(raw).messages[0]
    assert m.text == "क्षमा करें"
    assert m.has_devanagari


def test_zwj_preserved_in_emoji_sequences():
    """ZWJ is load-bearing; stripping it shatters family/profession emoji."""
    raw = "01/01/2025, 10:00 - A: \U0001f469‍\U0001f4bb coding\n"
    m = parse_text(raw).messages[0]
    assert "‍" in m.text


def test_invisible_marks_stripped_but_content_intact():
    assert normalize_line("‎Hello﻿") == "Hello"
    assert normalize_line("9:18 AM") == "9:18 AM"


def test_colon_inside_message_body_splits_at_first_colon(android):
    m = next(m for m in android.messages if m.text.startswith("check this"))
    assert m.sender == "Rohan"
    assert m.text == "check this: https://example.com/thing"


# ----------------------------------------------------------- date order ---


def test_unambiguous_day_first_detected(android):
    assert android.date_order == "day_first"
    assert "exceeds 12" in android.date_order_evidence


def test_month_first_detected_when_field2_exceeds_12():
    chat = parse_text("02/26/2025, 09:15 - A: hi\n")
    assert chat.date_order == "month_first"
    assert chat.messages[0].ts == datetime(2025, 2, 26, 9, 15)


def test_ambiguous_dates_resolved_by_monotonicity():
    """All fields <= 12, so only chronological ordering can disambiguate."""
    chat = parse_text(
        "01/02/2025, 10:00 - A: one\n"
        "05/02/2025, 10:00 - A: two\n"
        "09/02/2025, 10:00 - A: three\n"
    )
    assert chat.date_order == "day_first"
    assert [m.ts.day for m in chat.messages] == [1, 5, 9]


def test_two_digit_year_expanded():
    chat = parse_text("26/02/25, 09:15 - A: hi\n")
    assert chat.messages[0].ts.year == 2025


def test_date_order_is_decided_once_per_file(android):
    """Every message in a file shares one interpretation -- no per-line guessing."""
    march = next(m for m in android.messages if m.ts.month == 3)
    assert march.ts.day == 13, "13/03 under day-first is 13 March"


# --------------------------------------------------------------- shape ---


def test_participants_counted(android):
    assert set(android.participants) == {"Mehul", "Rohan", "Priya"}


def test_content_messages_exclude_noise(android):
    for m in android.content_messages:
        assert not (m.is_system or m.is_media or m.is_tombstone)
        assert m.text.strip()


def test_no_unparsed_lines_in_fixtures(android, ios):
    assert android.unparsed_lines == 0
    assert ios.unparsed_lines == 0


def test_empty_input_is_safe():
    chat = parse_text("")
    assert chat.messages == []
    assert chat.date_order == "day_first"
