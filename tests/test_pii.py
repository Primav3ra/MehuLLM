"""PII scrubber tests.

Two halves, and the second half matters as much as the first:
  1. does it catch real PII?
  2. does it LEAVE STYLE ALONE?

A scrubber that eats "100%" or "gn8" quietly degrades the fine-tune in a way
no aggregate metric would reveal.
"""

import pytest

from mehullm.pipeline.pii import NameMap, scrub, scrub_with_stats

# ------------------------------------------------------------- catches ---


@pytest.mark.parametrize(
    "raw,kind",
    [
        ("call me on 9876543210", "PHONE"),
        ("call me on +91 9876543210", "PHONE"),
        ("number is 09876543210", "PHONE"),
        ("mail me at mehul.k@example.co.in", "EMAIL"),
        ("pay to mehul@okicici na", "UPI"),
        ("my pan is ABCDE1234F", "PAN"),
        ("aadhaar 1234 5678 9012 hai", "AADHAAR"),
        ("car is MH12AB1234", "VEHICLE"),
        ("OTP is 449281 bhej", "OTP"),
        ("449281 is your OTP", "OTP"),
        ("card 4111 1111 1111 1111", "CARD"),
        ("pincode 400072 likh de", "PINCODE"),
        ("flat 4B mein aa ja", "ADDR"),
        ("dekh https://example.com/x?y=1", "URL"),
    ],
)
def test_catches_pii(raw, kind):
    out, stats = scrub_with_stats(raw)
    assert stats[kind] >= 1, f"{kind} not detected in {raw!r} -> {out!r}"
    assert f"<{kind}>" in out


def test_phone_digits_fully_removed():
    assert "9876543210" not in scrub("ping 9876543210 now")


def test_devanagari_digits_in_phone_are_caught():
    assert "<PHONE>" in scrub("call ९८७६५४३२१०")


# ------------------------------------------- STYLE MUST SURVIVE INTACT ---


@pytest.mark.parametrize(
    "text",
    [
        "fr fr no cap on god",
        "100% bhai",
        "gn8 tc",
        "2moro milte hain",
        "4u only",
        "yaaaar kya kar raha hai",
        "lmaooo 😂😂😂",
        "acha theek hai 👍",
        "मैं कल आ रहा हूँ पक्का",
        "brb 5 min",
        "ok 3 baje",
        "got 250109 views on that reel",
        "ist 2026 me milenge",
        "season 4 dekh raha hu",
    ],
)
def test_style_and_slang_untouched(text):
    assert scrub(text) == text, "scrubber must not touch authored style"


def test_bare_six_digit_number_is_not_an_otp():
    """A bare number is style data; only context words make it an OTP."""
    assert scrub("bhai 250109 log aaye the") == "bhai 250109 log aaye the"


def test_otp_far_from_keyword_is_still_caught():
    """Regression from a real leak scan: 2 OTPs survived because the keyword
    and the digits sat further apart than the tight pattern allowed."""
    out = scrub("your OTP for order 4471 placed on 12 March is 903112 do not share")
    assert "903112" not in out
    assert "4471" not in out


def test_otp_aggression_is_context_gated():
    """The aggressive rule must only fire in messages that mention an OTP."""
    assert scrub("order 4471 placed on 12 March, 903112 views") == (
        "order 4471 placed on 12 March, 903112 views"
    )


def test_bare_six_digit_number_is_not_a_pincode():
    assert scrub("score tha 400072 points") == "score tha 400072 points"


def test_short_numbers_never_matched():
    assert scrub("mai 21 saal ka hu, 10th me 92% aaye the") == (
        "mai 21 saal ka hu, 10th me 92% aaye the"
    )


def test_emoji_and_zwj_survive_scrubbing():
    t = "\U0001f469‍\U0001f4bb coding kar raha hu"
    assert scrub(t) == t


# ------------------------------------------------------ ordering rules ---


def test_email_wins_over_phone_inside_it():
    out = scrub("write to 9876543210@gmail.com")
    assert out.count("<") == 1, f"one placeholder expected, got {out!r}"
    assert "<EMAIL>" in out


def test_url_not_shredded_by_phone_pattern():
    out = scrub("dekh https://x.com/9876543210/post")
    assert out == "dekh <URL>"


# ------------------------------------------------------------- names ---


def test_name_map_is_consistent_across_calls():
    nm = NameMap()
    a = nm.pseudonymize("Rohan aa raha hai", ["Rohan"])
    b = nm.pseudonymize("bola Rohan ne", ["Rohan"])
    assert a.split()[0] == b.split()[1], "same name must map to the same alias"


def test_distinct_names_get_distinct_aliases():
    nm = NameMap()
    out = nm.pseudonymize("Rohan aur Priya aaye", ["Rohan", "Priya"])
    assert "Person_A" in out and "Person_B" in out


def test_owner_name_is_kept_verbatim():
    nm = NameMap(keep={"Mehul"})
    out = nm.pseudonymize("Mehul aur Rohan", ["Mehul", "Rohan"])
    assert "Mehul" in out, "the assistant must learn to answer to the owner's name"
    assert "Person_A" in out


def test_alias_generation_passes_26():
    nm = NameMap()
    names = [f"N{i}" for i in range(30)]
    out = nm.pseudonymize(" ".join(names), names)
    assert "Person_AA" in out, "alias sequence must not collide after Z"


def test_name_map_roundtrip(tmp_path):
    nm = NameMap(keep={"Mehul"})
    nm.pseudonymize("Rohan", ["Rohan"])
    p = tmp_path / "name_map.json"
    nm.save(p)
    assert NameMap.load(p).mapping == nm.mapping


def test_partial_name_not_replaced():
    nm = NameMap()
    out = nm.pseudonymize("Rohanpur gaya tha", ["Rohan"])
    assert out == "Rohanpur gaya tha", "word boundaries must prevent partial hits"
