"""Turn-merging, sessionisation and pair-construction tests.

The census reports its go/no-go number from exactly this code, so a bug here
would make the week-2 gate lie about project viability.
"""

from pathlib import Path

from mehullm.pipeline.census import run_census
from mehullm.pipeline.sessionize import build_pairs, merge_bursts, split_sessions
from mehullm.pipeline.whatsapp_parser import parse_text

FIXTURES = Path(__file__).parent / "fixtures"


def _chat(lines: str):
    return parse_text(lines)


# ------------------------------------------------------------- bursts ---


def test_consecutive_same_sender_merges_into_one_turn():
    chat = _chat(
        "01/01/2025, 10:00 - Mehul: haan yaar\n"
        "01/01/2025, 10:01 - Mehul: kaam khatam nahi ho raha\n"
        "01/01/2025, 10:02 - Mehul: bas thoda aur\n"
    )
    turns = merge_bursts(chat.content_messages)
    assert len(turns) == 1
    assert turns[0].n_messages == 3
    assert turns[0].text == "haan yaar\nkaam khatam nahi ho raha\nbas thoda aur"


def test_burst_breaks_after_window():
    chat = _chat(
        "01/01/2025, 10:00 - Mehul: one\n"
        "01/01/2025, 10:10 - Mehul: two\n"  # 10 min > 180 s
    )
    assert len(merge_bursts(chat.content_messages)) == 2


def test_burst_breaks_on_sender_change():
    chat = _chat(
        "01/01/2025, 10:00 - Mehul: hi\n"
        "01/01/2025, 10:00 - Rohan: hi\n"
        "01/01/2025, 10:01 - Mehul: sup\n"
    )
    turns = merge_bursts(chat.content_messages)
    assert [t.sender for t in turns] == ["Mehul", "Rohan", "Mehul"]


def test_media_and_system_excluded_from_turns():
    chat = _chat(
        "01/01/2025, 10:00 - Mehul: real\n"
        "01/01/2025, 10:01 - Mehul: <Media omitted>\n"
        "01/01/2025, 10:02 - Rohan added Priya\n"
    )
    turns = merge_bursts(chat.content_messages)
    assert len(turns) == 1 and turns[0].text == "real"


# ----------------------------------------------------------- sessions ---


def test_long_gap_starts_new_session():
    chat = _chat(
        "01/01/2025, 10:00 - Mehul: morning\n"
        "02/01/2025, 18:00 - Mehul: next day\n"  # > 6 h
    )
    assert len(split_sessions(merge_bursts(chat.content_messages))) == 2


def test_short_gap_stays_one_session():
    chat = _chat(
        "01/01/2025, 10:00 - Mehul: a\n"
        "01/01/2025, 13:00 - Rohan: b\n"  # 3 h < 6 h
    )
    assert len(split_sessions(merge_bursts(chat.content_messages))) == 1


# -------------------------------------------------------------- pairs ---


def _pairs(text: str, who="Mehul"):
    chat = _chat(text)
    return build_pairs(split_sessions(merge_bursts(chat.content_messages)), {who})


def test_valid_reply_produces_a_pair():
    pairs = _pairs(
        "01/01/2025, 10:00 - Rohan: kal aa raha hai?\n"
        "01/01/2025, 10:01 - Mehul: haan pakka\n"
    )
    assert len(pairs) == 1
    assert pairs[0].target.sender == "Mehul"
    assert pairs[0].context[-1].sender == "Rohan"


def test_first_turn_has_no_context_so_no_pair():
    assert _pairs("01/01/2025, 10:00 - Mehul: oye\n") == []


def test_stale_context_rejected():
    """Someone spoke, but 5 hours ago -- that is not a reply."""
    assert (
        _pairs(
            "01/01/2025, 10:00 - Rohan: sun\n"
            "01/01/2025, 15:00 - Mehul: haan bol\n"  # 5 h > 30 min
        )
        == []
    )


def test_monologue_rejected():
    """Consecutive self-turns with nobody else speaking are not replies."""
    assert (
        _pairs(
            "01/01/2025, 10:00 - Mehul: hello\n"
            "01/01/2025, 10:30 - Mehul: ?\n"
            "01/01/2025, 11:00 - Mehul: koi hai\n"
        )
        == []
    )


def test_self_match_is_case_insensitive():
    assert len(_pairs(
        "01/01/2025, 10:00 - Rohan: yo\n"
        "01/01/2025, 10:01 - mehul: yo\n",
        who="Mehul",
    )) == 1


def test_context_window_is_bounded():
    lines = "".join(
        f"01/01/2025, 10:{i:02d} - Rohan: msg{i}\n" for i in range(20)
    ) + "01/01/2025, 10:21 - Mehul: reply\n"
    pairs = _pairs(lines)
    assert len(pairs) == 1
    assert len(pairs[0].context) <= 6, "context must be capped at CONTEXT_TURNS"


# ------------------------------------------------------------- census ---


def test_census_does_not_double_count_files(tmp_path):
    """Windows globbing is case-insensitive; *.txt and *.TXT both match."""
    (tmp_path / "a.txt").write_text(
        "01/01/2025, 10:00 - Rohan: hi\n01/01/2025, 10:01 - Mehul: hey\n", encoding="utf-8"
    )
    (tmp_path / "b.TXT").write_text("01/01/2025, 10:00 - Rohan: yo\n", encoding="utf-8")
    c = run_census(tmp_path, self_alias="Mehul")
    assert len(c.chats) == 2, f"expected 2 chats, got {len(c.chats)}"


def test_census_counts_pairs_and_verdict(tmp_path):
    convo = "".join(
        f"0{d}/01/2025, 10:00 - Rohan: q{d}\n0{d}/01/2025, 10:01 - Mehul: a{d}\n"
        for d in range(1, 6)
    )
    (tmp_path / "chat.txt").write_text(convo, encoding="utf-8")
    c = run_census(tmp_path, self_alias="Mehul")
    assert c.pairs_1to1 == 5
    assert c.verdict[0] == "RED", "5 pairs must not pass the gate"


def test_census_detects_self_across_chats(tmp_path):
    """The owner appears in every chat; nobody else does."""
    for i, other in enumerate(["Rohan", "Priya", "Arjun"]):
        (tmp_path / f"c{i}.txt").write_text(
            f"01/01/2025, 10:00 - {other}: hi\n01/01/2025, 10:01 - Mehul: hey\n",
            encoding="utf-8",
        )
    c = run_census(tmp_path)
    assert c.likely_self == "Mehul"
    assert "high" in c.self_confidence


def test_census_handles_empty_directory(tmp_path):
    c = run_census(tmp_path)
    assert c.chats == []
    assert "No .txt exports found" in __import__(
        "mehullm.pipeline.census", fromlist=["format_report"]
    ).format_report(c)


def test_group_chats_excluded_from_1to1_pairs():
    c = run_census(FIXTURES, self_alias="Mehul")
    group = [x for x in c.chats if x.is_group]
    assert group, "the android fixture has 3 senders and must count as a group"
