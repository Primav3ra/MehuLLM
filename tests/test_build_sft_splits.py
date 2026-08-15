"""Regressions for two bugs the real 25 MB corpus exposed that the synthetic
fixtures did not:

  1. Group chats leaked into the SFT targets (2,259 of 90,122 pairs).
  2. The validation split overshot to 30.4% against a 15% target, because the
     greedy loop kept adding whole chats until it crossed the line.

Both only appear with realistically uneven chat sizes, which is why they are
tested separately with a deliberately lopsided corpus.
"""

import json

import pytest

from mehullm.pipeline.build_sft import VAL_SHARE, build


def _chat(d, name: str, n: int, senders: tuple[str, ...] = ("Rohan",)) -> None:
    """n exchanges, one per day. Extra senders make it a group chat."""
    out = []
    for i in range(n):
        day, month = (i % 28) + 1, (i // 28) + 1
        who = senders[i % len(senders)]
        out.append(f"{month:02d}/{day:02d}/2025, 10:00 - {who}: {name} q{i}")
        out.append(f"{month:02d}/{day:02d}/2025, 10:01 - Mehul: {name} reply {i} with some words")
    (d / f"{name}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


def _load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


@pytest.fixture
def lopsided(tmp_path):
    """Chat sizes roughly matching the real corpus: two dominant, several small."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _chat(raw, "huge", 300)
    _chat(raw, "big", 220)
    _chat(raw, "mid", 90)
    _chat(raw, "small", 40)
    _chat(raw, "tiny", 25)
    _chat(raw, "mini", 15)
    _chat(raw, "groupchat", 60, senders=("Rohan", "Priya", "Arjun"))
    return raw


# ----------------------------------------------------- group exclusion ---


def test_group_chats_excluded_from_sft(lopsided, tmp_path):
    out = tmp_path / "p.jsonl"
    stats = build(lopsided, out, {"Mehul"})
    assert "groupchat" in stats.skipped_group_chats
    assert not any(r["chat_id"] == "groupchat" for r in _load(out))


def test_group_exclusion_reduces_candidate_count(lopsided, tmp_path):
    """Group pairs must not be counted as candidates at all."""
    stats = build(lopsided, tmp_path / "p.jsonl", {"Mehul"})
    one_to_one = 300 + 220 + 90 + 40 + 25 + 15
    assert stats.pairs_in <= one_to_one, "group pairs leaked into pairs_in"


def test_two_person_chat_is_not_treated_as_group(lopsided, tmp_path):
    stats = build(lopsided, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.skipped_group_chats == ["groupchat"]


# --------------------------------------------------------- split sizes ---


def test_val_split_does_not_wildly_overshoot(lopsided, tmp_path):
    """The real corpus produced 30.4% against a 15% target."""
    stats = build(lopsided, tmp_path / "p.jsonl", {"Mehul"})
    non_heldout = stats.per_split["train"] + stats.per_split["val"]
    share = stats.per_split["val"] / non_heldout
    assert share <= VAL_SHARE * 2, f"val at {share:.1%}, target {VAL_SHARE:.0%}"


def test_val_is_never_empty(lopsided, tmp_path):
    stats = build(lopsided, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.per_split["val"] > 0


def test_train_remains_the_majority(lopsided, tmp_path):
    stats = build(lopsided, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.per_split["train"] > stats.per_split["val"]
    assert stats.per_split["train"] > stats.per_split["heldout"]


def test_single_chat_corpus_still_produces_a_split(tmp_path):
    """Degenerate case: one chat cannot be in three splits at once."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _chat(raw, "only", 60)
    stats = build(raw, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.pairs_out > 0
    assert sum(1 for v in stats.per_split.values() if v) >= 1


def test_all_group_corpus_yields_nothing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _chat(raw, "g1", 50, senders=("A", "B", "C"))
    stats = build(raw, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.pairs_out == 0
    assert stats.skipped_group_chats == ["g1"]
