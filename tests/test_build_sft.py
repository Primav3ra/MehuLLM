"""Dataset builder tests.

The two failures that would matter most are both silent:
  * TRAIN/VAL LEAKAGE -- splitting by message instead of by chat inflates every
    downstream metric and you would never see it.
  * STYLE COLLAPSE -- short replies either deleted (model writes essays) or
    left uncapped (model answers everything with "ok").
Both get explicit tests.
"""

import json

import pytest

from mehullm.pipeline.build_sft import BUCKET_CAPS, MAX_DUPLICATE_SHARE, build


def _write_chat(d, name: str, lines: list[tuple[str, str]], start_day: int = 1) -> None:
    """Each (other, me) tuple becomes one exchange on its own day."""
    out = []
    for i, (other, me) in enumerate(lines):
        day = start_day + i
        out.append(f"{day:02d}/01/2025, 10:00 - Rohan: {other}")
        out.append(f"{day:02d}/01/2025, 10:01 - Mehul: {me}")
    (d / f"{name}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


def _load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Targets are unique PER CHAT. Real people do not send byte-identical
    # messages across different conversations, and reusing text here would let
    # the cross-chat duplicate cap shred the fixture and mask the real bug.
    for c in range(6):
        _write_chat(
            raw,
            f"chat{c}",
            [(f"question {c}-{i} here", f"chat{c} reply number {i} with enough words")
             for i in range(25)],
        )
    return raw


# ------------------------------------------------------------- basics ---


def test_builds_and_writes_jsonl(corpus, tmp_path):
    out = tmp_path / "pairs.jsonl"
    stats = build(corpus, out, {"Mehul"})
    assert out.exists()
    recs = _load(out)
    assert len(recs) == stats.pairs_out > 0
    assert {"chat_id", "split", "context", "target", "bucket", "n_words", "ts"} <= recs[0].keys()


def test_context_precedes_target(corpus, tmp_path):
    out = tmp_path / "p.jsonl"
    build(corpus, out, {"Mehul"})
    r = _load(out)[0]
    assert r["context"], "every pair must carry context"
    assert "question" in r["context"][-1]["text"]


def test_deterministic_under_same_seed(corpus, tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    build(corpus, a, {"Mehul"}, seed=7)
    build(corpus, b, {"Mehul"}, seed=7)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


# ------------------------------------------------- SPLIT INTEGRITY ---


def test_no_chat_appears_in_two_splits(corpus, tmp_path):
    """The leakage test. A chat in both train and val invalidates every metric."""
    out = tmp_path / "p.jsonl"
    build(corpus, out, {"Mehul"})
    by_chat = {}
    for r in _load(out):
        by_chat.setdefault(r["chat_id"], set()).add(r["split"])
    bad = {c: s for c, s in by_chat.items() if len(s) > 1}
    assert not bad, f"chats split across sets: {bad}"


def test_heldout_chats_are_whole_and_excluded(corpus, tmp_path):
    out = tmp_path / "p.jsonl"
    stats = build(corpus, out, {"Mehul"})
    assert len(stats.heldout_chats) == 2
    recs = _load(out)
    held = {r["chat_id"] for r in recs if r["split"] == "heldout"}
    assert held == set(stats.heldout_chats)
    for r in recs:
        if r["chat_id"] in held:
            assert r["split"] == "heldout"


def test_all_three_splits_present(corpus, tmp_path):
    stats = build(corpus, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.per_split["train"] > 0
    assert stats.per_split["val"] > 0
    assert stats.per_split["heldout"] > 0


# --------------------------------------------- SHORT-REPLY HANDLING ---


def test_short_replies_are_kept_not_deleted(tmp_path):
    """They ARE the style. Deleting them produces a model that writes essays."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(4):
        _write_chat(
            raw,
            f"c{c}",
            [("kya haal", "ok"), ("chalega?", "haan"), ("sun na", "hmm")] * 6
            + [("explain karo", f"chat{c} much longer reply number {i} with many words")
               for i in range(30)],
        )
    out = tmp_path / "p.jsonl"
    build(raw, out, {"Mehul"})
    shorts = [r for r in _load(out) if r["bucket"] == "1-2"]
    assert shorts, "short replies must survive -- they are the voice"


def test_short_bucket_is_capped(tmp_path):
    """Uncapped, the model answers everything with 'ok'.

    Every target is unique so ONLY the bucket cap can bind -- otherwise the
    duplicate cap fires first and this test would pass for the wrong reason.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(4):
        _write_chat(
            raw,
            f"c{c}",
            [("q", f"ok{c}x{i}") for i in range(160)]           # flood of 1-word replies
            + [("q", f"chat{c} longer reply {i} with plenty of words here")
               for i in range(40)],
        )
    out = tmp_path / "p.jsonl"
    stats = build(raw, out, {"Mehul"})
    recs = _load(out)
    share = sum(1 for r in recs if r["bucket"] == "1-2") / len(recs)
    assert share <= BUCKET_CAPS["1-2"] + 0.01, f"1-2 bucket at {share:.1%}, cap is 15%"
    assert stats.dropped_bucket_cap > 0


def test_duplicate_target_is_capped(tmp_path):
    """Here duplicates ARE the point, so 'ok' is deliberately identical."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(4):
        _write_chat(
            raw, f"c{c}",
            [("q", "ok") for _ in range(100)]
            + [("q", f"chat{c} distinct reply {i} with several words in it")
               for i in range(100)],
        )
    out = tmp_path / "p.jsonl"
    stats = build(raw, out, {"Mehul"})
    recs = _load(out)
    n_ok = sum(1 for r in recs if r["target"].strip().casefold() == "ok")
    # 800 candidate pairs -> cap is int(800 * 0.005) = 4
    assert n_ok <= 5, f"'ok' survived {n_ok} times; cap should be ~4"
    assert stats.dropped_duplicate_cap > 0
    assert MAX_DUPLICATE_SHARE == 0.005


def test_emoji_only_reply_survives(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(4):
        _write_chat(raw, f"c{c}",
                    [("dekh", "😂😂")] + [("q", f"chat{c} reply {i} with words") for i in range(20)])
    out = tmp_path / "p.jsonl"
    build(raw, out, {"Mehul"})
    assert any("😂" in r["target"] for r in _load(out)), "emoji-only replies are style data"


# ---------------------------------------------------- PII / NAMES ---


def test_pii_is_scrubbed_from_target_and_context(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(4):
        _write_chat(
            raw, f"c{c}",
            [("mera number 9876543210 hai", "mail me at test.person@example.com")]
            + [("q", f"chat{c} reply {i} with several words") for i in range(20)],
        )
    out = tmp_path / "p.jsonl"
    stats = build(raw, out, {"Mehul"})
    blob = out.read_text(encoding="utf-8")
    assert "9876543210" not in blob
    assert "test.person@example.com" not in blob
    assert stats.pii["PHONE"] >= 1 and stats.pii["EMAIL"] >= 1


def test_other_names_pseudonymised_but_owner_kept(corpus, tmp_path):
    out = tmp_path / "p.jsonl"
    nm = tmp_path / "nm.json"
    build(corpus, out, {"Mehul"}, name_map_path=nm)
    senders = {t["sender"] for r in _load(out) for t in r["context"]}
    assert all(s.startswith("Person_") for s in senders), f"unmapped senders: {senders}"
    assert nm.exists()


def test_style_survives_the_full_pipeline(tmp_path):
    """End-to-end guard: slang must reach the JSONL byte-identical."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(4):
        _write_chat(
            raw, f"c{c}",
            [("kya scene", "yaaar fr fr no cap 100% chalega 😎")]
            + [("q", f"chat{c} reply {i} with several words") for i in range(20)],
        )
    out = tmp_path / "p.jsonl"
    build(raw, out, {"Mehul"})
    assert any(r["target"] == "yaaar fr fr no cap 100% chalega 😎" for r in _load(out))


# ---------------------------------------------------------- drops ---


def test_placeholder_only_target_dropped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for c in range(4):
        _write_chat(
            raw, f"c{c}",
            [("number?", "9876543210")]
            + [("q", f"chat{c} reply {i} with several words") for i in range(20)],
        )
    out = tmp_path / "p.jsonl"
    stats = build(raw, out, {"Mehul"})
    assert stats.dropped_placeholder_only >= 1
    assert not any(r["target"].strip() == "<PHONE>" for r in _load(out))


def test_overlong_target_dropped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    spam = " ".join(f"w{i}" for i in range(700))
    for c in range(4):
        _write_chat(raw, f"c{c}",
                    [("forward", spam)] + [("q", f"chat{c} reply {i} words here") for i in range(20)])
    stats = build(raw, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.dropped_too_long >= 1


def test_empty_corpus_is_safe(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    stats = build(raw, tmp_path / "p.jsonl", {"Mehul"})
    assert stats.pairs_out == 0
