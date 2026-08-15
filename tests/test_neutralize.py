"""Neutraliser tests.

Ollama is faked at the HTTP layer with respx, so these run in CI with no model
installed and no GPU. The properties that matter:

  * the cache makes re-runs free (this batch is hours long -- a crash must not
    cost the whole night)
  * implausible generations never reach the dataset
  * sampling does not let one huge chat dominate
  * the emitted rows are in the exact shape trl's SFTTrainer expects
"""

import json

import httpx
import pytest
import respx

from mehullm.pipeline.neutralize import (
    SYSTEM_PROMPT,
    VARIANT_B_SHARE,
    DraftCache,
    _is_degenerate,
    _plausible,
    _stratified_sample,
    neutralize,
)
from mehullm.voice.client import OllamaClient, OllamaError, strip_thinking

OLLAMA = "http://localhost:11434"


def _pairs_file(tmp_path, n=40, chats=("a", "b", "c"), split="train"):
    p = tmp_path / "pairs.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "chat_id": chats[i % len(chats)],
                        "split": split,
                        "context": [{"sender": "Person_A", "text": f"question {i}"}],
                        "target": f"reply number {i} yaar",
                        "bucket": "6-12",
                        "n_words": 4,
                        "ts": "2025-01-01T10:00:00",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return p


def _mock_ollama(response="This is a neutral rewritten sentence."):
    """Neutralisation goes through /api/chat, not /api/generate.

    Few-shot examples on the completion endpoint get CONTINUED rather than
    treated as examples -- a real smoke test showed the model emitting every
    example answer concatenated together.
    """
    respx.get(f"{OLLAMA}/api/version").mock(return_value=httpx.Response(200, json={"version": "x"}))
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3:1.7b"}]})
    )
    respx.post(f"{OLLAMA}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": response})
    )
    return respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": response}})
    )


# ------------------------------------------------------- ollama client ---


def test_strip_thinking_removes_block():
    assert strip_thinking("<think>hmm</think>hello") == "hello"


def test_strip_thinking_handles_unclosed_block():
    """num_predict can cut generation off mid-thought."""
    assert strip_thinking("<think>hmm and then") == ""


def test_strip_thinking_leaves_normal_text():
    assert strip_thinking("bhai sab badhiya 😎") == "bhai sab badhiya 😎"


@respx.mock
def test_preflight_fails_when_model_missing():
    respx.get(f"{OLLAMA}/api/version").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{OLLAMA}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    with pytest.raises(OllamaError, match="not installed"):
        OllamaClient().preflight()


@respx.mock
def test_preflight_fails_when_server_down():
    respx.get(f"{OLLAMA}/api/version").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(OllamaError, match="not responding"):
        OllamaClient().preflight()


@respx.mock
def test_generate_strips_thinking_from_response():
    _mock_ollama("<think>plan</think>The answer.")
    assert OllamaClient().generate("hi") == "The answer."


# -------------------------------------------------------------- cache ---


def test_cache_roundtrip(tmp_path):
    c = DraftCache(tmp_path / "c.db")
    k = DraftCache.key("m", "A", "text")
    assert c.get(k) is None
    c.put(k, "A", "draft")
    assert c.get(k) == "draft"


def test_cache_key_varies_by_variant_and_model():
    assert DraftCache.key("m", "A", "t") != DraftCache.key("m", "B", "t")
    assert DraftCache.key("m1", "A", "t") != DraftCache.key("m2", "A", "t")


@respx.mock
def test_second_run_is_fully_cached(tmp_path):
    """A crashed overnight batch must not have to redo its work."""
    route = _mock_ollama()
    pairs = _pairs_file(tmp_path, n=20)
    cache = tmp_path / "c.db"

    s1 = neutralize(pairs, tmp_path / "o1.jsonl", cache_path=cache, train_limit=20, concurrency=1)
    calls = route.call_count
    assert s1.generated > 0

    s2 = neutralize(pairs, tmp_path / "o2.jsonl", cache_path=cache, train_limit=20, concurrency=1)
    assert s2.cache_hits == s1.generated
    assert s2.generated == 0
    assert route.call_count == calls, "no new generation calls on a cached re-run"


# ------------------------------------------------- plausibility filter ---


@pytest.mark.parametrize(
    "draft",
    [
        "",
        "  ",
        "I cannot help with that.",
        "I'm sorry, but I can't.",
        "As an AI language model, I ...",
        "Translate the following Hinglish message into formal English.",  # echoed the prompt
        "Reply with the English translation only.",                       # echoed the prompt
        "x" * 5000,  # runaway
    ],
)
def test_implausible_drafts_rejected(draft):
    assert not _plausible(draft, "short target")


def test_plausible_draft_accepted():
    assert _plausible("I will confirm by tonight.", "raat tak bata dunga")


# ------------------------------------------------- DEGENERACY (critical) ---
# If the draft equals the reply, the pair teaches the identity function: the
# LoRA learns nothing, training loss looks healthy, and the failure only shows
# up as "the fine-tune was no better than the base model". The first prompt
# version produced exactly this for nearly every input.


@pytest.mark.parametrize(
    "draft,target",
    [
        ("raat tak bata dunga pakka", "raat tak bata dunga pakka"),   # identical
        ("Raat tak bata dunga pakka", "raat tak bata dunga pakka"),   # casing only
        ("raat tak bata dunga pakka.", "raat tak bata dunga pakka"),  # punctuation only
        ("ok", "ok"),
        ("nahi yaar aaj nahi ho payega", "nahi yaar aaj nahi ho payega, kal milte hain"),
    ],
)
def test_degenerate_drafts_rejected(draft, target):
    assert _is_degenerate(draft, target)
    assert not _plausible(draft, target)


def test_genuine_translation_is_not_degenerate():
    assert not _is_degenerate("Please transfer 500 rupees to <UPI>.", "bhai 500 rs bhej de <UPI> pe")


@respx.mock
def test_degenerate_drafts_never_reach_the_dataset(tmp_path):
    """Mock echoes the target back -- nothing may be written."""
    respx.get(f"{OLLAMA}/api/version").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3:1.7b"}]})
    )
    respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": "reply number 0 yaar"}}
        )
    )
    out = tmp_path / "o.jsonl"
    stats = neutralize(
        _pairs_file(tmp_path, n=6), out, cache_path=tmp_path / "c.db",
        train_limit=6, concurrency=1,
    )
    assert stats.rejected >= 1
    written = [x for x in out.read_text(encoding="utf-8").splitlines() if x]
    for row in written:
        r = json.loads(row)
        assert r["messages"][1]["content"] != r["messages"][2]["content"]


@respx.mock
def test_rejected_drafts_never_reach_the_dataset(tmp_path):
    _mock_ollama("I cannot help with that.")
    out = tmp_path / "o.jsonl"
    stats = neutralize(
        _pairs_file(tmp_path, n=10), out, cache_path=tmp_path / "c.db",
        train_limit=10, concurrency=1,
    )
    assert stats.rejected == 10
    assert out.read_text(encoding="utf-8").strip() == ""


# ----------------------------------------------------------- sampling ---


def test_stratified_sample_spreads_across_chats():
    """One conversation is ~40% of the real corpus; it must not own the sample."""
    import random

    recs = [{"chat_id": "huge", "bucket": "6-12"} for _ in range(900)]
    recs += [{"chat_id": c, "bucket": "6-12"} for c in ("s1", "s2", "s3") for _ in range(40)]
    picked = _stratified_sample(recs, 60, random.Random(1))
    share_huge = sum(1 for r in picked if r["chat_id"] == "huge") / len(picked)
    assert share_huge < 0.75, f"one chat took {share_huge:.0%} of the sample"
    assert len({r["chat_id"] for r in picked}) == 4


def test_stratified_sample_preserves_bucket_mix():
    import random

    recs = [{"chat_id": "a", "bucket": "1-2"} for _ in range(300)]
    recs += [{"chat_id": "a", "bucket": "6-12"} for _ in range(700)]
    picked = _stratified_sample(recs, 100, random.Random(1))
    short = sum(1 for r in picked if r["bucket"] == "1-2") / len(picked)
    assert 0.2 <= short <= 0.4, f"1-2 bucket at {short:.0%}, expected ~30%"


def test_sample_returns_all_when_under_limit():
    import random

    recs = [{"chat_id": "a", "bucket": "6-12"} for _ in range(5)]
    assert len(_stratified_sample(recs, 100, random.Random(1))) == 5


# ------------------------------------------------------- output shape ---


@respx.mock
def test_output_is_trl_messages_format(tmp_path):
    _mock_ollama()
    out = tmp_path / "o.jsonl"
    neutralize(_pairs_file(tmp_path, n=8), out, cache_path=tmp_path / "c.db",
               train_limit=8, concurrency=1)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    assert rows
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert r["messages"][0]["content"] == SYSTEM_PROMPT
        assert "<draft>" in r["messages"][1]["content"]
        assert "<context>" in r["messages"][1]["content"]
        assert r["variant"] in {"A", "B"}


@respx.mock
def test_assistant_turn_is_the_untouched_real_reply(tmp_path):
    """The target is the label. If the pipeline mangles it, the model learns wrong."""
    _mock_ollama()
    out = tmp_path / "o.jsonl"
    neutralize(_pairs_file(tmp_path, n=6), out, cache_path=tmp_path / "c.db",
               train_limit=6, concurrency=1)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    for r in rows:
        assert r["messages"][-1]["content"].startswith("reply number ")
        assert r["messages"][-1]["content"].endswith(" yaar")


@respx.mock
def test_both_variants_are_produced(tmp_path):
    _mock_ollama()
    out = tmp_path / "o.jsonl"
    stats = neutralize(_pairs_file(tmp_path, n=200), out, cache_path=tmp_path / "c.db",
                       train_limit=200, concurrency=1)
    assert stats.variants["A"] > 0 and stats.variants["B"] > 0
    b_share = stats.variants["B"] / (stats.variants["A"] + stats.variants["B"])
    assert abs(b_share - VARIANT_B_SHARE) < 0.15, f"B share {b_share:.0%}"


@respx.mock
def test_heldout_split_gets_no_drafts(tmp_path):
    """Held-out is scored against real messages; drafts there are wasted compute."""
    _mock_ollama()
    pairs = _pairs_file(tmp_path, n=20, split="heldout")
    stats = neutralize(pairs, tmp_path / "o.jsonl", cache_path=tmp_path / "c.db",
                       train_limit=20, concurrency=1)
    assert stats.selected == 0
