"""Corpus census -- the week-2 go/no-go gate.

Answers the one question that decides whether this project's fine-tuning arm
is viable at all: **how many usable (context -> my reply) pairs exist?**

    >= 8000   green  -- proceed with the LoRA as planned
    3000-8000 amber  -- proceed, but expect a weaker delta over few-shot
    <  3000   red    -- pivot: few-shot style exemplars carry the voice layer,
                        and the capstone reframes around memory + evaluation

Measuring this in week 2 rather than week 10 is the whole point. Also reports
the script mix, which is what justifies the multilingual embedding model and
the Qwen tokenizer choice in the write-up.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import regex

from mehullm.pipeline.sessionize import build_pairs, merge_bursts, split_sessions
from mehullm.pipeline.whatsapp_parser import Chat, parse_export

_DEVANAGARI = regex.compile(r"\p{Devanagari}")
_LATIN = regex.compile(r"\p{Latin}")
_EMOJI = regex.compile(r"[\p{Emoji_Presentation}\p{Extended_Pictographic}]")
_WORD = regex.compile(r"[\p{L}\p{N}']+")

# Common Hindi function words written in Latin script. A crude but effective
# code-switch detector -- these are the words that make Hinglish *Hinglish*.
_HINGLISH_LEXEMES = {
    "hai", "hain", "nahi", "nahin", "kya", "kyu", "kyun", "kaise", "kaisa", "acha",
    "accha", "theek", "thik", "yaar", "yar", "bhai", "bhaiya", "haan", "han", "ha",
    "abhi", "kal", "aaj", "raha", "rahi", "rahe", "kar", "karo", "karna", "gaya",
    "gayi", "hua", "hui", "mera", "meri", "tera", "teri", "apna", "bas", "phir",
    "fir", "matlab", "chal", "chalo", "dekh", "dekho", "sun", "suno", "bata",
    "batao", "mujhe", "tujhe", "humko", "tumko", "koi", "kuch", "sab", "bohot",
    "bahut", "thoda", "zyada", "jyada", "pakka", "sahi", "galat", "arre", "arey",
    "toh", "to", "na", "ke", "ki", "ka", "se", "me", "mein", "par", "aur", "ya",
}


@dataclass
class ChatStats:
    chat_id: str
    total_lines: int = 0
    messages: int = 0
    content: int = 0
    system: int = 0
    media: int = 0
    tombstone: int = 0
    unparsed: int = 0
    senders: Counter[str] = field(default_factory=Counter)
    date_order: str = ""
    is_group: bool = False


@dataclass
class Census:
    chats: list[ChatStats] = field(default_factory=list)
    likely_self: str = ""
    self_confidence: str = ""
    pairs_1to1: int = 0
    pairs_group: int = 0
    my_turns: int = 0
    my_msg_lengths: list[int] = field(default_factory=list)
    devanagari_msgs: int = 0
    hinglish_msgs: int = 0
    emoji_msgs: int = 0
    burst_sizes: list[int] = field(default_factory=list)

    @property
    def verdict(self) -> tuple[str, str]:
        n = self.pairs_1to1
        if n >= 8000:
            return "GREEN", "Proceed with the LoRA as planned."
        if n >= 3000:
            return "AMBER", "Proceed, but expect a smaller delta over few-shot prompting."
        return "RED", "Pivot: few-shot style exemplars carry the voice layer instead."


def _classify(text: str, c: Census) -> None:
    if _DEVANAGARI.search(text):
        c.devanagari_msgs += 1
    if _EMOJI.search(text):
        c.emoji_msgs += 1
    words = {w.casefold() for w in _WORD.findall(text)}
    if words & _HINGLISH_LEXEMES and _LATIN.search(text):
        c.hinglish_msgs += 1


def _detect_self(chats: list[Chat]) -> tuple[str, str]:
    """Guess the owner: the sender who appears in the most distinct chats.

    You are, by construction, a participant in every one of your own chats.
    This is only a *suggestion* -- the user confirms it via contacts.json,
    because getting it wrong would silently train on someone else's voice.
    """
    appearances: Counter[str] = Counter()
    for chat in chats:
        for sender in {m.sender for m in chat.messages if m.sender}:
            appearances[sender] += 1
    if not appearances:
        return "", "no senders found"
    ranked = appearances.most_common()
    top, n = ranked[0]
    if len(ranked) == 1:
        return top, f"only sender, in {n}/{len(chats)} chats"
    runner_up = ranked[1][1]
    conf = "high" if n > runner_up else "LOW - ambiguous, confirm manually"
    return top, f"{conf} (in {n}/{len(chats)} chats; next best {runner_up})"


def run_census(root: str | Path, self_alias: str | None = None) -> Census:
    root = Path(root)
    # Deduplicate by resolved path: Windows globbing is case-insensitive, so
    # rglob("*.txt") and rglob("*.TXT") each return every file and every chat
    # would be counted twice.
    files = sorted({f.resolve() for f in root.rglob("*") if f.suffix.lower() == ".txt"})
    parsed = [parse_export(f) for f in files]
    c = Census()

    if self_alias:
        c.likely_self, c.self_confidence = self_alias, "supplied by user"
    else:
        c.likely_self, c.self_confidence = _detect_self(parsed)

    self_aliases = {c.likely_self} if c.likely_self else set()

    for chat in parsed:
        cs = ChatStats(chat_id=chat.chat_id, date_order=chat.date_order)
        cs.unparsed = chat.unparsed_lines
        for m in chat.messages:
            cs.messages += 1
            if m.is_system:
                cs.system += 1
            elif m.is_media:
                cs.media += 1
            elif m.is_tombstone:
                cs.tombstone += 1
            elif m.is_content:
                cs.content += 1
                cs.senders[m.sender] += 1
                _classify(m.text, c)
        cs.is_group = len(cs.senders) > 2
        c.chats.append(cs)

        turns = merge_bursts(chat.content_messages)
        c.burst_sizes.extend(t.n_messages for t in turns)
        for t in turns:
            if c.likely_self and t.sender.casefold() == c.likely_self.casefold():
                c.my_turns += 1
                c.my_msg_lengths.append(len(_WORD.findall(t.text)))

        pairs = build_pairs(split_sessions(turns), self_aliases, chat.chat_id)
        if cs.is_group:
            c.pairs_group += len(pairs)
        else:
            c.pairs_1to1 += len(pairs)

    return c


def format_report(c: Census) -> str:
    if not c.chats:
        return "No .txt exports found. Put WhatsApp exports in data/raw/ and re-run."

    total_content = sum(x.content for x in c.chats)
    n_1to1 = sum(1 for x in c.chats if not x.is_group)
    L = c.my_msg_lengths
    verdict, advice = c.verdict

    def pct(n: int, d: int) -> str:
        return f"{100 * n / d:5.1f}%" if d else "    -"

    lines = [
        "=" * 66,
        "  MehuLLM corpus census",
        "=" * 66,
        "",
        f"  Chats parsed        {len(c.chats)}  ({n_1to1} one-to-one, {len(c.chats) - n_1to1} group)",
        f"  Content messages    {total_content:,}",
        f"  System / media      {sum(x.system for x in c.chats):,} / {sum(x.media for x in c.chats):,}",
        f"  Deleted             {sum(x.tombstone for x in c.chats):,}",
        f"  Unparsed lines      {sum(x.unparsed for x in c.chats):,}"
        + ("   <- investigate, should be 0" if sum(x.unparsed for x in c.chats) else "   (clean)"),
        "",
        f"  Detected 'me'       {c.likely_self or '(unknown)'}   [{c.self_confidence}]",
        f"  My turns            {c.my_turns:,}",
    ]

    if L:
        lines += [
            f"  My reply length     median {statistics.median(L):.0f} words, "
            f"mean {statistics.mean(L):.1f}, p90 {sorted(L)[int(0.9 * len(L))]}",
        ]
    if c.burst_sizes:
        merged = sum(1 for b in c.burst_sizes if b > 1)
        lines += [
            f"  Burst merging       {merged:,} of {len(c.burst_sizes):,} turns "
            f"were multi-message ({pct(merged, len(c.burst_sizes)).strip()})",
        ]

    lines += [
        "",
        "  Script / style mix (share of content messages)",
        f"    Hinglish          {pct(c.hinglish_msgs, total_content)}",
        f"    Devanagari        {pct(c.devanagari_msgs, total_content)}",
        f"    Contains emoji    {pct(c.emoji_msgs, total_content)}",
        "",
        "-" * 66,
        f"  USABLE SFT PAIRS (1:1)   {c.pairs_1to1:,}",
        f"  (group pairs, excluded)  {c.pairs_group:,}",
        "-" * 66,
        f"  VERDICT: {verdict}  -- {advice}",
        "",
        "  Thresholds:  >=8000 green   3000-8000 amber   <3000 red",
        "=" * 66,
        "",
        "  Per-chat breakdown (top 15 by content):",
    ]

    for x in sorted(c.chats, key=lambda x: -x.content)[:15]:
        kind = "group" if x.is_group else "1:1  "
        lines.append(f"    {kind}  {x.content:>7,}  {x.date_order:<12}  {x.chat_id[:40]}")

    if c.likely_self and "LOW" in c.self_confidence:
        lines += [
            "",
            "  !! 'me' detection was ambiguous. Confirm before building the dataset --",
            "     training on the wrong sender would learn someone else's voice.",
        ]
    return "\n".join(lines)
