"""Objective style-similarity score."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import regex

__all__ = ["StyleProfile", "StyleScore", "compare", "normalised", "profile"]

_WORD = regex.compile(r"[\p{L}\p{N}']+")
_DEVANAGARI = regex.compile(r"\p{Devanagari}")
_EMOJI = regex.compile(r"[\p{Emoji_Presentation}\p{Extended_Pictographic}]")
_LAUGH = regex.compile(r"\b(?:ha(?:ha)+h?|hehe+|lol+|lmao+|rofl)\b", regex.IGNORECASE)
_REPEAT = regex.compile(r"(\p{L})\1{2,}")  # yaaar, cuteee
_ALLCAPS = regex.compile(r"\b[\p{Lu}]{2,}\b")

# Hindi function words written in Latin script.
HINGLISH = {
    "hai",
    "hain",
    "nahi",
    "nahin",
    "kya",
    "kyu",
    "kyun",
    "kaise",
    "kaisa",
    "acha",
    "accha",
    "theek",
    "thik",
    "yaar",
    "yar",
    "bhai",
    "bhaiya",
    "haan",
    "abhi",
    "kal",
    "aaj",
    "raha",
    "rahi",
    "rahe",
    "karo",
    "karna",
    "gaya",
    "gayi",
    "hua",
    "hui",
    "mera",
    "meri",
    "tera",
    "teri",
    "apna",
    "phir",
    "matlab",
    "chal",
    "chalo",
    "dekh",
    "dekho",
    "suno",
    "bata",
    "batao",
    "mujhe",
    "tujhe",
    "koi",
    "kuch",
    "bohot",
    "bahut",
    "thoda",
    "zyada",
    "jyada",
    "pakka",
    "sahi",
    "galat",
    "arre",
    "arey",
    "toh",
    "mein",
    "aur",
    "kyunki",
    "lekin",
    "magar",
    "waise",
    "bilkul",
    "shayad",
    "zaroor",
    "milte",
    "jaana",
    "aana",
    "khana",
    "paisa",
    "paise",
    "ghar",
    "kaam",
    "baje",
    "wala",
    "wali",
    "hoga",
    "hogi",
    "tha",
    "thi",
    "kiya",
    "diya",
    "liya",
    "rakh",
    "bol",
    "bolo",
    "samajh",
    "pata",
    "yaad",
}

# Excluded on purpose -- ambiguous with common English:

BUCKETS = [(1, 2), (3, 5), (6, 12), (13, 30), (31, 10**9)]

WEIGHTS = {"C": 0.25, "L": 0.20, "P": 0.15, "N": 0.15, "X": 0.15, "E": 0.10}


def _bucket(n: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n <= hi:
            return i
    return len(BUCKETS) - 1


@dataclass
class StyleProfile:
    n: int = 0
    hinglish_rate: float = 0.0  # share of messages using Hindi lexemes
    devanagari_ratio: float = 0.0  # share of chars in Devanagari
    length_hist: list[float] = field(default_factory=lambda: [0.0] * len(BUCKETS))
    punct: dict[str, float] = field(default_factory=dict)
    emoji_rate: float = 0.0  # emoji per message
    ngrams: Counter = field(default_factory=Counter)


def profile(messages: list[str]) -> StyleProfile:
    msgs = [m for m in messages if m and m.strip()]
    if not msgs:
        return StyleProfile()

    p = StyleProfile(n=len(msgs))
    hinglish_tokens = total_tokens = 0
    deva_chars = total_chars = emoji = 0
    lower_start = terminal_dot = q = ex = ell = caps = rep = laugh = 0
    hist = [0] * len(BUCKETS)
    grams: Counter = Counter()

    for m in msgs:
        words = _WORD.findall(m)
        low = [w.casefold() for w in words]
        # Token FRACTION, not "contains any". A per-message boolean saturates:
        total_tokens += len(low)
        hinglish_tokens += sum(1 for w in low if w in HINGLISH)

        total_chars += len(m)
        deva_chars += len(_DEVANAGARI.findall(m))
        emoji += len(_EMOJI.findall(m))
        hist[_bucket(len(words))] += 1

        s = m.strip()
        if s and s[0].islower():
            lower_start += 1
        if s.endswith("."):
            terminal_dot += 1
        q += m.count("?")
        ex += m.count("!")
        ell += m.count("...")
        caps += len(_ALLCAPS.findall(m))
        rep += len(_REPEAT.findall(m))
        laugh += len(_LAUGH.findall(m))

        for k in (1, 2, 3):
            for i in range(len(low) - k + 1):
                grams[" ".join(low[i : i + k])] += 1

    n = p.n
    p.hinglish_rate = hinglish_tokens / max(1, total_tokens)
    p.devanagari_ratio = deva_chars / max(1, total_chars)
    p.length_hist = [h / n for h in hist]
    p.emoji_rate = emoji / n
    p.punct = {
        "lower_start": lower_start / n,
        "terminal_dot": terminal_dot / n,
        "question": q / n,
        "exclaim": ex / n,
        "ellipsis": ell / n,
        "allcaps": caps / n,
        "char_repeat": rep / n,
        "laugh": laugh / n,
    }
    p.ngrams = Counter(dict(grams.most_common(200)))
    return p


def _jsd(a: list[float], b: list[float]) -> float:
    """Jensen-Shannon divergence, base 2 -> already in [0, 1]."""

    def kl(x: list[float], y: list[float]) -> float:
        return sum(
            xi * math.log2(xi / yi) for xi, yi in zip(x, y, strict=True) if xi > 0 and yi > 0
        )

    m = [(x + y) / 2 for x, y in zip(a, b, strict=True)]
    return max(0.0, min(1.0, 0.5 * kl(a, m) + 0.5 * kl(b, m)))


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    va = [a.get(k, 0.0) for k in keys]
    vb = [b.get(k, 0.0) for k in keys]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(va, vb, strict=True)) / (na * nb)


def _weighted_jaccard(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    ta, tb = max(1, sum(a.values())), max(1, sum(b.values()))
    num = sum(min(a.get(k, 0) / ta, b.get(k, 0) / tb) for k in keys)
    den = sum(max(a.get(k, 0) / ta, b.get(k, 0) / tb) for k in keys)
    return num / den if den else 0.0


def _log_ratio(x: float, y: float, eps: float = 0.02) -> float:
    return math.exp(-abs(math.log((x + eps) / (y + eps))))


@dataclass
class StyleScore:
    C: float = 0.0
    L: float = 0.0
    P: float = 0.0
    N: float = 0.0
    X: float = 0.0
    E: float = 0.0
    total: float = 0.0
    n_generated: int = 0
    n_reference: int = 0

    def report(self) -> str:
        rows = [
            ("C  code-switch", self.C, WEIGHTS["C"]),
            ("L  length dist", self.L, WEIGHTS["L"]),
            ("P  punctuation", self.P, WEIGHTS["P"]),
            ("N  n-gram", self.N, WEIGHTS["N"]),
            ("X  perplexity", self.X, WEIGHTS["X"]),
            ("E  emoji", self.E, WEIGHTS["E"]),
        ]
        out = [f"  {'component':<18}{'score':>7}{'weight':>8}{'contrib':>9}"]
        out += [f"  {k:<18}{v:>7.3f}{w:>8.2f}{v * w:>9.3f}" for k, v, w in rows]
        out.append(f"  {'TOTAL':<18}{self.total:>7.3f}")
        out.append(f"  (n_gen={self.n_generated}, n_ref={self.n_reference})")
        return "\n".join(out)


def compare(
    generated: list[str], reference: list[str], *, perplexity_gain: float | None = None
) -> StyleScore:
    """Score `generated` against `reference` (held-out REAL messages)."""
    g, r = profile(generated), profile(reference)
    if not g.n or not r.n:
        return StyleScore()

    s = StyleScore(n_generated=g.n, n_reference=r.n)
    # Scaled by the REFERENCE rate, not absolute difference: his Hinglish rate is
    # low, so a fixed tolerance would score noise as a match.
    denom_h = max(r.hinglish_rate, 0.02)
    denom_d = max(r.devanagari_ratio, 0.01)
    ch = max(0.0, 1.0 - abs(g.hinglish_rate - r.hinglish_rate) / denom_h)
    cd = max(0.0, 1.0 - abs(g.devanagari_ratio - r.devanagari_ratio) / denom_d)
    # Devanagari is ~0% in this corpus, so weight it far below the Latin-script
    # Hinglish signal that actually carries the voice.
    s.C = 0.85 * ch + 0.15 * cd
    s.L = 1.0 - _jsd(g.length_hist, r.length_hist)
    s.P = _cosine(g.punct, r.punct)
    s.N = _weighted_jaccard(g.ngrams, r.ngrams)
    s.X = 0.5 if perplexity_gain is None else max(0.0, min(1.0, perplexity_gain))
    s.E = _log_ratio(g.emoji_rate, r.emoji_rate)

    s.total = sum(getattr(s, k) * w for k, w in WEIGHTS.items())
    return s


def normalised(score: float, floor: float, ceiling: float) -> float:
    """Position between 'generic model output' and 'human-level agreement'."""
    if ceiling <= floor:
        return 0.0
    return max(0.0, min(1.0, (score - floor) / (ceiling - floor)))


def anchors(real_messages: list[str], raw_model_messages: list[str]) -> tuple[float, float]:
    """Compute the ceiling and floor the headline number is reported against."""
    half = len(real_messages) // 2
    ceiling = compare(real_messages[:half], real_messages[half:]).total if half > 20 else 0.9
    floor = compare(raw_model_messages, real_messages).total if raw_model_messages else 0.3
    return floor, ceiling
