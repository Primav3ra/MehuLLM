"""Build the supervised fine-tuning dataset from parsed WhatsApp exports.

Pipeline order, and the order matters:

    parse -> turns -> sessions -> pairs
          -> scrub PII + pseudonymise names      (before any model sees it)
          -> drop invalid targets
          -> cap exact duplicates
          -> length-bucket rebalancing
          -> split BY CHAT (never by message)
          -> pairs.jsonl

Output is an *intermediate* format: (context -> my reply). The rewriter's input
side -- the neutral draft -- is added later by `neutralize.py`, which runs a
local model over these pairs. Keeping the two steps separate means the slow,
overnight generation step can be re-run without re-deriving the corpus, and the
expensive step is cached by content hash.

THE TWO DECISIONS MOST LIKELY TO RUIN THE FINE-TUNE
---------------------------------------------------
1. **Short replies.** "ok", "hmm", "haan" are ~35% of a raw WhatsApp corpus.
   Delete them and the model writes essays; keep them all and it collapses into
   a one-word machine. So they are DOWNSAMPLED, never dropped -- they are the
   style. The caps below are the whole reason this file has statistics in it.

2. **Splitting by message.** Two turns from the same conversation, one in train
   and one in val, leak: the model has effectively seen the val example's
   context. The split is therefore BY CHAT FILE, and two whole chats are held
   out untouched so style scoring is measured against a person the model never
   trained on.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import regex

from mehullm.pipeline.pii import NameMap, scrub_with_stats
from mehullm.pipeline.sessionize import Pair, build_pairs, merge_bursts, split_sessions
from mehullm.pipeline.whatsapp_parser import parse_export

_WORD = regex.compile(r"[\p{L}\p{N}']+")
# A target that is nothing but placeholders carries no style signal.
_ONLY_PLACEHOLDERS = regex.compile(r"^(?:\s*<[A-Z_]+>\s*)+$")

BUCKETS: list[tuple[str, int, int]] = [
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-12", 6, 12),
    ("13-30", 13, 30),
    ("31+", 31, 10**9),
]
# Share of the FINAL dataset each capped bucket may occupy.
BUCKET_CAPS = {"1-2": 0.15, "3-5": 0.20}
MAX_DUPLICATE_SHARE = 0.005  # no single exact target string beyond 0.5%
MAX_TARGET_WORDS = 600  # forwarded spam, not authored replies
HELDOUT_CHATS = 2  # whole chats withheld for honest style scoring
VAL_SHARE = 0.15
SEED = 3407


def _bucket_of(n: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return "31+"


@dataclass
class BuildStats:
    pairs_in: int = 0
    dropped_empty: int = 0
    dropped_placeholder_only: int = 0
    dropped_too_long: int = 0
    dropped_duplicate_cap: int = 0
    dropped_bucket_cap: int = 0
    pii: Counter[str] = field(default_factory=Counter)
    bucket_before: Counter[str] = field(default_factory=Counter)
    bucket_after: Counter[str] = field(default_factory=Counter)
    per_split: Counter[str] = field(default_factory=Counter)
    heldout_chats: list[str] = field(default_factory=list)
    skipped_group_chats: list[str] = field(default_factory=list)
    names_mapped: int = 0

    @property
    def pairs_out(self) -> int:
        return sum(self.per_split.values())


@dataclass
class Record:
    chat_id: str
    split: str
    context: list[dict[str, str]]
    target: str
    bucket: str
    n_words: int
    ts: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "chat_id": self.chat_id,
                "split": self.split,
                "context": self.context,
                "target": self.target,
                "bucket": self.bucket,
                "n_words": self.n_words,
                "ts": self.ts,
            },
            ensure_ascii=False,
        )


# --------------------------------------------------------------------------


def _scrub_pair(pair: Pair, names: NameMap, all_names: list[str], stats: BuildStats):
    """Scrub PII and pseudonymise names in BOTH context and target."""
    target, found = scrub_with_stats(pair.target.text)
    stats.pii.update(found)
    target = names.pseudonymize(target, all_names)

    context = []
    for turn in pair.context:
        text, found = scrub_with_stats(turn.text)
        stats.pii.update(found)
        context.append(
            {
                "sender": names.alias_for(turn.sender),
                "text": names.pseudonymize(text, all_names),
            }
        )
    return context, target


def _cap_duplicates(records: list[Record], stats: BuildStats) -> list[Record]:
    """Stop any one exact reply from dominating.

    Without this, "ok" alone can be ~9% of the dataset and the model learns
    that "ok" is a globally acceptable answer to anything.
    """
    limit = max(1, int(len(records) * MAX_DUPLICATE_SHARE))
    seen: Counter[str] = Counter()
    kept: list[Record] = []
    for r in records:
        key = r.target.strip().casefold()
        if seen[key] >= limit:
            stats.dropped_duplicate_cap += 1
            continue
        seen[key] += 1
        kept.append(r)
    return kept


def _rebalance_buckets(records: list[Record], rng: random.Random, stats: BuildStats):
    """Downsample the short-reply buckets to their configured share of the final set.

    Solved by WATER-FILLING, not a single division. The naive form --
    ``total = uncapped / (1 - sum_of_caps)`` -- silently overshoots whenever a
    capped bucket is empty or holds fewer records than its cap allows, because
    it assumes every capped bucket fills. With only the "1-2" bucket populated
    it produced 18.4% against a 15% cap.

    So: assume every capped bucket is binding, solve for the total, then demote
    any bucket that turns out to hold less than its allowance to "takes
    everything" and re-solve. Converges in at most len(BUCKET_CAPS) rounds.
    """
    by_bucket: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_bucket[r.bucket].append(r)
        stats.bucket_before[r.bucket] += 1

    uncapped = sum(len(v) for k, v in by_bucket.items() if k not in BUCKET_CAPS)
    binding = {k for k in BUCKET_CAPS if by_bucket.get(k)}
    allowed: dict[str, int] = {}

    while True:
        # Records that are taken in full: the uncapped buckets, plus any capped
        # bucket already demoted because it is under its allowance.
        fixed = uncapped + sum(len(by_bucket[k]) for k in BUCKET_CAPS if k not in binding)
        share_binding = sum(BUCKET_CAPS[k] for k in binding)
        if share_binding >= 1.0:  # degenerate config; keep everything
            allowed = {k: len(by_bucket[k]) for k in binding}
            break
        total = fixed / (1.0 - share_binding) if fixed else len(records)
        allowed = {k: int(total * BUCKET_CAPS[k]) for k in binding}

        demote = {k for k in binding if len(by_bucket[k]) <= allowed[k]}
        if not demote:
            break
        binding -= demote

    out: list[Record] = []
    for name, items in by_bucket.items():
        if name not in binding:
            out.extend(items)
            continue
        # Sample rather than truncate -- taking the first N would bias the set
        # toward whichever chats happen to be parsed first.
        rng.shuffle(items)
        keep = allowed[name]
        stats.dropped_bucket_cap += len(items) - keep
        out.extend(items[:keep])

    for r in out:
        stats.bucket_after[r.bucket] += 1
    return out


def _assign_splits(
    records: list[Record], chat_ids: list[str], rng: random.Random, stats: BuildStats
) -> None:
    """Split BY CHAT. Two whole chats are withheld for style scoring.

    Held-out chats are chosen from the mid-sized ones: withholding the largest
    would throw away a big share of training data, and withholding the tiniest
    gives a style estimate too noisy to trust.
    """
    counts = Counter(r.chat_id for r in records)
    ranked = [c for c, _ in counts.most_common()]
    mid = ranked[1:-1] or ranked  # skip the biggest and smallest when possible
    heldout = set(rng.sample(mid, min(HELDOUT_CHATS, len(mid))))

    remaining = [c for c in ranked if c not in heldout]
    val_target = sum(counts[c] for c in remaining) * VAL_SHARE

    # Whole chats are indivisible, so an exact 15% is usually unreachable. A
    # naive "keep adding until we cross the target" overshoots badly when a
    # few chats dominate -- it produced a 30.4% validation split on the real
    # corpus. Instead, walk smallest-first and take a chat only if doing so
    # moves the total CLOSER to the target than leaving it out.
    val, acc = set(), 0
    for c in sorted(remaining, key=lambda c: counts[c]):
        if abs(acc + counts[c] - val_target) < abs(acc - val_target):
            val.add(c)
            acc += counts[c]
    if not val and remaining:  # degenerate: one chat only -- never leave val empty
        smallest = min(remaining, key=lambda c: counts[c])
        val.add(smallest)

    for r in records:
        r.split = "heldout" if r.chat_id in heldout else "val" if r.chat_id in val else "train"
        stats.per_split[r.split] += 1
    stats.heldout_chats = sorted(heldout)


# --------------------------------------------------------------------------


def build(
    raw_dir: str | Path,
    out_path: str | Path,
    self_aliases: set[str],
    name_map_path: str | Path | None = None,
    seed: int = SEED,
) -> BuildStats:
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    rng = random.Random(seed)
    stats = BuildStats()

    files = sorted({p.resolve() for p in raw_dir.rglob("*") if p.suffix.lower() == ".txt"})
    chats = [parse_export(f) for f in files]

    names = NameMap.load(name_map_path) if name_map_path else NameMap()
    names.keep |= set(self_aliases)
    all_names = sorted({s for c in chats for s in c.participants}, key=len, reverse=True)

    records: list[Record] = []
    for chat in chats:
        # GROUP CHATS ARE EXCLUDED AS SFT TARGETS. Group voice differs from 1:1
        # voice, and other participants' turns contaminate the context with
        # names and in-jokes the model would learn to imitate. Groups still
        # feed the style-exemplar index and fact extraction later -- just not
        # the supervised targets.
        if len(chat.participants) > 2:
            stats.skipped_group_chats.append(chat.chat_id)
            continue

        turns = merge_bursts(chat.content_messages)
        pairs = build_pairs(split_sessions(turns), self_aliases, chat.chat_id)
        stats.pairs_in += len(pairs)

        for pair in pairs:
            context, target = _scrub_pair(pair, names, all_names, stats)

            if not target.strip():
                stats.dropped_empty += 1
                continue
            if _ONLY_PLACEHOLDERS.match(target):
                stats.dropped_placeholder_only += 1
                continue
            n_words = len(_WORD.findall(target))
            if n_words > MAX_TARGET_WORDS:
                stats.dropped_too_long += 1
                continue
            if n_words == 0:  # emoji-only reply: real style, count it in 1-2
                n_words = 1

            records.append(
                Record(
                    chat_id=pair.chat_id,
                    split="train",
                    context=context,
                    target=target,
                    bucket=_bucket_of(n_words),
                    n_words=n_words,
                    ts=pair.target.ts_start.isoformat(),
                )
            )

    stats.names_mapped = len(names.mapping)
    records = _cap_duplicates(records, stats)
    records = _rebalance_buckets(records, rng, stats)
    _assign_splits(records, [c.chat_id for c in chats], rng, stats)

    rng.shuffle(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")

    if name_map_path:
        names.save(name_map_path)
    return stats


def format_report(s: BuildStats, out_path: str | Path) -> str:
    def row(label: str, n: int, total: int) -> str:
        pc = f"{100 * n / total:5.1f}%" if total else "    -"
        return f"    {label:<22} {n:>8,}  {pc}"

    lines = [
        "=" * 66,
        "  SFT dataset build",
        "=" * 66,
        "",
        f"  Candidate pairs      {s.pairs_in:,}",
        "  Dropped:",
        f"    empty after scrub  {s.dropped_empty:,}",
        f"    placeholder-only   {s.dropped_placeholder_only:,}",
        f"    over {MAX_TARGET_WORDS} words     {s.dropped_too_long:,}",
        f"    duplicate cap      {s.dropped_duplicate_cap:,}",
        f"    bucket rebalance   {s.dropped_bucket_cap:,}",
        f"  Kept                 {s.pairs_out:,}",
        "",
        "  Reply-length buckets (before -> after rebalancing)",
    ]
    tb, ta = sum(s.bucket_before.values()), sum(s.bucket_after.values())
    for name, _, _ in BUCKETS:
        b, a = s.bucket_before[name], s.bucket_after[name]
        pb = f"{100 * b / tb:5.1f}%" if tb else "    -"
        pa = f"{100 * a / ta:5.1f}%" if ta else "    -"
        cap = f"  (cap {BUCKET_CAPS[name]:.0%})" if name in BUCKET_CAPS else ""
        lines.append(f"    {name:<6} {b:>8,} {pb}   ->  {a:>8,} {pa}{cap}")

    lines += ["", "  PII scrubbed"]
    if s.pii:
        for kind, n in s.pii.most_common():
            lines.append(f"    {kind:<12} {n:,}")
    else:
        lines.append("    (none found -- verify this is plausible)")

    lines += [
        f"    {'names mapped':<12} {s.names_mapped:,}",
        "",
        "  Split (by chat, never by message)",
    ]
    for name in ("train", "val", "heldout"):
        lines.append(row(name, s.per_split[name], s.pairs_out))
    lines += [
        f"    held-out chats:      {', '.join(s.heldout_chats) or '(none)'}",
    ]
    if s.skipped_group_chats:
        lines += [
            "",
            f"  Group chats excluded as SFT targets ({len(s.skipped_group_chats)}):",
            *(f"    {c}" for c in s.skipped_group_chats),
            "    (they still feed the style index and fact extraction later)",
        ]
    lines += [
        "",
        f"  Wrote {out_path}",
        "=" * 66,
    ]
    return "\n".join(lines)
