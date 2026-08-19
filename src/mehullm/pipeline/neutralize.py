"""Generate the rewriter's INPUT side, locally."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import regex

from mehullm.persistence import db
from mehullm.voice.client import OllamaClient, OllamaError

__all__ = ["SYSTEM_PROMPT", "NeutralizeStats", "neutralize"]

# The system prompt the voice model is ultimately trained and served with.
SYSTEM_PROMPT = (
    "Rewrite DRAFT as Mehul would send it on WhatsApp. Keep the meaning and all "
    "facts, names, numbers and links exactly. Match his length, script mix, "
    "punctuation and emoji habits. Output only the message."
)

# Variant A translates Hinglish; variant B rephrases as a verbose assistant.
# Few-shots matter more than the instruction: without them both drift into notes.

_SYS_A = (
    "You translate Hinglish (Hindi written in Latin script) into formal English. "
    "Keep every fact, name, number and <PLACEHOLDER> token exactly as given. "
    "Reply with the English translation only -- no preamble, no notes, no quotes."
)

_FEWSHOT_A: list[tuple[str, str]] = [
    ("kal milte hain 6 baje", "Let us meet tomorrow at 6 o'clock."),
    ("yaar mera mood kharab hai aaj", "I am not in a good mood today."),
    ("bhai 500 rs bhej de <UPI> pe", "Please transfer 500 rupees to <UPI>."),
    ("haan bhej diya dekh le", "Yes, I have sent it. Please check."),
]

_SYS_B = (
    "You rewrite a short chat reply as a helpful, slightly verbose AI assistant "
    "would phrase it in polished English. Convey exactly the same information as "
    "the original reply -- never answer differently, never add facts. Keep every "
    "name, number and <PLACEHOLDER> token. Reply with the rewritten text only."
)

_FEWSHOT_B: list[tuple[str, str]] = [
    (
        "Conversation:\nPerson_A: kal aa raha hai?\n\nReply that was sent:\npata nahi yaar",
        "I am not certain yet whether I will be able to come tomorrow.",
    ),
    (
        "Conversation:\nPerson_A: paise chahiye the\n\nReply that was sent:\n"
        "bhai 500 bhej de <UPI> pe",
        "Could you please transfer 500 rupees to <UPI> when you get a chance?",
    ),
    (
        "Conversation:\nPerson_A: aaj shaam free ho?\n\nReply that was sent:\n"
        "nahi yaar aaj nahi, kal 6 baje",
        "Unfortunately I am not available this evening. Would tomorrow at 6 work instead?",
    ),
]


def _messages_a(target: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": _SYS_A}]
    for u, a in _FEWSHOT_A:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": target})
    return msgs


def _messages_b(context: str, target: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": _SYS_B}]
    for u, a in _FEWSHOT_B:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append(
        {
            "role": "user",
            "content": f"Conversation:\n{context}\n\nReply that was sent:\n{target}",
        }
    )
    return msgs


VARIANT_B_SHARE = 0.40
DEFAULT_TRAIN_LIMIT = 12_000
DEFAULT_VAL_LIMIT = 1_500


@dataclass
class NeutralizeStats:
    considered: int = 0
    selected: int = 0
    cache_hits: int = 0
    generated: int = 0
    failed: int = 0
    rejected: int = 0
    variants: Counter[str] = field(default_factory=Counter)
    per_split: Counter[str] = field(default_factory=Counter)
    elapsed_s: float = 0.0


# Draft cache -- content-addressed, so re-runs and crashes are free.


class DraftCache:
    """SQLite cache keyed by sha256(model|variant|text)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS drafts ("
                " key TEXT PRIMARY KEY, variant TEXT, draft TEXT, created_at REAL)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = db.connect(self.path, rows=False)
            self._local.conn = conn
        return conn

    @staticmethod
    def key(model: str, variant: str, text: str) -> str:
        return hashlib.sha256(f"{model}|{variant}|{text}".encode()).hexdigest()

    def get(self, key: str) -> str | None:
        row = self._conn().execute("SELECT draft FROM drafts WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, variant: str, draft: str) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO drafts VALUES (?,?,?,?)", (key, variant, draft, time.time())
        )
        conn.commit()

    def count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM drafts").fetchone()[0]


def _render_context(context: list[dict[str, str]], max_turns: int = 4) -> str:
    return "\n".join(f"{t['sender']}: {t['text']}" for t in context[-max_turns:])


def _normalise(s: str) -> str:
    """Casefold and strip punctuation/space for degeneracy comparison only."""
    return regex.sub(r"[^\p{L}\p{N}]+", "", s).casefold()


def _is_degenerate(draft: str, target: str) -> bool:
    """True when the draft is effectively a copy of the reply."""
    d, t = _normalise(draft), _normalise(target)
    if not d:
        return True
    if d == t:
        return True
    # Near-copy: only casing/punctuation changed, or a token or two differs.
    return bool(len(t) > 12 and (d in t or t in d))


def _plausible(draft: str, target: str) -> bool:
    """Reject obviously broken generations before they poison the dataset."""
    d = draft.strip()
    if not d or len(d) < 2:
        return False
    if len(d) > max(400, 6 * len(target)):  # runaway generation
        return False
    low = d.lower()
    if low.startswith(("i cannot", "i can't", "i'm sorry", "as an ai", "sure, here")):
        return False
    if "translate the following" in low or "reply with the" in low:
        return False  # echoed its own instructions
    return not _is_degenerate(draft, target)


def _stratified_sample(records: list[dict], limit: int, rng: random.Random) -> list[dict]:
    """Sample while preserving the bucket mix and spreading across chats."""
    if len(records) <= limit:
        return list(records)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_bucket[r["bucket"]].append(r)

    out: list[dict] = []
    for _bucket, items in by_bucket.items():
        share = len(items) / len(records)
        want = max(1, round(limit * share))
        by_chat: dict[str, list[dict]] = defaultdict(list)
        for r in items:
            by_chat[r["chat_id"]].append(r)
        for v in by_chat.values():
            rng.shuffle(v)
        # Round-robin across chats so no single conversation dominates a bucket.
        picked, chats = [], list(by_chat.values())
        while len(picked) < want and any(chats):
            for v in chats:
                if v and len(picked) < want:
                    picked.append(v.pop())
            chats = [v for v in chats if v]
        out.extend(picked)

    rng.shuffle(out)
    return out[:limit]


@dataclass
class _Drafter:
    """Generate the neutral input side for one record, sharing cache and stats."""

    ollama: OllamaClient
    cache: DraftCache
    model: str
    stats: NeutralizeStats
    out: Any
    started: float
    on_progress: Any = None
    progress_every: int = 200
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _write_lock: threading.Lock = field(default_factory=threading.Lock)

    def _draft(self, rec: dict, ctx: str, client: httpx.Client) -> str | None:
        """A cached or freshly generated draft; None when the record is dropped."""
        variant, target = rec["_variant"], rec["target"]
        cache_input = target if variant == "A" else f"{ctx}\n---\n{target}"
        key = DraftCache.key(self.model, variant, cache_input)

        cached = self.cache.get(key)
        if cached is not None:
            with self._lock:
                self.stats.cache_hits += 1
            return cached

        msgs = _messages_a(target) if variant == "A" else _messages_b(ctx, target)
        try:
            draft = self.ollama.chat(msgs, temperature=0.3, num_predict=120, client=client)
        except OllamaError:
            with self._lock:
                self.stats.failed += 1
            return None
        if not _plausible(draft, target):
            with self._lock:
                self.stats.rejected += 1
            return None

        self.cache.put(key, variant, draft)
        with self._lock:
            self.stats.generated += 1
        return draft

    def handle(self, rec: dict, client: httpx.Client) -> None:
        ctx = _render_context(rec["context"])
        draft = self._draft(rec, ctx, client)
        if draft is None:
            return

        variant = rec["_variant"]
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"<context>\n{ctx}\n</context>\n<draft>\n{draft.strip()}\n</draft>",
                },
                {"role": "assistant", "content": rec["target"]},
            ],
            "split": rec["split"],
            "variant": variant,
            "chat_id": rec["chat_id"],
            "bucket": rec["bucket"],
        }
        with self._write_lock:
            self.out.write(json.dumps(row, ensure_ascii=False) + "\n")
        with self._lock:
            self.stats.variants[variant] += 1
            done = self.stats.variants["A"] + self.stats.variants["B"]
            if self.on_progress and done % self.progress_every == 0:
                self.on_progress(done, self.stats.selected, time.time() - self.started)

    def run_chunk(self, chunk: list[dict]) -> None:
        with httpx.Client(base_url=self.ollama.host, timeout=self.ollama.timeout) as client:
            for rec in chunk:
                self.handle(rec, client)


def _load_by_split(pairs_path: Path, stats: NeutralizeStats) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = defaultdict(list)
    with pairs_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                stats.considered += 1
                by_split[rec["split"]].append(rec)
    return by_split


def _select(
    by_split: dict[str, list[dict]],
    train_limit: int,
    val_limit: int,
    rng: random.Random,
    stats: NeutralizeStats,
) -> list[dict]:
    """heldout deliberately gets no drafts: it is scored against Mehul's real messages."""
    selected: list[dict] = []
    for split, limit in (("train", train_limit), ("val", val_limit)):
        chosen = _stratified_sample(by_split.get(split, []), limit, rng)
        for rec in chosen:
            rec["_variant"] = "B" if rng.random() < VARIANT_B_SHARE else "A"
        selected.extend(chosen)
        stats.per_split[split] = len(chosen)
    stats.selected = len(selected)
    return selected


def neutralize(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    model: str = "qwen3:1.7b",
    cache_path: str | Path = "data/derived/draft_cache.db",
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    val_limit: int = DEFAULT_VAL_LIMIT,
    concurrency: int = 2,
    seed: int = 3407,
    progress_every: int = 200,
    on_progress=None,
) -> NeutralizeStats:
    pairs_path, out_path = Path(pairs_path), Path(out_path)
    rng = random.Random(seed)
    stats = NeutralizeStats()
    started = time.time()

    ollama = OllamaClient(model=model)
    ollama.preflight()

    selected = _select(_load_by_split(pairs_path, stats), train_limit, val_limit, rng, stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        drafter = _Drafter(
            ollama=ollama,
            cache=DraftCache(cache_path),
            model=model,
            stats=stats,
            out=out,
            started=started,
            on_progress=on_progress,
            progress_every=progress_every,
        )
        chunks: list[list[dict]] = [[] for _ in range(max(1, concurrency))]
        for i, rec in enumerate(selected):
            chunks[i % len(chunks)].append(rec)
        threads = [
            threading.Thread(target=drafter.run_chunk, args=(c,), daemon=True) for c in chunks
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    stats.elapsed_s = time.time() - started
    return stats


def format_report(s: NeutralizeStats, out_path: str | Path) -> str:
    total = s.variants["A"] + s.variants["B"]
    rate = total / s.elapsed_s if s.elapsed_s else 0
    return "\n".join(
        [
            "=" * 66,
            "  Draft neutralisation (local)",
            "=" * 66,
            "",
            f"  Pairs in file        {s.considered:,}",
            f"  Selected             {s.selected:,}   "
            f"(train {s.per_split['train']:,}, val {s.per_split['val']:,})",
            f"  Cache hits           {s.cache_hits:,}",
            f"  Generated            {s.generated:,}",
            f"  Rejected (implausible) {s.rejected:,}",
            f"  Failed (ollama)      {s.failed:,}",
            "",
            f"  Variant A (paraphrase)  {s.variants['A']:,}",
            f"  Variant B (assistant)   {s.variants['B']:,}",
            "",
            f"  Written              {total:,}",
            f"  Elapsed              {s.elapsed_s / 60:.1f} min  ({rate:.1f}/s)",
            f"  Wrote {out_path}",
            "=" * 66,
        ]
    )
