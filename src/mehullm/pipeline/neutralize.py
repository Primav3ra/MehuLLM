"""Generate the rewriter's INPUT side, locally.

The voice model is a rewriter: (neutral draft + context) -> Mehul's reply. Chat
logs give us the output side for free, but not the input. This module
synthesises it with a local model.

Runs locally by requirement: a hosted API would ship raw 1:1 chats to a third
party, and Gemini's free tier trains on submitted data.

Two variants, and the mix matters:
  A (60%) neutral paraphrase       -> teaches content fidelity
  B (40%) verbose assistant draft  -> teaches compression + style transfer, and
          matches the real inference-time input distribution
Without B the model only learns to undo a paraphraser's tics; without A it drops
facts.

Note this deviates from the original plan, which had B answer the context
freely. That would produce drafts whose CONTENT differs from the real reply,
training the rewriter to invent -- the precise failure the runtime invariant
firewall exists to catch. Both variants therefore hold content fixed and vary
only register.
"""

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

import httpx
import regex

from mehullm.voice.client import OllamaClient, OllamaError

__all__ = ["neutralize", "NeutralizeStats", "SYSTEM_PROMPT"]

# The system prompt the voice model is ultimately trained and served with.
SYSTEM_PROMPT = (
    "Rewrite DRAFT as Mehul would send it on WhatsApp. Keep the meaning and all "
    "facts, names, numbers and links exactly. Match his length, script mix, "
    "punctuation and emoji habits. Output only the message."
)

# PROMPTING NOTE -- this was rebuilt after a smoke test showed the first
# attempt was worthless. Two failures, both fixed here:
#
#   1. "Rewrite in plain, standard English" produced OUTPUT IDENTICAL TO INPUT.
#      A 1.7B model does not infer that Hinglish counts as not-English. The
#      word that works is TRANSLATE.
#   2. Few-shot examples on the /api/generate completion endpoint were
#      CONTINUED rather than treated as examples -- the model emitted all three
#      example answers concatenated. Hence chat turns, via OllamaClient.chat().
#
# If the draft comes back equal to the target, the training pair degenerates
# into an identity mapping and the LoRA learns nothing. `_is_degenerate()`
# rejects those outright rather than letting them silently poison the dataset.

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


# Draft cache -- content-addressed, so re-runs and crashes are free


class DraftCache:
    """SQLite cache keyed by sha256(model|variant|text).

    Resumability falls out of this for free: a crashed run simply finds its
    earlier work on restart. No separate job queue to keep consistent.
    """

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
            conn = sqlite3.connect(self.path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
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
    """True when the draft is effectively a copy of the reply.

    THE most important check in this module. If draft == target the training
    pair teaches the identity function: the LoRA has nothing to learn, training
    loss looks fine, and you only discover the problem when the fine-tuned
    model turns out no better than the base. The first prompt version produced
    this for essentially every input.
    """
    d, t = _normalise(draft), _normalise(target)
    if not d:
        return True
    if d == t:
        return True
    # Near-copy: only casing/punctuation changed, or a token or two differs.
    return bool(len(t) > 12 and (d in t or t in d))


def _plausible(draft: str, target: str) -> bool:
    """Reject obviously broken generations before they poison the dataset.

    A local 1.7B will sometimes refuse, echo the prompt, or run away. Cheap
    structural checks catch most of it; the rest is caught at eval time.
    """
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


def _stratified_sample(
    records: list[dict], limit: int, rng: random.Random
) -> list[dict]:
    """Sample while preserving the bucket mix and spreading across chats.

    A flat random sample would over-represent whichever chat is largest -- one
    conversation is 40% of this corpus -- and the voice model would learn how
    Mehul talks to one specific person.
    """
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

    cache = DraftCache(cache_path)

    by_split: dict[str, list[dict]] = defaultdict(list)
    with pairs_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                stats.considered += 1
                by_split[r["split"]].append(r)

    # heldout deliberately gets no drafts: it is scored against Mehul's REAL
    # messages, so generating inputs for it would be wasted compute.
    selected: list[dict] = []
    for split, limit in (("train", train_limit), ("val", val_limit)):
        chosen = _stratified_sample(by_split.get(split, []), limit, rng)
        for r in chosen:
            r["_variant"] = "B" if rng.random() < VARIANT_B_SHARE else "A"
        selected.extend(chosen)
        stats.per_split[split] = len(chosen)
    stats.selected = len(selected)

    lock = threading.Lock()
    write_lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh_out = out_path.open("w", encoding="utf-8", newline="\n")

    def work(rec: dict, client: httpx.Client) -> None:
        variant = rec["_variant"]
        target = rec["target"]
        ctx = _render_context(rec["context"])
        cache_input = target if variant == "A" else f"{ctx}\n---\n{target}"
        key = DraftCache.key(model, variant, cache_input)

        draft = cache.get(key)
        if draft is not None:
            with lock:
                stats.cache_hits += 1
        else:
            msgs = _messages_a(target) if variant == "A" else _messages_b(ctx, target)
            try:
                draft = ollama.chat(msgs, temperature=0.3, num_predict=120, client=client)
            except OllamaError:
                with lock:
                    stats.failed += 1
                return
            if not _plausible(draft, target):
                with lock:
                    stats.rejected += 1
                return
            cache.put(key, variant, draft)
            with lock:
                stats.generated += 1

        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"<context>\n{ctx}\n</context>\n<draft>\n{draft.strip()}\n</draft>",
                },
                {"role": "assistant", "content": target},
            ],
            "split": rec["split"],
            "variant": variant,
            "chat_id": rec["chat_id"],
            "bucket": rec["bucket"],
        }
        with write_lock:
            fh_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        with lock:
            stats.variants[variant] += 1
            done = stats.variants["A"] + stats.variants["B"]
            if on_progress and done % progress_every == 0:
                on_progress(done, stats.selected, time.time() - started)

    def runner(chunk: list[dict]) -> None:
        with httpx.Client(base_url=ollama.host, timeout=ollama.timeout) as client:
            for rec in chunk:
                work(rec, client)

    try:
        chunks: list[list[dict]] = [[] for _ in range(max(1, concurrency))]
        for i, rec in enumerate(selected):
            chunks[i % len(chunks)].append(rec)
        threads = [threading.Thread(target=runner, args=(c,), daemon=True) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        fh_out.close()

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
