"""Hybrid retrieval: BM25 (FTS5) + dense (sqlite-vec), fused with RRF (k=60)."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from mehullm.memory.embed import embed_query, transliterate_hinglish
from mehullm.memory.store import MemoryStore, pack

RRF_K = 60
FACT_HALFLIFE_DAYS = 180.0


@dataclass
class Hit:
    id: int
    text: str
    kind: str  # 'fact' | 'style' | 'doc'
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    recency: float = 1.0
    meta: dict | None = None


# Stripped from FTS queries. Not for tidiness -- an OR query over stopwords.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "you",
        "your",
        "yours",
        "we",
        "us",
        "our",
        "ours",
        "they",
        "them",
        "their",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "about",
        "as",
        "into",
        "over",
        "after",
        "before",
        "and",
        "or",
        "but",
        "if",
        "so",
        "than",
        "then",
        "there",
        "here",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "not",
        "no",
        "nor",
        "too",
        "very",
        "just",
        "also",
        "again",
        "once",
        "mera",
        "meri",
        "mere",
        "main",
        "mujhe",
        "hai",
        "hain",
        "tha",
        "the",
        "ka",
        "ki",
        "ke",
        "ko",
        "se",
        "par",
        "kya",
        "kaun",
        "kaha",
        "kab",
        "kyu",
        "kyun",
        "kaise",
        "hu",
        "ho",
        "hoon",
        "me",
        "aur",
        "ya",
    ]
)


def _fts_query(text: str) -> str:
    """FTS5 MATCH is a query language, not a literal -- unescaped user text with a quote or an operator raises Operat."""
    raw = "".join(c if c.isalnum() or c.isspace() else " " for c in text).split()
    words = [w for w in raw if w.casefold() not in _STOPWORDS]
    # An all-stopword query ("how are you") should retrieve on meaning alone
    # rather than fall back to matching everything.
    return " OR ".join(f'"{w}"' for w in words[:24])


# FTS5 DOES NOT RANK BY DEFAULT.
#
_ORDER_BY_RANK = "ORDER BY rank"


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


# Formatting instructions are addressed to the MODEL, not to memory, and they.
_INSTRUCTION_TAIL = re.compile(
    r"[\s,.;–—-]*\b(?:"
    r"one line|in one line|keep it short|keep it brief|be brief|briefly|"
    r"in short|short answer|concisely|concise|tl;?dr|"
    r"in a sentence|one sentence|few words|quickly|asap"
    r")\b[\s.!?]*$",
    re.IGNORECASE,
)


def _clean_query(text: str) -> str:
    out = text.strip()
    for _ in range(3):  # "briefly, one line."
        stripped = _INSTRUCTION_TAIL.sub("", out).strip()
        if stripped == out:
            break
        out = stripped
    return out or text.strip()


def search_facts(store: MemoryStore, query: str, k: int = 8) -> list[Hit]:
    c = store.conn()
    scores: dict[int, Hit] = {}
    query = _clean_query(query)

    q = _fts_query(query)
    if q:
        for rank, row in enumerate(
            c.execute(
                "SELECT f.id, f.text, f.confidence, f.observed_at, f.predicate FROM facts_fts"
                " JOIN facts f ON f.id = facts_fts.rowid"
                f" WHERE facts_fts MATCH ? AND f.status='active' {_ORDER_BY_RANK} LIMIT 50",
                (q,),
            )
        ):
            scores[row["id"]] = Hit(
                row["id"],
                row["text"],
                "fact",
                _rrf(rank),
                bm25_rank=rank,
                meta={
                    "confidence": row["confidence"],
                    "observed_at": row["observed_at"],
                    "predicate": row["predicate"],
                },
            )

    vec = embed_query(query)
    for rank, row in enumerate(
        c.execute(
            "SELECT v.fact_id AS id, f.text, f.confidence, f.observed_at, f.predicate"
            " FROM facts_vec v JOIN facts f ON f.id = v.fact_id AND f.status='active'"
            " WHERE v.embedding MATCH ? AND k = 50",
            (pack(vec),),
        )
    ):
        h = scores.get(row["id"])
        if h:
            h.score += _rrf(rank)
            h.dense_rank = rank
        else:
            scores[row["id"]] = Hit(
                row["id"],
                row["text"],
                "fact",
                _rrf(rank),
                dense_rank=rank,
                meta={
                    "confidence": row["confidence"],
                    "observed_at": row["observed_at"],
                    "predicate": row["predicate"],
                },
            )

    now = time.time()
    for h in scores.values():
        obs = (h.meta or {}).get("observed_at")
        if obs:
            age_days = max(0.0, (now - float(obs)) / 86400.0)
            h.recency = 1.0 + 0.5 * math.exp(-age_days / FACT_HALFLIFE_DAYS)
            h.score *= h.recency

    return sorted(scores.values(), key=lambda h: -h.score)[:k]


def search_style(store: MemoryStore, query: str, k: int = 8) -> list[Hit]:
    """Style exemplars for the voice model. NO recency weighting -- see module docstring."""
    c = store.conn()
    scores: dict[int, Hit] = {}

    translit = transliterate_hinglish(query)
    q = _fts_query(query)
    if q:
        # Match Latin AND the Devanagari transliteration: e5 was not trained on
        # romanized Hindi, so lexical does the heavy lifting on Hinglish.
        clause = "chunks_fts MATCH ?"
        params: tuple = (q,)
        if translit:
            tq = _fts_query(translit)
            if tq:
                clause = "chunks_fts MATCH ?"
                params = (f"({q}) OR ({tq})",)
        for rank, row in enumerate(
            c.execute(
                f"SELECT ch.id, ch.text, ch.kind FROM chunks_fts"
                f" JOIN chunks ch ON ch.id = chunks_fts.rowid"
                f" WHERE {clause} {_ORDER_BY_RANK} LIMIT 50",
                params,
            )
        ):
            scores[row["id"]] = Hit(row["id"], row["text"], row["kind"], _rrf(rank), bm25_rank=rank)

    vec = embed_query(query)
    for rank, row in enumerate(
        c.execute(
            "SELECT v.chunk_id AS id, ch.text, ch.kind FROM chunks_vec v"
            " JOIN chunks ch ON ch.id = v.chunk_id"
            " WHERE v.embedding MATCH ? AND k = 50",
            (pack(vec),),
        )
    ):
        h = scores.get(row["id"])
        if h:
            h.score += _rrf(rank)
            h.dense_rank = rank
        else:
            scores[row["id"]] = Hit(
                row["id"], row["text"], row["kind"], _rrf(rank), dense_rank=rank
            )

    return sorted(scores.values(), key=lambda h: -h.score)[:k]


def render_memory_block(hits: list[Hit]) -> str:
    """Fact ids are load-bearing: they let a trace show WHICH fact drove an answer, and make factual recall determini."""
    lines = []
    for h in hits:
        meta = h.meta or {}
        pred = str(meta.get("predicate") or "").replace("_", " ").strip()
        tag = f" · {pred}" if pred else ""
        when = ""
        if meta.get("observed_at"):
            when = " · " + time.strftime("%Y-%m-%d", time.localtime(float(meta["observed_at"])))
        conf = f" · conf {meta.get('confidence', 0):.1f}" if meta.get("confidence") else ""
        lines.append(f"[F{h.id}{tag}{when}{conf}] {h.text}")
    return "\n".join(lines)
