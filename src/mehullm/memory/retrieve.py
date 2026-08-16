"""Hybrid retrieval: BM25 (FTS5) + dense (sqlite-vec), fused with RRF (k=60).

RRF rather than score-normalisation: the two scales are incomparable.

Recency applies to FACTS ONLY -- style exemplars are sampled uniformly across
years, or the voice tracks last month's mood. No reranker in the hot path.
"""

from __future__ import annotations

import math
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
    kind: str            # 'fact' | 'style' | 'doc'
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    recency: float = 1.0
    meta: dict | None = None


def _fts_query(text: str) -> str:
    """FTS5 MATCH is a query language, not a literal. Unescaped user text with
    a quote or an operator raises sqlite3.OperationalError mid-request."""
    words = [w for w in "".join(c if c.isalnum() or c.isspace() else " " for c in text).split() if w]
    return " OR ".join(f'"{w}"' for w in words[:24])


# FTS5 DOES NOT RANK BY DEFAULT.
#
# Without an explicit `ORDER BY rank`, SQLite returns matching rows in ROWID
# order, so `LIMIT 50` silently returns the first 50 chunks in the table --
# the SAME rows for every query. Observed directly: two unrelated queries both
# returned ids [5, 10, 11, 16, 19], and lexical/dense overlap was exactly zero
# because one half of the hybrid retriever was constant.
#
# `rank` is BM25 (negative; more negative = better), so ORDER BY rank ASC.
_ORDER_BY_RANK = "ORDER BY rank"


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def search_facts(store: MemoryStore, query: str, k: int = 8) -> list[Hit]:
    c = store.conn()
    scores: dict[int, Hit] = {}

    q = _fts_query(query)
    if q:
        for rank, row in enumerate(
            c.execute(
                "SELECT f.id, f.text, f.confidence, f.observed_at FROM facts_fts"
                " JOIN facts f ON f.id = facts_fts.rowid"
                f" WHERE facts_fts MATCH ? AND f.status='active' {_ORDER_BY_RANK} LIMIT 50",
                (q,),
            )
        ):
            scores[row["id"]] = Hit(
                row["id"], row["text"], "fact", _rrf(rank), bm25_rank=rank,
                meta={"confidence": row["confidence"], "observed_at": row["observed_at"]},
            )

    vec = embed_query(query)
    for rank, row in enumerate(
        c.execute(
            "SELECT v.fact_id AS id, f.text, f.confidence, f.observed_at"
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
                row["id"], row["text"], "fact", _rrf(rank), dense_rank=rank,
                meta={"confidence": row["confidence"], "observed_at": row["observed_at"]},
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
            scores[row["id"]] = Hit(row["id"], row["text"], row["kind"], _rrf(rank), dense_rank=rank)

    return sorted(scores.values(), key=lambda h: -h.score)[:k]


def render_memory_block(hits: list[Hit]) -> str:
    """Fact ids are load-bearing.

    They let a trace show WHICH fact drove an answer, and they make the
    "factual recall" eval category deterministically gradeable instead of a
    judgement call.
    """
    lines = []
    for h in hits:
        meta = h.meta or {}
        when = ""
        if meta.get("observed_at"):
            when = " · " + time.strftime("%Y-%m-%d", time.localtime(float(meta["observed_at"])))
        conf = f" · conf {meta.get('confidence', 0):.1f}" if meta.get("confidence") else ""
        lines.append(f"[F{h.id}{when}{conf}] {h.text}")
    return "\n".join(lines)
