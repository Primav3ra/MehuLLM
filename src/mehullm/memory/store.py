"""Local memory store: sqlite-vec + FTS5 in one file.

BM25 and vector search share a transaction, so hybrid retrieval is one SQL
round-trip and no extra process runs. sqlite-vec is pre-1.0 -- pin it; every
vector is regenerable from `chunks.text` via reindex().

Three collections, separate because they feed different prompts:
  chunks(kind='style') -> voice model, as exemplars
  chunks(kind='doc')   -> brain, as context
  facts                -> brain, inside <memory>
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

DIM = 384  # intfloat/multilingual-e5-small

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
  id         INTEGER PRIMARY KEY,
  chat_id    TEXT,
  session_id TEXT,
  kind       TEXT CHECK(kind IN ('style','doc')) NOT NULL DEFAULT 'style',
  text       TEXT NOT NULL,
  translit   TEXT,
  speaker    TEXT,
  ts         INTEGER,
  sha256     TEXT UNIQUE
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, translit, content='chunks', content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
  chunk_id INTEGER PRIMARY KEY, embedding float[{DIM}]
);

CREATE TABLE IF NOT EXISTS facts (
  id            INTEGER PRIMARY KEY,
  subject       TEXT NOT NULL,
  predicate     TEXT NOT NULL,
  object        TEXT NOT NULL,
  text          TEXT NOT NULL,          -- natural-language form; this is embedded
  single_valued INTEGER DEFAULT 0,      -- 1 => a newer value supersedes
  confidence    REAL DEFAULT 0.5,
  observed_at   INTEGER,
  source_chunks TEXT,                   -- JSON array of chunk ids
  superseded_by INTEGER REFERENCES facts(id),
  -- Facts land as 'pending' and are NOT retrievable until reviewed.
  -- Precision over recall: a wrong fact in memory is worse than a missing one,
  -- because the agent will state it confidently and cite an id for it.
  status        TEXT DEFAULT 'pending'
                CHECK(status IN ('pending','active','superseded','rejected')),
  verdict       TEXT,      -- why the verifier passed/failed it
  grounded      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_facts_sp ON facts(subject, predicate);
CREATE INDEX IF NOT EXISTS ix_facts_status ON facts(status);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  text, content='facts', content_rowid='id'
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
  fact_id INTEGER PRIMARY KEY, embedding float[{DIM}]
);

-- Short-term memory.
CREATE TABLE IF NOT EXISTS sessions (
  id                TEXT PRIMARY KEY,
  started_at        REAL,
  last_at           REAL,
  summary           TEXT,
  summary_upto_turn INTEGER DEFAULT 0,
  est_tokens        INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
  id         INTEGER PRIMARY KEY,
  session_id TEXT REFERENCES sessions(id),
  idx        INTEGER,
  role       TEXT,
  content    TEXT,
  tool_calls TEXT,
  n_tokens   INTEGER,
  compacted  INTEGER DEFAULT 0,
  trace_id   TEXT,
  ts         REAL
);
CREATE INDEX IF NOT EXISTS ix_turns_session ON turns(session_id, idx);

-- Extraction job queue: a crash costs ONE session, not the whole night.
CREATE TABLE IF NOT EXISTS extract_jobs (
  session_key TEXT PRIMARY KEY,
  status      TEXT DEFAULT 'pending',
  attempts    INTEGER DEFAULT 0,
  last_error  TEXT
);
"""


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


@dataclass
class Fact:
    id: int
    text: str
    subject: str
    predicate: str
    object: str
    confidence: float
    observed_at: int | None
    status: str = "active"


# Columns/constraints added after the first release. `CREATE TABLE IF NOT
# EXISTS` does NOT alter an existing table, so a db created before these
# existed raises "no such column" at INSERT time -- discovered the hard way,
# 15 minutes into a run.
_FACTS_REQUIRED_COLUMNS = {"verdict", "grounded"}
_FACTS_REQUIRED_STATUS = "pending"


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        c = self.conn()
        c.executescript(SCHEMA)
        self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """Bring an older `facts` table up to the current shape.

        Two kinds of drift:
          * missing COLUMNS      -> ALTER TABLE ADD COLUMN (cheap, in place)
          * changed CHECK constraint -> SQLite cannot alter one, so the table
            must be rebuilt. Only attempted when `facts` is empty, which is the
            realistic case (facts are re-derivable by re-running extraction);
            otherwise we leave it alone and let the caller decide rather than
            silently destroying reviewed data.
        """
        cols = {r[1] for r in c.execute("PRAGMA table_info(facts)")}
        if not cols:
            return

        for col, ddl in (("verdict", "TEXT"), ("grounded", "INTEGER DEFAULT 0")):
            if col not in cols:
                c.execute(f"ALTER TABLE facts ADD COLUMN {col} {ddl}")

        sql = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts'"
        ).fetchone()
        if sql and _FACTS_REQUIRED_STATUS not in (sql[0] or ""):
            n = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            if n == 0:
                # Safe: nothing to lose, and re-running extraction rebuilds it.
                c.executescript(
                    "DROP TABLE IF EXISTS facts;"
                    "DROP TABLE IF EXISTS facts_fts;"
                    "DROP TABLE IF EXISTS facts_vec;"
                )
                c.executescript(SCHEMA)
            else:
                raise RuntimeError(
                    f"`facts` has an outdated CHECK constraint and holds {n} rows. "
                    "SQLite cannot alter a constraint in place. Back up the db, then "
                    "either re-run extraction on a fresh file or migrate manually."
                )
        c.commit()

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.enable_load_extension(True)
            sqlite_vec.load(c)
            c.enable_load_extension(False)
            c.execute("PRAGMA journal_mode=WAL")
            c.row_factory = sqlite3.Row
            self._local.c = c
        return c


    def add_chunk(
        self,
        text: str,
        embedding: list[float],
        *,
        kind: str = "style",
        chat_id: str = "",
        session_id: str = "",
        speaker: str = "",
        ts: int | None = None,
        translit: str = "",
        sha256: str = "",
    ) -> int | None:
        """Returns the row id, or None if this exact text is already stored."""
        c = self.conn()
        try:
            cur = c.execute(
                "INSERT INTO chunks(chat_id,session_id,kind,text,translit,speaker,ts,sha256)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (chat_id, session_id, kind, text, translit, speaker, ts, sha256 or text[:64]),
            )
        except sqlite3.IntegrityError:
            return None  # duplicate sha256
        rid = cur.lastrowid
        c.execute(
            "INSERT INTO chunks_fts(rowid, text, translit) VALUES (?,?,?)", (rid, text, translit)
        )
        c.execute(
            "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?,?)", (rid, pack(embedding))
        )
        return rid

    def commit(self) -> None:
        self.conn().commit()


    def add_fact(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        text: str,
        embedding: list[float],
        single_valued: bool = False,
        confidence: float = 0.6,
        observed_at: int | None = None,
        source_chunks: list[int] | None = None,
        status: str = "pending",
        verdict: str = "",
        grounded: bool = False,
    ) -> int:
        c = self.conn()
        cur = c.execute(
            "INSERT INTO facts(subject,predicate,object,text,single_valued,confidence,"
            "observed_at,source_chunks,status,verdict,grounded) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                subject, predicate, object_, text, int(single_valued),
                confidence, observed_at, json.dumps(source_chunks or []),
                status, verdict, int(grounded),
            ),
        )
        fid = cur.lastrowid
        c.execute("INSERT INTO facts_fts(rowid, text) VALUES (?,?)", (fid, text))
        c.execute("INSERT INTO facts_vec(fact_id, embedding) VALUES (?,?)", (fid, pack(embedding)))

        if single_valued and status == "active":
            # Newer value wins, older is SUPERSEDED not deleted -- the chain is
            # how "where did I used to live?" gets answered.
            # Only fires for ACTIVE facts: a pending fact must not silently
            # retire an approved one before anyone has looked at it.
            c.execute(
                "UPDATE facts SET status='superseded', superseded_by=?"
                " WHERE status='active' AND single_valued=1 AND subject=? AND predicate=?"
                "   AND id<>? AND (observed_at IS NULL OR observed_at <= ?)",
                (fid, subject, predicate, fid, observed_at or 0),
            )
        c.commit()
        return fid


    def pending_facts(self, limit: int = 500) -> list[dict]:
        rows = self.conn().execute(
            "SELECT id, text, subject, predicate, object, confidence, grounded, verdict,"
            " observed_at, source_chunks FROM facts WHERE status='pending'"
            " ORDER BY confidence DESC, id LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_status(self, fact_ids: list[int], status: str) -> int:
        if not fact_ids:
            return 0
        c = self.conn()
        c.executemany(
            "UPDATE facts SET status=? WHERE id=?", [(status, i) for i in fact_ids]
        )
        # Approving a single-valued fact retires older approved ones.
        if status == "active":
            for fid in fact_ids:
                r = c.execute(
                    "SELECT subject,predicate,single_valued,observed_at FROM facts WHERE id=?",
                    (fid,),
                ).fetchone()
                if r and r["single_valued"]:
                    c.execute(
                        "UPDATE facts SET status='superseded', superseded_by=?"
                        " WHERE status='active' AND single_valued=1 AND subject=?"
                        " AND predicate=? AND id<>? AND (observed_at IS NULL OR observed_at <= ?)",
                        (fid, r["subject"], r["predicate"], fid, r["observed_at"] or 0),
                    )
        c.commit()
        return len(fact_ids)

    def source_text(self, fact_id: int) -> str:
        r = self.conn().execute(
            "SELECT source_chunks FROM facts WHERE id=?", (fact_id,)
        ).fetchone()
        ids = json.loads(r["source_chunks"] or "[]") if r else []
        if not ids:
            return ""
        rows = self.conn().execute(
            f"SELECT text FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids
        ).fetchall()
        return "\n---\n".join(x["text"] for x in rows)

    def nearest_fact(self, embedding: list[float]) -> tuple[int, float] | None:
        """For dedup: closest live fact and its distance.

        Includes PENDING as well as ACTIVE. Restricting this to active would
        mean a 3,500-session extraction run creates the same pending fact
        dozens of times, and the review queue becomes unreadable.
        Rejected/superseded are excluded on purpose -- a fact you already threw
        out should not silently absorb a new observation.
        """
        row = self.conn().execute(
            "SELECT v.fact_id, v.distance FROM facts_vec v"
            " JOIN facts f ON f.id = v.fact_id AND f.status IN ('active','pending')"
            " WHERE v.embedding MATCH ? AND k = 1",
            (pack(embedding),),
        ).fetchone()
        return (row["fact_id"], row["distance"]) if row else None

    def merge_fact(self, fact_id: int, extra_chunks: list[int], confidence: float) -> None:
        c = self.conn()
        row = c.execute("SELECT source_chunks, confidence FROM facts WHERE id=?", (fact_id,)).fetchone()
        merged = sorted(set(json.loads(row["source_chunks"] or "[]")) | set(extra_chunks))
        c.execute(
            "UPDATE facts SET source_chunks=?, confidence=? WHERE id=?",
            (json.dumps(merged), min(0.99, max(row["confidence"], confidence) + 0.05), fact_id),
        )
        c.commit()

    def active_facts(self, limit: int = 200) -> list[Fact]:
        rows = self.conn().execute(
            "SELECT * FROM facts WHERE status='active' ORDER BY confidence DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_fact(r) for r in rows]

    def fact_history(self, subject: str, predicate: str) -> list[Fact]:
        rows = self.conn().execute(
            "SELECT * FROM facts WHERE subject=? AND predicate=? ORDER BY observed_at DESC",
            (subject, predicate),
        ).fetchall()
        return [_fact(r) for r in rows]


    def queue_jobs(self, keys: list[str]) -> int:
        c = self.conn()
        c.executemany(
            "INSERT OR IGNORE INTO extract_jobs(session_key) VALUES (?)", [(k,) for k in keys]
        )
        c.commit()
        return len(keys)

    def next_jobs(self, n: int = 50) -> list[str]:
        return [
            r["session_key"]
            for r in self.conn().execute(
                "SELECT session_key FROM extract_jobs WHERE status='pending' LIMIT ?", (n,)
            )
        ]

    def finish_job(self, key: str, ok: bool, error: str = "") -> None:
        c = self.conn()
        c.execute(
            "UPDATE extract_jobs SET status=?, attempts=attempts+1, last_error=? WHERE session_key=?",
            ("done" if ok else "failed", error[:300], key),
        )
        c.commit()


    def stats(self) -> dict[str, Any]:
        c = self.conn()
        one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
        return {
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "style_chunks": one("SELECT COUNT(*) FROM chunks WHERE kind='style'"),
            "facts_pending": one("SELECT COUNT(*) FROM facts WHERE status='pending'"),
            "facts_active": one("SELECT COUNT(*) FROM facts WHERE status='active'"),
            "facts_rejected": one("SELECT COUNT(*) FROM facts WHERE status='rejected'"),
            "facts_superseded": one("SELECT COUNT(*) FROM facts WHERE status='superseded'"),
            "jobs_pending": one("SELECT COUNT(*) FROM extract_jobs WHERE status='pending'"),
            "jobs_done": one("SELECT COUNT(*) FROM extract_jobs WHERE status='done'"),
        }


def _fact(r: sqlite3.Row) -> Fact:
    return Fact(
        id=r["id"], text=r["text"], subject=r["subject"], predicate=r["predicate"],
        object=r["object"], confidence=r["confidence"], observed_at=r["observed_at"],
        status=r["status"],
    )
