"""Load the curated fact bank from facts/*.yaml into memory."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from mehullm.memory.embed import embed_passages
from mehullm.memory.store import MemoryStore, pack

FACTS_DIR = "facts"

# Predicates that hold ONE value at a time: a newer one supersedes the older,
# which is retired rather than deleted so "where did I used to live?" works.
SINGLE_VALUED = {
    "lives_in",
    "studies_at",
    "works_at",
    "current_role",
    "current_project",
    "relationship_status",
    "phone_of",
    "email_of",
}


@dataclass
class LoadStats:
    files: int = 0
    inserted: int = 0
    updated: int = 0
    superseded: int = 0
    blank: int = 0
    errors: list[str] = field(default_factory=list)


def _parse_when(v) -> int | None:
    """YAML gives a date for `2025-08-14` and a str for `2025-08`."""
    if v is None:
        return None
    if isinstance(v, date):
        return int(time.mktime(v.timetuple()))
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return int(time.mktime(time.strptime(s, fmt)))
        except ValueError:
            continue
    return None


def _embed_text(e: dict) -> str:
    """What gets EMBEDDED -- not what gets displayed."""
    pred = str(e.get("predicate", "")).replace("_", " ").strip()
    obj = str(e.get("object", "")).strip()
    subj = str(e.get("subject", "Mehul")).strip()
    head = " ".join(p for p in (subj, pred, obj) if p)
    return f"{head}. {e['text']}" if pred else e["text"]


def _read_entries(facts_dir: str, st: LoadStats) -> list[dict]:
    """Parse every fact file, rejecting duplicates and unusable entries."""
    entries: list[dict] = []
    seen: dict[int, str] = {}
    for path in sorted(Path(facts_dir).glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not doc:
            continue  # an untouched template is not an error
        st.files += 1
        for raw in doc:
            if not isinstance(raw, dict):
                st.errors.append(f"{path.name}: entry is not a mapping: {raw!r}")
                continue
            if not str(raw.get("text", "")).strip():
                st.blank += 1  # an unfilled template line is the normal state
                continue
            try:
                fid = int(str(raw["id"]).lstrip("Ff"))
            except (KeyError, ValueError):
                st.errors.append(f"{path.name}: bad or missing id on {raw['text'][:50]!r}")
                continue
            if fid in seen:
                st.errors.append(f"{path.name}: id F{fid:03d} already used in {seen[fid]}")
                continue
            seen[fid] = path.name
            raw["_id"] = fid
            entries.append(raw)
    return entries


def _ensure_porter_fts(con) -> None:
    """Rebuild an facts_fts created before the porter tokenizer, whose stemming differs."""
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts_fts'"
    ).fetchone()
    if row and "porter" not in (row[0] or ""):
        con.executescript(
            "DROP TABLE IF EXISTS facts_fts;"
            "CREATE VIRTUAL TABLE facts_fts USING fts5("
            "  text, content='facts', content_rowid='id',"
            "  tokenize='porter unicode61 remove_diacritics 2');"
        )


def _upsert(con, entries: list[dict], st: LoadStats) -> None:
    vecs = embed_passages([_embed_text(e) for e in entries])
    now = int(time.time())

    for e, vec in zip(entries, vecs, strict=True):
        fid = e["_id"]
        pred = str(e.get("predicate", "")).strip()
        single = bool(e.get("single_valued", pred in SINGLE_VALUED))
        exists = con.execute("SELECT 1 FROM facts WHERE id=?", (fid,)).fetchone()

        con.execute(
            "INSERT INTO facts(id,subject,predicate,object,text,single_valued,confidence,"
            "observed_at,source_chunks,status,verdict,grounded,superseded_by)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET subject=excluded.subject,"
            " predicate=excluded.predicate, object=excluded.object, text=excluded.text,"
            " single_valued=excluded.single_valued, confidence=excluded.confidence,"
            " observed_at=excluded.observed_at, status=excluded.status,"
            " superseded_by=excluded.superseded_by",
            (
                fid,
                str(e.get("subject", "Mehul")),
                pred,
                str(e.get("object", "")),
                e["text"],
                int(single),
                float(e.get("confidence", 1.0)),
                _parse_when(e.get("observed_at")) or now,
                json.dumps([]),
                str(e.get("status", "active")),
                "curated",
                1,
                int(str(e["superseded_by"]).lstrip("Ff")) if e.get("superseded_by") else None,
            ),
        )
        # facts_fts is external-content (content='facts') and cannot be DELETEd
        # directly; it is rebuilt separately, so only facts_vec is touched here.
        con.execute("DELETE FROM facts_vec WHERE fact_id=?", (fid,))
        con.execute("INSERT INTO facts_vec(fact_id, embedding) VALUES (?,?)", (fid, pack(vec)))

        st.updated += 1 if exists else 0
        st.inserted += 0 if exists else 1


def _apply_supersession(con, entries: list[dict], st: LoadStats) -> None:
    """Run after the full pass, so forward references work regardless of file order."""
    for e in entries:
        if e.get("superseded_by"):
            con.execute("UPDATE facts SET status='superseded' WHERE id=?", (e["_id"],))
            st.superseded += 1


def load_dir(db_path: str, facts_dir: str = FACTS_DIR) -> LoadStats:
    store = MemoryStore(db_path)
    con = store.conn()
    st = LoadStats()

    entries = _read_entries(facts_dir, st)
    if not entries:
        return st

    _ensure_porter_fts(con)

    # Residue from the retired extraction pipeline: those landed as 'pending' and
    # were never approved, while curated facts are 'active' by construction.
    dropped = con.execute("DELETE FROM facts WHERE status='pending'").rowcount
    if dropped:
        st.errors.append(f"cleared {dropped} unapproved facts left by the old extractor")

    _upsert(con, entries, st)
    _apply_supersession(con, entries, st)

    con.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
    con.commit()
    return st


def format_report(st: LoadStats) -> str:
    out = [
        "=" * 56,
        "  Fact bank loaded",
        "=" * 56,
        f"  files            {st.files}",
        f"  new              {st.inserted}",
        f"  updated in place {st.updated}",
        f"  superseded       {st.superseded}",
        f"  still blank      {st.blank}",
    ]
    if st.errors:
        out.append(f"\n  {len(st.errors)} PROBLEM(S):")
        out += [f"    {e}" for e in st.errors]
    out.append("=" * 56)
    return "\n".join(out)
