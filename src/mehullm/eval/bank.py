"""Scenario bank: load, validate, seed."""

from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlite_vec
import yaml

from mehullm.eval.graders import known_assertions

# _embed_text is imported rather than reimplemented on purpose: a seeded fact
# embedded differently from a real one is not exercising the real retriever.
from mehullm.memory.embed import embed_passages
from mehullm.memory.facts import _embed_text
from mehullm.memory.store import pack

CATEGORIES = {
    "style",
    "factual_recall",
    "tool_selection",
    "refusal",
    "multistep",
    "hallucination_trap",
    "prompt_injection",
}


@dataclass
class Scenario:
    id: str
    category: str
    prompt: str
    assertions: list[dict[str, Any]] = field(default_factory=list)
    seed_facts: list[dict[str, Any]] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    inject: dict[str, Any] | None = None
    rubric: str = ""
    weight: int = 1
    style_probe: bool = False
    source: str = ""

    @property
    def judged(self) -> bool:
        """Style is never LLM-judged (§10) -- it gets the objective score."""
        return bool(self.rubric) and self.category != "style"


@dataclass
class ValidationError:
    scenario_id: str
    problem: str


def load(path: str | Path) -> list[Scenario]:
    """Load every scenario under a file or directory."""
    p = Path(path)
    files = sorted(p.glob("**/*.yaml")) if p.is_dir() else [p]
    out: list[Scenario] = []
    for f in files:
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for raw in doc.get("scenarios", []):
            known = Scenario.__dataclass_fields__
            out.append(
                Scenario(
                    **{k: v for k, v in raw.items() if k in known},
                    source=str(f),
                )
            )
    return out


def validate(scenarios: list[Scenario]) -> list[ValidationError]:
    errs: list[ValidationError] = []
    seen: dict[str, str] = {}
    valid = set(known_assertions())

    for s in scenarios:
        if not s.id:
            errs.append(ValidationError("<missing>", f"scenario without id in {s.source}"))
            continue
        if s.id in seen:
            errs.append(ValidationError(s.id, f"duplicate id (also in {seen[s.id]})"))
        seen[s.id] = s.source

        if s.category not in CATEGORIES:
            errs.append(ValidationError(s.id, f"unknown category '{s.category}'"))
        if not s.prompt.strip():
            errs.append(ValidationError(s.id, "empty prompt"))
        if not s.assertions:
            errs.append(ValidationError(s.id, "no assertions -- it can never fail"))

        for a in s.assertions:
            kind = a.get("type")
            if kind not in valid:
                errs.append(ValidationError(s.id, f"unknown assertion '{kind}'"))

        # A cites_fact assertion that names a fact the scenario never seeds can
        # only ever fail. Caught here rather than as a mystery red in the report.
        seeded = {f.get("id") for f in s.seed_facts}
        for a in s.assertions:
            if a.get("type") != "cites_fact":
                continue
            want = a["value"]
            want = [want] if isinstance(want, str) else want
            for w in want:
                if w not in seeded:
                    errs.append(ValidationError(s.id, f"cites {w} but never seeds it"))

        if s.weight < 1:
            errs.append(ValidationError(s.id, f"weight {s.weight} < 1"))
    return errs


def coverage(scenarios: list[Scenario]) -> dict[str, int]:
    out = dict.fromkeys(sorted(CATEGORIES), 0)
    for s in scenarios:
        if s.category in out:
            out[s.category] += 1
    return out


# ------------------------------------------------------------------ seeding

_SEED_SQL = """
INSERT OR REPLACE INTO facts
  (id, subject, predicate, object, text, single_valued, confidence,
   observed_at, source_chunks, superseded_by, status, verdict, grounded)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def seed_db(base_db: str | Path, scratch: str | Path, scenario: Scenario) -> Path:
    """Copy the db and replace facts with the scenario's own."""
    scratch = Path(scratch)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    if scratch.exists():
        try:
            scratch.unlink()
        except OSError:
            # Windows will not delete a file another process still holds open, so fall
            # back to a numbered scratch name rather than failing the scenario.
            for n in range(1, 50):
                alt = scratch.with_name(f"{scratch.stem}.{n}{scratch.suffix}")
                if not alt.exists():
                    scratch = alt
                    break
            else:
                raise
    shutil.copy2(base_db, scratch)

    con = sqlite3.connect(scratch)
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)

        con.execute("DELETE FROM facts")
        con.execute("DELETE FROM facts_vec")

        now = int(time.time())
        rows = []
        for f in scenario.seed_facts:
            fid = int(str(f["id"]).lstrip("F"))
            sup = f.get("superseded_by")
            con.execute(
                _SEED_SQL,
                (
                    fid,
                    f.get("subject", ""),
                    f.get("predicate", ""),
                    f.get("object", ""),
                    f["text"],
                    int(f.get("single_valued", 0)),
                    float(f.get("confidence", 0.9)),
                    f.get("observed_at", now),
                    "[]",
                    int(str(sup).lstrip("F")) if sup else None,
                    # 'active' unless the scenario deliberately seeds a retired fact.
                    f.get("status", "active"),
                    "seeded by eval scenario",
                    1,
                ),
            )
            rows.append((fid, f))

        # Predicate-prefixed, matching memory.facts._embed_text -- a seeded fact
        # embedded differently from a real one is not testing the real retriever.
        if rows:
            texts = [_embed_text(f) for _, f in rows]
            for (fid, _), vec in zip(rows, embed_passages(texts), strict=True):
                con.execute(
                    "INSERT INTO facts_vec(fact_id, embedding) VALUES (?,?)", (fid, pack(vec))
                )

        con.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        con.commit()
    finally:
        con.close()
    return scratch
