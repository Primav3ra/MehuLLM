"""Scenario bank: load, validate, seed.

validate() needs no API key, so a typo'd assertion type is caught offline rather
than silently passing. Scenarios carry their own facts, seeded into a scratch
copy of the db, so grading is identical on every machine.
"""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mehullm.eval.graders import known_assertions

CATEGORIES = {
    "style", "factual_recall", "tool_selection", "refusal",
    "multistep", "hallucination_trap", "prompt_injection",
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
            out.append(Scenario(
                **{k: v for k, v in raw.items() if k in known},
                source=str(f),
            ))
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


def seed_db(base_db: str | Path, scratch: str | Path,
            scenario: Scenario) -> Path:
    """Copy the db and overwrite facts with the scenario's own. A copy, so an
    eval can never mutate real facts. Style chunks are kept."""
    scratch = Path(scratch)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    if scratch.exists():
        scratch.unlink()
    shutil.copy2(base_db, scratch)

    con = sqlite3.connect(scratch)
    try:
        con.execute("DELETE FROM facts")
        for tbl in ("facts_vec", "facts_fts"):
            # vec0 needs the extension loaded; absence is not fatal here
            with contextlib.suppress(sqlite3.OperationalError):
                con.execute(f"DELETE FROM {tbl}")

        now = int(time.time())
        for f in scenario.seed_facts:
            fid = int(str(f["id"]).lstrip("F"))
            sup = f.get("superseded_by")
            con.execute(_SEED_SQL, (
                fid, f.get("subject", ""), f.get("predicate", ""),
                f.get("object", ""), f["text"], int(f.get("single_valued", 0)),
                float(f.get("confidence", 0.9)), f.get("observed_at", now),
                "[]", int(str(sup).lstrip("F")) if sup else None,
                # Seeded facts are 'active' by default: a scenario asserting
                # recall needs them retrievable, and the pending/review flow is
                # the extraction pipeline's concern, not the bank's.
                f.get("status", "active"), "seeded by eval scenario", 1,
            ))
        con.commit()
    finally:
        con.close()
    return scratch
