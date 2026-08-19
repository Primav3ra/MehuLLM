"""Free-tier quota accounting."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mehullm.persistence import db

PT = ZoneInfo("America/Los_Angeles")

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
  id            INTEGER PRIMARY KEY,
  ts            REAL NOT NULL,
  day_pt        TEXT NOT NULL,
  provider      TEXT NOT NULL,
  model         TEXT NOT NULL,
  input_tokens  INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  ok            INTEGER DEFAULT 1,
  error_kind    TEXT
);
CREATE INDEX IF NOT EXISTS ix_usage_day ON llm_usage(provider, day_pt);
CREATE INDEX IF NOT EXISTS ix_usage_ts  ON llm_usage(provider, ts);

CREATE TABLE IF NOT EXISTS provider_state (
  provider      TEXT PRIMARY KEY,
  demoted_until REAL,
  last_error    TEXT,
  observed_rpd  INTEGER,
  rpd_day_pt    TEXT
);
"""

MIGRATIONS = ["ALTER TABLE provider_state ADD COLUMN rpd_day_pt TEXT"]


def day_pt(when: float | None = None) -> str:
    return datetime.fromtimestamp(when or time.time(), PT).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Limits:
    rpm: int
    rpd: int


class QuotaStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        c = self._conn()
        c.executescript(SCHEMA)
        for stmt in MIGRATIONS:
            with contextlib.suppress(sqlite3.OperationalError):
                c.execute(stmt)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = db.connect(self.path, rows=False)
            self._local.c = c
        return c

    def record(
        self,
        provider: str,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ok: bool = True,
        error_kind: str | None = None,
    ) -> None:
        now = time.time()
        c = self._conn()
        c.execute(
            "INSERT INTO llm_usage(ts,day_pt,provider,model,input_tokens,output_tokens,ok,error_kind)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (now, day_pt(now), provider, model, input_tokens, output_tokens, int(ok), error_kind),
        )
        c.commit()

    def note_daily_limit(self, provider: str, reported: int | None = None) -> None:
        """Record today's daily ceiling."""
        used = reported if reported else self.requests_today(provider)
        c = self._conn()
        c.execute(
            "INSERT INTO provider_state(provider, observed_rpd, rpd_day_pt)"
            " VALUES (?,?,?) ON CONFLICT(provider) DO UPDATE SET"
            " observed_rpd=excluded.observed_rpd, rpd_day_pt=excluded.rpd_day_pt",
            (provider, max(1, used), day_pt()),
        )
        c.commit()

    def demote(self, provider: str, until: float, reason: str) -> None:
        c = self._conn()
        c.execute(
            "INSERT INTO provider_state(provider, demoted_until, last_error) VALUES (?,?,?)"
            " ON CONFLICT(provider) DO UPDATE SET demoted_until=excluded.demoted_until,"
            " last_error=excluded.last_error",
            (provider, until, reason),
        )
        c.commit()

    def requests_today(self, provider: str) -> int:
        row = (
            self._conn()
            .execute(
                "SELECT COUNT(*) FROM llm_usage WHERE provider=? AND day_pt=?",
                (provider, day_pt()),
            )
            .fetchone()
        )
        return row[0] if row else 0

    def requests_last_minute(self, provider: str) -> int:
        row = (
            self._conn()
            .execute(
                "SELECT COUNT(*) FROM llm_usage WHERE provider=? AND ts>?",
                (provider, time.time() - 60),
            )
            .fetchone()
        )
        return row[0] if row else 0

    def demoted_until(self, provider: str) -> float:
        row = (
            self._conn()
            .execute("SELECT demoted_until FROM provider_state WHERE provider=?", (provider,))
            .fetchone()
        )
        return (row[0] or 0.0) if row else 0.0

    def observed_rpd(self, provider: str) -> int | None:
        row = (
            self._conn()
            .execute(
                "SELECT observed_rpd FROM provider_state WHERE provider=? AND rpd_day_pt=?",
                (provider, day_pt()),
            )
            .fetchone()
        )
        return row[0] if row and row[0] else None

    def is_available(self, provider: str, limits: Limits, headroom: float = 0.9) -> bool:
        """Preflight."""
        if time.time() < self.demoted_until(provider):
            return False
        rpd = self.observed_rpd(provider) or limits.rpd
        if self.requests_today(provider) >= rpd * headroom:
            return False
        return not self.requests_last_minute(provider) >= limits.rpm * headroom

    def snapshot(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for (p,) in self._conn().execute(
            "SELECT DISTINCT provider FROM llm_usage UNION SELECT provider FROM provider_state"
        ):
            out[p] = {
                "requests_today": self.requests_today(p),
                "requests_last_minute": self.requests_last_minute(p),
                "observed_rpd": self.observed_rpd(p),
                "demoted_until": self.demoted_until(p),
            }
        return out
