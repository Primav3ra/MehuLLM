"""Free-tier quota accounting.

There is no billing API to lean on, so usage is counted locally in SQLite and
the guard runs BEFORE each request. Three things worth knowing:

* Gemini's free-tier RPM/RPD are no longer published. So the configured numbers
  are a guess, and the runtime LEARNS the real ceiling by recording the request
  count at the moment of each 429 (`observed_rpd`).
* The day boundary is midnight America/Los_Angeles, not local midnight.
  `date.today()` would roll over at the wrong time and let the guard pass while
  the provider is still refusing.
* Token counts are estimated locally. Calling a count_tokens endpoint per
  request would spend the very quota we are trying to protect.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
  observed_rpd  INTEGER
);
"""


def day_pt(when: float | None = None) -> str:
    return datetime.fromtimestamp(when or time.time(), PT).strftime("%Y-%m-%d")


def estimate_tokens(text: str) -> int:
    """~4 chars/token plus per-message overhead. Deliberately rough: this feeds
    a safety guard, and the authoritative number arrives later on UsageEvent."""
    return len(text) // 4 + 8


@dataclass(frozen=True)
class Limits:
    rpm: int
    rpd: int


class QuotaStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
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

    def note_rate_limited(self, provider: str) -> None:
        """Learn the real ceiling: whatever we managed today IS the limit."""
        used = self.requests_today(provider)
        c = self._conn()
        c.execute(
            "INSERT INTO provider_state(provider, observed_rpd) VALUES (?,?)"
            " ON CONFLICT(provider) DO UPDATE SET observed_rpd="
            "  CASE WHEN observed_rpd IS NULL OR excluded.observed_rpd < observed_rpd"
            "       THEN excluded.observed_rpd ELSE observed_rpd END",
            (provider, used),
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
        row = self._conn().execute(
            "SELECT COUNT(*) FROM llm_usage WHERE provider=? AND day_pt=?",
            (provider, day_pt()),
        ).fetchone()
        return row[0] if row else 0

    def requests_last_minute(self, provider: str) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM llm_usage WHERE provider=? AND ts>?",
            (provider, time.time() - 60),
        ).fetchone()
        return row[0] if row else 0

    def demoted_until(self, provider: str) -> float:
        row = self._conn().execute(
            "SELECT demoted_until FROM provider_state WHERE provider=?", (provider,)
        ).fetchone()
        return (row[0] or 0.0) if row else 0.0

    def observed_rpd(self, provider: str) -> int | None:
        row = self._conn().execute(
            "SELECT observed_rpd FROM provider_state WHERE provider=?", (provider,)
        ).fetchone()
        return row[0] if row and row[0] else None

    def is_available(self, provider: str, limits: Limits, headroom: float = 0.9) -> bool:
        """Preflight. Refuse at 90% so the last request does not become a 429
        mid-turn, which would surface to the user as a broken conversation."""
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
