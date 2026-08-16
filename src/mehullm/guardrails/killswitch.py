"""Kill switch and token-bucket rate limiting.

The kill switch is PERSISTED. An in-memory flag that clears on the next
`uvicorn --reload` is not a kill switch -- it is a suggestion. It is checked at
three points: request admission, before every tool call, and between loop steps.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


class Killed(RuntimeError):
    pass


class KillSwitch:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.event = asyncio.Event()
        self.reason = ""
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS flags(key TEXT PRIMARY KEY, value TEXT)")

    def load(self) -> None:
        """Survives restarts -- that is the entire point."""
        with sqlite3.connect(self.path) as c:
            row = c.execute("SELECT value FROM flags WHERE key='kill'").fetchone()
        if row and row[0]:
            self.reason = row[0]
            self.event.set()

    def engage(self, reason: str) -> None:
        self.reason = reason or "engaged"
        with sqlite3.connect(self.path) as c:
            c.execute(
                "INSERT INTO flags VALUES('kill',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self.reason,),
            )
        self.event.set()

    def release(self) -> None:
        self.reason = ""
        with sqlite3.connect(self.path) as c:
            c.execute("DELETE FROM flags WHERE key='kill'")
        self.event.clear()

    @property
    def engaged(self) -> bool:
        return self.event.is_set()

    def check(self) -> None:
        if self.event.is_set():
            raise Killed(f"kill switch engaged: {self.reason}")


@dataclass
class Bucket:
    capacity: int
    per_seconds: float
    tokens: float = field(init=False)
    updated: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(
            self.capacity, self.tokens + (now - self.updated) * self.capacity / self.per_seconds
        )
        self.updated = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


class RateLimited(RuntimeError):
    pass


class RateLimiter:
    """Per-tool and per-server token buckets.

    Matters most for metered free tiers: Alpha Vantage allows ~25 requests/day,
    and without this the agent discovers that cap by burning it in one turn.
    """

    def __init__(self, per_tool: dict[str, tuple[int, float]] | None = None):
        self._specs = per_tool or {}
        self._buckets: dict[str, Bucket] = {}
        self._server = defaultdict(lambda: Bucket(60, 60.0))

    def acquire(self, tool: str, server: str) -> None:
        spec = self._specs.get(tool)
        if spec:
            b = self._buckets.setdefault(tool, Bucket(spec[0], spec[1]))
            if not b.take():
                raise RateLimited(f"rate limit reached for {tool}")
        if not self._server[server].take():
            raise RateLimited(f"rate limit reached for server {server}")
