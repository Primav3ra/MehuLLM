"""Shared SQLite connection setup."""

from __future__ import annotations

import sqlite3
from pathlib import Path

TIMEOUT_S = 30


def connect(path: str | Path, *, rows: bool = True, vec: bool = False) -> sqlite3.Connection:
    """A WAL connection. check_same_thread is off because callers cache one per thread."""
    c = sqlite3.connect(str(path), timeout=TIMEOUT_S, check_same_thread=False)
    if vec:
        import sqlite_vec

        c.enable_load_extension(True)
        sqlite_vec.load(c)
        # Left enabled, any later SQL could load an arbitrary shared library.
        c.enable_load_extension(False)
    c.execute("PRAGMA journal_mode=WAL")
    if rows:
        c.row_factory = sqlite3.Row
    return c
