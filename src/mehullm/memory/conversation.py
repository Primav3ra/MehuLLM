"""Short-term memory: turns that persist across requests."""

from __future__ import annotations

import time

from mehullm.llm.types import Msg
from mehullm.memory.store import MemoryStore

KEEP_TURNS = 12


class ConversationStore:
    def __init__(self, store: MemoryStore, keep_turns: int = KEEP_TURNS):
        self.store = store
        self.keep_turns = keep_turns

    def history(self, conversation_id: str) -> list[Msg]:
        """Replay the recent turns of a conversation as canonical messages."""
        rows = (
            self.store.conn()
            .execute(
                "SELECT role, content FROM turns"
                " WHERE session_id=? AND role IN ('user','assistant') AND content <> ''"
                " ORDER BY idx DESC LIMIT ?",
                (conversation_id, self.keep_turns),
            )
            .fetchall()
        )
        return [Msg(role=r["role"], text=r["content"]) for r in reversed(rows)]

    def append(self, conversation_id: str, role: str, text: str, trace_id: str = "") -> None:
        if not (text or "").strip():
            return
        c = self.store.conn()
        now = time.time()
        c.execute(
            "INSERT INTO sessions(id, started_at, last_at) VALUES (?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET last_at=excluded.last_at",
            (conversation_id, now, now),
        )
        nxt = c.execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM turns WHERE session_id=?",
            (conversation_id,),
        ).fetchone()[0]
        c.execute(
            "INSERT INTO turns(session_id, idx, role, content, n_tokens, trace_id, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (conversation_id, nxt, role, text, len(text) // 4, trace_id, now),
        )
        c.commit()

    def recent(self, conversation_id: str, limit: int = 50) -> list[dict]:
        rows = (
            self.store.conn()
            .execute(
                "SELECT idx, role, content, ts, trace_id FROM turns"
                " WHERE session_id=? ORDER BY idx DESC LIMIT ?",
                (conversation_id, limit),
            )
            .fetchall()
        )
        return [dict(r) for r in reversed(rows)]

    def conversations(self, limit: int = 50) -> list[dict]:
        rows = (
            self.store.conn()
            .execute(
                "SELECT s.id, s.started_at, s.last_at, COUNT(t.id) AS turns"
                " FROM sessions s LEFT JOIN turns t ON t.session_id = s.id"
                " GROUP BY s.id ORDER BY s.last_at DESC LIMIT ?",
                (limit,),
            )
            .fetchall()
        )
        return [dict(r) for r in rows]
