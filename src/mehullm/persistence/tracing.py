"""Tracing: plain SQLite with OTel-shaped column names."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mehullm.persistence import db
from mehullm.pipeline.pii import scrub

SpanKind = Literal[
    "llm_call", "tool_call", "retrieval", "guardrail", "voice_rewrite", "embed", "turn"
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  trace_id      TEXT PRIMARY KEY,
  session_id    TEXT,
  user_input    TEXT,
  final_output  TEXT,
  started_at    REAL,
  ended_at      REAL,
  status        TEXT,
  total_tokens  INTEGER DEFAULT 0,
  error         TEXT
);
CREATE TABLE IF NOT EXISTS spans (
  span_id        TEXT PRIMARY KEY,
  trace_id       TEXT,
  parent_span_id TEXT,
  name           TEXT,
  kind           TEXT,
  started_at     REAL,
  ended_at       REAL,
  duration_ms    INTEGER,
  provider       TEXT,
  model          TEXT,
  tokens_in      INTEGER DEFAULT 0,
  tokens_out     INTEGER DEFAULT 0,
  input_redacted TEXT,
  output_redacted TEXT,
  attributes     TEXT,
  status         TEXT,
  error          TEXT
);
CREATE INDEX IF NOT EXISTS ix_spans_trace ON spans(trace_id, started_at);
CREATE INDEX IF NOT EXISTS ix_traces_time ON traces(started_at DESC);
"""

MAX_PAYLOAD = 4000


class TraceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = db.connect(self.path)
            self._local.c = c
        return c

    def start_trace(self, trace_id: str, session_id: str, user_input: str) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO traces(trace_id,session_id,user_input,started_at,status)"
            " VALUES (?,?,?,?,'running')",
            (trace_id, session_id, _payload(user_input), time.time()),
        )
        c.commit()

    def end_trace(
        self, trace_id: str, *, status: str, final_output: str = "", error: str = ""
    ) -> None:
        c = self._conn()
        c.execute(
            "UPDATE traces SET ended_at=?, status=?, final_output=?, error=?,"
            " total_tokens=(SELECT COALESCE(SUM(tokens_in+tokens_out),0) FROM spans"
            "               WHERE spans.trace_id=traces.trace_id)"
            " WHERE trace_id=?",
            (time.time(), status, _payload(final_output), error[:500], trace_id),
        )
        c.commit()

    def span(self, trace_id: str, name: str, kind: SpanKind, **attrs: Any) -> Span:
        return Span(self, trace_id, name, kind, attrs, parent_span_id=_parent.get())

    def _write_span(self, s: Span) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO spans(span_id,trace_id,parent_span_id,name,kind,"
            "started_at,ended_at,duration_ms,provider,model,tokens_in,tokens_out,"
            "input_redacted,output_redacted,attributes,status,error)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                s.span_id,
                s.trace_id,
                s.parent_span_id,
                s.name,
                s.kind,
                s.started_at,
                s.ended_at,
                s.duration_ms,
                s.provider,
                s.model,
                s.tokens_in,
                s.tokens_out,
                _payload(s.input_text),
                _payload(s.output_text),
                json.dumps(s.attrs, default=str),
                s.status,
                s.error[:500],
            ),
        )
        c.commit()

    def get(self, trace_id: str) -> dict[str, Any]:
        c = self._conn()
        t = c.execute("SELECT * FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
        spans = c.execute(
            "SELECT * FROM spans WHERE trace_id=? ORDER BY started_at", (trace_id,)
        ).fetchall()
        return {
            "trace": dict(t) if t else None,
            "spans": [dict(s) for s in spans],
        }

    def recent(self, limit: int = 20) -> list[dict]:
        return [
            dict(r)
            for r in self._conn().execute(
                "SELECT trace_id, session_id, started_at, ended_at, status, total_tokens"
                " FROM traces ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        ]

    def prune(self, older_than_days: int = 30) -> int:
        """Keep metrics forever, drop payloads after 30 days."""
        cutoff = time.time() - older_than_days * 86400
        c = self._conn()
        n = c.execute(
            "UPDATE spans SET input_redacted=NULL, output_redacted=NULL"
            " WHERE trace_id IN (SELECT trace_id FROM traces WHERE started_at < ?)",
            (cutoff,),
        ).rowcount
        c.execute(
            "UPDATE traces SET user_input=NULL, final_output=NULL WHERE started_at < ?",
            (cutoff,),
        )
        c.commit()
        return n


@dataclass
class Span:
    store: TraceStore
    trace_id: str
    name: str
    kind: SpanKind
    attrs: dict[str, Any] = field(default_factory=dict)
    span_id: str = field(default_factory=lambda: f"sp_{uuid.uuid4().hex[:12]}")
    parent_span_id: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    duration_ms: int = 0
    provider: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    input_text: str = ""
    output_text: str = ""
    status: str = "ok"
    error: str = ""
    _token: Any = None

    def __enter__(self) -> Span:
        self._token = _parent.get()
        _parent.set(self.span_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _parent.set(self._token)
        if exc is not None:
            self.status = "error"
            self.error = f"{exc_type.__name__}: {exc}"
        self.ended_at = time.time()
        self.duration_ms = int((self.ended_at - self.started_at) * 1000)
        self.store._write_span(self)
        return False


# ------------------------------------------------------ ambient tracer

_store: TraceStore | None = None
_trace: ContextVar[str | None] = ContextVar("trace_id", default=None)
_parent: ContextVar[str | None] = ContextVar("parent_span_id", default=None)


def set_store(store: TraceStore | None) -> None:
    global _store
    _store = store


def bind_trace(trace_id: str | None) -> None:
    _trace.set(trace_id)


def record_span(name: str, kind: SpanKind, duration_ms: int, **attrs: Any) -> None:
    """Write a completed span."""
    tid = _trace.get()
    if _store is None or not tid:
        return
    s = Span(_store, tid, name, kind, attrs, parent_span_id=_parent.get())
    s.provider = str(attrs.pop("provider", "") or "")
    s.model = str(attrs.pop("model", "") or "")
    s.tokens_in = int(attrs.pop("tokens_in", 0) or 0)
    s.tokens_out = int(attrs.pop("tokens_out", 0) or 0)
    s.status = str(attrs.pop("status", "ok") or "ok")
    s.error = str(attrs.pop("error", "") or "")
    s.ended_at = time.time()
    s.duration_ms = duration_ms
    _store._write_span(s)


@contextlib.contextmanager
def span(name: str, kind: SpanKind, **attrs: Any) -> Iterator[Any]:
    """Emit a span, or a no-op shim when tracing is unconfigured."""
    tid = _trace.get()
    if _store is None or not tid:
        yield _Noop()
        return
    with _store.span(tid, name, kind, **attrs) as s:
        yield s


class _Noop:
    """Absorbs attribute writes so call sites need no conditionals."""

    def __setattr__(self, k: str, v: Any) -> None:
        pass

    def __getattr__(self, k: str) -> Any:
        return ""


def _payload(text: str | None) -> str | None:
    """Redact before storing. Traces must never become a plaintext archive."""
    if not text:
        return text
    return scrub(text[:MAX_PAYLOAD])


def _header(t: dict) -> list[str]:
    dur = (t["ended_at"] or time.time()) - (t["started_at"] or 0)
    return [
        f"trace {t['trace_id']}  [{t['status']}]  {dur:.1f}s  {t['total_tokens']} tokens",
        f"  in : {(t['user_input'] or '')[:100]}",
        f"  out: {(t['final_output'] or '')[:100]}",
        "",
    ]


def _span_lines(s: dict, widest: int) -> list[str]:
    ms = s["duration_ms"] or 0
    bar = "█" * max(1, int(24 * ms / max(1, widest)))
    tok = f"{s['tokens_in']}→{s['tokens_out']}" if (s["tokens_in"] or s["tokens_out"]) else ""
    mark = "!" if s["status"] == "error" else " "
    lines = [f" {mark}{s['kind']:<14}{s['name'][:26]:<28}{ms:>6}ms {bar:<25}{tok}"]
    if s["status"] == "error":
        lines.append(f"    error: {(s['error'] or '')[:120]}")
    return lines


def render_tree(data: dict) -> str:
    """Human-readable span tree. `uv run mehullm-trace show <id>` uses this."""
    t, spans = data.get("trace"), data.get("spans", [])
    if not t:
        return "trace not found"

    widest = max((s["duration_ms"] or 0) for s in spans) if spans else 1
    out = _header(t)
    for s in spans:
        out.extend(_span_lines(s, widest))
    return "\n".join(out)
