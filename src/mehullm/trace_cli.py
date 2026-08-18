"""`mehullm-trace` -- read traces from the CLI."""

from __future__ import annotations

import argparse
import sys

from mehullm.obs import utf8_stdout
from mehullm.persistence.tracing import TraceStore, render_tree
from mehullm.settings import settings


def main(argv: list[str] | None = None) -> int:
    utf8_stdout()
    p = argparse.ArgumentParser(prog="mehullm-trace")
    p.add_argument("--db", default=settings.mehullm_db)
    sub = p.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show", help="render one trace as a span tree")
    show.add_argument("trace_id")

    ls = sub.add_parser("list", help="recent traces")
    ls.add_argument("--limit", type=int, default=20)

    a = p.parse_args(argv)
    store = TraceStore(a.db)

    if a.cmd == "list":
        rows = store.recent(a.limit)
        if not rows:
            print("no traces yet")
            return 0
        print(f"  {'trace_id':<18}{'status':<10}{'ms':>7}{'tokens':>8}  session")
        for r in rows:
            ms = int(((r["ended_at"] or r["started_at"]) - r["started_at"]) * 1000)
            print(
                f"  {r['trace_id']:<18}{r['status'] or '?':<10}{ms:>7}"
                f"{r['total_tokens'] or 0:>8}  {r['session_id'] or ''}"
            )
        return 0

    data = store.get(a.trace_id)
    if not data["trace"]:
        print(f"no such trace: {a.trace_id}", file=sys.stderr)
        return 1
    print(render_tree(data))
    return 0
