"""Memory CLI.

    uv run mehullm-memory search "where do I live"
    uv run mehullm-memory stats
"""

from __future__ import annotations

import argparse
import sys

from mehullm.memory.retrieve import render_memory_block, search_facts, search_style
from mehullm.memory.store import MemoryStore

DEFAULT_DB = "data/derived/memory.db"


def _cmd_search(a: argparse.Namespace) -> int:
    store = MemoryStore(a.db)
    facts = search_facts(store, a.query, k=a.k)
    print("FACTS")
    print(render_memory_block(facts) if facts else "  (none)")
    if a.style:
        print("\nSTYLE EXEMPLARS")
        for h in search_style(store, a.query, k=5):
            print(f"  [{h.score:.4f}] {h.text[:110]}")
    return 0


def _cmd_stats(a: argparse.Namespace) -> int:
    for k, v in MemoryStore(a.db).stats().items():
        print(f"  {k:<20} {v:,}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mehullm-memory")
    p.add_argument("--db", default=DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="query memory")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--style", action="store_true")
    s.set_defaults(fn=_cmd_search)

    st = sub.add_parser("stats", help="counts")
    st.set_defaults(fn=_cmd_stats)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
