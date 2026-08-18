"""Memory CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mehullm.memory.retrieve import render_memory_block, search_facts, search_style
from mehullm.memory.store import MemoryStore
from mehullm.obs import utf8_stdout
from mehullm.settings import settings

DEFAULT_DB = settings.memory_db


def _cmd_load(a: argparse.Namespace) -> int:
    from mehullm.memory.facts import format_report, load_dir

    st = load_dir(a.db, a.dir)
    print(format_report(st))
    if not st.files:
        print(f"\n  nothing in {a.dir}/ yet -- fill in the templates first")
    return 1 if st.errors else 0


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


def _cmd_index(a: argparse.Namespace) -> int:
    """Index chat exports as style exemplars."""
    import json

    from mehullm.memory.index_chats import index

    aliases = set(a.self_alias or [])
    if a.contacts and Path(a.contacts).exists():
        tagged = json.loads(Path(a.contacts).read_text(encoding="utf-8"))
        aliases |= {k for k, v in tagged.items() if v == "self"}
    if not aliases:
        print("no self aliases: pass --self-alias or a --contacts file", file=sys.stderr)
        return 1
    st = index(a.raw_dir, a.db, aliases, on_progress=lambda n, f: print(f"  {n:>6} {f}"))
    for k, v in vars(st).items():
        print(f"  {k:<20} {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    utf8_stdout()
    p = argparse.ArgumentParser(prog="mehullm-memory")
    p.add_argument("--db", default=DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    ld = sub.add_parser("load", help="load facts/*.yaml into memory (idempotent)")
    ld.add_argument("--dir", default="facts")
    ld.set_defaults(fn=_cmd_load)

    s = sub.add_parser("search", help="query memory")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--style", action="store_true")
    s.set_defaults(fn=_cmd_search)

    ix = sub.add_parser("index", help="index chat exports as style exemplars")
    ix.add_argument("--raw-dir", default=str(Path(settings.mehullm_derived_dir).parent / "raw"))
    ix.add_argument("--self-alias", action="append", help="repeatable")
    ix.add_argument("--contacts", default=str(Path(settings.mehullm_derived_dir) / "contacts.json"))
    ix.set_defaults(fn=_cmd_index)

    st = sub.add_parser("stats", help="counts")
    st.set_defaults(fn=_cmd_stats)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
