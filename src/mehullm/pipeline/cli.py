"""Pipeline CLI.

    uv run mehullm-parse peek  data/raw/chat.txt     # eyeball the parse
    uv run mehullm-parse build --self "Mehul Karwa"  # build the SFT dataset
    uv run mehullm-parse neutralize                  # rewriter drafts via Ollama
"""

from __future__ import annotations

import argparse
import sys

from mehullm.pipeline.whatsapp_parser import parse_export


def _cmd_peek(args: argparse.Namespace) -> int:
    chat = parse_export(args.path)
    print(f"chat_id={chat.chat_id}  date_order={chat.date_order}  ({chat.date_order_evidence})")
    print(f"messages={len(chat.messages)}  content={len(chat.content_messages)}  "
          f"unparsed_lines={chat.unparsed_lines}\n")
    for m in chat.messages[: args.n]:
        tag = ("SYS " if m.is_system else "MED " if m.is_media
               else "DEL " if m.is_tombstone else "    ")
        print(f"{tag}{m.ts:%Y-%m-%d %H:%M}  {(m.sender or '-'):<14}  "
              f"{m.text.replace(chr(10), ' / ')[:90]}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from mehullm.pipeline.build_sft import build, format_report

    aliases = {a.strip() for a in args.self_alias if a.strip()}
    if not aliases:
        print("error: --self is required (pass it once per alias)", file=sys.stderr)
        return 1
    print(format_report(build(
        raw_dir=args.path, out_path=args.out, self_aliases=aliases,
        name_map_path=args.name_map, seed=args.seed,
    ), args.out))
    return 0


def _cmd_neutralize(args: argparse.Namespace) -> int:
    from mehullm.pipeline.neutralize import format_report, neutralize
    from mehullm.voice.client import OllamaError

    def progress(done: int, total: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate / 60 if rate else 0
        print(f"  {done:,}/{total:,}  {rate:.1f}/s  ETA {eta:.0f} min", flush=True)

    try:
        stats = neutralize(
            args.pairs, args.out, model=args.model, cache_path=args.cache,
            train_limit=args.train_limit, val_limit=args.val_limit,
            concurrency=args.concurrency, on_progress=progress,
        )
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(format_report(stats, args.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mehullm-parse", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("peek", help="print the first N parsed messages of one file")
    v.add_argument("path")
    v.add_argument("-n", type=int, default=30)
    v.set_defaults(fn=_cmd_peek)

    b = sub.add_parser("build", help="build the SFT dataset")
    b.add_argument("path", nargs="?", default="data/raw")
    b.add_argument("--self", dest="self_alias", action="append", required=True,
                   help="your alias; repeat once per alias you have used")
    b.add_argument("--out", default="data/derived/pairs.jsonl")
    b.add_argument("--name-map", default="data/derived/name_map.json")
    b.add_argument("--seed", type=int, default=3407)
    b.set_defaults(fn=_cmd_build)

    z = sub.add_parser("neutralize", help="generate rewriter drafts locally via Ollama")
    z.add_argument("--pairs", default="data/derived/pairs.jsonl")
    z.add_argument("--out", default="data/derived/sft_pairs.jsonl")
    z.add_argument("--cache", default="data/derived/draft_cache.db")
    z.add_argument("--model", default="qwen3:1.7b")
    z.add_argument("--train-limit", type=int, default=12000)
    z.add_argument("--val-limit", type=int, default=1500)
    z.add_argument("--concurrency", type=int, default=2)
    z.set_defaults(fn=_cmd_neutralize)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
