"""Pipeline CLI.

    uv run mehullm-parse census   data/raw/            # the week-2 go/no-go gate
    uv run mehullm-parse census   data/raw/ --self "Mehul"
    uv run mehullm-parse contacts data/raw/            # emit contacts.json to tag 'self'
    uv run mehullm-parse peek     data/raw/chat.txt    # eyeball the parse
    uv run mehullm-parse build    --self "Mehul Karwa" # build the SFT dataset
    uv run mehullm-parse audit    -n 100               # PII audit sample (week-3 gate)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from mehullm.pipeline.census import format_report, run_census
from mehullm.pipeline.whatsapp_parser import parse_export


def _cmd_census(args: argparse.Namespace) -> int:
    census = run_census(args.path, self_alias=args.self_alias)
    print(format_report(census))
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "likely_self": census.likely_self,
                    "pairs_1to1": census.pairs_1to1,
                    "pairs_group": census.pairs_group,
                    "my_turns": census.my_turns,
                    "verdict": census.verdict[0],
                    "chats": [
                        {"chat_id": c.chat_id, "content": c.content, "is_group": c.is_group}
                        for c in census.chats
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0 if census.verdict[0] != "RED" else 2


def _cmd_contacts(args: argparse.Namespace) -> int:
    """Emit contacts.json listing every sender, for the user to tag 'self'.

    Deliberately not automatic: guessing wrong means training on someone
    else's voice, and the failure is invisible until the model sounds off.
    """
    root = Path(args.path)
    counts: Counter[str] = Counter()
    per_chat: dict[str, list[str]] = {}
    # Dedup by resolved path -- see the note in census.run_census.
    for f in sorted({p.resolve() for p in root.rglob("*") if p.suffix.lower() == ".txt"}):
        chat = parse_export(f)
        senders = [s for s in chat.participants]
        per_chat[chat.chat_id] = senders
        counts.update(chat.participants)

    out = {
        "_instructions": (
            "Set 'self': true for EVERY alias that is you. WhatsApp renders you "
            "by your profile name, which changes over the years, so there may be "
            "several. Everything left false is treated as someone else."
        ),
        "aliases": [
            {"name": name, "messages": n, "self": False} for name, n in counts.most_common()
        ],
        "per_chat": per_chat,
    }
    dest = Path(args.out)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest} with {len(counts)} aliases -- tag yours with \"self\": true")
    return 0


def _cmd_peek(args: argparse.Namespace) -> int:
    chat = parse_export(args.path)
    print(f"chat_id={chat.chat_id}  date_order={chat.date_order}  ({chat.date_order_evidence})")
    print(f"messages={len(chat.messages)}  content={len(chat.content_messages)}  "
          f"unparsed_lines={chat.unparsed_lines}\n")
    for m in chat.messages[: args.n]:
        tag = (
            "SYS " if m.is_system
            else "MED " if m.is_media
            else "DEL " if m.is_tombstone
            else "    "
        )
        text = m.text.replace("\n", " ⏎ ")
        print(f"{tag}{m.ts:%Y-%m-%d %H:%M}  {(m.sender or '-'):<14}  {text[:90]}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from mehullm.pipeline.build_sft import build, format_report

    aliases = {a.strip() for a in args.self_alias if a.strip()}
    if not aliases:
        print("error: --self is required (pass it once per alias you used)", file=sys.stderr)
        return 1
    stats = build(
        raw_dir=args.path,
        out_path=args.out,
        self_aliases=aliases,
        name_map_path=args.name_map,
        seed=args.seed,
    )
    print(format_report(stats, args.out))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Dump a random sample for the manual PII audit.

    This is a human gate, deliberately. Automated PII detection has a recall
    ceiling and the cost of a miss here is a private detail baked into model
    weights, which cannot be un-baked.
    """
    import random

    path = Path(args.pairs)
    if not path.exists():
        print(f"error: {path} not found -- run `build` first", file=sys.stderr)
        return 1

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"PII AUDIT SAMPLE -- {len(sample)} of {len(records):,} pairs\n")
        fh.write("Read every one. Look for: real names, phone numbers, addresses,\n")
        fh.write("account details, anything you would not want in model weights.\n")
        fh.write("=" * 70 + "\n\n")
        for i, r in enumerate(sample, 1):
            fh.write(f"--- {i}/{len(sample)}  [{r['split']}]  {r['chat_id'][:40]}\n")
            for turn in r["context"]:
                fh.write(f"    {turn['sender']}: {turn['text']}\n")
            fh.write(f"  >> ME: {r['target']}\n\n")
    print(f"wrote {out} -- {len(sample)} pairs. Read all of them before training.")
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
            args.pairs,
            args.out,
            model=args.model,
            cache_path=args.cache,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            concurrency=args.concurrency,
            on_progress=progress,
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

    c = sub.add_parser("census", help="corpus statistics + go/no-go verdict")
    c.add_argument("path", nargs="?", default="data/raw")
    c.add_argument("--self", dest="self_alias", default=None, help="your alias, if known")
    c.add_argument("--json", default=None, help="also write machine-readable output here")
    c.set_defaults(fn=_cmd_census)

    k = sub.add_parser("contacts", help="emit contacts.json for tagging 'self'")
    k.add_argument("path", nargs="?", default="data/raw")
    k.add_argument("--out", default="data/derived/contacts.json")
    k.set_defaults(fn=_cmd_contacts)

    v = sub.add_parser("peek", help="print the first N parsed messages of one file")
    v.add_argument("path")
    v.add_argument("-n", type=int, default=30)
    v.set_defaults(fn=_cmd_peek)

    b = sub.add_parser("build", help="build the SFT dataset (pairs.jsonl)")
    b.add_argument("path", nargs="?", default="data/raw")
    b.add_argument("--self", dest="self_alias", action="append", required=True,
                   help="your alias; repeat the flag once per alias you have used")
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

    a = sub.add_parser("audit", help="dump a random sample for the manual PII audit")
    a.add_argument("--pairs", default="data/derived/pairs.jsonl")
    a.add_argument("-n", type=int, default=100)
    a.add_argument("--out", default="data/derived/audit_sample.txt")
    a.add_argument("--seed", type=int, default=1)
    a.set_defaults(fn=_cmd_audit)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
