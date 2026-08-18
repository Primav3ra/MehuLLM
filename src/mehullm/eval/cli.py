"""Eval CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mehullm.eval.bank import coverage, load, validate
from mehullm.obs import utf8_stdout
from mehullm.settings import settings

DEFAULT_BANK = "evals/scenarios"
DEFAULT_DB = settings.memory_db

# §10 targets. Deviating is allowed, but it should be a decision, not a drift.
TARGETS = {
    "style": 12,
    "factual_recall": 12,
    "tool_selection": 10,
    "refusal": 8,
    "multistep": 8,
    "hallucination_trap": 6,
    "prompt_injection": 4,
}


def _cmd_validate(a: argparse.Namespace) -> int:
    scenarios = load(a.bank)
    errs = validate(scenarios)
    cov = coverage(scenarios)

    print("=" * 62)
    print(f"  Scenario bank — {len(scenarios)} scenarios from {a.bank}")
    print("=" * 62)
    for cat, n in cov.items():
        want = TARGETS.get(cat, 0)
        mark = "ok " if n >= want else "LOW"
        print(f"  {mark} {cat:<20} {n:>3} / {want}")
    print(f"\n  weighted total {sum(s.weight for s in scenarios)}")
    print(f"  judged         {sum(s.judged for s in scenarios)}   (style is never LLM-judged)")
    print(f"  style probes   {sum(s.style_probe for s in scenarios)}")
    print(f"  with injection {sum(bool(s.inject) for s in scenarios)}")

    if errs:
        print(f"\n  {len(errs)} PROBLEM(S):")
        for e in errs:
            print(f"    [{e.scenario_id}] {e.problem}")
        return 1
    print("\n  all scenarios valid")
    return 0


def _cmd_list(a: argparse.Namespace) -> int:
    for s in load(a.bank):
        if a.category and s.category != a.category:
            continue
        flags = "".join(
            ("J" if s.judged else "-", "S" if s.style_probe else "-", "I" if s.inject else "-")
        )
        print(f"  {s.id:<38} {s.category:<20} w{s.weight} {flags}  {s.prompt[:44]}")
    return 0


def _cmd_run(a: argparse.Namespace) -> int:
    import asyncio

    from mehullm.eval.runner import BankRunner, compare_to_last_green, persist

    scenarios = [s for s in load(a.bank) if not a.category or s.category == a.category]
    if a.only:
        scenarios = [s for s in scenarios if s.id in set(a.only.split(","))]
    if not scenarios:
        print("no scenarios matched", file=sys.stderr)
        return 1

    errs = validate(scenarios)
    if errs:
        print("bank does not validate; fix it before running:", file=sys.stderr)
        for e in errs:
            print(f"  [{e.scenario_id}] {e.problem}", file=sys.stderr)
        return 1

    # The CALL must be inside the guard, not just the import -- a missing API key.
    try:
        from mehullm.eval.wiring import build_loop_factory

        factory = build_loop_factory(voice=not a.no_voice)
    except ImportError as e:
        print(f"cannot build an agent: {e}", file=sys.stderr)
        print("The bank validates offline; running it needs a configured brain.", file=sys.stderr)
        return 2
    runner = BankRunner(loop_factory=factory, base_db=a.db)

    def show(r) -> None:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.scenario.id:<38} {r.transcript.latency_ms:>6} ms")
        for f in r.failures:
            print(f"         {f}")

    print(f"running {len(scenarios)} scenarios...\n")
    rep = asyncio.run(runner.run_bank(scenarios, on_result=show))

    print("\n" + "=" * 62)
    print(f"  weighted pass rate  {rep.pass_rate * 100:.1f}%")
    for cat, (p, n) in rep.by_category().items():
        print(f"    {cat:<20} {p}/{n}")
    print(f"  elapsed {rep.elapsed_s / 60:.1f} min")

    rid = persist(a.db, rep, tag=a.tag)
    print(f"  saved as eval run #{rid}")

    for alarm in compare_to_last_green(a.db, rep, a.tag):
        print(f"  DRIFT: {alarm}")
    print("=" * 62)
    return 0 if rep.pass_rate >= a.threshold else 1


def _cmd_style(a: argparse.Namespace) -> int:
    from mehullm.eval.style_score import compare, normalised

    def lines(p: str) -> list[str]:
        return [ln for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]

    s = compare(lines(a.gen), lines(a.ref), perplexity_gain=a.ppl_gain)
    print(s.report())
    # Never report the raw total alone (§10). Against the human ceiling and the.
    print(f"\n  floor {a.floor:.3f}   ceiling {a.ceiling:.3f}")
    print(
        f"  NORMALISED {normalised(s.total, a.floor, a.ceiling):.3f}"
        "   (0 = raw model, 1 = human-level)"
    )
    return 0


def _cmd_history(a: argparse.Namespace) -> int:
    import sqlite3

    from mehullm.eval.runner import SCHEMA

    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)
    rows = con.execute(
        "SELECT id, started_at, git_sha, brain_model, pass_rate, style_score, tag"
        " FROM eval_runs ORDER BY id DESC LIMIT ?",
        (a.limit,),
    ).fetchall()
    if not rows:
        print("no eval runs yet")
        return 0
    import time as _t

    print(f"  {'#':<5}{'when':<18}{'sha':<10}{'brain':<22}{'pass':>7}{'style':>8}  tag")
    for rid, ts, sha, brain, pr, ss, tag in rows:
        when = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(ts or 0))
        print(
            f"  {rid:<5}{when:<18}{sha or '?':<10}{(brain or '?')[:20]:<22}"
            f"{(pr or 0) * 100:>6.1f}%{(ss if ss is not None else 0):>8.3f}  {tag or ''}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    utf8_stdout()
    p = argparse.ArgumentParser(
        prog="mehullm-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bank", default=DEFAULT_BANK)
    p.add_argument("--db", default=DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check the bank without running it")
    v.set_defaults(fn=_cmd_validate)

    ls = sub.add_parser("list", help="list scenarios")
    ls.add_argument("--category")
    ls.set_defaults(fn=_cmd_list)

    r = sub.add_parser("run", help="run the bank against the agent")
    r.add_argument("--category")
    r.add_argument("--only", help="comma-separated scenario ids")
    r.add_argument("--tag", default="")
    r.add_argument("--no-voice", action="store_true", help="skip the voice layer")
    r.add_argument(
        "--threshold", type=float, default=0.0, help="exit nonzero below this weighted pass rate"
    )
    r.set_defaults(fn=_cmd_run)

    st = sub.add_parser("style", help="objective style score vs held-out chats")
    st.add_argument("--gen", required=True)
    st.add_argument("--ref", required=True)
    st.add_argument("--ceiling", type=float, default=0.914)
    st.add_argument("--floor", type=float, default=0.298)
    st.add_argument(
        "--ppl-gain",
        type=float,
        default=None,
        help="(ppl_base-ppl_tuned)/ppl_base from the training notebook; "
        "omitted neutralises component X at 0.5",
    )
    st.set_defaults(fn=_cmd_style)

    h = sub.add_parser("history", help="past eval runs")
    h.add_argument("--limit", type=int, default=20)
    h.set_defaults(fn=_cmd_history)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
