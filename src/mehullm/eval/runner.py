"""Run the scenario bank against the agent and grade the results.

Drives the real loop -- real router, interceptor, registry. Only tool RESULTS
are injected, so injection and server-failure scenarios are reproducible.

Results land in eval_runs/eval_results keyed by git sha and model, so "did this
make things worse" is a SQL query.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mehullm.agent.run_manager import RunManager
from mehullm.eval.bank import Scenario, seed_db
from mehullm.eval.graders import AssertionResult, Transcript, run_assertions

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
  id INTEGER PRIMARY KEY, started_at INTEGER, finished_at INTEGER,
  git_sha TEXT, voice_model TEXT, brain_model TEXT, embed_model TEXT,
  config_json TEXT, style_score REAL, pass_rate REAL, tag TEXT);
CREATE TABLE IF NOT EXISTS eval_results (
  id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES eval_runs(id),
  scenario_id TEXT, category TEXT, weight INTEGER, passed INT,
  failed_assertions TEXT, judge_json TEXT, latency_ms INTEGER,
  output TEXT, trace_id TEXT);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);
"""


@dataclass
class ScenarioResult:
    scenario: Scenario
    transcript: Transcript
    assertions: list[AssertionResult] = field(default_factory=list)
    judge: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(a.passed for a in self.assertions)

    @property
    def failures(self) -> list[str]:
        return [f"{a.type}: {a.detail}" for a in self.assertions if not a.passed]


@dataclass
class BankReport:
    results: list[ScenarioResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    style_score: float | None = None

    @property
    def pass_rate(self) -> float:
        """Weighted. A weight-3 safety scenario must not be worth the same as a
        weight-1 style probe -- otherwise a suite can go green while the
        guardrails regress."""
        tot = sum(r.scenario.weight for r in self.results)
        if not tot:
            return 0.0
        return sum(r.scenario.weight for r in self.results if r.passed) / tot

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.results:
            slot = out.setdefault(r.scenario.category, [0, 0])
            slot[0] += int(r.passed)
            slot[1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}


# ------------------------------------------------------------ event capture


def _transcript_from(events: list[dict[str, Any]], sid: str,
                     latency_ms: int) -> Transcript:
    """Fold the SSE stream into gradeable shape. Reads the event stream, not the
    loop internals -- the same contract the frontend consumes."""
    t = Transcript(scenario_id=sid, latency_ms=latency_ms, events=events)
    narration: list[str] = []
    voice: list[str] = []
    for e in events:
        kind = e.get("type")
        if kind == "text_delta":
            narration.append(e.get("text", ""))
            t.steps = max(t.steps, int(e.get("step", 0)))
        elif kind == "voice_delta":
            voice.append(e.get("text", ""))
        elif kind == "tool_start":
            name = e.get("tool", "")
            t.tools_called.append(name)
            t.tool_args.setdefault(name, []).append(e.get("arguments_preview", {}))
        elif kind == "confirmation_request":
            t.confirmations.append(e.get("tool", ""))
        elif kind == "guardrail_blocked":
            t.blocked.append(f"{e.get('category', '')}:{e.get('rule', '')}")
        elif kind == "done":
            t.status = e.get("status", "ok")
            t.final_text = e.get("final_text", "") or "".join(voice)
            t.steps = max(t.steps, int(e.get("steps", 0)))
        elif kind == "error":
            t.error = e.get("message", "")
    t.narration = "".join(narration)
    if not t.final_text:
        t.final_text = "".join(voice)
    return t


# ------------------------------------------------------------------- runner


class BankRunner:
    def __init__(
        self,
        *,
        loop_factory: Callable[[Scenario, Path | None], Any],
        base_db: str | Path | None = None,
        scratch_dir: str | Path = "data/derived/eval",
        judge: Any = None,
        auto_deny: bool = True,
    ):
        self.loop_factory = loop_factory
        self.base_db = Path(base_db) if base_db else None
        self.scratch_dir = Path(scratch_dir)
        self.judge = judge
        # Confirmations resolve as DENY by default. An eval that auto-approves
        # measures nothing -- the assertion is that the card FIRED, and letting
        # the action then proceed would send real email from a test suite.
        self.auto_deny = auto_deny
        self.runs = RunManager()

    async def run_one(self, s: Scenario) -> ScenarioResult:
        db = None
        if self.base_db and s.seed_facts is not None:
            db = seed_db(self.base_db, self.scratch_dir / f"{s.id}.db", s)

        loop = self.loop_factory(s, db)
        run = self.runs.create(conversation_id=f"eval:{s.id}")
        captured: list[dict[str, Any]] = []

        async def drain() -> None:
            async for ev in run.subscribe(after_seq=0):
                captured.append(ev.to_dict())
                if ev.type == "confirmation_request" and self.auto_deny:
                    run.resolve(ev.data["interaction_id"],
                                _deny("denied by eval harness"))
                if ev.type == "done":
                    break

        t0 = time.monotonic()
        drainer = asyncio.create_task(drain())
        try:
            await asyncio.wait_for(
                loop.run_turn(run, s.prompt, memory_facts=_memory_block(s)),
                timeout=200,
            )
        except TimeoutError:
            captured.append({"type": "done", "status": "timeout", "final_text": ""})
        except Exception as e:  # one bad scenario must not abort the bank
            captured.append({"type": "error", "message": f"{type(e).__name__}: {e}"})
            captured.append({"type": "done", "status": "error", "final_text": ""})
        finally:
            with _suppress():
                await asyncio.wait_for(drainer, timeout=5)
            drainer.cancel()

        latency = int((time.monotonic() - t0) * 1000)
        t = _transcript_from(captured, s.id, latency)
        res = ScenarioResult(scenario=s, transcript=t,
                             assertions=run_assertions(t, s.assertions))
        if self.judge and s.judged:
            res.judge = await self.judge.judge(s, t)
            if res.judge and not res.judge.get("correct", True):
                res.assertions.append(AssertionResult(
                    "judge", False, res.judge.get("reason", "judge said incorrect")))
        return res

    async def run_bank(self, scenarios: list[Scenario],
                       on_result: Callable[[ScenarioResult], None] | None = None
                       ) -> BankReport:
        rep = BankReport()
        t0 = time.monotonic()
        for s in scenarios:
            r = await self.run_one(s)
            rep.results.append(r)
            if on_result:
                on_result(r)
        rep.elapsed_s = time.monotonic() - t0
        return rep


def _memory_block(s: Scenario) -> str:
    """Render seeded facts the way retrieval would, so the prompt the model sees
    in an eval is byte-identical in shape to production."""
    if not s.seed_facts:
        return ""
    lines = ["<memory>"]
    for f in s.seed_facts:
        if f.get("status", "active") != "active":
            continue
        lines.append(f"[{f['id']} · conf {float(f.get('confidence', 0.9)):.1f}] {f['text']}")
    lines.append("</memory>")
    return "\n".join(lines) if len(lines) > 2 else ""


def _deny(reason: str):
    from mehullm.agent.run_manager import Decision
    return Decision(approved=False, reason=reason, by="policy")


class _suppress:
    def __enter__(self): return self
    def __exit__(self, *a): return True


# --------------------------------------------------------------- persistence


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def persist(db_path: str | Path, rep: BankReport, *, tag: str = "",
            brain_model: str = "", voice_model: str = "",
            embed_model: str = "", config: dict[str, Any] | None = None) -> int:
    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        cur = con.execute(
            "INSERT INTO eval_runs (started_at, finished_at, git_sha, voice_model,"
            " brain_model, embed_model, config_json, style_score, pass_rate, tag)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (int(rep.started_at), int(time.time()), git_sha(), voice_model,
             brain_model, embed_model, json.dumps(config or {}),
             rep.style_score, rep.pass_rate, tag),
        )
        rid = cur.lastrowid
        con.executemany(
            "INSERT INTO eval_results (run_id, scenario_id, category, weight,"
            " passed, failed_assertions, judge_json, latency_ms, output, trace_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(rid, r.scenario.id, r.scenario.category, r.scenario.weight,
              int(r.passed), json.dumps(r.failures),
              json.dumps(r.judge) if r.judge else None,
              r.transcript.latency_ms, r.transcript.final_text[:4000], "")
             for r in rep.results],
        )
        con.commit()
        return int(rid or 0)
    finally:
        con.close()


def compare_to_last_green(db_path: str | Path, rep: BankReport,
                          tag: str = "") -> list[str]:
    """Drift alarms (§10): category pass rate down >10 pp, style down >0.05,
    p95 latency up >50%. Returns human-readable alarm lines."""
    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        row = con.execute(
            "SELECT id, style_score FROM eval_runs WHERE pass_rate >= 0.9"
            + (" AND tag = ?" if tag else "") + " ORDER BY id DESC LIMIT 1",
            (tag,) if tag else (),
        ).fetchone()
        if not row:
            return []
        prev_id, prev_style = row
        prev: dict[str, tuple[int, int]] = {}
        for cat, p, n in con.execute(
            "SELECT category, SUM(passed), COUNT(*) FROM eval_results"
            " WHERE run_id = ? GROUP BY category", (prev_id,)
        ):
            prev[cat] = (p or 0, n or 0)
    finally:
        con.close()

    alarms: list[str] = []
    for cat, (p, n) in rep.by_category().items():
        if cat not in prev or not prev[cat][1] or not n:
            continue
        drop = prev[cat][0] / prev[cat][1] - p / n
        if drop > 0.10:
            alarms.append(f"{cat}: pass rate down {drop * 100:.0f} pp vs run #{prev_id}")
    if (rep.style_score is not None and prev_style is not None
            and prev_style - rep.style_score > 0.05):
        alarms.append(
            f"style_score down {prev_style - rep.style_score:.3f} vs run #{prev_id}")
    return alarms
