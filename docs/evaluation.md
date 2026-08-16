# Evaluation

How MehuLLM is measured, and why each choice is defensible.

The system makes three separable claims, so it needs three separable measurements:

| Claim | Measured by | Kind |
|---|---|---|
| *It sounds like me* | Six-component style score vs. two withheld chats | Objective, no LLM |
| *It acts safely* | 60-scenario bank, ~70% deterministic assertions | Mostly objective |
| *It answers correctly* | LLM judge, dual-pass, on the remaining 30% | Subjective, bounded |

Conflating these is the usual failure. "Rate this response 1–10" produces a number that
moves with the judge's mood and cannot separate *wrong* from *badly worded*.

---

## 1. The scenario bank

60 scenarios in `evals/scenarios/`, matching the §10 targets exactly:

| Category | N | What it protects |
|---|---:|---|
| `style` | 12 | Voice survives the round trip; never LLM-judged |
| `factual_recall` | 12 | Memory is used, and its absence is admitted |
| `tool_selection` | 10 | The right tool — and often, no tool |
| `refusal` | 8 | Declines vs. *pauses* — a distinction worth keeping |
| `multistep` | 8 | Chains, recovery, step caps |
| `hallucination_trap` | 6 | Invented papers, tools, scores, sent emails |
| `prompt_injection` | 4 | Provenance escalation actually fires |

Run `uv run mehullm-eval validate` to check the bank offline — no API key, no model. It
catches unknown assertion types, duplicate ids, scenarios with no assertions, and
`cites_fact` referring to a fact the scenario never seeds. **A typo'd assertion type would
otherwise report green forever**, which is worse than a red suite.

### Determinism

Two sources of drift are closed deliberately:

- **Seeded facts.** Scenarios carry their own facts, materialised into a *scratch copy* of
  `memory.db`. Grading "did it recall F077" is meaningless if F077 differs per machine, and
  an eval must never mutate facts the user actually approved.
- **Injected tool results.** `inject:` substitutes a tool's return value. This is what makes
  prompt-injection reproducible without leaving a poisoned Notion page on the internet. The
  call still goes through the real interceptor, so policy, provenance escalation, secret
  scanning, and truncation all run — only the transport is stubbed.

### Confirmations auto-deny

The harness answers every confirmation card with **deny**. The assertion is that the card
*fired*; letting the action proceed would mean a test suite that sends real email. An eval
that auto-approves measures nothing it claims to.

---

## 2. Graders

`eval/graders.py`. Everything mechanically checkable is checked mechanically, because
assertions are free, stable, and do not drift.

```
cites_fact  tool_called  tool_not_called  no_tools  confirmation_requested
guardrail_blocked  no_pii_leak  no_think_leak  contains  not_contains
refuses  json_valid  max_latency_ms  max_steps  succeeds
```

Three that carry more weight than their size suggests:

- **`no_pii_leak`** reuses the *pipeline* scrubber — one implementation, three call sites
  (§7). It scans narration and answer **separately**: `\D` matches a newline, so joining
  fields lets a 4-digit number on one line and a 6-digit one on the next read as a phone
  number. That false positive cost an hour during the pipeline audit.
- **`no_think_leak`** asserts Qwen3's reasoning markup never reaches output. ~15% of
  generations leak it if the non-thinking branch is not pinned (§9). Asserted, not hoped for.
- **Unknown assertion types fail.** A grader that silently passes what it does not
  understand is a suite that reports green while rotting.

Grading reads the **SSE event stream**, not the loop's internals — the same contract the
frontend consumes. A scenario cannot pass on a stream the UI could not have rendered.

### Weighting

Pass rate is weighted. Safety scenarios carry weight 3, style probes weight 1. Unweighted,
a suite can go green while the guardrails regress — twelve style passes would paper over
four injection failures.

---

## 3. The judge

`eval/judge.py`, scoped to **correctness, faithfulness, helpfulness**. Two rules:

1. **Style is never judged.** The prompt explicitly forbids rewarding or penalising tone,
   slang, length, or language mixing. A judge that likes polished English would silently
   punish the exact Hinglish voice this project exists to produce — and would report it as
   a *correctness* score, where nobody would think to look.
2. **Judged twice, criteria order swapped.** Disagreement is flagged for review, not
   averaged. Averaging two contradictory verdicts manufactures a confident number out of a
   coin flip. On disagreement the harness takes the **stricter** verdict, so ambiguous
   scenarios cannot drift green over time.

An unparseable judge response fails the scenario rather than passing it.

---

## 4. Style score

`eval/style_score.py`. Six components, weighted per §10: code-switch (0.25), length
distribution (0.20), punctuation/caps (0.15), n-gram (0.15), perplexity (0.15), emoji (0.10).

**Always reported against two anchors, never alone:**

| Anchor | Value | Meaning |
|---|---:|---|
| Ceiling | **0.914** | Two disjoint halves of Mehul's real messages, scored against each other — human-level self-agreement |
| Floor | **0.298** | Raw hosted-model output |
| Spread | **0.616** | The band a result must be placed within to mean anything |

`NormalizedStyle = (S − floor) / (ceiling − floor)`. A raw 0.62 is meaningless on its own;
against these anchors it is ~52% of the way from generic model output to human-level.

Two calibration bugs found and fixed while building it, both of which had inflated scores:

- The Hinglish lexicon contained `me`, `par`, `le`, `na`, `se` — words that are also common
  English. *"Let me know if you have any questions"* scored as Hinglish. Words that are
  also ordinary English are now excluded by construction.
- Code-switch used absolute difference. With a reference rate of 0.13, text containing
  **zero** Hinglish scored `1 − 0.13/2 = 0.93`. It is now scaled by the reference rate, so
  output sharing none of the defining trait scores near 0.

**Sub-scores are reported individually.** A single number is not a defensible result; six
components with anchors is.

---

## 5. Regression

Results persist to `eval_runs` / `eval_results`, keyed by git sha and model, so *"did this
change make things worse"* is a SQL query rather than a memory of last week's numbers.

Drift alarms vs. the last green run (≥90% weighted): any category down >10 pp, style down
>0.05, p95 latency up >50%.

```bash
uv run mehullm-eval validate                 # offline; CI gate
uv run mehullm-eval run --tag nightly
uv run mehullm-eval run --category prompt_injection
uv run mehullm-eval history
```

---

## 6. The headline experiment

Three-way comparison on the same bank (§14, week 11):

1. Raw hosted brain — no voice layer
2. Few-shot styled — 8 retrieved real messages as exemplars
3. LoRA voice layer

Whether the LoRA beats few-shot is **genuinely uncertain**, which is why the few-shot
baseline ships first. If it does not win, *"with N pairs, in-context style exemplars matched
a rank-32 LoRA"* is also a result — and the anchored, component-wise score is what makes
that claim publishable rather than an anecdote.

---

## 7. Known limits

- The judge runs on the free Gemini tier; its verdicts are subject to the same
  undocumented quota as everything else.
- 60 scenarios is enough to catch regressions, not enough to certify safety. The injection
  category in particular is 4 scenarios against an open-ended attack surface.
- The style ceiling is computed from one person's messages across several years; voice
  drifts over that span, so 0.914 is an *estimate* of human self-agreement, not a constant.
