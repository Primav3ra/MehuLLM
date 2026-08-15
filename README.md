# MehuLLM

A personal AI agent that **talks like me** and can **act on my behalf**.

Those are two different capabilities, so they use two different models:

- **The brain** — a hosted LLM (Gemini free tier, Groq fallback) that reasons,
  plans, and calls tools over MCP.
- **The voice** — a LoRA-tuned Qwen3-1.7B running locally, which rewrites the
  brain's final answer in my own WhatsApp style, behind a firewall that verifies
  no fact was altered in the process.

Everything runs on free tiers and open-source software. **No paid APIs, no
credit card.**

## Status

Week 1 of ~16. The data pipeline is in and tested; the agent is not built yet.

| Component | State |
|---|---|
| WhatsApp export parser | ✅ done, 33 tests |
| PII scrubber (Indian-context) | ✅ done, 42 tests |
| Turn merging / sessionisation / SFT pairs | ✅ done, 17 tests |
| Corpus census + go/no-go gate | ✅ done |
| LLM router (Gemini ↔ Groq) | ⬜ week 5–6 |
| MCP client | ⬜ week 7 |
| Guardrails + confirmation protocol | ⬜ week 8 |
| Voice layer (LoRA) | ⬜ week 9–10 |
| Eval harness | ⬜ week 11 |

Full design: [`docs/PLAN.md`](docs/PLAN.md).

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q
```

## Getting your data in

1. In WhatsApp, open a chat → ⋮ / chat name → **Export chat** → **Without media**.
   There is no bulk export, so this is manual. Aim for your ~25 highest-volume
   1:1 chats.
2. Drop the `.txt` files into `data/raw/`. That directory is **gitignored** —
   it was added to `.gitignore` before any code was written, deliberately.
3. Run the census:

```bash
uv run mehullm-parse census data/raw/
```

This prints the **go/no-go gate** for the fine-tuning arm of the project:

```
  USABLE SFT PAIRS (1:1)   12,431
  VERDICT: GREEN  -- Proceed with the LoRA as planned.

  Thresholds:  >=8000 green   3000-8000 amber   <3000 red
```

Under 3,000 pairs, no LoRA configuration will save the voice layer, and the plan
pivots to few-shot style exemplars. Measuring this in **week 2 rather than week
10** is the entire reason this command exists.

Other commands:

```bash
uv run mehullm-parse contacts data/raw/        # emit contacts.json to tag which alias is "me"
uv run mehullm-parse peek data/raw/chat.txt    # eyeball how a single file parsed
```

## Design notes worth knowing

**Style preservation is a hard requirement, enforced by tests.** This corpus
exists to capture how one specific person writes — Hinglish, Gen-Z slang,
lowercase, `yaaar`, missing punctuation, emoji. So the pipeline normalises as
little as possible:

- **NFC only**, never NFKC/NFKD — those decompose Devanagari matras and mangle
  ZWJ conjuncts.
- ZWJ/ZWNJ are preserved (load-bearing in both conjuncts and emoji sequences).
- No case folding, no punctuation stripping, no whitespace collapsing.
- The PII scrubber's numeric patterns are context-anchored so they cannot eat
  `100%`, `gn8`, `2moro`, or `got 250109 views`. A bare six-digit number is
  *not* treated as an OTP.

There is a test for each of these. They exist because this class of corruption
is invisible in aggregate statistics and would only surface as "the model
doesn't quite sound like me" three months from now.

**Names are pseudonymised consistently** (`Rohan → Person_A` everywhere), not
flattened to a single `<NAME>` token — collapsing them would destroy the
turn-taking structure the model needs to learn. The owner's own name is kept
verbatim, so the assistant learns to answer to it.

**WhatsApp is a data source only.** No live bridge, no unofficial client. That
avoids any risk to the account which is also the source of the training corpus.
See `docs/threat-model.md`.

## Layout

```
src/mehullm/
  pipeline/     parser, PII scrubber, sessionisation, census   ← built
  llm/          provider-agnostic router (Gemini | Groq)
  mcp/          hand-built MCP client
  guardrails/   policy engine, PII vault, confirmation gate
  memory/       sqlite-vec + FTS5 hybrid retrieval
  voice/        Ollama client + fact-invariant firewall
  agent/        the orchestration loop
tests/          92 tests
data/raw/       your exports — gitignored
docs/PLAN.md    the full design
```
