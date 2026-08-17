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

Full design: [`docs/PLAN.md`](docs/PLAN.md). Current state:
[`docs/STATUS.md`](docs/STATUS.md).

## Quickstart

```bash
uv sync --all-extras
uv run mehullm-eval validate          # offline, no API key needed
```

## Getting your data in

1. WhatsApp → open a chat → ⋮ / chat name → **Export chat** → **Without media**.
   No bulk export exists, so this is manual.
2. Drop the `.txt` files into `data/raw/` (gitignored).
3. Build the dataset:

```bash
uv run mehullm-parse build --self "Your Name" --self "Your Other Alias"
uv run mehullm-parse peek data/raw/chat.txt    # eyeball a single file's parse
```

## Design notes worth knowing

**Style preservation is a hard requirement.** This corpus exists to capture how
one specific person writes — Hinglish, Gen-Z slang, lowercase, `yaaar`, missing
punctuation, emoji. So the pipeline normalises as little as possible: NFC only
(never NFKC/NFKD, which decompose Devanagari matras), ZWJ/ZWNJ preserved, no
case folding or punctuation stripping. The PII scrubber's numeric patterns are
context-anchored so they cannot eat `100%`, `gn8`, or `got 250109 views`.

**Names are pseudonymised consistently** (`Rohan → Person_A`), not flattened to
`<NAME>` — collapsing them would destroy the turn-taking structure.

**Facts are curated, not extracted.** Local extraction from chat logs was built
and measured at ~50% precision with 63% of proposals being the model echoing its
own prompt. See `docs/STATUS.md` for the numbers and why curation won.

**WhatsApp is a data source only.** No live bridge, no unofficial client — that
would risk the account which is also the training corpus.

## Layout

```
src/mehullm/
  pipeline/     parser, PII scrubber, sessionisation, SFT builder
  memory/       sqlite-vec + FTS5 hybrid retrieval
  llm/          provider-agnostic router (Gemini | Groq)
  mcp/          hand-built MCP client
  guardrails/   policy engine, PII vault, confirmation gate
  agent/        the orchestration loop + SSE events
  voice/        Ollama client + fact-invariant firewall
  eval/         scenario bank, graders, judge, style score
evals/scenarios/  60 eval scenarios
data/raw/         your exports — gitignored
```
