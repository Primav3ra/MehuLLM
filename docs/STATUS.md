# Status

## Where things stand

| Component | State |
|---|---|
| WhatsApp parser, PII scrubber, SFT builder | done — 419k messages → 75,465 pairs |
| Memory store (sqlite-vec + FTS5), hybrid retrieval | done — 82,179 chunks indexed |
| LLM router (Gemini + Groq), schema sanitiser, quota | built, **never called a real API** |
| MCP hub, registry, guardrail interceptor | built, **never called a real server** |
| Agent loop, run manager, SSE events | built, **never run end to end** |
| Eval bank + graders + judge + runner | 60 scenarios, validates offline |
| Style score | calibrated — ceiling 0.914, floor 0.298 |
| Fact population | **question bank** (see below) |
| Voice LoRA | not trained |
| Frontend | not started |

## Next

1. A free Gemini key in `.env` — ~6,000 LOC has never made a real call.
2. Write the question bank, load it into memory.
3. Smoke-test the loop end to end, then run the eval bank for real.

---

## Decision: curated facts, not extracted ones

Fact extraction from chat logs was built, measured, and **retired**. The numbers
are worth keeping because they are the justification:

| Measure | Result |
|---|---|
| Yield | ~1 fact per 21–50 sessions |
| Precision of surviving facts | ~50% at best |
| Proposals that were the model echoing its own few-shot example | **63%** |
| Time | ~19 s/session → ~19 h for the full corpus |

Five real bugs were found and fixed along the way, none visible from the code
alone — all surfaced by reading what the pipeline actually produced:

1. Reactions to photos stored as durable preferences (`likes the hues`).
2. `Mehul Karwa` vs `Mehul` stored twice; normalisation touched `subject` but
   not the embedded `text`, so dedup could never fire.
3. Grounding demanded 40% word overlap between generated English and Hinglish
   source. The subject is a stopword, so a normal fact has 3–4 content words and
   one miss sinks it — *"Mehul is getting married in December"* scored 0.33
   against a source containing **December**. 67% false-negative rate.
4. The durability gate rejected any all-caps object over 4 chars as "shouting",
   which silently ate **every acronym** — NEU, IIT, BMW. `Mehul studies at NEU`
   could never have survived.
5. The extractor echoed its own few-shot answer. It used *Boston* and
   *Northeastern* — real places in Mehul's life — so an echo could have been
   stored as a true fact with a bogus source.

**The finding that ended it:** `Mehul studies at Christ University` passed every
gate. Its source was a forwarded survey ("we are students from Christ
University…") plus a conversation about someone else studying there. Three
rounds of prompt hardening; real data found a new shape each time.

A 4B local model cannot reliably tell *whose* life a fact describes. And since
every extracted fact needs human review anyway, extraction does not save review
time — it adds to it. Writing ~100 facts by hand takes 30–45 minutes and every
one is correct by construction.

This is a defensible capstone result: *we built local extraction, measured 50%
precision and 63% prompt echo on a 4B model, and concluded curation wins at
personal scale.* The chat corpus's real value was always **style**, and that
part works.

---

## Open item

`.gitignore` protects `data/` and `.secrets/` correctly, but **ignores itself**,
so it is untracked and will never be committed. Locally you are fully protected;
a fresh clone would have no ignore rules. Either leave it and never re-clone, or
drop the trailing `.gitignore` line and commit it. Unchanged pending your call.
