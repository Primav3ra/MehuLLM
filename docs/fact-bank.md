# The fact bank

Memory is **curated, not extracted**. Facts are hand-written in `facts/*.yaml`
and loaded into `memory.db`. See `STATUS.md` for the measurements that led here.

`facts/` is gitignored — it is personal data once filled in. This file is the
committed record of the schema and the questions, so the templates can be
recreated on a fresh clone.

## Schema

```yaml
- id: F001                     # stable, unique across ALL files
  text: "Mehul lives in Boston."   # REQUIRED — embedded, quoted, and cited
  predicate: lives_in          # optional; drives supersession
  object: Boston               # optional
  subject: Mehul               # optional, defaults to "Mehul"
  confidence: 1.0              # optional, defaults to 1.0
  single_valued: true          # optional; inferred from predicate
  observed_at: 2025-08-14      # optional; when it became true
  superseded_by: F002          # optional; retires THIS fact in favour of F002
  status: active               # optional; active | superseded
```

Only `id` and `text` are required.

**`text` is the load-bearing field.** It gets embedded, retrieved, quoted, and
cited as `[F001]`. Write a complete sentence that stands alone — the agent may
show it with no surrounding context.

**IDs are stable on purpose.** `[F102]` in an answer, in an eval scenario, and
in a trace all mean the same fact six months from now. Loading is an upsert
keyed on the id: edit a line, re-run `load`, and the fact updates in place
rather than duplicating.

### Supersession

Single-valued predicates hold one value at a time. A newer fact retires the
older one, which is marked `superseded` and **never deleted** — the chain is how
*"where did I used to live?"* gets answered.

```yaml
- id: F001
  text: "Mehul lives in Boston."
  predicate: lives_in

- id: F002
  text: "Mehul lived in Hyderabad until August 2025."
  predicate: lived_in
  valid_to: 2025-08
  superseded_by: F001          # F002 becomes history; F001 is current
```

Inferred single-valued predicates: `lives_in`, `studies_at`, `works_at`,
`current_role`, `current_project`, `relationship_status`, `phone_of`, `email_of`.

## ID ranges

One range per file, so files can be edited independently without collisions.

| File | IDs | What |
|---|---|---|
| `identity.yaml` | F001–F099 | name, location, study, languages |
| `people.yaml` | F100–F199 | family, partner, friends, advisor |
| `work.yaml` | F200–F299 | projects, skills, tools, handles |
| `preferences.yaml` | F300–F399 | food, media, sport, communication style |
| `routine.yaml` | F400–F499 | the shape of a normal week |
| `admin.yaml` | F500–F599 | which email, which repo, deadlines, boundaries |

## What makes a good fact

**The test: would this change an answer?**

| Keep | Drop |
|---|---|
| "Mehul is vegetarian." | "Mehul likes good food." |
| "Mehul uses uv, not pip or poetry." | "Mehul knows Python." |
| "Mehul's advisor is Dr. X." | "Mehul has an advisor." |

Three rules:

1. **Constraints beat likes.** "Won't", "can't", "allergic to" — getting one
   wrong is actively bad, not merely unhelpful.
2. **Write relationships, not labels.** *"Arjun is Mehul's roommate and they've
   been friends since Christ University"* is worth three flat facts, because
   retrieval reaches it from more directions.
3. **Skip anything you'd have to look up.** If you can't recall it in five
   seconds, you wouldn't expect the agent to know it either.

Target ~80–120 facts. That is a working personal memory.

## Never put in here

No passwords, API keys, OTPs, or card numbers. Credentials belong in `.env`,
which the agent reads without ever putting them in a prompt. Everything in
`facts/` is embedded and can be quoted back in an answer.

## Loading

```bash
uv run mehullm-memory load        # idempotent — re-run after every edit
uv run mehullm-memory search "where do I live"
uv run mehullm-memory stats
```

Facts load as `status='active'` — there is no review queue, because the author
is the reviewer.
