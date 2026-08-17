# Training the voice layer

Two notebooks on a **free Colab T4**. Nothing trains on the 1650 — 4 GB VRAM is
for inference only.

Expect **~2.5–3h wall clock** for run #1. Colab will probably disconnect at
least once; that is planned for, not a failure.

---

## Before you start

Confirm the runtime is a **T4**:
`Runtime → Change runtime type → T4 GPU`. Anything else and the fp16 settings
are wrong.

---

## Step 1 — upload the dataset

```
Google Drive → create folder:  MyDrive/mehullm/
upload:                        data/derived/sft_pairs.jsonl
```

Both notebooks read from `/content/drive/MyDrive/mehullm`, so the folder name
must match exactly.

## Step 2 — run `02_lora_t4.ipynb`

Open in Colab, `Runtime → Run all`. It will:

1. Mount Drive and install a **pinned** stack from the lockfile. Do not upgrade
   anything by hand — `transformers` v5 changed default dtype behaviour and TRL
   renamed `tokenizer=` → `processing_class=`; the pins exist because of that.
2. Print the split counts and the **degenerate-pair rate** (draft == reply).
   Sanity-check this: a high rate means the neutraliser echoed targets and the
   model would learn the identity function.
3. Train rank-32 LoRA, 3 epochs, `assistant_only_loss=True` so loss lands on the
   reply only — not on the draft it is rewriting.
4. Checkpoint to Drive every 200 steps.
5. Save the adapter to `MyDrive/mehullm/adapter`.

**If Colab disconnects:** reopen and `Run all` again. It picks up from the last
checkpoint via `resume_from_checkpoint`. That is the entire reason checkpoints
go to Drive rather than local scratch.

**If loss goes NaN:** drop `learning_rate` to `5e-5` before touching anything
else. sm_75 fp16 is the usual cause.

## Step 3 — run `03_merge_gguf.ipynb`

Separate notebook on purpose: llama.cpp's converter breaks whenever their build
changes, and that must not cost a training run.

It merges the adapter into fp16, converts to GGUF, quantises to **Q4_K_M
(~1.1 GB)**, and writes `voice-q4km.gguf` + a `Modelfile` to Drive.

## Step 4 — install locally

Download both files from Drive into a folder, then:

```bash
ollama create mehul-voice -f Modelfile
ollama list                      # expect mehul-voice
```

The Modelfile pins `num_ctx 2048`, `temperature 0.85`, and prefills an empty
`<think></think>` block — without that last part roughly 15% of generations leak
reasoning into replies.

## Step 5 — it activates itself

No code change needed. The API checks at startup:

```
voice model not installed; serving brain output unstyled   ← now
```

Once `mehul-voice` exists that line disappears and the voice layer engages,
behind the invariant firewall: if the rewrite drops a number, URL, or email, the
original draft is served instead and `voice_end.fell_back=true` is emitted.

Verify:

```bash
curl -s localhost:8000/api/health
uv run mehullm-eval run --tag lora-v1
```

---

## What "working" looks like

Compare against the baseline before declaring victory:

| Run | What it measures |
|---|---|
| raw brain | floor — generic assistant English |
| few-shot exemplars | the honest baseline the LoRA must beat |
| LoRA voice | the thing being tested |

Style score is reported **normalised** against ceiling **0.914** (human
self-agreement) and floor **0.298** (raw model). A raw number alone means
nothing.

**If the LoRA does not beat few-shot, that is still a result** — "with ~7K pairs,
in-context exemplars matched a rank-32 LoRA" is publishable and you have a
working system either way.

---

## Dataset note for v1 vs v2

Run #1 trains on the **~6,742-pair snapshot** (plan's amber band, 3k–8k).
Neutralisation continues in the background toward 13.5K for a v2 run. Keep the
v1 eval numbers — the v1→v2 delta on the same 60-scenario bank is the cleanest
evidence that more data helped, and it costs nothing extra to record.
