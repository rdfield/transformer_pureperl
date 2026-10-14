# d=768 encoder-decoder summariser — training stages

Compiled 2026-10-07 from run logs, checkpoint timestamps and live process command
lines. **Provenance is marked on every command**, because it matters for anything
written up:

- **VERBATIM** — captured from the running process (`ps -o cmd=`) or copied from
  `slm_training_command_line_examples.txt`. Trust these.
- **RECONSTRUCTED** — the flags are inferred from log output (corpus sizes, model
  banner, per-epoch `lr`). The parameters are right; the exact flag spelling and
  any flags with no visible effect are not guaranteed.

The pretrain and anneal commands are RECONSTRUCTED **because `MVEncDecTrainer`
does not log its own `argv`.** Every log begins `5 special tokens / RNG seed: 42`,
so the invocation is lost once the shell closes. One `printf` of the argv at
startup would have made this whole exercise unnecessary — worth adding.

Model geometry is identical in every stage: **d=768, 4 encoder + 4 decoder
layers, 8 heads, d_ff=3072, vocab 16,005, tied head, 90.7M parameters.**

---

## Stage 0 — Pretrain (denoising, full-window LM)

**Corpus** `blend_v2.train` — 1,680,135 docs, **2,112,105,989 tokens**,
8,837,264 windows of 239. Val `pretrain_blend_16000.val`, 1,500 docs / 1.35M tokens.
**Output** `d768_1.7B.safetensors` -> `_2` -> `_3`
*(the "1.7B" in the filename is a naming artefact — the corpus is 2.11B tokens)*

Sequence 256/240, batch 8, 1,104,658 batches, cosine LR, ~50 windows/s, ~48h projected.

**RECONSTRUCTED:**

    ./build/MVEncDecTrainer --pretrain \
      --train $P/blend_v2.train --val $P/pretrain_blend_16000.val \
      --vocab $P/cosmo_client_blended2.model \
      --d 768 --enc_layers 4 --dec_layers 4 --heads 8 --d_ff 3072 \
      --seq 256 --batch 8 --lr 0.0003 --seed 42 \
      d768_1.7B.safetensors

**Ran in three segments, not one.** Four consecutive crashes, each a
`cublasDgemmStridedBatched FAILED status=14`, caused by **4 out-of-range token ids
in 2.11B** — misdiagnosed three times as thermal/driver/VRAM because the shuffle
reseeds per run, so the crash position moved. Segments warm-started from the
previous checkpoint:

| log | resumed from | crashed at | lr there |
|---|---|---|---|
| `d768_pretrain.log` | — | 20.0% | 0.000285 |
| `d768_pretrain_2.log` | `d768_1.7B` | 44.0% | 0.000205 |
| `d768_pretrain_3.log` | `d768_1.7B_2` | 64.0% | 0.000116 |

Guard added afterwards: `perl filter_encoded_corpus.pl --vocab N` rejects
out-of-range ids. **Run it before any long pretrain.**

---

## Stage 1 — Anneal

**Corpus** `anneal.train` — 168,038 docs, **210,759,410 tokens**, 881,839 windows.
**Init** `d768_1.7B_3` **Output** `d768_annealed.safetensors`

Objective is the pretrain's full-window LM at seq 256/240, **not** the enc-dec
pair format. One epoch, 110,229 batches, **20,888s (5.80h)**, 42-52 windows/s.

**Result: val_loss 0.7293, val_tok_acc 86.1%, conditioning signal 77.6%**
(shuf_loss 4.3009, gap +3.5666), gnorm 0.65 throughout.
LR decayed **0.000050 -> 0.000030** across the epoch.

That 77.6% is the calibration point for the whole project: it is what "the encoder
is fully engaged" looks like. **But it is a PRETRAIN property, not an anneal one** —
the signal was 77.3% at the first eval and 77.6% at the last, never leaving
77.3-77.7%. The anneal did not build the conditioning; it was already there.

**The anneal barely moved anything.** Train loss 0.7015 -> 0.6793 over the whole
epoch, val_loss 0.7458 -> 0.7293. Five hours and fifty minutes for 0.016 of
val_loss: the model was converged before it started. And the best checkpoint
landed at **90.9%**, with the final 100% eval *worse* (0.7325), so the last tenth
was counterproductive.

Eval cost: 11 evaluations x ~171s = 1,881s = **9.0%** of the run. (Not the 17% I
claimed elsewhere — that figure was the throughput drop from 52 to 43 windows/s
after the first eval, which is a separate effect and may be thermal.)

**RECONSTRUCTED:** as Stage 0 plus `--init d768_1.7B_3.safetensors`,
`--train $P/anneal.train`, `--lr 0.00005`, `--epochs 1`,
`--evals_per_epoch 11`.

---

## Stage A — Direct client fine-tune — **FAILED, abandoned**

**Init** `d768_annealed` **Data** 8,118 train / 902 val client thread pairs
**Output** `d768_thread_ft.safetensors`

**Result: 63.4% val_tok_acc but conditioning signal 25.7%** — fluent output that
did not read its input. Hallucinated "Vendor-B Product-X" and "Vendor-C" into a
Vendor-A thread. Identifier fidelity 55.3% exact / 26.6% invented.

The lesson: **token accuracy cannot see this failure**, because teacher forcing
supplies the correct prefix. The conditioning check can.

**VERBATIM** (from `slm_training_command_line_examples.txt`):

    cd /home/mvine/embedding_cpp
    P=/home/mvine/shared/rdf/pretrain_data

    stdbuf -oL ./build/MVEncDecTrainer \
      --init d768_annealed.safetensors \
      --train $P/train_16000_finetune.txt \
      --val   $P/val_16000_finetune.txt \
      --vocab $P/cosmo_client_blended2.model \
      --d 768 --enc_layers 4 --dec_layers 4 --heads 8 --d_ff 3072 \
      --seq_enc 2048 --seq_tgt 240 \
      --email_budget 2048 --summary_budget 224 \
      --batch 4 --accum 8 \
      --epochs 30 --lr 0.0003 --patience 4 \
      --pointer --flash --bucket \
      --cond_check --seed 42 \
      d768_thread_ft.safetensors 2>&1 | tee ~/d768_ft.log

---

## Stage 2 — Summarisation skill, from generated data

Rolled **back past Stage A** to `d768_annealed`, on the reasoning that the model
had never been taught *how to summarise* — only client house style, from 8k examples.

### 2a. Build the data (~$2.60 on OpenRouter)

Sampled Enron + client emails, had Qwen summarise them with the production prompt,
filtered the output into pairs. Teacher validated first by blind bake-off at an
**85% A-rate**.

    perl bakeoff_sample_enron.pl ...            # sample Enron bodies
    perl bakeoff_sample.pl ...                  # sample client bodies
    split -n r/4 ...                            # round-robin! -n l/4 splits by BYTES
    perl bakeoff_generate.pl ...                # Qwen via OpenRouter, 4 shards
    perl build_stage2_pairs.pl vocab.model sample.csv gen.csv pairs.txt \
         --min-chars 25 --max-names 8 --max-src-tokens 2048

`build_stage2_pairs.pl` filters: refusals (**40%** of Enron — the production prompt
is built for transactional requests), under 25 chars, >=8 capitalised name runs
(recipient-list dumps), `finish_reason: length`, and **sources over the 2048-token
budget** (56,316 -> 51,105 pairs).

Exact generate flags are **NOT RECORDED** — the shards were launched interactively.

### 2b. Train — **VERBATIM** (captured from the running process)

    ./build/MVEncDecTrainer \
      --init d768_annealed.safetensors \
      --train $P/stage2_fit.train \
      --val   $P/val_16000_finetune.txt \
      --vocab $P/cosmo_client_blended2.model \
      --d 768 --enc_layers 4 --dec_layers 4 --heads 8 --d_ff 3072 \
      --seq_enc 2048 --seq_tgt 240 \
      --email_budget 2048 --summary_budget 224 \
      --batch 4 --accum 8 \
      --epochs 8 --lr 0.0003 --patience 3 \
      --pointer --flash --bucket \
      --cond_check --seed 42 \
      d768_stage2.safetensors

**51,105 train / 902 val. 8 epochs, 2.2h each (~18h).**

| | ep 1 | ep 8 |
|---|---|---|
| val_loss | 2.1603 | **1.8704** |
| val_tok_acc | 56.2% | **62.0%** |
| signal | 20.0% | 28.1% |
| loss gap | — | +1.2921 |

**It set a new best on all 8 epochs and never early-stopped — it ran out of
schedule, not out of progress.** The loss gap rose monotonically every epoch
(1.0173 -> 1.2921) while the accuracy-based signal bounced (+2.0, +1.6, +1.2,
+2.0, +0.6, +0.3). **Read the loss gap, not the signal.**

`val_loss 1.8704` beats the 2.0207 of the best previous client fine-tune — reached
while training on *Qwen* data and validating on *client*.

### 2c. Verify — mode collapse gone

    ./build/MVEncDecInfer --model d768_stage2.safetensors \
      --input $P/val_16000_finetune.txt --vocab $P/cosmo_client_blended2.model \
      --d 768 --enc_layers 4 --dec_layers 4 --heads 8 --d_ff 3072 \
      --seq_enc 2048 --seq_tgt 240 --email_budget 2048 --max_new 224 \
      --pointer --no_repeat_ngram 3 --n 40 > stage2_gen40.txt

    perl audit_generations.pl stage2_gen40.txt
    perl gen_to_bakeoff.pl --pairs $P/val_16000_finetune.txt \
         --gen stage2_gen40.txt --out /tmp/s2 --label stage2
    perl measure_identifiers.pl /tmp/s2_pairs.csv /tmp/s2_gen.csv

| | collapsed | Stage A | **Stage 2** |
|---|---|---|---|
| distinct 5-prefixes | 4/40 | 35/40 | **38/40** |
| diversity (vs ref) | 0.220 | 0.683 | **0.96** |
| conditioning | 0.08 | — | **0.27** (bar 0.15) |
| identifiers exact | — | 55.3% | **65.2%** |
| identifiers invented | — | 26.6% | **17.4%** |

Eyeball of 10: roughly **2 A / 6 B / 2 C** graded against references (harsher than
against the source thread). Remaining failure is **person names, not
identifiers** — and four of the invented names are **Enron** names with no client
presence at all (Kean 250/0, Hansen 199/0, Guerrero 134/4, Krecek 14/0), i.e. an
artefact of training on Enron in 2b, which is what Stage 3 corrects.

---

## Stage 3 — client house style — **RUNNING (started 2026-10-07 06:53)**

**VERBATIM** (captured from the running process):

    cd /home/mvine/embedding_cpp
    P=/home/mvine/shared/rdf/pretrain_data

    stdbuf -oL ./build/MVEncDecTrainer \
      --init d768_stage2.safetensors \
      --train $P/train_16000_finetune.txt \
      --val   $P/val_16000_finetune.txt \
      --vocab $P/cosmo_client_blended2.model \
      --d 768 --enc_layers 4 --dec_layers 4 --heads 8 --d_ff 3072 \
      --seq_enc 2048 --seq_tgt 240 \
      --email_budget 2048 --summary_budget 224 \
      --batch 4 --accum 8 \
      --epochs 12 --lr 0.00015 --patience 3 \
      --pointer --flash --bucket \
      --cond_check --seed 42 \
      d768_stage3.safetensors 2>&1 | tee ~/d768_stage3.log

LR is **half** Stage 2's: 8,118 pairs at full rate risks washing out the
summarisation skill just installed.

Startup reports **1,043 of 8,118 sources truncated at 2048 tokens (12.8%)**. Of
those, 249 cite a name the model cannot see and 55 cite an unseen identifier — so
**3.1% of all pairs train the model to invent a name.** Real but far too small to
explain the observed rate, which supports the Enron-bleed diagnosis instead.
Within-thread dedup (40.5% of body text is duplicated lines) would *recover* those
pairs rather than dropping them.

**Success criterion was: do the Enron names disappear?** They did.

### Result — COMPLETE, 12 epochs, new best on every one (never early-stopped)

| | direct client FT (Stage A, failed) | Stage 2 | **Stage 3** |
|---|---|---|---|
| val_loss | — | 1.8704 | **1.4978** |
| val_tok_acc | 63.4% | 62.0% | **67.4%** |
| conditioning signal | 25.7% | 28.1% | **34.2%** |
| loss gap | — | +1.2921 | **+1.4325** |

Previous best val_loss by any client fine-tune was 2.0207, so this is 0.52 nats
better — roughly a 41% cut in perplexity. Train/val crossed at epoch 4 and the
gap reached +0.37 by the end, the familiar data-limited signature on 8k pairs,
but val_loss kept setting new bests so `--patience 3` never fired.

**The Enron bleed is gone.** Names appearing in the same 40 generations:

| | Enron-name hits |
|---|---|
| Stage 2 | Kean, Hansen, Krecek, Guerrero, [client-name] |
| **Stage 3** | **[client-name] only** |

[client-name] is the one that was never Enron bleed — a genuine client name (757 uses in
the corpus) attached to the wrong thread. That residual "wrong client person" error
is what remains.

Audit: PASS, conditioning 0.27 -> **0.33**, diversity 0.91, opening var 0.85
(both down slightly from Stage 2's 0.96/0.97 — output is wordier and more
templated).

### Identifiers: the repair pass is where the win lands

    perl repair_identifiers.pl /tmp/s3_pairs.csv /tmp/s3_gen.csv /tmp/s3_rep.csv
    perl measure_identifiers.pl /tmp/s3_pairs.csv /tmp/s3_gen.csv /tmp/s3_rep.csv

| | ids | exact | drift | invented | covered |
|---|---|---|---|---|---|
| teacher | 29 | 96.6% | 0.0% | 3.4% | 13.6% |
| Stage 3 raw | 27 | 59.3% | 25.9% | 14.8% | 7.8% |
| **Stage 3 + repair** | 26 | **80.8%** | **3.8%** | 15.4% | **10.2%** |
| *best before (Stage A + repair)* | | *68.9%* | | | |

**Drift is repairable, invention is not.** Stage 3 moved errors out of invention
and into drift, which reads as a regression on raw exact-rate (65.2% -> 59.3% vs
Stage 2) and is actually the better outcome, because repair then has something to
match. 80.8% against a previous best of 68.9%.

This confirms the split: **the model selects, copying renders.** It also shrinks
the case for constrained decoding further — that was estimated at +4.5pp over
repair using Stage A's buckets, and repair alone now reaches 80.8%.

### New failure introduced: within-sentence degeneration

    Helena from Orange is asking Helena to confirm...
    whether an an APAC distributor is needed...
    (1 World Wide Way, 1 World Wide Way, Maland Heights, MO 63146)

Self-reference, doubled words, duplicated address fragments. **Invisible to both
instruments**: `audit_generations.pl` measures 4-grams and reports
`within-seq repeats 0.000`, and `--no_repeat_ngram 3` cannot catch a single-token
repeat — the same blind spot found during the sticky-copy work. Needs a
decode-time guard, not more training.

Stage 3 also lost an identifier Stage 2 had exactly right (`307423`), consistent
with the wordier output.

---

## Control experiment — `baseline_hf/`

Independent of the stages above. The detailed results doc is kept OUT of this
repo: it quoted verbatim generations carrying real client identifiers.

    PY=/home/mvine/miniconda3/envs/fastbook/bin/python   # transformers 4.37.2, torch 2.2.0
    $PY prepare_data.py
    $PY run_lead.py --split val --out out/L_lead.jsonl              # lead baseline, no model
    $PY run_zeroshot.py --model facebook/bart-large-cnn --split val \
       --out out/A_bart_large_cnn.jsonl                             # arm A, CPU
    $PY measure.py out/A_bart_large_cnn.jsonl
    $PY finetune_bart.py --model facebook/bart-base --out out/B_bart_base   # arm B, NOT RUN

Headline: `bart-large-cnn` (406M, zero-shot) **transcribes rather than
summarises** — extractiveness 0.83 against the teacher's 0.06 — and still invents
**12.7%** of identifiers. A zero-parameter lead baseline is **100% exact** on
identifiers but recalls only **32.2%** of the teacher's choices, so selection is
the model's job and fidelity is copying's.
