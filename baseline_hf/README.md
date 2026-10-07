# HuggingFace baselines for the client thread summariser

A control experiment. Everything here is self-contained; nothing in it is part of
the production pipeline.

## Why this exists

The from-scratch encoder-decoder (90.7M params, d=768, 4+4) reached
**val_tok_acc 63.4%** on client thread summarisation and still produces unusable
output — it hallucinated "Vendor-B Product-X" and "Vendor-C" into a thread about
a **Vendor-A** order, neither of which appears in the source. Its conditioning
signal was **25.7%**, meaning shuffling the source costs it only a quarter of the
available headroom: it is largely generating from corpus priors rather than
reading the input.

Four independent measurements say the bottleneck is **data, not parameters**:

| evidence | finding |
|---|---|
| size sweep | d=128 44.4% -> d=256 49.8%, train/val gap 0.045 -> 0.326 |
| gap progression | 0.045 -> 0.326 -> 0.445 -> 0.462 as capacity grew |
| pretrain conversion | a **38.5% better** pretrain bought only **+4.0pp** on task accuracy |
| literature | BART-base (139M) and See et al. (~20M) both summarise — on **287k** pairs, against our 8k |

But "the data is the problem" is a conclusion we reached from our own numbers only.
It has never been tested against an outside reference. **That is what this
directory does**: take publicly available models of comparable size, give them the
same data and the same evaluation, and see where they land.

## The question each arm answers

| arm | model | params | input limit | answers |
|---|---|---|---|---|
| **L** | lead baseline — no model | **0** | none | Does this task need a generative model **at all**? Strip the boilerplate, print the first two sentences. |
| **A** | `facebook/bart-large-cnn` | 406M | 1024 tok | What does an **off-the-shelf** summariser do on client threads, with no training? Not size-matched — this is the ceiling a free download gives you. |
| **B** | `facebook/bart-base` fine-tuned | 139M | 1024 tok | Would a **properly pretrained** model of comparable size, given the **same 8,118 client pairs**, beat ours? |

**Arm L was added after arm A, and should have come first.** It is the standard
control in this literature because it is frequently hard to beat, and a system
that cannot beat it has not demonstrated that it is summarising. It costs no
compute. Run it before anything else.

It also carries a product question the other arms do not. Extracted text **cannot
hallucinate**. Under a "no garbage to customer" constraint, a system that gets the
form wrong but the facts right is a different kind of object from one that is
fluent and sometimes false — and ours is currently the second kind.

**Interpreting arm B:**

- **B clearly beats ours** -> the bottleneck is pretraining quality. 2.1B tokens
  from scratch was not enough, and the answer is a better base model (or starting
  from someone else's).
- **B lands in the same place** -> the ceiling is the data and the task, not our
  architecture. That would be the strongest evidence yet that ~8k-59k pairs cannot
  support abstractive summarisation of 2000-token multi-party threads at any size
  in this range — and it would redirect effort from generation towards extraction,
  templating, and the quality gate.

## What we are looking for, restated after arm A

The original framing was a single quality axis: does a public model land above or
below ours. Arm A broke that framing, because it fails in a way that is not
"worse" but *different* — it transcribes where ours fabricates. So the thing to
measure is **which failure mode** a system has, not only how much it fails:

| | extractiveness | failure |
|---|---|---|
| reference (gpt-4.1-mini) | 0.06 | — |
| arm L, lead | 1.00 | right facts, wrong form; no role attribution |
| arm A, bart-large-cnn | 0.77 | transcribes the lead, keeps the footer, fabricates relations |
| ours, 90.7M | TBD | invents entities absent from the source |

These are three different points — transcription, correct abstraction, untethered
abstraction — not three positions on one line. A grader's A/B/C cannot separate
the first from the second, because a fluent copy and a fluent abstraction both
read as "a sentence"; that is why `measure.py` exists.

The practical question this opens: if arm L has the facts and ours has the form,
the shortest path to a shippable system may be neither of them alone.

## Why NOT t5-small

`t5-small` (60M) is the closest match on parameters and was the obvious first
choice, but its input limit is **512 tokens** while our threads run to 2048. It
would see roughly a quarter of a typical thread, which handicaps the model rather
than testing it. `facebook/bart-base` is 139M with a **1024-token** limit and is
the actual model from the literature comparison above. `google/flan-t5-base` is
also 512 and ruled out for the same reason.

Licences: `bart-base` Apache-2.0, `bart-large-cnn` MIT. Both fine for a
commercially deployed product, unlike most summarisation **datasets** (see
`register_corpus_gap` in the project memory).

## Known unfairness, and how it is handled

**Context limit.** BART takes 1024 tokens; ours takes 2048. Measured in our
tokeniser the client threads are p50 551, p95 1758 — so BART truncates roughly the
top 15-20%. Results are therefore **reported split by thread length**: short
threads are the fair comparison, long ones quantify the handicap. Do not read a
single headline number.

**Tokeniser.** BART has its own BPE, so "tokens" are not comparable between the
two systems. All length statistics here are in **characters** unless stated.

**Evaluation must use our instruments, not ROUGE.** ROUGE has never tracked the
A/B/C grades on this project — an audit scored against machine references once
gave 3-10% where grading against the source thread gave 1 A / 19 B / 0 C. So
outputs are converted to the bake-off CSV format and run through
`measure_identifiers.pl` plus a **blind** `bakeoff_bundle.pl` sheet with our
model's output as the other arm. Same val set, same rubric, same blinding.

## Setup

The existing `fastbook` conda env:

    PY=/home/mvine/miniconda3/envs/fastbook/bin/python

**THE ENV CHANGED MID-EXPERIMENT. The two arms did not run on the same stack.**

| | transformers | torch |
|---|---|---|
| arm A + the lead baseline (2026-10-06) | **4.37.2** | **2.2.0+cu121** |
| arm B (2026-10-07) | **5.18.0** | **2.14.1+cu130** |

Nothing in this directory caused the upgrade; it happened between the two runs.
Consequences:

- `evaluation_strategy` was renamed `eval_strategy`, and `warmup_ratio` was
  dropped in favour of `warmup_steps` only. `finetune_bart.py` now detects which
  names the installed version accepts rather than hard-coding either.
- `accelerate` (1.15.0) had to be installed — transformers 5.x `Trainer`
  requires >=1.1.0 and the env had none. **That is the only package this
  directory installed.**
- Arm A's numbers are reproducible only under 4.37.2. Re-running it on 5.18.0
  may not reproduce them exactly. Its generations are preserved in
  `out/A_bart_large_cnn.jsonl`, so the *measurements* stand regardless.

**Check the installed version before trusting any command here**, and use
`dataclasses.fields(Seq2SeqTrainingArguments)` rather than
`inspect.signature(...)` to discover accepted arguments — in 5.x the generated
`__init__` is narrower than the dataclass, so a signature check reports
`warmup_ratio` as present and the constructor then rejects it.

Input data is the **decoded text** of the client pairs, not the BPE ids, because
BART has its own tokeniser. `prepare_data.py` drives the project's Perl tokeniser
to decode them, applying the same skip rule `MVEncDecInfer` uses, so row N here
is row N everywhere else in the pipeline:

    data/val.jsonl     902 pairs   {"id","source","reference"}
    data/train.jsonl  8118 pairs

## How to run

    PY=/home/mvine/miniconda3/envs/fastbook/bin/python

    # 0. build data/{train,val}.jsonl from the project's pair files
    $PY prepare_data.py

    # arm A — zero-shot, CPU-only, safe to run while the GPU is busy
    $PY run_zeroshot.py --model facebook/bart-large-cnn --split val --out out/A_bart_large_cnn.jsonl

    # arm B — fine-tune, NEEDS THE GPU (do not run while a training job is active)
    $PY finetune_bart.py --model facebook/bart-base --epochs 3 --out out/B_bart_base
    $PY run_zeroshot.py --model out/B_bart_base --split val --out out/B_bart_base.jsonl

    # the lead baseline — no model, no GPU, runs in a second. Do this FIRST.
    $PY run_lead.py --split val --out out/L_lead.jsonl

    # characterise the output — extractiveness and boilerplate contamination
    $PY measure.py out/A_bart_large_cnn.jsonl --examples 5

    # convert to the project's CSV format for identifier measurement and blind grading.
    # The pairs file comes from OUR model's run, so both arms are graded against
    # byte-identical sources; to_bakeoff.py verifies the ids agree and aborts if not.
    $PY to_bakeoff.py out/A_bart_large_cnn.jsonl /tmp/ft_pairs.csv out/A --label bart-large-cnn
    cd /home/mvine/embedding && perl measure_identifiers.pl /tmp/ft_pairs.csv \
        baseline_hf/out/A_gen.csv

**Arm A is CPU-only on purpose.** At the time of writing the GPU has 1.2 GB free
with the stage-2 run occupying it, and an OOM there would cost hours.
`finetune_bart.py` enforces this itself: it checks free VRAM at startup and
refuses to run below 6 GB rather than trusting the operator to remember.

## The scripts

| script | does |
|---|---|
| `prepare_data.py` | decode the client BPE pairs to `data/{train,val}.jsonl` text |
| `run_zeroshot.py` | generate summaries; used for BOTH arms, so they share a decode path |
| `run_lead.py` | the lead baseline — no model, strip boilerplate, first N sentences |
| `measure.py` | extractiveness + boilerplate contamination, split by truncation |
| `to_bakeoff.py` | emit `_gen.csv` for the project's identifier/grading tooling |
| `finetune_bart.py` | arm B fine-tune (Seq2SeqTrainer, VRAM-guarded) |

`measure.py` was not in the original plan. It exists because the arm A smoke test
produced output that *read* as fluent summary but turned out to be near-verbatim
copying, which the A/B/C rubric cannot distinguish from abstraction — see
`RESULTS.md`.

## Results

See `RESULTS.md`. It is written as the experiment proceeds and records what was
found, including anything that contradicts the hypothesis above.
