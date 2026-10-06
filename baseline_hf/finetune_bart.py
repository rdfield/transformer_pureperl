#!/usr/bin/env python
"""Arm B: fine-tune a public pretrained seq2seq model on the SAME client pairs.

This is the control that matters. Arm A only showed that an off-the-shelf
checkpoint does the wrong task. Arm B asks the actual question: given a model of
comparable size whose pretraining was done properly by someone else, do the same
8,118 client pairs produce a usable summariser?

DESIGN: this uses HuggingFace's own Seq2SeqTrainer rather than a hand-written
loop, on purpose. The point of a control is to run the baseline the way the rest
of the world runs it — a bespoke loop would make any gap arguable ("you trained
it badly") and that is exactly the ambiguity the experiment exists to remove.

MATCHED TO OUR RUN where it is meaningful:
  --patience 3          same as MVEncDecTrainer --patience 3
  best-by-val-loss      same checkpoint-selection rule
  warmup_ratio 0.06     we warm up; an unwarmed baseline would be a straw man
  no label smoothing    our loss is plain masked cross-entropy
NOT matched, and cannot be: tokeniser (BART's own BPE), input limit (1024 vs our
2048), and the pretraining corpus. Those are the independent variable.

  python finetune_bart.py --model facebook/bart-base --out out/B_bart_base

Add --train-file to point at a different pair set (e.g. the stage-2 Qwen pairs,
decoded to jsonl) to run the second condition.
"""
import argparse
import json
import os
import sys

import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

HERE = os.path.dirname(os.path.abspath(__file__))


class Pairs(torch.utils.data.Dataset):
    """Tokenise on the fly. 9k pairs is small; caching buys nothing and the
    memory is better spent on sequence length."""

    def __init__(self, path, tok, max_in, max_out):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok, self.max_in, self.max_out = tok, max_in, max_out

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tok(r["source"], truncation=True, max_length=self.max_in)
        y = self.tok(text_target=r["reference"], truncation=True, max_length=self.max_out)
        x["labels"] = y["input_ids"]
        return x


def guard_vram(need_gb):
    """Refuse to start if the GPU is in use. A standing project rule: an OOM in
    the from-scratch trainer costs hours of real work, and this baseline is never
    worth that. Checked here rather than left to the operator because the whole
    reason arm A runs on CPU is that the GPU was busy."""
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 2**30, total / 2**30
    print(f"GPU: {free_gb:.1f} GB free of {total_gb:.1f} GB")
    if free_gb < need_gb:
        sys.exit(f"REFUSING TO START: {free_gb:.1f} GB free, need ~{need_gb:.0f} GB.\n"
                 f"Something else is using the GPU (check `nvidia-smi`). Wait for it\n"
                 f"to finish, or pass --force if you know the other job has ended.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/bart-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-file", default=os.path.join(HERE, "data", "train.jsonl"))
    ap.add_argument("--val-file", default=os.path.join(HERE, "data", "val.jsonl"))
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="3e-5 is the standard BART fine-tune rate; our 3e-4 is for "
                         "a from-scratch model and would wreck a pretrained one")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8, help="effective batch 32, as our run")
    ap.add_argument("--max-in", type=int, default=1024)
    ap.add_argument("--max-out", type=int, default=160, help="our --summary_budget is 224")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--need-gb", type=float, default=6.0)
    ap.add_argument("--force", action="store_true", help="skip the VRAM guard")
    args = ap.parse_args()

    if not args.force:
        guard_vram(args.need_gb)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    tr = Pairs(args.train_file, tok, args.max_in, args.max_out)
    va = Pairs(args.val_file, tok, args.max_in, args.max_out)
    print(f"train {len(tr)} pairs, val {len(va)} pairs")

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        warmup_ratio=0.06,
        evaluation_strategy="epoch",   # 4.37.2 spelling; renamed eval_strategy in 4.46+
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[],
        predict_with_generate=False,   # generation is run_zeroshot.py's job, greedy
        seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=tr, eval_dataset=va,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nbest model saved to {args.out}")
    print(f"next: python run_zeroshot.py --model {args.out} --device cuda "
          f"--out out/B_$(basename {args.out}).jsonl")


if __name__ == "__main__":
    main()
