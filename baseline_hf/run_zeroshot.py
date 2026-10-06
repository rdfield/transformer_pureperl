#!/usr/bin/env python
"""Generate summaries with a HuggingFace seq2seq model.

Used for both arms: zero-shot on a public checkpoint (arm A), and inference on
our own fine-tuned checkpoint (arm B). Same code path so the two are comparable.

  --device cpu   is the DEFAULT, deliberately. At the time of writing the GPU is
                 occupied by a stage-2 training run with ~1.6 GB free, and an OOM
                 there would cost hours. Pass --device cuda only when the GPU is
                 known idle.

  --max-input    1024 for BART. Sources longer than this are TRUNCATED, which is
                 a real handicap versus our model's 2048 — the output records
                 `truncated` per row so results can be split by whether the model
                 actually saw the whole thread.

Generation settings mirror what our own inference does as closely as the two
APIs allow: beam search off (greedy), no sampling, and a repetition guard
equivalent to our --no_repeat_ngram 3. Greedy rather than beams because our
MVEncDecInfer is greedy, and beam search would give BART an advantage our model
does not have.

Output: JSON lines {"id","summary","truncated","input_tokens","gen_chars","ms"}
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or a local checkpoint dir")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda — see docstring")
    ap.add_argument("--max-input", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=120,
                    help="reference summaries are p50 219 / p95 351 CHARS, roughly 60-90 BPE tokens")
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", f"{args.split}.jsonl"))]
    if args.limit:
        rows = rows[: args.limit]

    # resume: skip ids already written, so a long CPU run is interruptible
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
        print(f"resuming: {len(done)} already done")
    todo = [r for r in rows if r["id"] not in done]

    print(f"model {args.model} on {args.device}; {len(todo)} of {len(rows)} to generate")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device).eval()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"parameters: {nparam/1e6:.1f}M")

    out = open(args.out, "a")
    t_all = time.time()
    n_trunc = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        enc = tok([c["source"] for c in chunk], return_tensors="pt",
                  truncation=True, max_length=args.max_input, padding=True)
        # was anything cut? compare against an untruncated encode
        full = [len(tok(c["source"])["input_ids"]) for c in chunk]
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                **{k: v.to(args.device) for k, v in enc.items()},
                max_new_tokens=args.max_new,
                num_beams=1,                       # greedy, matching MVEncDecInfer
                do_sample=False,
                no_repeat_ngram_size=args.no_repeat_ngram,
            )
        ms = (time.time() - t0) * 1000 / len(chunk)
        for c, g, fl in zip(chunk, gen, full):
            s = tok.decode(g, skip_special_tokens=True).strip()
            trunc = fl > args.max_input
            n_trunc += trunc
            out.write(json.dumps({"id": c["id"], "summary": s, "truncated": bool(trunc),
                                  "input_tokens": fl, "gen_chars": len(s),
                                  "ms": round(ms)}) + "\n")
        out.flush()
        done_n = i + len(chunk)
        if (i // args.batch) % 10 == 0:
            el = time.time() - t_all
            rate = done_n / el if el else 0
            eta = f"eta {(len(todo)-done_n)/rate/60:.0f} min" if rate else ""
            # flush=True because the documented way to run this is under nohup,
            # where stdout is block-buffered and progress would not appear until
            # the run finished. The JSONL itself is flushed per batch regardless.
            print(f"  {done_n}/{len(todo)}  {rate:.2f}/s  {eta}", flush=True)
    out.close()
    print(f"\nwrote {args.out}")
    print(f"truncated at --max-input {args.max_input}: {n_trunc} of {len(todo)} "
          f"({100*n_trunc/max(1,len(todo)):.1f}%) — report results split by this")


if __name__ == "__main__":
    main()
