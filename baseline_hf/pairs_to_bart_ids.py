#!/usr/bin/env python
"""Re-encode the client pairs with BART's tokeniser, in the trainer's pair format.

The 16k pair files cannot be used with BART's weights: its embedding table is
50,265 entries in its own BPE, so feeding our id 4217 into it returns whatever
token 4217 means in GPT-2's vocabulary. Unrelated.

Only the PAIRS need this. The 2.11B pretrain corpus does NOT: its whole purpose
was to give the model general language ability, which is exactly what BART's
weights already carry.

Output format, one line per pair, matching what MVEncDecTrainer reads:

    src_ids,50264,tgt_ids

SEPARATOR IS 50264 (`<mask>`), NOT 2 (`</s>`), even though </s> is BART's
sep_token_id. The trainer splits on the FIRST occurrence, and a source encoded
with special tokens ENDS in </s> -- the split would land there and silently
discard the target. Measured over 2000 pairs: neither </s> nor <mask> occurs
inside an encoded source, so <mask> is safe and cannot collide however the
source is encoded. Same reasoning as [UNK]-doubling-as-[MASK] in the denoising
packer.

Encoded WITHOUT special tokens: the trainer adds its own boundaries, and a
stray <s>/</s> in the middle of a pair line would be read as content.

  python pairs_to_bart_ids.py [--out-dir /home/mvine/shared/rdf/pretrain_data]
"""
import argparse
import json
import os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = 50264          # <mask>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/bart-base")
    ap.add_argument("--out-dir", default="/home/mvine/shared/rdf/pretrain_data")
    ap.add_argument("--prefix", default="bart50k")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.mask_token_id == SEP, f"separator {SEP} is not <mask> for this tokeniser"

    for split in ("train", "val"):
        src_path = os.path.join(HERE, "data", f"{split}.jsonl")
        if not os.path.exists(src_path):
            raise SystemExit(f"missing {src_path} — run prepare_data.py first")
        out_path = os.path.join(args.out_dir, f"{args.prefix}_{split}_finetune.txt")

        n = 0
        slen, tlen = [], []
        with open(out_path, "w") as out:
            for line in open(src_path):
                r = json.loads(line)
                s = tok(r["source"],  add_special_tokens=False)["input_ids"]
                t = tok(text_target=r["reference"], add_special_tokens=False)["input_ids"]
                if not s or not t:
                    continue
                if SEP in s or SEP in t:
                    # would make the line unsplittable; measured as never happening
                    raise SystemExit(f"{split}: separator id {SEP} occurs in the text")
                out.write(",".join(map(str, s)) + f",{SEP}," + ",".join(map(str, t)) + "\n")
                slen.append(len(s)); tlen.append(len(t)); n += 1

        slen.sort(); tlen.sort()
        q = lambda a, p: a[int(len(a) * p)] if a else 0
        over = sum(1 for x in slen if x > 1024)
        print(f"{out_path}")
        print(f"  {n} pairs | source p50 {q(slen,.5)} p95 {q(slen,.95)} max {slen[-1]}"
              f" | summary p50 {q(tlen,.5)} p95 {q(tlen,.95)} max {tlen[-1]}")
        print(f"  over BART's 1024 position limit: {over} ({100*over/max(1,n):.1f}%)")

    print("\nvocab 50265, pad_id 1, separator 50264 (<mask>)")


if __name__ == "__main__":
    main()
