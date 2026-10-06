#!/usr/bin/env python
"""Build a sample.csv (id,band,source_b64,reference_b64) from data/val.jsonl.

FOR HUGGINGFACE-ONLY COMPARISONS. Do NOT use this when our own model is one of
the arms — use the `*_pairs.csv` that `gen_to_bakeoff.pl` writes instead.

The reason is the one documented in to_bakeoff.py: the Perl bridge decodes the
source from the BPE ids our model actually saw and verifies alignment against the
REF ids MVEncDecInfer printed. This script has no such check — it trusts that
val.jsonl row N is thread N. That is true (both apply the same [SEP] skip rule
over the same file) but it is trust, not verification, and it is only acceptable
when every arm being compared reads from this same file.

  python make_sample_csv.py --out out/hf_sample.csv --limit 20 [--ids 1,2,3]
"""
import argparse
import base64
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
b64 = lambda s: base64.b64encode(s.encode("utf-8")).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--ids", default="", help="comma-separated ids, overrides --limit")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(os.path.join(HERE, "data", f"{args.split}.jsonl"))]
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["id"] in want]
    else:
        rows = rows[: args.limit]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for r in rows:
            # band = source length in CHARACTERS. Our tokeniser's count is not
            # available here and BART's is not comparable to it.
            w.writerow([r["id"], f"{len(r['source'])}ch",
                        b64(r["source"]), b64(r["reference"])])
    print(f"wrote {args.out}  ({len(rows)} threads)")


if __name__ == "__main__":
    main()
