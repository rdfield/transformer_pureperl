#!/usr/bin/env python
"""Convert a HuggingFace run into the project's gen.csv format for grading.

Writes ONLY the generation side (`<out>_gen.csv`). The source/reference side is
REUSED from a `*_pairs.csv` that `gen_to_bakeoff.pl` already wrote for our own
model, deliberately:

  - the Perl bridge decodes the source from the BPE ids our model actually saw,
    and verifies alignment against the REF ids MVEncDecInfer printed. That check
    cannot be replicated here, so inheriting its output inherits its guarantee.
  - both arms are then measured against BYTE-IDENTICAL sources. Writing our own
    pairs file would risk the two arms being graded against subtly different
    text (whitespace, truncation) while every number still looked plausible.
  - `band` in that file is the source length in OUR tokeniser. BART's token
    counts are not comparable, so this script has no business setting it.

Alignment is still VERIFIED, not assumed: the reference text for each id must
match between the pairs file and data/val.jsonl. The comparison is
whitespace-normalised because prepare_data.py collapses runs of whitespace and
the Perl bridge does not. A mismatch aborts — grading arm A against arm B's
threads would produce a confident, meaningless result.

  python to_bakeoff.py out/A_bart_large_cnn.jsonl /tmp/ft_pairs.csv out/A --label bart-large-cnn

Then, from /home/mvine/embedding:
  perl measure_identifiers.pl /tmp/ft_pairs.csv baseline_hf/out/A_gen.csv
  perl bakeoff_bundle.pl /tmp/ft_pairs.csv OURS_gen.csv baseline_hf/out/A_gen.csv ...
"""
import argparse
import base64
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
norm = lambda s: re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", help="run_zeroshot.py output .jsonl")
    ap.add_argument("pairs", help="a *_pairs.csv written by gen_to_bakeoff.pl")
    ap.add_argument("out", help="output prefix; writes <prefix>_gen.csv")
    ap.add_argument("--label", default="bart", help="arm name, appears in the key file")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    val = {json.loads(l)["id"]: json.loads(l)
           for l in open(os.path.join(HERE, "data", f"{args.split}.jsonl"))}

    # reference text per id, as the Perl side decoded it
    csv.field_size_limit(10_000_000)
    pairs = {}
    with open(args.pairs, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                pairs[int(row[0])] = base64.b64decode(row[3]).decode("utf-8", "replace")

    rows = [json.loads(l) for l in open(args.gen)]
    bad, checked = [], 0
    for r in rows:
        i = r["id"]
        if i not in pairs or i not in val:
            bad.append(f"id {i} missing from " + ("pairs" if i not in pairs else "val.jsonl"))
            continue
        checked += 1
        if norm(pairs[i]) != norm(val[i]["reference"]):
            bad.append(f"id {i}: reference differs\n"
                       f"    pairs: {norm(pairs[i])[:70]}\n"
                       f"    val  : {norm(val[i]['reference'])[:70]}")
    if bad:
        print(f"MISALIGNED — {len(bad)} of {len(rows)} ids do not agree:", file=sys.stderr)
        for b in bad[:5]:
            print("  " + b, file=sys.stderr)
        if len(bad) > 5:
            print(f"  ... and {len(bad)-5} more", file=sys.stderr)
        sys.exit("refusing to write: every identifier number would be measured "
                 "against the wrong thread")
    print(f"alignment verified: all {checked} references match {args.pairs}")

    path = f"{args.out}_gen.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for r in rows:
            b64 = base64.b64encode(r["summary"].encode("utf-8")).decode("ascii")
            w.writerow([r["id"], args.label, b64, 0, 0, 0, "stop", "n/a"])
    print(f"wrote {path}  ({len(rows)} generations, label '{args.label}')")


if __name__ == "__main__":
    main()
