#!/usr/bin/env python
"""Characterise generated summaries against their sources.

NOT a quality score. These are three mechanical properties that separate the
failure MODES we have seen, and that the A/B/C grading cannot distinguish
because a fluent copy and a fluent abstraction both read as "a sentence".

  extractiveness  fraction of the summary's 4-grams that appear VERBATIM in the
                  source. A real abstractive summary rewrites: our gpt-4.1-mini
                  references land low. A lead-bias extractor copies the opening
                  sentences and lands high. This is the measure that tells the
                  two apart, and it is the one that caught bart-large-cnn.

  boilerplate     did the summary pull in signature blocks, phone numbers, email
                  addresses, legal disclaimers? A summariser trained on news has
                  never seen an email footer and has no reason to skip it. The
                  reference summaries never contain these, so any hit is a defect
                  with no judgement call attached.

  chars           length, for comparison with the reference distribution.

Report is split by whether the source exceeded the model's input limit, because
BART's 1024 tokens against our 2048 is a real handicap and a single headline
number hides it.

  python measure.py out/A_bart_large_cnn.jsonl [--split val]
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Each of these appears in the client sources and in NO reference summary.
BOILERPLATE = [
    (r"\+\d[\d\s()-]{7,}", "phone number"),
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "email address"),
    (r"\b(?:kind|best|warm)\s+regards\b|thanks\s*&\s*regards|"
     r"met\s+vriendelijke\s+groet|\bbien\s+cordialement\b", "sign-off"),
    (r"confidential\s+or\s+privileged|should\s+not\s+be\s+distributed|"
     r"if\s+you\s+have\s+received\s+this\s+(?:email|message)\s+in\s+error|"
     r"\brestricted\b", "legal disclaimer"),
    (r"\bB-BBEE\b|\bVan:\s|\bDatum:\s|\bSent:\s|\bFrom:\s.*\bTo:\s", "header/footer fragment"),
]


def ngrams(text, n=4):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def extractiveness(summary, source):
    sg = ngrams(summary)
    if not sg:
        return None
    return len(sg & ngrams(source)) / len(sg)


def boilerplate(summary):
    return [name for pat, name in BOILERPLATE if re.search(pat, summary, re.I)]


def stats(rows, label):
    rows = [r for r in rows if r["ext"] is not None]
    if not rows:
        print(f"  {label:<28} (none)")
        return
    ex = sorted(r["ext"] for r in rows)
    ch = sorted(r["chars"] for r in rows)
    nb = sum(1 for r in rows if r["bp"])
    med = lambda a: a[len(a) // 2]
    print(f"  {label:<28} n={len(rows):<5} extractive p50 {med(ex):.2f} "
          f"mean {sum(ex)/len(ex):.2f}   chars p50 {med(ch):<5} "
          f"boilerplate {nb:>4} ({100*nb/len(rows):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", help="a run_zeroshot.py output .jsonl")
    ap.add_argument("--split", default="val")
    ap.add_argument("--examples", type=int, default=0,
                    help="print this many of the most extractive summaries")
    args = ap.parse_args()

    val = {json.loads(l)["id"]: json.loads(l)
           for l in open(os.path.join(HERE, "data", f"{args.split}.jsonl"))}

    gen, ref = [], []
    for l in open(args.gen):
        r = json.loads(l)
        v = val[r["id"]]
        gen.append({"id": r["id"], "trunc": r.get("truncated", False),
                    "ext": extractiveness(r["summary"], v["source"]),
                    "chars": len(r["summary"]), "bp": boilerplate(r["summary"]),
                    "text": r["summary"]})
        ref.append({"id": r["id"], "trunc": False,
                    "ext": extractiveness(v["reference"], v["source"]),
                    "chars": len(v["reference"]), "bp": boilerplate(v["reference"]),
                    "text": v["reference"]})

    print(f"\n{args.gen}  —  {len(gen)} summaries\n")
    stats(ref, "REFERENCE (gpt-4.1-mini)")
    stats(gen, "GENERATED (all)")
    stats([r for r in gen if not r["trunc"]], "GENERATED, source fitted")
    stats([r for r in gen if r["trunc"]], "GENERATED, source truncated")

    hits = {}
    for r in gen:
        for name in r["bp"]:
            hits[name] = hits.get(name, 0) + 1
    if hits:
        print("\n  boilerplate pulled in, by kind:")
        for name, c in sorted(hits.items(), key=lambda x: -x[1]):
            print(f"    {name:<26} {c:>4}  ({100*c/len(gen):.1f}%)")

    if args.examples:
        print("\n  most extractive:")
        for r in sorted(gen, key=lambda x: -(x["ext"] or 0))[: args.examples]:
            print(f"    id {r['id']} ext {r['ext']:.2f}: {r['text'][:140]}")


if __name__ == "__main__":
    main()
