#!/usr/bin/env python
"""The lead baseline: no model at all. Strip the boilerplate, keep the first
content sentences, stop.

WHY THIS EXISTS, and why it should have been the FIRST thing built. In
summarisation research a lead baseline ("lead-3" on news) is the standard control
because it is frequently hard to beat, and a system that cannot beat it has not
demonstrated that it is summarising. We have been comparing a from-scratch 90.7M
model against a 406M model against a commercial teacher, and never once against
"print the first two sentences".

It is also a live product question, not just a methodological one. client threads
are transactional business requests, and a request email conventionally states
the request at the top. If the lead carries the facts, then a large share of
threads need no generative model — which would be the cheapest possible answer
to "no garbage to customer", because extracted text cannot hallucinate.

What it CANNOT do is the abstraction the references perform: attributing the
request to a named person in a role ("Jill asks Keith to..."). So expect high
faithfulness and wrong form. That combination is exactly what a templating or
hybrid approach would exploit, and this script measures how much is there.

The boilerplate stripping is the whole substance of the baseline — an unstripped
lead is just the greeting. Patterns below are drawn from the client corpus and are
the same families `measure.py` detects.

  python run_lead.py --split val --out out/L_lead.jsonl [--sentences 2]
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Everything from here on is not the message: quoted history, signature, legal.
#
# NOTE THE ABSENT ^ ANCHORS. The obvious way to find a quoted-history header is
# `^\s*(?:From|Sent|Subject):` with re.M, and it matches NOTHING here, silently:
# prepare_data.py collapses every run of whitespace to a single space so the
# decoded sources contain no newlines at all. Measured cost of getting this
# wrong: 9.9% of leads ran straight on into the quoted thread below them. Any
# pattern in this file that assumes line structure is dead code.
CUT = re.compile(
    r"(?:"
    r"\b(?:thanks?|regards|kind\s+regards|best\s+regards|warm\s+regards|"
    r"thanks?\s*&\s*regards|met\s+vriendelijke\s+groet|bien\s+cordialement|"
    r"cordialement|mit\s+freundlichen\s+gr)\b"
    # quoted-history headers, unanchored. 'Subject:'/'Objet:' and the date forms
    # are safe; 'To:'/'Cc:'/'De:' are NOT included because they occur in prose
    # and inside address blocks that are genuine content.
    r"|\b(?:van|datum|from|sent|verzonden|subject|objet|envoye|gesendet)\s*:\s"
    r"|-----\s*original\s+message"
    # external-sender banners injected by the mail gateway
    r"|\*?\s*caution\s*\*?\s*:|\bthis\s+e-?mail\s+originated\s+from\s+outside"
    r"|\byou\s+don'?t\s+often\s+get\s+email\s+from\b"
    r"|\bthis\s+(?:message|e-?mail)\s+and\s+its\s+attachments?\b"
    r"|\bce\s+message\s+et\s+ses\s+pieces\s+jointes\b"
    r"|\bconfidential\s+or\s+privileged\b"
    r"|_{10,}|-{10,}|={10,}"
    r")", re.I)

# Zero-width and soft-hyphen padding. HTML marketing mail pads its preheader
# with runs of these so the preview text looks right; one val source is 32%
# invisible characters by count. They SURVIVED BPE ENCODING, so the model is
# being fed them too — this is a preprocessing finding, not just
# a baseline one. No reference summary contains a single one.
INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\u034f\u00ad\ufeff]+")

# Preamble, stripped from the FRONT. Distinct from CUT, which truncates at the
# first match: when the junk is at the START, truncating there leaves nothing.
# Two different operations, so two different patterns.
PREAMBLE = re.compile(
    r"\A\s*(?:[\w .&'-]{0,30}?)??"              # a bare brand name, e.g. "Vendor-A"
    r"(?:to view this email as a web ?page|view (?:this email )?in (?:your )?browser"
    r"|having trouble (?:viewing|reading) this)"
    # GREEDY span: ", click here" must be consumed. Written lazily first, which
    # matched zero characters and left "click here" at the head of every lead.
    r"[^.!?\n]{0,40}[\s,.:;|-]*", re.I)

# A greeting is not content. Drop it so sentence 1 is the actual request.
GREET = re.compile(
    r"\A\s*(?:general\s+)?"
    r"(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening|day)|bonjour)"
    r"\b[^.!?\n]{0,40}?[,.!:\n]\s*", re.I)


def sentences(text, n):
    """Split on sentence enders, but not on the dots inside an abbreviation,
    a decimal, or a product code — client bodies are full of 'Unit 02-07' and
    'No. 773511650', and a naive split returns half a line."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    out, got = [], 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
        got += 1
        if got >= n:
            break
    return " ".join(out)


def lead(source, n):
    body = INVISIBLE.sub(" ", source)
    body = PREAMBLE.sub("", body)
    body = CUT.split(body)[0]
    body = GREET.sub("", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:                       # the whole email was boilerplate
        body = re.sub(r"\s+", " ", GREET.sub("", source)).strip()
    return sentences(body, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sentences", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(os.path.join(HERE, "data", f"{args.split}.jsonl"))]
    empty = 0
    with open(args.out, "w") as f:
        for r in rows:
            s = lead(r["source"], args.sentences)
            empty += not s
            f.write(json.dumps({"id": r["id"], "summary": s, "truncated": False,
                                "input_tokens": 0, "gen_chars": len(s),
                                "ms": 0}) + "\n")
    print(f"wrote {args.out}  ({len(rows)} leads, {args.sentences} sentences each)")
    print(f"empty after stripping: {empty}")
    print("\nNOTE: extractiveness for this baseline is 1.00 BY CONSTRUCTION — it is")
    print("copied text. The number to read is whether it carries the same FACTS as")
    print("the reference, which needs the A/B/C grading, not measure.py.")


if __name__ == "__main__":
    main()
