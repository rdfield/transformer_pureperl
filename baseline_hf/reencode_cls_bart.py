#!/usr/bin/env python
"""Re-encode the classifier split with BART's tokeniser, in MVTrainer's CSV format.

  out rows:  label,id,id,id,...          (no trailing newline issues)

The first id is BART's <s> (0), which serves as the CLS position the classifier
head reads -- so nothing needs prepending. PAD is BART's own <pad> (1). Both are
INSIDE the 50265 vocab, unlike our 8K/16K scheme which appended PAD/CLS past the
end, so --pad_id 1 --cls_id 0 --vocab_size 50265.

Labels come straight from clsdata/*.jsonl, so this is the SAME stratified split
as the bart-base control and the from-scratch arm. NOTE the label indices are
alphabetical (LABELS.index order) and do NOT match cls_train.csv's ordering --
map by name before comparing per-class figures across the two.
"""
import json, os, sys
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MAXLEN = int(os.environ.get("MAXLEN", "512"))
tok = AutoTokenizer.from_pretrained("facebook/bart-base")

for split in ("train", "val"):
    src = os.path.join(HERE, "clsdata", f"{split}.jsonl")
    dst = os.path.join("/home/mvine/embedding_cpp", f"cls_{split}_bart.csv")
    n = 0; tot = 0; capped = 0
    with open(src) as f, open(dst, "w") as o:
        for line in f:
            r = json.loads(line)
            ids = tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"]
            if len(ids) < 2:
                continue
            o.write(f"{r['label']}," + ",".join(str(i) for i in ids) + "\n")
            n += 1; tot += len(ids); capped += (len(ids) >= MAXLEN)
    print(f"  {dst}: {n} rows, mean {tot/n:.0f} tok, {100*capped/n:.1f}% at the {MAXLEN} cap")
print(f"\n  flags: --vocab_size {len(tok)} --pad_id {tok.pad_token_id} --cls_id {tok.bos_token_id}")
