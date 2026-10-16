#!/usr/bin/env python
"""Control experiment: fine-tune facebook/bart-base for sequence classification
on the same labelled email corpus the from-scratch classifier plateaued on.

THE QUESTION. Our d=256 encoder-only classifier reaches 73.5% accuracy, and 11
configurations from 30K to 7M parameters all land in 71-73%. That plateau was
read as a data ceiling -- but every one of those 11 measurements compares our
model against our other models. The same reasoning was wrong for the
summariser, where an off-the-shelf checkpoint on the same data scored 90%
against our 20%. So: give a pretrained model the same data and see where it
lands.

  ~73%  -> the ceiling is the data and labels; porting BART's encoder into our
           stack would be wasted work.
  much higher -> the ceiling is pretrain quality, and the port is justified.

REPORTS MACRO-F1 AS WELL AS ACCURACY, deliberately. 80.3% of this corpus sits
in three of eleven classes and the smallest class has 12 examples in total, so
a model that quietly ignores the eight rare classes scores 80.3% accuracy --
better than our actual 73.5% -- while being useless for the rare categories a
human most wants routed correctly. Accuracy alone cannot tell those apart.
"""
import json
import os, numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "facebook/bart-base"
MAXLEN = 512          # matches the from-scratch classifier's MAX_PAD_LEN
EPOCHS = 8
LABELS = json.load(open(f"{HERE}/clsdata/labels.json"))

free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
print(f"free VRAM {free:.1f} GB")
if torch.cuda.is_available() and free < 5:
    raise SystemExit("under 5 GB free -- refusing to start (another job is running)")

tok = AutoTokenizer.from_pretrained(MODEL)

class DS(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = tok(r["text"], truncation=True, max_length=MAXLEN)
        enc["labels"] = r["label"]
        return enc

train_ds, val_ds = DS(f"{HERE}/clsdata/train.jsonl"), DS(f"{HERE}/clsdata/val.jsonl")
print(f"train {len(train_ds)}  val {len(val_ds)}  classes {len(LABELS)}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS),
    id2label={i: c for i, c in enumerate(LABELS)},
    label2id={c: i for i, c in enumerate(LABELS)})

def metrics(p):
    preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(p.label_ids, preds, average="weighted", zero_division=0)}

# Filter to what THIS transformers version accepts. 5.x dropped
# overwrite_output_dir and renamed evaluation_strategy -> eval_strategy, and the
# generated __init__ is narrower than the dataclass, so dataclasses.fields() is
# the authoritative list -- inspect.signature() is not (see README).
import dataclasses
want = dict(output_dir=f"{HERE}/out/cls_bart", overwrite_output_dir=True,
            num_train_epochs=EPOCHS, learning_rate=2e-5, warmup_ratio=0.1,
            per_device_train_batch_size=8, per_device_eval_batch_size=16,
            gradient_accumulation_steps=2, logging_steps=100,
            eval_strategy="epoch", evaluation_strategy="epoch",
            save_strategy="no", report_to=[], seed=42, fp16=True)
ok = {f.name for f in dataclasses.fields(TrainingArguments)}
dropped = sorted(set(want) - ok)
if dropped: print(f"this transformers does not accept: {dropped}")
args = TrainingArguments(**{k: v for k, v in want.items() if k in ok})

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()

pred = trainer.predict(val_ds)
logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
preds = np.argmax(logits, axis=-1)
# Persist the fine-tuned model. save_strategy="no" above is deliberate -- it
# avoids writing a 560 MB checkpoint after EVERY epoch -- but with no final save
# either, the result was unreproducible for inference: evaluating it, measuring
# its latency, or using it to mine rare-class candidates from unlabelled email
# all meant a ~36 min retrain. Set SAVE_MODEL=0 to skip.
if os.environ.get("SAVE_MODEL", "1") != "0":
    trainer.save_model(want["output_dir"])
    tok.save_pretrained(want["output_dir"])
    print(f"saved fine-tuned model to {want['output_dir']}")

print("\n================ CONTROL RESULT ================")
print(f"accuracy    {accuracy_score(pred.label_ids, preds)*100:.1f}%")
print(f"macro-F1    {f1_score(pred.label_ids, preds, average='macro', zero_division=0):.3f}")
print(f"weighted-F1 {f1_score(pred.label_ids, preds, average='weighted', zero_division=0):.3f}")
print("\n" + classification_report(pred.label_ids, preds, target_names=LABELS,
                                   digits=3, zero_division=0))

# Persist per-example (true, predicted). Without this, NOTHING about this arm
# can be re-analysed -- bootstrap CIs, a different averaging, a confusion matrix
# -- without a ~36 min retrain, and the only record is whatever got printed to a
# log. Mirrors --dump_preds on the C++ trainer so the two arms are comparable
# with the same script. Same column order: true,pred.
DUMP = os.environ.get("DUMP_PREDS", "cls_preds_bart.csv")
with open(DUMP, "w") as f:
    for t, q in zip(pred.label_ids, preds):
        f.write(f"{int(t)},{int(q)}\n")
print(f"\nwrote {len(preds)} predictions to {DUMP}")
print("NOTE: these class ids are LABELS.index() order (alphabetical), which is NOT")
print("      the C++ trainer's ordering. Map by name or by support before comparing.")
