#!/usr/bin/env python
"""Build data/{train,val}.jsonl from the project's client pair files.

The pair files are comma-separated BPE ids in OUR 16K vocabulary
(`email_ids,16003,summary_ids`). HuggingFace models have their own tokenisers, so
they need the TEXT. Decoding is done by the project's Perl tokeniser — the only
implementation that can read a `minbpe v1` vocab — and this script drives it.

Output: one JSON object per line, {"id", "source", "reference"}.

  source     the email/thread, decoded from the ids before [SEP]
  reference  the gpt-4.1-mini summary, decoded from the ids after [SEP]

WHY NOT read the original text files instead: the ids are what the from-scratch
model was actually trained and evaluated on, including whatever truncation and
cleaning the encoding pipeline applied. Decoding them keeps the two systems on
byte-identical inputs. Using the upstream text would silently give BART a
cleaner source than our model ever saw.
"""
import json
import os
import subprocess
import sys

P = "/home/mvine/shared/rdf/pretrain_data"
VOCAB = f"{P}/cosmo_client_blended2.model"
SEP = 16003
HERE = os.path.dirname(os.path.abspath(__file__))

DECODER = r"""
use strict; use warnings;
use lib '/home/mvine/embedding'; use ML::Tokeniser;
my ($model,$in,$sep)=@ARGV;
binmode(STDOUT, ':encoding(UTF-8)');
my $t=ML::Tokeniser->new(vocab_size=>16000); $t->load(file=>$model);
open my $fh,'<',$in or die "$in: $!";
while (my $l=<$fh>) {
   my @ids=grep {/^-?\d+$/} split /,/,$l;
   my ($s)=grep {$ids[$_]==$sep} 0..$#ids;
   next unless defined $s && $s>0 && $s<$#ids;
   my $src=$t->decode(ids=>[@ids[0..$s-1]]);
   my $ref=$t->decode(ids=>[@ids[$s+1..$#ids]]);
   for ($src,$ref) { s/\s+/ /g; s/^\s+|\s+$//g }
   # tab-separated so the caller can split without re-quoting
   print "$src\t$ref\n";
}
"""


def decode(pairs_path):
    """Run the Perl decoder and yield (source, reference) text pairs."""
    script = os.path.join(HERE, ".decode.pl")
    with open(script, "w") as f:
        f.write(DECODER)
    out = subprocess.run(
        ["perl", script, VOCAB, pairs_path, str(SEP)],
        capture_output=True, text=True, check=True,
    )
    for line in out.stdout.splitlines():
        if "\t" not in line:
            continue
        src, ref = line.split("\t", 1)
        if src and ref:
            yield src, ref


def build(name, pairs_path):
    path = os.path.join(HERE, "data", f"{name}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    slen = []
    rlen = []
    with open(path, "w") as f:
        for i, (src, ref) in enumerate(decode(pairs_path), 1):
            f.write(json.dumps({"id": i, "source": src, "reference": ref}) + "\n")
            slen.append(len(src))
            rlen.append(len(ref))
            n += 1
    slen.sort()
    rlen.sort()
    q = lambda a, p: a[int(len(a) * p)] if a else 0
    print(f"{name}.jsonl: {n} pairs")
    print(f"  source chars    p50 {q(slen,.5):>6}  p95 {q(slen,.95):>6}  max {slen[-1]:>6}")
    print(f"  reference chars p50 {q(rlen,.5):>6}  p95 {q(rlen,.95):>6}  max {rlen[-1]:>6}")
    return n


if __name__ == "__main__":
    for name, src in (("val", f"{P}/val_16000_finetune.txt"),
                      ("train", f"{P}/train_16000_finetune.txt")):
        if not os.path.exists(src):
            sys.exit(f"missing {src}")
        build(name, src)
    print("\nNOTE: lengths are CHARACTERS. BART's tokeniser differs from ours, so")
    print("token counts are not comparable between the two systems.")
