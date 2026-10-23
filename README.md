# Transformer — Pure Perl

A from-scratch encoder-decoder Transformer implementation in Perl, with no ML frameworks or automatic differentiation. All forward passes, backpropagation, and optimisation are implemented by hand using nested Perl array references.

Intended for people who want to understand exactly what happens inside a Transformer at the implementation level, not just how to call PyTorch.

## What this is

- Full encoder-decoder Transformer (attention, residual connections, layer normalisation, positional encodings)
- Manual backpropagation throughout — every gradient derived and coded explicitly
- Adam optimiser in all weight-bearing layers (MHA, Linear, LayerNorm)
- PAD-masked cross-entropy loss
- Learning rate warmup
- Byte-level BPE tokeniser (GPT-4 regex split pattern)
- Numerical gradient verification tests for all key modules

## What this is not

- Fast. Training the full model (`train_model.pl`, d=256) takes ~9.5 hours per epoch on CPU.
- A production system. This is a learning and verification tool.

## Architecture

All modules live under `ML/`:

| Module | Role |
|--------|------|
| `ML::Transformer` | Top-level: encode → decode → project, save/load weights |
| `ML::Encoder` | N × EncoderBlock + LayerNorm |
| `ML::Decoder` | N × DecoderBlock + LayerNorm |
| `ML::EncoderBlock` | Self-MHA → Residual+Norm → FFN → Residual+Norm |
| `ML::DecoderBlock` | Masked self-MHA + cross-MHA + FFN, each with residual+norm |
| `ML::MHA` | Multi-Head Attention with Adam; stores W_q, W_k, W_v, W_o |
| `ML::PositionwiseFeedForward` | Linear → ReLU → Linear |
| `ML::LayerNormalisation` | Per-embedding γ/β with Adam |
| `ML::Linear` | Weight matrix + bias with Adam |
| `ML::LinearReLu` | Linear layer followed by ReLU, the first half of the FFN |
| `ML::ResidualConnection` | Sublayer + residual add + LayerNorm, with dropout on the sublayer branch |
| `ML::LossGradient` | Holds the loss gradient so it can start `backward()` like any other layer |
| `ML::InputEmbeddings` | Learnable token embedding table, scaled by √d_model |
| `ML::PositionalEmbeddings` | Sinusoidal positional encodings |
| `ML::Tokeniser` | BPE tokeniser (GPT-4 split pattern), extends ML::BaseTokeniser |
| `ML::BaseTokeniser` | Base class: get_stats, merge, save, load |
| `ML::DataLoader` | Batching, padding, tokenisation for training |
| `ML::Util` | matmul, softmax, adam_optimiser, masking utilities |
| `ML::Identifiers` | What counts as an identifier (PO number, SKU, serial) and how two are compared |
| `ML::PostFilter` | Neutralises structured literals in a summary that its source does not support |

## Running

All scripts should be run from the repo root:

```bash
# Quick convergence check (d=12, 50 epochs, ~9 min on CPU)
perl test_convergence.pl

# Full training (d=256, 15 epochs, ~6 days on CPU)
perl train_model.pl

# Gradient norm diagnostic
perl grad_norm_probe.pl

# Numerical gradient verification
perl test_linear.pl       # ML::Linear          (56/56)
perl test_mha.pl          # ML::MHA             (656/656)
perl test_layernorm.pl    # ML::LayerNorm       (42/42)
perl test_embedding.pl    # ML::Embedding       (31/31)
perl test_tokeniser.pl    # ML::Tokeniser       (54/54)
```

Each `train_model.pl` epoch saves a checkpoint: `d256_l2_h4_{epoch}.json`.

## Scripts

The `ML/` modules and the first two groups below are the Perl Transformer itself. The
rest grew up around a C++/CUDA reimplementation that lives in a separate repository:
preparing its training data, building its pretraining corpus, choosing the model that
writes its training summaries, and checking what it generates. Run everything from the
repo root. Many of these scripts read or write client email, which is never committed —
`.gitignore` excludes the data, the vocabularies trained on it and the encoded corpora —
so on a fresh clone those scripts have nothing to run on.

### Training and convergence

`train_model.pl` is the full training run: the encoder-decoder at d=256 with 2 layers and
4 heads, 15 epochs after a 3-epoch learning-rate warmup, saving a checkpoint every epoch.
At about 9.5 hours an epoch on CPU it is not the everyday check — that is
`test_convergence.pl`, the same architecture at d=12 on a copy task, which runs 50 epochs
in about 9 minutes and passes when the loss falls below 0.85 of its first-epoch value (it
currently reaches 0.018, with 100% accuracy). `test_classifier.pl` does the same for the
encoder-only path: a `[CLS]` token feeding a linear classification head. When training
stalls, start with `grad_norm_probe.pl`: one forward and backward pass, printing the
gradient norm at every layer, where vanishing or exploding gradients show up as layers more
than an order of magnitude apart. `test_weight_update.pl` runs one tiny epoch and saves the
weights before and after as JSON, for diffing what a single update changed.

### Gradient checks

Every backward pass is verified against numerical differentiation: nudge a parameter by
±ε (1e-4), measure how the loss moves, and compare that central difference with the
gradient the module computes. `test_linear.pl`, `test_mha.pl`, `test_layernorm.pl`,
`test_ffn.pl` and `test_residual.pl` cover the layers — `test_mha.pl` includes the
cross-attention path that carries the gradient back into the encoder, and
`test_layernorm.pl` the Adam update as well. `test_loss_grad.pl` covers the PAD-masked
cross-entropy loss and the gradient it starts backpropagation with. An embedding has no
differentiable input, since token ids are integers, so `test_embedding.pl` instead checks
that the gradient accumulates into the right rows of the table, including for tokens that
repeat. Run these after changing any `backward`.

### Tokeniser and vocabularies

`ML::Tokeniser` is byte-level BPE in the style of minbpe, with GPT-4's pre-tokenisation
split. Its encoding is byte-identical to minbpe's `RegexTokenizer`, checked on 3,000 real
emails (2.7M tokens) with two vocabularies. Training uses minbpe's rule, the most frequent
pair with ties to the one seen first, but tracks "first seen" incrementally, so some ties
break differently: a vocabulary trained here is valid BPE, not always merge-for-merge
identical to minbpe's. Training is deterministic. The split pattern is saved in the `.model`
file and restored by `load()`; pass `pattern =>` to `new()` to train with another one.
`train()` and `encode()` take decoded text, and warn once if given undecoded UTF-8 bytes.

**Ids changed on 2026-10-23.** Before then the encoder picked each merge by its frequency in
the text being encoded and ran without the pre-tokenisation split. Its output decoded
correctly but was not canonical BPE, used about 1.4× more tokens, and took time growing with
the square of the text length. Ids encoded before that date differ from today's for the same
text, so a model trained on them should be retrained on fresh encodings; decoding old ids is
unaffected.

`build_client_vocab.pl` trains a vocabulary of a chosen size on a text file and writes the
`.model`/`.vocab` pair the other scripts load; `email_csv_to_txt.pl` produces that training
text from the labelled email export, one body per line with the labels dropped.
`test_tokeniser.pl` checks the tokeniser end to end — 54 tests, including encode/decode round
trips on seen, unseen and multi-byte UTF-8 text, save/load, merge order, prefix locality,
UTF-8 training, pattern persistence and deterministic training. Vocabularies trained on
client email are not committed.

### Classifier data

`client_tokenise_emails.pl` encodes the labelled email export into the classifier's
training format: one email per line, as its category id, the `[CLS]` token, then the
body's token ids. `split_data.pl` makes a stratified train/validation split — each
category shuffled and divided separately, 20% to validation by default — so that rare
categories reliably land in both halves.

### Summary pairs for fine-tuning

`email_summaries_to_ids_v2.pl` takes a CSV of base64-encoded threads and their summaries
and writes each pair as one line: the thread's token ids, the `[SEP]` token, then the
summary's. That is the format the C++ trainer fine-tunes on. `dedup_split_pairs.pl` then
deduplicates and splits it into training and validation files. It deduplicates on the
thread alone, because the same thread with a different summary is still leakage between
the two.

### Pretraining corpus

Pretraining starts from the public Enron email dataset, `cleaned_enron_emails.json`: about
517,000 emails with their metadata. `enron_emails_to_base64.pl` writes each body as one
base64 line, so the file can be split and encoded in parallel, and
`convert_enron_base64_to_ids.sh <input> <shards> <model> <prefix>` does exactly that,
running `encode_input.pl` on every shard and writing `<input>.encoded`. `dedup_singles.pl`
removes exact duplicates and splits off a validation set (10% by default): the 516,794
emails come down to 199,570. The larger blended corpora used later get two more passes.
`clean_corpus_lines.pl` strips junk lines from inside documents before encoding, and
`filter_encoded_corpus.pl` drops what a straight encode leaves behind — binary
attachments decoded as text, and token ids outside the vocabulary. Always give it
`--vocab`: four such ids in a two-billion-token corpus once crashed pretraining four times
running. `register_compare.pl` measures how close a candidate corpus's register is to the
target email before days of compute are spent on it.

### Reading and auditing generated summaries

The C++ generators print token ids; `decode_gen.pl` turns their `GEN` and `REF` lines back
into text (set `VOCAB` to choose the tokeniser). The rest are the audits that decided what
to fix next. `audit_generations.pl` checks a batch for mode collapse — fluent, well-formed
summaries that barely depend on their input, which teacher-forced accuracy cannot see.
`audit_sample.pl` shows each thread beside its generated and reference summaries, so a
person grades against the source thread rather than the reference, which is itself
machine-written. `audit_tally.pl` steps through `decode_gen.pl` output recording a
single-key verdict for each summary, and `post_filter_audit.pl` does the same after
`ML::PostFilter` has neutralised structured literals — email addresses, phone numbers,
part numbers — that the source does not support.

### Identifier fidelity

A summary of business email is only useful if its purchase-order numbers, quote
references, SKUs and serial numbers survive exactly. `measure_identifiers.pl` checks that
against the source thread, including the common failure where an identifier starts right
and drifts partway through. `repair_identifiers.pl` is a post-processing pass that repairs
drifted identifiers against the source. Both rely on `ML::Identifiers` for what counts as
an identifier and how two are compared.

### Choosing the teacher: the bake-off, and Stage-2 pairs

The summaries the model learns from are written by a larger language model, and these
scripts choose which one, in stages numbered in their own headers. `bakeoff_sample.pl`
draws a length-stratified sample of threads so every candidate is judged on identical
inputs, and `bakeoff_sample_enron.pl` does the same over Enron email for the second stage
of the training curriculum. `bakeoff_generate.pl` summarises the sample through an
OpenAI-compatible API: the key comes from `OPENROUTER_API_KEY`, and the prompt from
`bakeoff_prompt.txt`, whose hash is recorded with the results. `bakeoff_screen.pl` applies
an automatic first screen, so human reading time goes only to candidates that pass it.
`bakeoff_bundle.pl` builds a blind grading sheet that is graded against the source thread,
and `bakeoff_tally.pl` unblinds it and applies a decision rule written into the script
before any grading started. `gen_to_bakeoff.pl` converts the C++ model's own output into
the same formats, so the identifier and grading tools can score it alongside the
candidates. Finally, `build_stage2_pairs.pl` turns the chosen teacher's summaries into
Stage-2 training pairs. It filters out refusals, truncated output, summaries that are just
lists of names, and any pair whose source exceeds the training length budget.

### `baseline_hf/`

Python controls: public Hugging Face models run on the same data and scored with the same
tools, to test the from-scratch results against an outside reference. It has its own
README.

### Other documents

- `ATTENTION_BACKWARD.md` — the derivation of the multi-head attention backward pass, keyed to the lines of `ML/MHA.pm` and `ML/Util.pm`
- `TRAINING_STAGES.md` — the training stages of the d=768 summariser, with the commands
- `TRIM_PIPELINE.md` — the specification for trimming raw email before it is summarised
- `recommended_parameters.txt` — recommended training hyperparameters for the Perl model

## Key implementation details

**Cross-attention gradient channel** — the encoder receives its gradient via a dedicated path:
`MHA::gradient_kv()` → `ResidualConnection` → `DecoderBlock::gradient_enc()` → `Decoder::gradient_enc()` → `Transformer::update()` → `encoder->backward()`

**Loss function** — cross-entropy with PAD masking. Loss is normalised by non-PAD token count; gradient is normalised by `batch_size × seq_len`. These use different denominators intentionally — normalising the gradient by non-PAD count makes Adam converge ~40× slower.

**Learning rate warmup** — linear ramp over the first N epochs is critical. Without it, decoder layer 0 self-attention Q/K gradients are ~40× smaller than other layers.

**Adam moments** are not serialised in checkpoint files. They zero-initialise on reload, which is standard practice.

## Dependencies

```bash
cpan Modern::Perl Storable Data::Dumper File::Slurp JSON Time::HiRes Math::Random
```

## Tested on

Debian Linux, Perl 5.36, CPU only.

## Acknowledgements

Architecture follows the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. BPE tokeniser follows Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).
