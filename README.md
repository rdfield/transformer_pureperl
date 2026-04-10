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
| `ML::InputEmbeddings` | Learnable token embedding table, scaled by √d_model |
| `ML::PositionalEmbeddings` | Sinusoidal positional encodings |
| `ML::Tokeniser` | BPE tokeniser (GPT-4 split pattern), extends ML::BaseTokeniser |
| `ML::BaseTokeniser` | Base class: get_stats, merge, save, load |
| `ML::DataLoader` | Batching, padding, tokenisation for training |
| `ML::Util` | matmul, softmax, adam_optimiser, masking utilities |

## Running

All scripts should be run from the repo root:

```bash
# Quick convergence check (d=12, 150 epochs, ~22 min on CPU)
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
perl test_tokeniser.pl    # ML::Tokeniser       (41/41)
```

Each `train_model.pl` epoch saves a checkpoint: `d256_l2_h4_{epoch}.json`.

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
