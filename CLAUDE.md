# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A from-scratch Transformer (encoder-decoder) implementation in Perl, trained on a sequence copy task. No ML frameworks — all tensor operations are pure Perl nested array refs.

## Running

Run from the repo root (the scripts add `.` to `@INC` via `use lib '.'`):

```bash
perl test_convergence.pl    # gradient flow / convergence check (d=12, 150 epochs, ~22 min)
perl train_model.pl          # full training (d=256, 15 epochs, ~6 days on CPU)
perl grad_norm_probe.pl      # one-batch gradient norm report — diagnose vanishing/exploding gradients
perl test_linear.pl          # gradient check for ML::Linear (numerical differentiation)
perl test_mha.pl             # gradient check for ML::MHA (numerical differentiation)
```

Each `train_model.pl` epoch saves a checkpoint JSON named `d{d_model}_l{num_layers}_h{num_heads}_{epoch}.json`.

## Architecture

All modules live under `ML/` as `ML::*` packages.

**Forward pass** (top-down):
- `ML::Transformer` — orchestrates encode → decode → project. `save_model`/`load_model` serialize weights to JSON.
- `ML::Encoder` — N × `EncoderBlock` + final `LayerNormalisation`
- `ML::Decoder` — N × `DecoderBlock` + final `LayerNormalisation`
- `ML::EncoderBlock` — self-`MHA` → `ResidualConnection`/`LayerNormalisation` → `PositionwiseFeedForward` → `ResidualConnection`/`LayerNormalisation`
- `ML::DecoderBlock` — masked self-`MHA` + cross-`MHA` (attends to encoder output) + `PositionwiseFeedForward`, each wrapped in residual+norm
- `ML::MHA` — Multi-Head Attention; stores W_q, W_k, W_v, W_o as plain 2-D arrays (not `Linear` objects)
- `ML::InputEmbeddings` — learnable token embedding table
- `ML::PositionalEmbeddings` — sinusoidal positional encodings
- `ML::Linear` — weight matrix + bias vector; used for the final projection layer
- `ML::PositionwiseFeedForward` — two-layer FFN (Linear → ReLU → Linear)
- `ML::LayerNormalisation` — per-embedding γ/β parameters
- `ML::LossGradient` — thin wrapper that holds the loss gradient array so it satisfies the `->gradient()` interface expected by `->backward(next => ...)` calls

**Backprop / optimisation** — mixed Adam. Each module implements:
- `forward(batch => ...)` — stores intermediate activations needed for backprop
- `backward(next => $layer)` — pulls `$layer->gradient()`, computes local gradients
- `optimise(learning_rate => ...)` — applies Adam update in-place
- `gradient()` — returns the gradient to pass upstream

`ML::Transformer::update()` drives the full backward pass: projection → decoder → encoder (in that order), then calls `optimise` on each component.

**Data pipeline** (`ML::DataLoader`):
- `init_iter(batch_size => N)` shuffles data and resets the cursor
- `next_batch()` tokenizes, pads, and returns `{src_ids, decoder_input_ids, label_ids}` for one batch

**Utilities** (`ML::Util`):
- Core math: `matmul`, `transpose`, `add_2_arrays`, `mult_2_arrays`, `randn`
- Attention: `scaled_dot_product_attention` — supports 2-D (src) and 3-D (causal) masks
- Masking: `create_src_mask` (padding), `create_tgt_mask` (causal + padding)
- `softmax` — by-column default; pass `1` as second arg for by-row
- `adam_optimiser` — in-place Adam update for 2-D weight matrices

## Tensor conventions

All tensors are nested Perl array refs. Typical shapes:
- Batch of sequences: `[batch][seq_len][embeddings]`
- MHA intermediate: `[batch][num_heads][seq_len][d_k]`

`d_k = embeddings / num_heads` — `embeddings` must be divisible by `num_heads`.

## Optimisers

All weight-bearing modules now use **Adam** (β1=0.9, β2=0.999):
- `ML::MHA` — moment state in `m_W_q/v_W_q` etc.; `adam_epoch` incremented each `optimise()` call
- `ML::Linear` — Adam via `adam_optimiser` from `ML::Util`
- `ML::LayerNormalisation` — Adam for γ/β; moment state in `m_gamma/v_gamma/m_beta/v_beta`; `adam_epoch` incremented each call

Adam moments are **not serialised** in `get_weights()`/`save_model()` — they zero-init on reload, which is standard.

Recommended learning rate: **0.001** (0.01 overshoots MHA weights with Adam).

## Learning rate warmup

Both training scripts use a linear warmup to prevent Q/K weights being pushed into a bad region before they have any structure:
- `train_model.pl`: 3-epoch warmup (lr ramps from 0.000333 → 0.001)
- `test_convergence.pl`: 5-epoch warmup (lr ramps from 0.000200 → 0.001)

Warmup is critical — without it, decoder layer 0 self-attention Q/K gradients are ~40× smaller than all other layers, bottlenecking convergence.

## Loss function

Cross-entropy loss with **PAD token masking** (matches PyTorch `CrossEntropyLoss(ignore_index=PAD_ID)`):
- Loss is normalised by the number of non-PAD tokens in the batch
- Gradient is zeroed for PAD positions; normalised by `batch_size × seq_len` (keeps Adam moment scale stable)
- Separating loss scale (non-PAD) from gradient scale (total positions) is intentional — normalising gradient by non-PAD count makes Adam converge ~40× slower

## Cross-attention gradient channel

The encoder receives its gradient via a dedicated `gradient_kv` path (not the decoder embedding gradient):
`MHA::gradient_kv()` → `ResidualConnection::gradient_kv()` → `DecoderBlock::gradient_enc()` → `Decoder::gradient_enc()` → `Transformer::update()` → `encoder->backward()`

Any future changes to the backward pass must preserve this channel.

## Gradient norm diagnostic

`grad_norm_probe.pl` runs one forward+backward pass (no optimise) and prints Frobenius/L2 norms at every layer. Use it to diagnose:
- **Vanishing gradient**: encoder layer 0 norms << decoder layer N-1
- **LayerNorm bottleneck**: LN dgamma/dbeta << MHA dW norms
- **Cross-attn silent**: cross-attn dWk/dWv ≈ 0 (gradient_kv channel broken)

Confirmed healthy norms on current codebase — all layers within ~10× of each other.

## Gradient correctness tests

Both verified against numerical differentiation (central finite differences, ε=1e-4):
- `perl test_linear.pl` — 56/56 pass; covers 2D and 3D input paths, weights, biases, input gradient
- `perl test_mha.pl` — 656/656 pass; covers self-attention and cross-attention, all four weight matrices, input gradient, and `gradient_kv`

## Test scripts

- `perl test_convergence.pl` — convergence check: 6-token vocab (PAD/SOS/EOS/A/B/C), d=12, 150 epochs, lr warmup 5 epochs. Pass criteria: loss ratio <0.85, accuracy ≥0%. Last known result: ratio ~0.484 at 50 epochs — **improving**
- `perl train_model.pl` — full test: 13-token vocab (digits 0-9), d=256, d_ff=1024, 15 epochs, lr=0.001 with 3-epoch warmup. Saves checkpoint JSON each epoch. Reports greedy-decode accuracy at end (pass ≥50%)
- `perl probe_lr.pl` — learning rate probe with inline accuracy every 10 epochs

## Debugging

Each module accepts a `debug => 1` constructor argument which enables verbose `print_2d_array` logging throughout forward and backward passes. At d=256 this produces enormous output — use only at d=12 or smaller.
