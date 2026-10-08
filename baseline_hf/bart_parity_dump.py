#!/usr/bin/env python
"""Dump ONE BART encoder layer -- weights, a fixed input, and the reference
output -- in our tensor layout, for test_bart_parity.cpp to check against.

WHY ONLY ONE LAYER, FED AN ACTIVATION. The question this answers is "is the
weight mapping right", and nothing else. Feeding a fixed [B,S,768] activation
straight into one encoder block bypasses the vocabulary (50,265 vs our 16,005),
the positional embeddings (BART learned, ours sinusoidal), layernorm_embedding
(we have none) and the whole decoder. Those are all real problems; none of them
need solving to find out whether the projections are transposed.

WHY FLOAT64. `load_blob_into_tensor` accepts a matching dtype, or downcasts
F64 -> F32 -- and it THROWS on F32 -> F64. BART ships F32, so a naive dump
cannot be loaded by the FP64 CPU build at all. Casting to float64 here means
ONE file serves both builds: exact memcpy on build-fp64-bias, the existing
downcast on build-bias. It also gives a reference good to ~1e-16, so a real
mapping error cannot hide behind "that's just precision".

THE TRANSPOSES ARE DERIVED, NOT GUESSED. Our projection runs
`Q = input x W_q` (launch_matmul_strided_batched with both transpose flags 0),
so Q[s,n] = sum_k input[s,k] * W_q[k,n]. torch.nn.Linear stores weight as
[out, in] and computes x @ W.T, so y[n] = sum_k x[k] * W[n,k]. Therefore
W_q[k,n] = weight[n,k] -- transpose all four attention projections, exactly as
fc1/fc2 visibly need. The attention matrices are square so shape cannot show
it, which is why the alt.* copies below exist: the C++ side tries both and the
numbers settle it.

  python bart_parity_dump.py [--layer 0] [--out bart_layer0.safetensors]
"""
import argparse
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/bart-base")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "bart_layer0.safetensors"))
    ap.add_argument("--relu", action="store_true",
                    help="run the reference with ReLU instead of BART's gelu. "
                         "Use this to prove the weight mapping BEFORE the GELU "
                         "kernels exist -- then both sides compute the same "
                         "function and any error is the mapping alone.")
    args = ap.parse_args()

    kw = {"activation_function": "relu"} if args.relu else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **kw)
    # eval() is LOAD-BEARING: BART has dropout 0.1 on three paths inside this
    # layer, and in training mode the reference would differ every run.
    model = model.eval().double()
    layer = model.model.encoder.layers[args.layer]
    cfg = model.config
    E, H, FF = cfg.d_model, cfg.encoder_attention_heads, cfg.encoder_ffn_dim

    torch.manual_seed(1234)
    x = torch.randn(args.batch, args.seq, E, dtype=torch.float64)
    with torch.no_grad():
        y = layer(x, attention_mask=None)
    # transformers 4.x returned a TUPLE here, 5.x returns a bare Tensor. Writing
    # `[0]` unconditionally silently indexes the BATCH on 5.x and hands you a
    # [S,E] "reference" that the C++ side would then be compared against. Caught
    # by the shape assert below, which is why it is an assert and not a comment.
    if isinstance(y, (tuple, list)):
        y = y[0]
    assert y.shape == x.shape, f"reference shape {tuple(y.shape)} != input {tuple(x.shape)}"

    sd = layer.state_dict()
    f64 = lambda t: t.detach().to(torch.float64).contiguous()
    T   = lambda n: f64(sd[n].T)                 # [out,in] -> [in,out]
    col = lambda n: f64(sd[n].reshape(-1, 1))    # [E] -> [E,1], our bias/LN shape

    P = "enc.block_0"
    out = {
        "input":      f64(x),
        "output_ref": f64(y),

        # self-attention: four projections, transposed, plus their biases
        f"{P}.mha.W_q": T("self_attn.q_proj.weight"),
        f"{P}.mha.W_k": T("self_attn.k_proj.weight"),
        f"{P}.mha.W_v": T("self_attn.v_proj.weight"),
        f"{P}.mha.W_o": T("self_attn.out_proj.weight"),
        f"{P}.mha.b_q": col("self_attn.q_proj.bias"),
        f"{P}.mha.b_k": col("self_attn.k_proj.bias"),
        f"{P}.mha.b_v": col("self_attn.v_proj.bias"),
        f"{P}.mha.b_o": col("self_attn.out_proj.bias"),

        # the SAME four, NOT transposed, so the C++ side can try both and report
        f"alt.{P}.mha.W_q": f64(sd["self_attn.q_proj.weight"]),
        f"alt.{P}.mha.W_k": f64(sd["self_attn.k_proj.weight"]),
        f"alt.{P}.mha.W_v": f64(sd["self_attn.v_proj.weight"]),
        f"alt.{P}.mha.W_o": f64(sd["self_attn.out_proj.weight"]),

        # FFN -- visibly transposed, since these are not square
        f"{P}.ff.l1.W":      T("fc1.weight"),       # [3072,768] -> [768,3072]
        f"{P}.ff.l2.W":      T("fc2.weight"),       # [768,3072] -> [3072,768]
        f"{P}.ff.l1.biases": f64(sd["fc1.bias"]),   # stays 1-D, as MLLinear has it
        f"{P}.ff.l2.biases": f64(sd["fc2.bias"]),

        # LayerNorms. rc1 wraps self-attention, rc2 wraps the FFN -- the same
        # order BART applies them, since both compute LN(sublayer(x) + x).
        f"{P}.rc1.ln.gamma": col("self_attn_layer_norm.weight"),
        f"{P}.rc1.ln.beta":  col("self_attn_layer_norm.bias"),
        f"{P}.rc2.ln.gamma": col("final_layer_norm.weight"),
        f"{P}.rc2.ln.beta":  col("final_layer_norm.bias"),

        # config the C++ side needs, as 1-element tensors
        "cfg.embeddings": torch.tensor([float(E)],  dtype=torch.float64),
        "cfg.num_heads":  torch.tensor([float(H)],  dtype=torch.float64),
        "cfg.d_ff":       torch.tensor([float(FF)], dtype=torch.float64),
        "cfg.batch":      torch.tensor([float(args.batch)], dtype=torch.float64),
        "cfg.seq":        torch.tensor([float(args.seq)],   dtype=torch.float64),
    }
    save_file(out, args.out)

    print(f"{args.model} layer {args.layer}: d={E} heads={H} d_ff={FF} "
          f"activation={cfg.activation_function}")
    print(f"input {tuple(x.shape)}  output {tuple(y.shape)}  "
          f"|input| {x.norm():.4f}  |output| {y.norm():.4f}")
    print(f"wrote {args.out}  ({len(out)} tensors, all F64)")
    print("\nnext: ./build-fp64-bias/test_bart_parity   (expect < 1e-12)")
    print("      ./build-bias/test_bart_parity         (expect < 1e-5)")


if __name__ == "__main__":
    main()
