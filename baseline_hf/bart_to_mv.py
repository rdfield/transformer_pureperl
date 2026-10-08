#!/usr/bin/env python
"""Convert facebook/bart-base into an MVEncDecTrainer checkpoint.

HOW THIS AVOIDS GUESSWORK. It does not build a checkpoint from scratch -- it
starts from a SKELETON written by embedding_cpp/bart_skeleton, which calls the
very collect_tensors() that load_tensors() will later read. So every tensor
name, every shape and every scalar blob comes from the C++ side itself, and
this script only REPLACES the tensors it understands. Anything it does not
touch keeps a correct initialisation (Adam moments stay zero, which is what a
fresh checkpoint should have). Reverse-engineering that manifest in Python and
getting one scalar wrong would load a differently-shaped model.

The tensor mapping below is the one proven by test_bart_parity to 1.7e-15 on
the encoder and 4.5e-15 on the decoder.

  python bart_to_mv.py --skeleton skel.safetensors --out bart_init.safetensors

Also writes a 6-line minbpe-v1 stub next to the output: MVEncDecTrainer reads
VOCAB_SIZE and the special ids from a --vocab file, and there is no such file
for BART's vocabulary. process_bpe_file only parses special-token names and
ids, and sets token_count = max(id)+1, so [SEP]=50264 yields 50265.
"""
import argparse
import json
import os
import struct

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForSeq2SeqLM

OFFSET = 2        # BartLearnedPositionalEmbedding: position i reads row i+2
LN_EPS = 1e-5     # BART's nn.LayerNorm default; ours defaults to 1e-6


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/bart-base")
    ap.add_argument("--skeleton", required=True,
                    help="written by: ./build-bias/bart_skeleton skel.safetensors --pointer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors.torch import load_file
    skel = load_file(args.skeleton)
    hdr = read_header(args.skeleton)
    print(f"skeleton: {len(skel)} tensors from {args.skeleton}")

    m = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    sd = m.state_dict()
    cfg = m.config
    D = cfg.d_model

    # the skeleton's own geometry decides how much of BART we take
    SE = skel["pos_src.position_embeds"].shape[1]
    ST = skel["pos_tgt.position_embeds"].shape[1]
    n_enc = sum(1 for k in hdr if k.endswith(".mha.W_q") and k.startswith("enc.block_"))
    n_dec = sum(1 for k in hdr if k.endswith(".mha.W_q") and k.startswith("dec.block_"))
    print(f"  geometry: d={D} enc={n_enc} dec={n_dec} seq {SE}/{ST}")
    if n_enc != cfg.encoder_layers or n_dec != cfg.decoder_layers:
        raise SystemExit(f"skeleton has {n_enc}+{n_dec} layers, BART has "
                         f"{cfg.encoder_layers}+{cfg.decoder_layers} -- rebuild it")
    if SE > cfg.max_position_embeddings:
        raise SystemExit(f"skeleton seq_enc {SE} exceeds BART's "
                         f"{cfg.max_position_embeddings} learned positions")

    out = dict(skel)
    took = []

    def put(name, t):
        if name not in out:
            raise SystemExit(f"skeleton has no tensor '{name}'")
        # clone(): BART ties encoder/decoder/head to one `shared` table, so
        # emb_src and emb_tgt would alias the same storage and safetensors
        # refuses to write that. They are separate tables on our side anyway
        # (only --share_emb makes one a view).
        t = t.detach().to(out[name].dtype).contiguous().clone()
        if tuple(t.shape) != tuple(out[name].shape):
            raise SystemExit(f"{name}: got {tuple(t.shape)}, "
                             f"skeleton wants {tuple(out[name].shape)}")
        out[name] = t
        took.append(name)

    T   = lambda n: sd[n].T                        # [out,in] -> [in,out]
    col = lambda n: sd[n].reshape(-1, 1)           # [E] -> [E,1]
    eps = lambda: torch.tensor([LN_EPS])

    # ---- embeddings. BART ties encoder, decoder and the lm head to `shared`,
    # and MLTiedProjection binds the head to emb_tgt, so the same table goes in
    # both places and the head follows automatically.
    put("emb_src.emb.weights", sd["model.shared.weight"])
    put("emb_tgt.emb.weights", sd["model.shared.weight"])

    # ---- positions: drop BART's 2-row offset, then take what fits
    for side, src, n in (("pos_src", "encoder", SE), ("pos_tgt", "decoder", ST)):
        pe = sd[f"model.{src}.embed_positions.weight"][OFFSET:OFFSET + n]
        put(f"{side}.position_embeds", pe.reshape(1, n, D))
        put(f"{side}.ln_emb.gamma", col(f"model.{src}.layernorm_embedding.weight"))
        put(f"{side}.ln_emb.beta",  col(f"model.{src}.layernorm_embedding.bias"))
        put(f"{side}.ln_emb.ln_eps", eps())

    # ---- encoder / decoder layers
    for side, n_layers in (("enc", n_enc), ("dec", n_dec)):
        hf = "encoder" if side == "enc" else "decoder"
        for i in range(n_layers):
            P, L = f"{side}.block_{i}", f"model.{hf}.layers.{i}"
            for ours, theirs in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "out")):
                put(f"{P}.mha.W_{ours}", T(f"{L}.self_attn.{theirs}_proj.weight"))
                put(f"{P}.mha.b_{ours}", col(f"{L}.self_attn.{theirs}_proj.bias"))
            put(f"{P}.rc1.ln.gamma", col(f"{L}.self_attn_layer_norm.weight"))
            put(f"{P}.rc1.ln.beta",  col(f"{L}.self_attn_layer_norm.bias"))
            put(f"{P}.rc1.ln.ln_eps", eps())

            if side == "dec":
                for ours, theirs in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "out")):
                    put(f"{P}.cab.W_{ours}", T(f"{L}.encoder_attn.{theirs}_proj.weight"))
                    put(f"{P}.cab.b_{ours}", col(f"{L}.encoder_attn.{theirs}_proj.bias"))
                put(f"{P}.rc2.ln.gamma", col(f"{L}.encoder_attn_layer_norm.weight"))
                put(f"{P}.rc2.ln.beta",  col(f"{L}.encoder_attn_layer_norm.bias"))
                put(f"{P}.rc2.ln.ln_eps", eps())

            put(f"{P}.ff.l1.W", T(f"{L}.fc1.weight"))
            put(f"{P}.ff.l2.W", T(f"{L}.fc2.weight"))
            put(f"{P}.ff.l1.biases", sd[f"{L}.fc1.bias"])
            put(f"{P}.ff.l2.biases", sd[f"{L}.fc2.bias"])

            # BART's final_layer_norm is the FFN's, so rc3 in a decoder block
            # and rc2 in an encoder block. Loading it over the cross-attention
            # norm would have plausible shapes and the wrong model.
            last = "rc3" if side == "dec" else "rc2"
            put(f"{P}.{last}.ln.gamma", col(f"{L}.final_layer_norm.weight"))
            put(f"{P}.{last}.ln.beta",  col(f"{L}.final_layer_norm.bias"))
            put(f"{P}.{last}.ln.ln_eps", eps())

    # ---- our final norms are SURPLUS and must be DISABLED, not neutralised.
    # BART (post-LN) has none -- encoder.layer_norm is None, a pre-LN/mBART
    # feature. gamma=1/beta=0 is NOT an identity: a LayerNorm still subtracts
    # the mean and divides by the standard deviation. bart_skeleton therefore
    # sets use_final_norm=false, which is serialised; these weights are then
    # never read, and are left as the skeleton wrote them.
    if skel.get("enc.use_final_norm", torch.tensor([1]))[0].item() != 0:
        raise SystemExit("skeleton has enc.use_final_norm=1; rebuild bart_skeleton "
                         "-- BART has no trailing norm and a 'neutral' one still "
                         "standardises the output")

    save_file(out, args.out)

    untouched = [k for k in out if k not in set(took)]
    weighty = [k for k in untouched
               if hdr[k]["dtype"] != "I32" and not k.startswith(("m_", "v_"))
               and ".m_" not in k and ".v_" not in k and not k.endswith(("m_PE", "v_PE"))]
    print(f"\nwrote {args.out}")
    print(f"  replaced {len(took)} tensors, left {len(untouched)} "
          f"(scalars + zeroed Adam moments)")
    if weighty:
        print(f"  NOT from BART ({len(weighty)}): {', '.join(sorted(weighty)[:6])}"
              + (" ..." if len(weighty) > 6 else ""))

    # ---- the vocab stub
    stub = os.path.splitext(args.out)[0] + "_vocab.model"
    with open(stub, "w") as f:
        f.write("minbpe v1\n")
        f.write("bart-base 50265 — stub: process_bpe_file reads only the\n")
        f.write("5\n")
        f.write("[PAD] 1\n")       # <pad>
        f.write("[CLS] 2\n")       # </s> -- BART's decoder_start_token_id
                                   # is 2, NOT <s>. SOS_ID comes from [CLS].
        f.write("[EOS] 2\n")       # </s>
        f.write("[UNK] 3\n")       # <unk>, doubles as MASK_ID
        f.write("[SEP] 50264\n")   # <mask>, our pair delimiter; sets token_count
    print(f"  wrote {stub}  (token_count = 50264+1 = 50265)")


if __name__ == "__main__":
    main()
