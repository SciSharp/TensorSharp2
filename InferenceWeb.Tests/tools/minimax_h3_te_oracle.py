#!/usr/bin/env python3
"""MiniMax-H3 text-encoder numeric oracle.

Builds a PyTorch reference of the Qwen3-VL trunk from the REAL GGUF weights
(dequantized with the `gguf` package) and dumps fixtures for the C# parity test.

The full encoder is 32B parameters, which cannot be held in F32 on a workstation,
so this runs the first N layers. Every distinct piece of math (QK-RMSNorm, GQA,
rotate-half RoPE at theta 5e6, SwiGLU, causal masking) appears in layer 0, so a
2-layer comparison pins the layer implementation; the remaining layers are
structurally identical.

Usage
-----
    minimax_h3_te_oracle.py --gguf <qwen3vl_32b_minimax_h3.gguf> \
                            --out-dir <fixtures> [--layers 2] [--tokens 6]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6
ROPE_THETA = 5_000_000.0


def write(out_dir, name, t):
    a = t.detach().to(torch.float32).cpu().numpy().astype('<f4', copy=False)
    p = os.path.join(out_dir, name)
    a.tofile(p)
    with open(p + '.shape', 'w') as f:
        f.write(','.join(str(d) for d in a.shape))
    print(f'  wrote {name} {tuple(a.shape)}  '
          f'min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}')


class Weights:
    def __init__(self, path):
        from gguf import GGUFReader
        self.raw = {t.name: t for t in GGUFReader(path).tensors}
        print(f'[te] {len(self.raw)} tensors in {os.path.basename(path)}')

    def __contains__(self, n):
        return n in self.raw

    def get(self, name):
        """Dequantized tensor in PyTorch orientation ([out, in] for a Linear)."""
        import gguf.quants as q
        t = self.raw[name]
        a = q.dequantize(t.data, t.tensor_type).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(a))

    def embed_rows(self, name, ids):
        """Gather only the rows we need, so the 1.5 GB table is never fully
        materialized in F32."""
        import gguf.quants as q
        t = self.raw[name]
        hidden = int(t.shape[0])
        out = np.empty((len(ids), hidden), dtype=np.float32)
        flat = q.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1, hidden)
        for i, tid in enumerate(ids):
            out[i] = flat[tid]
        return torch.from_numpy(out)


def rms_norm(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def build_rope(seq, head_dim, theta=ROPE_THETA):
    """Rotate-half tables. Dim j and j+half share an angle."""
    half = head_dim // 2
    inv = theta ** (-(2.0 * np.arange(half)) / head_dim)
    pos = np.arange(seq)[:, None] * inv[None, :]
    ang = np.concatenate([pos, pos], axis=-1).astype(np.float32)
    return torch.from_numpy(np.cos(ang)), torch.from_numpy(np.sin(ang))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    # x: [seq, heads, head_dim]; cos/sin: [seq, head_dim]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return x * c + rotate_half(x) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--tokens', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    W = Weights(args.gguf)

    hidden = int(W.raw['model.embed_tokens.weight'].shape[0])
    head_dim = int(W.raw['model.layers.0.self_attn.q_norm.weight'].shape[0])
    heads = int(W.raw['model.layers.0.self_attn.q_proj.weight'].shape[1]) // head_dim
    kvh = int(W.raw['model.layers.0.self_attn.k_proj.weight'].shape[1]) // head_dim
    print(f'[te] hidden={hidden} heads={heads}/{kvh}x{head_dim} '
          f'final_norm={"model.norm.weight" in W}')

    # Deterministic, low token ids so the fixture is reproducible without a tokenizer.
    ids = [(i * 977 + 13) % 100000 for i in range(args.tokens)]
    print(f'[te] token ids {ids}')
    with open(os.path.join(args.out_dir, 'h3_te_token_ids.txt'), 'w') as f:
        f.write(','.join(str(i) for i in ids))

    x = W.embed_rows('model.embed_tokens.weight', ids)
    write(args.out_dir, 'h3_te_embeddings.bin', x)

    seq = len(ids)
    cos, sin = build_rope(seq, head_dim)
    mask = torch.triu(torch.full((seq, seq), float('-inf')), diagonal=1)
    scale = 1.0 / np.sqrt(head_dim)

    for l in range(args.layers):
        p = f'model.layers.{l}.'
        n1 = rms_norm(x, W.get(p + 'input_layernorm.weight'))
        q = F.linear(n1, W.get(p + 'self_attn.q_proj.weight')).view(seq, heads, head_dim)
        k = F.linear(n1, W.get(p + 'self_attn.k_proj.weight')).view(seq, kvh, head_dim)
        v = F.linear(n1, W.get(p + 'self_attn.v_proj.weight')).view(seq, kvh, head_dim)

        # Qwen3 normalizes Q and K per head, before RoPE.
        q = rms_norm(q, W.get(p + 'self_attn.q_norm.weight'))
        k = rms_norm(k, W.get(p + 'self_attn.k_norm.weight'))
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: each kv head serves heads/kvh query heads.
        rep = heads // kvh
        kx = k.repeat_interleave(rep, dim=1)
        vx = v.repeat_interleave(rep, dim=1)
        att = torch.einsum('qhd,khd->hqk', q, kx) * scale + mask[None]
        att = att.softmax(dim=-1)
        o = torch.einsum('hqk,khd->qhd', att, vx).reshape(seq, heads * head_dim)
        x = x + F.linear(o, W.get(p + 'self_attn.o_proj.weight'))

        n2 = rms_norm(x, W.get(p + 'post_attention_layernorm.weight'))
        g = F.linear(n2, W.get(p + 'mlp.gate_proj.weight'))
        u = F.linear(n2, W.get(p + 'mlp.up_proj.weight'))
        x = x + F.linear(F.silu(g) * u, W.get(p + 'mlp.down_proj.weight'))
        write(args.out_dir, f'h3_te_layer{l}_out.bin', x)

    # H3's checkpoint has no final norm — the DiT consumes this raw state.
    write(args.out_dir, 'h3_te_hidden_out.bin', x)
    print('done')


if __name__ == '__main__':
    main()
