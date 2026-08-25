#!/usr/bin/env python3
"""MiniMax-H3 DiT numeric oracle.

Builds a PyTorch reference of the H3 diffusion transformer from the REAL GGUF
weights (dequantized with the `gguf` package) and dumps fixtures for the C#
parity tests.

The full model is 19.3B parameters, which does not fit in F32 on a workstation,
so this runs a configurable slice: the input projections, the token refiner, N
transformer blocks and the final layer. That is enough to pin every distinct
piece of math — the remaining blocks are structurally identical.

Reference for the math: stable-diffusion.cpp src/model/diffusion/minimax_h3.hpp
and MiniMaxAI/MiniMax-H3 transformer/config.json.

Usage
-----
    minimax_h3_dit_oracle.py --gguf <denoiser.gguf> --out-dir <fixtures> [--blocks 2]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

# ---- geometry constants (mirrored in MiniMaxH3Geometry / MiniMaxH3Layout) ----
PATCH_T, PATCH_H, PATCH_W = 1, 2, 2
MODALITY_VISUAL, MODALITY_TEXT, MODALITY_AUDIO = 0, 1, 2
MODALITIES = 3
VECTORS_PER_ROW = 6
FRAME_RESCALE = 5.0 / 3.0
SPATIAL_EXTENT = 32.0
SPAN_TABLE = (1, 4, 4, 4, 4)
VISUAL_COND_TIMESTEP = 0.999


def write(out_dir, name, t):
    a = t.detach().to(torch.float32).cpu().numpy().astype('<f4', copy=False)
    p = os.path.join(out_dir, name)
    a.tofile(p)
    with open(p + '.shape', 'w') as f:
        f.write(','.join(str(d) for d in a.shape))
    print(f'  wrote {name} {tuple(a.shape)}  '
          f'min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}')


def seeded(shape, seed):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal(shape, dtype=np.float32))


class Weights:
    """Lazily dequantized GGUF tensors, transposed into PyTorch orientation."""

    def __init__(self, path):
        from gguf import GGUFReader
        self.reader = GGUFReader(path)
        self.raw = {t.name: t for t in self.reader.tensors}
        self.cache = {}
        print(f'[dit] {len(self.raw)} tensors in {os.path.basename(path)}')

    def __contains__(self, name):
        return name in self.raw

    def get(self, name):
        """Return the tensor in PyTorch orientation.

        GGUF stores ne-order, so a Linear weight is [in, out] there and the
        dequantizer already hands back [out, in] — i.e. exactly nn.Linear's
        layout. 1-D tensors pass through.
        """
        if name in self.cache:
            return self.cache[name]
        if name not in self.raw:
            raise KeyError(name)
        import gguf.quants as q
        t = self.raw[name]
        a = q.dequantize(t.data, t.tensor_type).astype(np.float32)
        out = torch.from_numpy(np.ascontiguousarray(a))
        self.cache[name] = out
        return out


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def spatial_axis(latent_dim, sqrt_area):
    count = latent_dim // PATCH_H
    ratio = latent_dim / sqrt_area
    return np.array([(i * (ratio / count) + (1.0 - ratio) * 0.5) * SPATIAL_EXTENT
                     for i in range(count)], dtype=np.float32)


def video_span(frame):
    return FRAME_RESCALE * SPAN_TABLE[frame % 5]


def time_snr_shift(shift, t):
    return shift * t / (1.0 + (shift - 1.0) * t)


def map_sigma(sigma, frm=12.0, to=3.0):
    b = sigma / (frm + sigma * (1.0 - frm))
    return time_snr_shift(to, b)


def build_layout(text_len, lat_t, lat_h, lat_w, audio_t, sigma):
    """Mirror of MiniMaxH3Layout.Build: positions, modulation rows, timesteps."""
    video_t = 1.0 - sigma
    audio_ts = 1.0 - map_sigma(sigma)

    timesteps = []

    def add_ts(v):
        for i, e in enumerate(timesteps):
            if e == v:
                return i
        timesteps.append(v)
        return len(timesteps) - 1

    video_row = add_ts(video_t)
    audio_row = add_ts(audio_ts)
    add_ts(max(video_t, VISUAL_COND_TIMESTEP))
    add_ts(max(audio_ts, 1.0))

    sqrt_area = float(np.sqrt(lat_h * lat_w))
    h_axis = spatial_axis(lat_h, sqrt_area)
    w_axis = spatial_axis(lat_w, sqrt_area)

    pos = []
    spans = []   # (start, end, row)
    for i in range(text_len):
        pos.append((float(i), 0.0, 0.0))
    if text_len:
        spans.append((0, text_len, video_row * MODALITIES + MODALITY_TEXT))

    row = text_len
    cursor = float(text_len)

    audio_start = row
    for ch in range(2):
        w = w_axis[0] if ch == 0 else w_axis[-1]
        for t in range(audio_t):
            pos.append((cursor + t, 0.0, float(w)))
    n_audio = audio_t * 2
    spans.append((audio_start, audio_start + n_audio,
                  audio_row * MODALITIES + MODALITY_AUDIO))
    row += n_audio

    video_start = row
    for t in range(lat_t):
        for h in h_axis:
            for w in w_axis:
                pos.append((cursor, float(h), float(w)))
        cursor += video_span(t)
    n_video = lat_t * len(h_axis) * len(w_axis)
    spans.append((video_start, video_start + n_video,
                  video_row * MODALITIES + MODALITY_VISUAL))
    row += n_video

    return (np.array(pos, dtype=np.float32), spans, np.array(timesteps, dtype=np.float32),
            (audio_start, audio_start + n_audio), (video_start, video_start + n_video))


def curve_embed(table, timesteps):
    """Interpolate the learned AdaLN curve table at each timestep.

    table: [grid, dim] after transpose. Returns [len(timesteps), dim].
    """
    grid = table.shape[0]
    out = []
    for t in timesteps:
        p = float(np.clip(t, 0.0, 1.0)) * (grid - 1)
        lo = int(min(max(int(np.floor(p)), 0), grid - 2))
        frac = p - lo
        out.append(table[lo] * (1.0 - frac) + table[lo + 1] * frac)
    return torch.stack(out, 0)


def build_rope(inv_freq, positions, head_dim):
    """3-axis partial RoPE. positions [n,3]; returns cos/sin [n, rot_dim]."""
    n = positions.shape[0]
    nf = inv_freq.shape[0]
    angles = np.zeros((n, 3 * nf), dtype=np.float32)
    for axis in range(3):
        angles[:, axis * nf:(axis + 1) * nf] = \
            positions[:, axis:axis + 1] * inv_freq.numpy()[None, :]
    a = torch.from_numpy(angles)
    return torch.cos(a), torch.sin(a)


def apply_rope_half(x, cos, sin):
    """NeoX rotate-half over the first 2*cos.shape[-1] dims; tail passes through.

    x: [n, heads, head_dim]; cos/sin: [n, half]
    """
    half = cos.shape[-1]
    rot = 2 * half
    xr, xt = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., :half], xr[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    out = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return torch.cat([out, xt], dim=-1)


def attention(x, W, cfg, cos, sin, eps):
    """Full bidirectional attention with QK-RMSNorm and partial RoPE."""
    n, hidden = x.shape
    heads, hd = cfg['heads'], cfg['head_dim']
    qkv = F.linear(x, W['qkv'])
    q, k, v = qkv.split(heads * hd, dim=-1)
    q = q.view(n, heads, hd)
    k = k.view(n, heads, hd)
    v = v.view(n, heads, hd)
    q = rms_norm(q, W['q_norm'], eps)
    k = rms_norm(k, W['k_norm'], eps)
    q = apply_rope_half(q, cos, sin)
    k = apply_rope_half(k, cos, sin)
    # [heads, n, hd]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
    o = o.transpose(0, 1).reshape(n, heads * hd)
    return F.linear(o, W['out'])


def mlp(x, W):
    """SwiGLU with the gate first: fc1 emits [gate | value]."""
    gv = F.linear(x, W['fc1'])
    gate, val = gv.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * val, W['fc2'])


def modulate(x, mods, spans, idx_shift, idx_scale, hidden):
    """Per-segment AdaLN affine: x*(1+scale) + shift."""
    out = x.clone()
    for (s, e, row) in spans:
        m = mods[row]
        shift = m[idx_shift * hidden:(idx_shift + 1) * hidden]
        scale = m[idx_scale * hidden:(idx_scale + 1) * hidden]
        out[s:e] = x[s:e] + x[s:e] * scale + shift
    return out


def gated_residual(x, upd, mods, spans, idx_gate, hidden):
    out = x.clone()
    for (s, e, row) in spans:
        gate = mods[row][idx_gate * hidden:(idx_gate + 1) * hidden]
        out[s:e] = x[s:e] + upd[s:e] * gate
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--blocks', type=int, default=2,
                    help='transformer blocks to run (full model does not fit in F32)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    W = Weights(args.gguf)

    hidden = W.get('video_patch_proj.weight').shape[0]
    head_dim = W.get('blocks.0.attn.q_norm.weight').shape[0]
    heads = W.get('blocks.0.attn.qkv_proj.weight').shape[0] // (3 * head_dim)
    ffn = W.get('blocks.0.mlp.fc1.weight').shape[0] // 2
    text_dim = W.get('condition_proj.weight').shape[1]
    inv_freq = W.get('rope.inv_freq')
    cfg = dict(hidden=hidden, heads=heads, head_dim=head_dim, ffn=ffn)
    eps = 1e-5
    print(f'[dit] hidden={hidden} heads={heads}x{head_dim} ffn={ffn} text_dim={text_dim} '
          f'inv_freq={inv_freq.shape[0]}')

    # A deliberately tiny request so the fixtures stay small but every code path runs:
    # 2 latent frames, 4x4 latent -> 2x2 token grid, 3 audio latents, 5 text tokens.
    text_len, lat_t, lat_h, lat_w, audio_t = 5, 2, 4, 4, 3
    sigma = 0.5
    pos, spans, timesteps, aspan, vspan = build_layout(
        text_len, lat_t, lat_h, lat_w, audio_t, sigma)
    n_tok = pos.shape[0]
    print(f'[dit] tokens={n_tok} (text {text_len}, audio {aspan[1]-aspan[0]}, '
          f'video {vspan[1]-vspan[0]}) timesteps={timesteps.tolist()}')

    write(args.out_dir, 'h3_dit_positions.bin', torch.from_numpy(pos))
    write(args.out_dir, 'h3_dit_timesteps.bin', torch.from_numpy(timesteps))

    # ---- inputs ----
    video_lat = seeded((lat_t, lat_h, lat_w, 24), 11) * 0.5
    audio_lat = seeded((2, audio_t, 32), 12) * 0.5
    text_hidden = seeded((text_len, text_dim), 13) * 0.5
    write(args.out_dir, 'h3_dit_video_latent.bin', video_lat)
    write(args.out_dir, 'h3_dit_audio_latent.bin', audio_lat)
    write(args.out_dir, 'h3_dit_text_hidden.bin', text_hidden)

    # ---- patchify + project ----
    # video: (t, h, w, c) -> tokens over (t, h/2, w/2). Within a token the values are
    # CHANNEL-MAJOR, patch-minor (c * pt*ph*pw + ky*pw + kx) -- the reference's
    # "patch_last" layout. The transposed order has identical statistics and
    # silently scrambles every token.
    vp = video_lat.reshape(lat_t, lat_h // PATCH_H, PATCH_H,
                           lat_w // PATCH_W, PATCH_W, 24)
    vp = vp.permute(0, 1, 3, 5, 2, 4).reshape(-1, 24 * PATCH_H * PATCH_W)
    video_tok = F.linear(vp, W.get('video_patch_proj.weight'),
                         W.get('video_patch_proj.bias'))
    audio_tok = F.linear(audio_lat.reshape(-1, 32), W.get('audio_patch_proj.weight'),
                         W.get('audio_patch_proj.bias'))
    write(args.out_dir, 'h3_dit_video_tokens.bin', video_tok)
    write(args.out_dir, 'h3_dit_audio_tokens.bin', audio_tok)

    # ---- text condition: project then refine ----
    ctx = F.linear(text_hidden, W.get('condition_proj.weight'), W.get('condition_proj.bias'))
    write(args.out_dir, 'h3_dit_condition_proj.bin', ctx)

    n_ref = 0
    while f'token_refiner.blocks.{n_ref}.norm1.weight' in W:
        n_ref += 1
    text_pos = torch.from_numpy(pos[:text_len])
    tcos, tsin = build_rope(inv_freq, text_pos.numpy(), head_dim)
    for i in range(n_ref):
        p = f'token_refiner.blocks.{i}.'
        aw = dict(qkv=W.get(p + 'attn.qkv_proj.weight'),
                  out=W.get(p + 'attn.out_proj.weight'),
                  q_norm=W.get(p + 'attn.q_norm.weight'),
                  k_norm=W.get(p + 'attn.k_norm.weight'))
        h = rms_norm(ctx, W.get(p + 'norm1.weight'), eps)
        ctx = ctx + attention(h, aw, cfg, tcos, tsin, eps)
        h = rms_norm(ctx, W.get(p + 'norm2.weight'), eps)
        ctx = ctx + mlp(h, dict(fc1=W.get(p + 'mlp.fc1.weight'),
                                fc2=W.get(p + 'mlp.fc2.weight')))
    ctx = rms_norm(ctx, W.get('token_refiner.final_norm.weight'), eps)
    write(args.out_dir, 'h3_dit_refined_text.bin', ctx)

    # ---- packed sequence ----
    x = torch.cat([ctx, audio_tok, video_tok], dim=0)
    assert x.shape[0] == n_tok, (x.shape, n_tok)
    write(args.out_dir, 'h3_dit_packed_in.bin', x)

    cos, sin = build_rope(inv_freq, pos, head_dim)
    write(args.out_dir, 'h3_dit_rope_cos.bin', cos)
    write(args.out_dir, 'h3_dit_rope_sin.bin', sin)

    # ---- AdaLN time embedding ----
    # adaln_t_table is [dim, grid] in ne order -> [grid, dim] here.
    table = W.get('adaln_t_table')
    if table.shape[0] < table.shape[1]:
        table = table.t().contiguous()
    t_emb = curve_embed(table, timesteps.tolist())
    write(args.out_dir, 'h3_dit_time_embed.bin', t_emb)

    n_blocks = min(args.blocks, 50)
    print(f'[dit] running {n_blocks} block(s)')
    for i in range(n_blocks):
        p = f'blocks.{i}.'
        # AdaLN: [n_timesteps, 6*3*hidden] -> per (timestep, modality) row.
        proj = F.linear(t_emb, W.get(p + 'adaln_proj.linear.weight'),
                        W.get(p + 'adaln_proj.linear.bias'))
        mods = proj.reshape(-1, VECTORS_PER_ROW * hidden)
        if i == 0:
            write(args.out_dir, 'h3_dit_block0_mods.bin', mods)

        h = rms_norm(x, W.get(p + 'norm1.weight'), eps)
        h = modulate(h, mods, spans, 0, 1, hidden)
        aw = dict(qkv=W.get(p + 'attn.qkv_proj.weight'),
                  out=W.get(p + 'attn.out_proj.weight'),
                  q_norm=W.get(p + 'attn.q_norm.weight'),
                  k_norm=W.get(p + 'attn.k_norm.weight'))
        a = attention(h, aw, cfg, cos, sin, eps)
        if i == 0:
            write(args.out_dir, 'h3_dit_block0_attn.bin', a)
        x = gated_residual(x, a, mods, spans, 2, hidden)

        h = rms_norm(x, W.get(p + 'norm2.weight'), eps)
        h = modulate(h, mods, spans, 3, 4, hidden)
        m = mlp(h, dict(fc1=W.get(p + 'mlp.fc1.weight'), fc2=W.get(p + 'mlp.fc2.weight')))
        x = gated_residual(x, m, mods, spans, 5, hidden)
        write(args.out_dir, f'h3_dit_block{i}_out.bin', x)

    # ---- final layer ----
    fproj = F.linear(t_emb, W.get('final_layer.adaln_proj.linear.weight'),
                     W.get('final_layer.adaln_proj.linear.bias'))
    fmods = fproj.reshape(-1, 2 * hidden)
    write(args.out_dir, 'h3_dit_final_mods.bin', fmods)

    h = rms_norm(x, W.get('final_layer.norm.weight'), eps)
    # The final layer modulates with a single shift/scale pair per timestep row; the
    # spans still select which row each token uses.
    fspans = [(s, e, row // MODALITIES) for (s, e, row) in spans]
    h = modulate(h, fmods, fspans, 0, 1, hidden)
    write(args.out_dir, 'h3_dit_final_normed.bin', h)

    audio_out = F.linear(h[aspan[0]:aspan[1]], W.get('final_layer.audio_out.weight'),
                         W.get('final_layer.audio_out.bias'))
    video_out = F.linear(h[vspan[0]:vspan[1]], W.get('final_layer.video_out.weight'),
                         W.get('final_layer.video_out.bias'))

    # The model emits velocities: video negated, audio scaled by
    # d(sigma_audio)/d(sigma_video), so one Euler step on the video schedule
    # advances both streams. Bake that in so the fixture is what production emits.
    def shift_slope(sigma, frm=12.0, to=3.0):
        b = sigma / (frm + sigma * (1.0 - frm))
        a = 1.0 + (frm - 1.0) * b
        c = 1.0 + (to - 1.0) * b
        return to * a * a / (frm * c * c)

    write(args.out_dir, 'h3_dit_video_out.bin', video_out * -1.0)
    write(args.out_dir, 'h3_dit_audio_out.bin', audio_out * -shift_slope(sigma))
    print('done')


if __name__ == '__main__':
    main()
