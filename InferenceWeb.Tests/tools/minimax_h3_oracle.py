#!/usr/bin/env python3
"""MiniMax-H3 numeric oracle: dumps reference tensors for the C# parity tests.

Runs the UPSTREAM reference PyTorch (MiniMaxAI/MiniMax-H3's own bundled VAE
sources) on deterministic inputs and writes raw little-endian F32 blobs that
InferenceWeb.Tests reads back. Nothing here is TensorSharp code, deliberately:
the point is an independent reference.

Setup
-----
    python3 -m venv venv && ./venv/bin/pip install torch safetensors numpy pyyaml
    # reference sources (small .py files, no weights):
    #   https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA/audio_vae
    #   https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA/video_vae
    # weights:
    #   https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae

Usage
-----
    minimax_h3_oracle.py --ref-dir <dir with audio_vae/ and video_vae/ packages> \
                         --weights-dir <dir with the two safetensors> \
                         --out-dir <fixture dir>
"""
import argparse
import os
import sys

import numpy as np
import torch


def write(out_dir, name, tensor):
    """Write a tensor as a raw little-endian F32 blob plus a .shape sidecar."""
    a = tensor.detach().to(torch.float32).cpu().numpy().astype('<f4', copy=False)
    path = os.path.join(out_dir, name)
    a.tofile(path)
    with open(path + '.shape', 'w') as f:
        f.write(','.join(str(d) for d in a.shape))
    print(f'  wrote {name} {tuple(a.shape)}  '
          f'min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}')


def defuse_weight_norm(model):
    """Strip weight_norm parametrizations so plain `.weight` keys load.

    The reference modules wrap their convolutions in weight_norm, but the shipped
    checkpoints store the ALREADY-FUSED weight. Removing the parametrization makes
    the module's state-dict keys match the file — and confirms a reimplementation
    needs no weight_g/weight_v fusion of its own.
    """
    from torch.nn.utils import parametrize
    n = 0
    for module in model.modules():
        if parametrize.is_parametrized(module, 'weight'):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized=True)
            n += 1
    print(f'  removed weight_norm from {n} modules')
    return model


def seeded(shape, seed):
    """Deterministic normal noise, independent of torch version RNG changes."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal(shape, dtype=np.float32))


def audio_vae(args):
    from audio_vae.dac_audio_vae import DacAudioVAE
    from safetensors.torch import load_file

    print('[audio-vae] building reference model')
    model = DacAudioVAE(
        encoder_dim=64,
        encoder_rates=[2, 4, 4, 5, 5],
        decoder_dim=1024,
        decoder_rates=[5, 5, 2, 2, 2, 2, 2],
        sample_rate=32000,
        vae_latent_channels=32,
        attn_proj=True,
        decoder_type='bigvgan',
        latent_dim=2048,
    ).eval()
    defuse_weight_norm(model)
    sd = load_file(os.path.join(args.weights_dir, 'minimax_h3_audio_vae_fp32.safetensors'),
                   device='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # latents_mean/std are buffers of the wrapper, not the inner model.
    unexpected = [k for k in unexpected if k not in ('latents_mean', 'latents_std')]
    if missing or unexpected:
        raise SystemExit(f'audio vae state_dict mismatch: missing={missing[:5]} '
                         f'unexpected={unexpected[:5]}')

    write(args.out_dir, 'h3_audio_latents_mean.bin', sd['latents_mean'])
    write(args.out_dir, 'h3_audio_latents_std.bin', sd['latents_std'])

    # A short clip: 8 audio latent frames -> 8*800 = 6400 samples per channel.
    # Both stereo planes go through the mono decoder as a batch of 2, which is
    # exactly how the real pipeline decodes stereo.
    frames = 8
    z = seeded((2, 32, frames), seed=1234)
    write(args.out_dir, 'h3_audio_latent_in.bin', z)
    with torch.no_grad():
        out = model.decode(z)
    print(f'[audio-vae] decode {tuple(z.shape)} -> {tuple(out.shape)}')
    write(args.out_dir, 'h3_audio_wave_out.bin', out)

    # Intermediates, so a parity failure localizes instead of just saying "wrong".
    with torch.no_grad():
        h = model.dec_in_proj(z)
        write(args.out_dir, 'h3_audio_dec_in_proj.bin', h)
        h = model.decoder.conv_pre(h)
        write(args.out_dir, 'h3_audio_conv_pre.bin', h)
        for i, up in enumerate(model.decoder.ups):
            h = up[0](h)
            if i == 0:
                write(args.out_dir, 'h3_audio_up0.bin', h)
            xs = None
            for j in range(model.decoder.num_kernels):
                r = model.decoder.resblocks[i * model.decoder.num_kernels + j](h)
                xs = r if xs is None else xs + r
            h = xs / model.decoder.num_kernels
            if i == 0:
                write(args.out_dir, 'h3_audio_mrf0.bin', h)


def video_vae(args):
    from video_vae.klvae import AutoencoderKLLegacy
    from safetensors.torch import load_file
    import json

    print('[video-vae] building reference model')
    with open(os.path.join(args.ref_dir, 'video_vae', 'source_config.json')) as f:
        cfg = json.load(f)
    cfg.pop('_class_name', None)
    cfg.pop('_diffusers_version', None)
    model = AutoencoderKLLegacy(**cfg).eval()
    defuse_weight_norm(model)

    sd = load_file(os.path.join(args.weights_dir, 'minimax_h3_video_vae_fp16.safetensors'),
                   device='cpu')
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if 'encoder' not in k]
    if missing:
        raise SystemExit(f'video vae decoder weights missing: {missing[:8]}')
    print(f'[video-vae] loaded ({len(unexpected)} unexpected, {len(missing)} missing on decode path)')

    # 7 latent frames is exactly one decode chunk. The spatial size is deliberately
    # NON-SQUARE: a square fixture cannot catch a height/width transposition, and
    # that is exactly the class of bug that shows up as sheared output at 16:9.
    z = seeded((1, 24, 7, 6, 10), seed=4321) * 0.5
    write(args.out_dir, 'h3_video_latent_in.bin', z)
    with torch.no_grad():
        h = model.post_quant_conv(z)
        write(args.out_dir, 'h3_video_post_quant.bin', h)
        # The transformer decoder emits ImageNet-normalized pixels; the outer decode
        # applies x*std+mean and clamps. Dump both so a parity failure says which half.
        out = model.decoder(h)
        print(f'[video-vae] decoder {tuple(z.shape)} -> {tuple(out.shape)}')
        write(args.out_dir, 'h3_video_pixels_out.bin', out)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        rgb = (out * std + mean).clamp(0.0, 1.0)
        write(args.out_dir, 'h3_video_rgb_out.bin', rgb)

    write(args.out_dir, 'h3_video_latents_mean.bin', sd['latents_mean'])
    write(args.out_dir, 'h3_video_latents_std.bin', sd['latents_std'])

    # ---- single-frame encode (image conditioning) ----
    # A deterministic 64x96 "image" in [0,1]; the encoder sees it ImageNet-normalized,
    # exactly as the decoder's output is denormalized.
    rng = np.random.default_rng(777)
    img01 = torch.from_numpy(rng.random((1, 3, 1, 64, 96), dtype=np.float32))
    write(args.out_dir, 'h3_video_encode_image.bin', img01)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        xn = (img01 - mean) / std
        eh = model.encoder(xn)
        eh = model.quant_conv(eh)
        emean = eh[:, :24]
    print(f'[video-vae] encode {tuple(img01.shape)} -> {tuple(emean.shape)}')
    write(args.out_dir, 'h3_video_encode_latent.bin', emean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True,
                    help='directory containing audio_vae/ and video_vae/ reference packages')
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--only', choices=['audio', 'video'], default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.ref_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    if args.only in (None, 'audio'):
        audio_vae(args)
    if args.only in (None, 'video'):
        video_vae(args)
    print('done')


if __name__ == '__main__':
    main()
