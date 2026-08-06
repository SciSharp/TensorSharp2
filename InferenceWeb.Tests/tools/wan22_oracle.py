# Wan 2.2 oracle fixture generator: runs the REAL diffusers reference models
# (AutoencoderKLWan for both VAE generations, WanTransformer3DModel for the
# TI2V-5B DiT) on deterministic inputs and writes raw little-endian F32 blobs
# that the guarded C# tests (WanVideoOracleTests) compare against.
#
# Usage: python wan22_oracle.py [all|vae22|vae21|dit]
# Needs: torch, diffusers >= 0.38, numpy; the Wan model files under
# C:\Works\models\wan (incl. the Wan-AI/Wan2.2-TI2V-5B-Diffusers transformer/vae
# in diffusers-ti2v-5b/). Fixtures land in C:\Works\models\wan\fixtures.
import sys, os, json
import numpy as np
import torch

FIX = r"C:\Works\models\wan\fixtures"
os.makedirs(FIX, exist_ok=True)

def save(name, t):
    a = t.detach().float().cpu().numpy().astype("<f4")
    a.tofile(os.path.join(FIX, name))
    print(f"{name}: shape {tuple(t.shape)} mean {a.mean():.5f} std {a.std():.5f}")

def gen(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32)

which = sys.argv[1] if len(sys.argv) > 1 else "all"

if which in ("all", "vae22"):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        r"C:\Works\models\wan\diffusers-ti2v-5b", subfolder="vae",
        torch_dtype=torch.float32)
    vae.eval()

    # ---- decode: raw (de-normalized) latent [1, 48, 3, 10, 10] -> pixels ----
    z = gen((1, 48, 3, 10, 10), 1234) * 0.6
    save("vae22_dec_z.bin", z)
    with torch.no_grad():
        px = vae.decode(z).sample
    save("vae22_dec_out.bin", px)          # [1, 3, 9, 160, 160]
    print("decode out shape:", tuple(px.shape))

    # ---- encode: deterministic 5-frame clip [1, 3, 5, 160, 192] -> mu -------
    tt, H, W = 5, 160, 192
    yy = torch.linspace(-1, 1, H).view(1, 1, 1, H, 1)
    xx = torch.linspace(-1, 1, W).view(1, 1, 1, 1, W)
    ft = torch.linspace(-1, 1, tt).view(1, 1, tt, 1, 1)
    r = (yy * xx).expand(1, 1, tt, H, W)
    g2 = (0.5 * yy + 0.3 * ft).expand(1, 1, tt, H, W)
    b = torch.sin(3 * xx + 2 * ft).expand(1, 1, tt, H, W)
    clip = torch.cat([r, g2, b], dim=1).contiguous()
    save("vae22_enc_in.bin", clip)
    with torch.no_grad():
        mu = vae.encode(clip).latent_dist.mode()
    save("vae22_enc_mu.bin", mu)           # [1, 48, 2, 10, 12]
    print("encode mu shape:", tuple(mu.shape))
    del vae

if which in ("all", "vae21"):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_single_file(
        r"C:\Works\models\wan\wan_2.1_vae.safetensors", torch_dtype=torch.float32)
    vae.eval()
    tt, H, W = 5, 128, 160
    yy = torch.linspace(-1, 1, H).view(1, 1, 1, H, 1)
    xx = torch.linspace(-1, 1, W).view(1, 1, 1, 1, W)
    ft = torch.linspace(-1, 1, tt).view(1, 1, tt, 1, 1)
    r = (yy * xx).expand(1, 1, tt, H, W)
    g2 = (0.5 * yy + 0.3 * ft).expand(1, 1, tt, H, W)
    b = torch.sin(3 * xx + 2 * ft).expand(1, 1, tt, H, W)
    clip = torch.cat([r, g2, b], dim=1).contiguous()
    save("vae21_enc_in.bin", clip)
    with torch.no_grad():
        mu = vae.encode(clip).latent_dist.mode()
    save("vae21_enc_mu.bin", mu)           # [1, 16, 2, 16, 20]
    print("2.1 encode mu shape:", tuple(mu.shape))
    del vae

if which in ("all", "dit"):
    from diffusers import WanTransformer3DModel
    tr = WanTransformer3DModel.from_pretrained(
        r"C:\Works\models\wan\diffusers-ti2v-5b", subfolder="transformer",
        torch_dtype=torch.float32)
    tr.eval()

    T, H, W = 2, 8, 8                      # latent grid -> seq = 2*4*4 = 32 tokens
    x = gen((1, 48, T, H, W), 42)
    ctx = gen((1, 512, 4096), 43) * 0.5
    save("dit_x.bin", x)
    save("dit_ctx.bin", ctx)

    with torch.no_grad():
        # (a) uniform timestep 500
        t1 = torch.tensor([500.0])
        v1 = tr(hidden_states=x, timestep=t1, encoder_hidden_states=ctx, return_dict=False)[0]
        save("dit_v_uniform.bin", v1)

        # (b) per-token timesteps: frame 0 tokens at 0, the rest at 500 (TI2V i2v)
        seq = T * (H // 2) * (W // 2)
        s0 = (H // 2) * (W // 2)
        ts = torch.full((1, seq), 500.0)
        ts[:, :s0] = 0.0
        v2 = tr(hidden_states=x, timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0]
        save("dit_v_masked.bin", v2)
    meta = dict(T=T, H=H, W=W, seq=seq, s0=s0)
    json.dump(meta, open(os.path.join(FIX, "dit_meta.json"), "w"))
    print("dit fixtures done", meta)

print("done")
