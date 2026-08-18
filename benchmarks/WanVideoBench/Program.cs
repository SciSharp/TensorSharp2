// Wan video-generation benchmark harness.
//
// A 5-second 720p video is one long DiT loop plus one long VAE decode, and the
// knobs that decide their cost (attention KV precision, the VAE conv im2col
// budget, band tiling, the Metal 4 tensor API) are process-global and read at
// backend init. Sweeping them through a whole generation costs hours per data
// point, so this harness runs each stage in isolation against synthetic inputs
// of the real shapes.
//
// Modes:
//   vae-decode <vae.safetensors> [t] [lh] [lw] [backend]
//       Decode a synthetic latent; reports MP/s and sanity-checks that the
//       frames are a real picture (the Metal tensor-API mul_mm bug decodes to a
//       flat/NaN plane, which this catches).
//   dit <dit.gguf> [t] [hLen] [wLen] [backend] [passes]
//       Time DiT forwards at a given token grid.
using System;
using System.Diagnostics;
using System.IO;
using TensorSharp.GGML;
using TensorSharp.Models.WanVideo;
using TensorSharp.Runtime;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0) { Usage(); return 1; }
        return args[0] switch
        {
            "vae-decode" => VaeDecode(args),
            "dit" => Dit(args),
            "pipeline" => Pipeline(args),
            _ => Usage(),
        };
    }

    private static int Usage()
    {
        Console.WriteLine("""
            WanVideoBench <mode> ...

              vae-decode <vae.safetensors> [t=31] [lh=52] [lw=68] [backend=metal]
                  Wan 2.2 VAE decode of a synthetic [z, t, lh, lw] latent. The
                  defaults are the 1088x832x121-frame image-to-video shape.

              dit <dit.gguf> [t=31] [hLen=26] [wLen=34] [backend=metal] [passes=2]
                  Wan DiT forwards over a t x hLen x wLen token grid (the same
                  defaults = 27404 tokens).

            Knobs (read at backend init, so set them in the environment):
              TS_WAN_DIT_KV_F16=0        F32 attention keys/values
              TS_WAN_VAE_GEMM_MAX_MB=<n> VAE conv im2col budget
              TS_WAN_VAE_TILE=0          disable VAE band tiling
              GGML_METAL_TENSOR_ENABLE / GGML_METAL_TENSOR_DISABLE
            """);
        return 1;
    }

    private static GgmlBackendType Backend(string name) => (name ?? "metal").ToLowerInvariant() switch
    {
        "cuda" => GgmlBackendType.Cuda,
        "vulkan" => GgmlBackendType.Vulkan,
        "cpu" => GgmlBackendType.Cpu,
        _ => GgmlBackendType.Metal,
    };

    private static int Arg(string[] a, int i, int fallback)
        => a.Length > i && int.TryParse(a[i], out int v) && v > 0 ? v : fallback;

    private static string Knobs() =>
        $"kv_f16={Environment.GetEnvironmentVariable("TS_WAN_DIT_KV_F16") ?? "1"} " +
        $"gemm_mb={Environment.GetEnvironmentVariable("TS_WAN_VAE_GEMM_MAX_MB") ?? "auto"} " +
        $"tile={Environment.GetEnvironmentVariable("TS_WAN_VAE_TILE") ?? "1"} " +
        $"tensor={(Environment.GetEnvironmentVariable("GGML_METAL_TENSOR_DISABLE") != null ? "off" : Environment.GetEnvironmentVariable("GGML_METAL_TENSOR_ENABLE") != null ? "on" : "auto")}";

    private static int VaeDecode(string[] args)
    {
        if (args.Length < 2 || !File.Exists(args[1]))
        {
            Console.Error.WriteLine($"vae-decode: VAE safetensors not found: {(args.Length > 1 ? args[1] : "<missing>")}");
            return 1;
        }
        int t = Arg(args, 2, 31), lh = Arg(args, 3, 52), lw = Arg(args, 4, 68);
        GgmlBasicOps.EnsureBackendAvailable(Backend(args.Length > 5 ? args[5] : null));

        using var vae = new WanVae(args[1]);
        var rng = new Random(1234);
        var latent = new float[(long)vae.ZDim * t * lh * lw];
        for (long i = 0; i < latent.LongLength; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);

        var sw = Stopwatch.StartNew();
        var frames = vae.Decode(latent, t, lh, lw);
        sw.Stop();

        double mp = (double)frames.Length * frames[0].Width * frames[0].Height / 1e6;
        // A corrupt decode (the tensor-API mul_mm failure mode) comes back flat or
        // NaN. "Not flat" is necessary but nowhere near sufficient — that bug was
        // layout-dependent and could perturb a subset of columns — so also emit a
        // digest over EVERY pixel, which a caller diffs across configurations.
        float min = 1f, max = 0f;
        double mean = 0, absMean = 0;
        long nan = 0, n = 0;
        ulong hash = 1469598103934665603;   // FNV-1a over pixels quantized to 1/1024
        foreach (var f in frames)
        {
            foreach (float v in f.Pixels)
            {
                if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
                if (v < min) min = v;
                if (v > max) max = v;
                mean += v; absMean += Math.Abs(v); n++;
                hash = (hash ^ (ulong)(uint)(int)MathF.Round(v * 1024f)) * 1099511628211;
            }
        }
        mean /= Math.Max(1, n); absMean /= Math.Max(1, n);
        bool ok = max > min + 0.05f && nan == 0;

        Console.WriteLine(
            $"[vae-decode] {frames[0].Width}x{frames[0].Height}x{frames.Length}f ({mp:F1} MP) " +
            $"{sw.Elapsed.TotalSeconds:F1}s = {mp / sw.Elapsed.TotalSeconds:F2} MP/s | " +
            $"min={min:F4} max={max:F4} mean={mean:F6} absmean={absMean:F6} nan={nan} digest={hash:x16} " +
            $"{(ok ? "OK" : "*** FLAT/NaN ***")} | {Knobs()}");

        // TS_WAN_BENCH_DUMP=<file>: raw F32 pixels, so two configurations can be
        // compared with a real PSNR instead of summary statistics.
        string dump = Environment.GetEnvironmentVariable("TS_WAN_BENCH_DUMP");
        if (!string.IsNullOrEmpty(dump))
        {
            using var fs = new FileStream(dump, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            foreach (var f in frames)
                foreach (float v in f.Pixels) bw.Write(v);
            Console.WriteLine($"[vae-decode] wrote {dump}");
        }
        return ok ? 0 : 2;
    }

    /// <summary>
    /// DiT-then-VAE in one process, the way a real generation runs. The Metal 4
    /// tensor-API VAE corruption follows the process's ALLOCATION HISTORY, so a
    /// fresh-process `vae-decode` cannot see it — it decoded clean at every shape
    /// while full generations came out black. This reproduces the real ordering
    /// (load DiT, forward, release its device residency, then decode) in minutes
    /// instead of the ~17 a 121-frame video takes.
    /// </summary>
    private static int Pipeline(string[] args)
    {
        if (args.Length < 3 || !File.Exists(args[1]) || !File.Exists(args[2]))
        {
            Console.Error.WriteLine("pipeline <dit.gguf> <vae.safetensors> [ditT=31] [vaeT=3] [lh=52] [lw=68]");
            return 1;
        }
        int ditT = Arg(args, 3, 31), vaeT = Arg(args, 4, 3), lh = Arg(args, 5, 52), lw = Arg(args, 6, 68);
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Metal);

        int hLen = lh / 2, wLen = lw / 2;
        var sw = Stopwatch.StartNew();

        // Image-to-video runs the VAE ENCODER before the DiT, so the encoder's
        // buffers are part of the allocation history the decode inherits. Leaving
        // it out is what made the first version of this repro come back clean.
        {
            int H = lh * 16, W = lw * 16;
            using var vaeEnc = new WanVae(args[2]);
            var pixels = new float[3L * 1 * H * W];
            var prng = new Random(11);
            for (long i = 0; i < pixels.LongLength; i++) pixels[i] = (float)(prng.NextDouble() * 2 - 1);
            float[] mu = vaeEnc.Encode(pixels, 1, H, W);
            Console.WriteLine($"[pipeline] VAE encode {W}x{H}x1: {sw.Elapsed.TotalSeconds:F1}s, mu[0]={mu[0]:F4}");
        }
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();
        using (var gguf = new GgufFile(args[1]))
        using (var dit = new WanDiT(gguf))
        {
            int seq = ditT * hLen * wLen;
            var x = new float[(long)dit.InTok * seq];
            var rng = new Random(7);
            for (long i = 0; i < x.LongLength; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var ctx = new float[WanDiT.TextDim * WanTextEncoder.TextLen];
            for (long i = 0; i < ctx.LongLength; i++) ctx[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            float[] v = dit.Predict(x, ctx, 999f, WanRope.Build(ditT, hLen, wLen), seq);
            long nan = 0;
            foreach (float f in v) if (float.IsNaN(f) || float.IsInfinity(f)) nan++;
            Console.WriteLine($"[pipeline] DiT {ditT}x{hLen}x{wLen} = {seq} tokens: {sw.Elapsed.TotalSeconds:F1}s, nan={nan}");
        }
        // Exactly what WanVideoPipeline does between stages.
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        GgmlBasicOps.ClearHostBufferCache();

        var vargs = new[] { "vae-decode", args[2], vaeT.ToString(), lh.ToString(), lw.ToString(), "metal" };
        int rc = VaeDecode(vargs);
        Console.WriteLine($"[pipeline] {(rc == 0 ? "DECODE OK" : "*** DECODE CORRUPT ***")} | total {sw.Elapsed.TotalSeconds:F1}s");
        return rc;
    }

    private static int Dit(string[] args)
    {
        if (args.Length < 2 || !File.Exists(args[1]))
        {
            Console.Error.WriteLine($"dit: GGUF not found: {(args.Length > 1 ? args[1] : "<missing>")}");
            return 1;
        }
        int t = Arg(args, 2, 31), hLen = Arg(args, 3, 26), wLen = Arg(args, 4, 34);
        int passes = Arg(args, 6, 2);
        GgmlBasicOps.EnsureBackendAvailable(Backend(args.Length > 5 ? args[5] : null));

        using var gguf = new GgufFile(args[1]);
        using var dit = new WanDiT(gguf);
        int seq = t * hLen * wLen;
        var x = new float[(long)dit.InTok * seq];
        var rng = new Random(7);
        for (long i = 0; i < x.LongLength; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        var ctx = new float[WanDiT.TextDim * WanTextEncoder.TextLen];
        for (long i = 0; i < ctx.LongLength; i++) ctx[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        var rope = WanRope.Build(t, hLen, wLen);

        Console.WriteLine($"[dit] {t}x{hLen}x{wLen} = {seq} tokens, {dit.Layers} blocks, dim={dit.Dim} | {Knobs()}");
        double best = double.MaxValue, total = 0;
        for (int i = 0; i < passes; i++)
        {
            var sw = Stopwatch.StartNew();
            float[] v = dit.Predict(x, ctx, 999f, rope, seq);
            sw.Stop();
            double s = sw.Elapsed.TotalSeconds;
            best = Math.Min(best, s);
            if (i > 0) total += s;   // pass 0 carries the weight upload + graph build
            double mn = double.MaxValue, mx = double.MinValue;
            long nan = 0;
            foreach (float f in v) { if (float.IsNaN(f) || float.IsInfinity(f)) { nan++; continue; } if (f < mn) mn = f; if (f > mx) mx = f; }
            Console.WriteLine($"  pass {i + 1}/{passes}: {s:F1}s   v[min={mn:E2} max={mx:E2} nan={nan}]");
        }
        Console.WriteLine($"[dit] best {best:F1}s, steady-state mean {(passes > 1 ? total / (passes - 1) : best):F1}s");
        return 0;
    }
}
