// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Parity and timing of the Qwen-Image-2.1 transformer on the pure-C# `cpu` backend
// (QwenImage21ManagedDiT) against the native ggml_cpu graph, on identical inputs.
//
//   dotnet build -c Release benchmarks/QwenImageDiTBench/QwenImageDiTBench.csproj
//   (copy a GgmlOps.dll of the same revision next to the exe for ggml_cpu)
//   QwenImageDiTBench --dit <qwen_image_2.1 gguf> [--size 256] [--lora <plugin.json>] [--edit]
//                     [--backends cpu,ggml_cpu] [--prompt "..."] [--random-cond] [--quick] [--label name]
//   QwenImageDiTBench --kernels [--size 512]   (attention / LoRA-product throughput per kernel width)
//   QwenImageDiTBench --pipeline [--size 128] [--steps N] [--lora <plugin.json>]
//                     (text encode + denoise + VAE through QwenImageModel on the cpu backend,
//                      then asserts the native GGML library was never loaded)
//
// A trailing '~' on a backend perturbs its inputs to measure that pipeline's own sensitivity:
// --perturb latents (default) scales the latents by 1 + U(-s, s); --perturb sigma scales the
// timestep by 1 + s. s is --perturb-scale (default 1e-6). ggml-cpu rounds img_in's input to
// BF16 (its vec_dot_type), which erases a 1e-6 latent change, so its noise floor needs sigma.
// Runs whose backends are all cpu* fail (exit 3) if GgmlOps was loaded into the process.
//
// Per backend: (1) a full forward at sigma0 without the prefix cache, (2) the extract forward
// (cache on) at sigma0, (3) a cached forward at sigma1 and (4, unless --quick) a full forward at
// sigma1. sigma1's latents are one Euler step from the first backend's velocity, reused by the
// second so every backend sees the same inputs. Reported: velocity cosine / relative L2 /
// max-abs across backends and cached vs uncached within one, and each forward's time.
// Text conditioning comes from the Qwen3-VL encoder on ggml_cpu (cached under the output
// folder), or random rows with --random-cond. --edit inserts a 16x16 reference image
// (64 vision slots after the fourth text token) with noise latents.
using System.Diagnostics;
using System.Globalization;
using System.Text;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

var options = Options.Parse(args);
Directory.CreateDirectory(options.Out);
if (options.Kernels)
{
    KernelBench.Run(options.Size);
    KernelBench.Projections(options.Dit, options.Size);
    return 0;
}
if (options.ProbeMatmul)
{
    MatmulProbe.Run(options.Dit);
    return 0;
}
if (options.Pipeline) return PipelineRun.Run(options);
int latent = options.Size / 16, sequence = latent * latent;

// ---- conditioning -------------------------------------------------------------------------
var (text, textSeq) = options.RandomCond ? RandomConditioning(23) : Conditioning(options);
int[] slots = null;
float[][] references = Array.Empty<float[]>();
int[] refHeights = Array.Empty<int>(), refWidths = Array.Empty<int>();
if (options.Edit)
{
    // 64 vision slots (a 256x256 reference = 16x16 latent tokens) after the fourth token.
    const int insertAt = 4, slotCount = 64;
    var widened = new float[(textSeq + slotCount) * 4096];
    Array.Copy(text, 0, widened, 0, insertAt * 4096);
    Array.Copy(text, insertAt * 4096, widened, (insertAt + slotCount) * 4096, (textSeq - insertAt) * 4096);
    text = widened;
    slots = new int[textSeq + slotCount];
    Array.Fill(slots, 1, insertAt, slotCount);
    textSeq += slotCount;
    references = new[] { QwenImage21Pipeline.ToTokens(QwenImage21Sampling.Noise(16 * 16 * 64, options.Seed + 1), 16, 16) };
    refHeights = new[] { 16 };
    refWidths = new[] { 16 };
}
float[] latents0 = QwenImage21Pipeline.ToTokens(QwenImage21Sampling.Noise(sequence * 64, options.Seed), latent, latent);
float[] latents1 = null;

// ---- runs ---------------------------------------------------------------------------------------
var results = new Dictionary<string, Dictionary<string, (float[] V, double Ms, string Path)>>();
var lines = new List<string>();
void Log(string line) { Console.WriteLine(line); lines.Add(line); }
Log($"Qwen-Image-2.1 DiT bench: {Path.GetFileName(options.Dit)}, {options.Size}x{options.Size} ({sequence} target tokens), " +
    $"text {textSeq} tokens{(options.Edit ? " incl. 64 vision slots + a 16x16 reference" : "")}, " +
    $"LoRA {(options.Lora == null ? "none" : Path.GetFileName(options.Lora))}, kernels {QwenImage21CpuKernels.Width * 32}-bit, " +
    $"pool {QwenImage21CpuKernels.Workers} threads");
foreach (string backendName in options.Backends)
{
    // cpu: the pure-C# forward (F32 activations into dequantized weights); cpu_q8: its
    // Q8_K/Q8_0 integer-dot variant (TS_QWEN21_CPU_MATMUL=q8). A trailing '~' perturbs the
    // latents or the timestep (--perturb) to measure a pipeline's noise floor.
    bool perturb = backendName.EndsWith('~');
    bool perturbSigma = perturb && options.Perturb == "sigma";
    var backend = backendName.TrimEnd('~') switch
    {
        "cpu" or "cpu_q8" => BackendType.Cpu,
        "ggml_cpu" => BackendType.GgmlCpu,
        _ => throw new ArgumentException($"Unsupported backend '{backendName}' (cpu, cpu_q8 or ggml_cpu, optionally with a trailing ~)."),
    };
    QwenImage21ManagedDiT.F32Matmul = !backendName.StartsWith("cpu_q8", StringComparison.Ordinal);
    float[] Perturbed(float[] x)
    {
        if (!perturb || perturbSigma) return x;
        var rng = new Random(77);
        float s = options.PerturbScale;
        return x.Select(v => v * (1f + s * (float)(rng.NextDouble() * 2 - 1))).ToArray();
    }
    // The timestep is clamped to [0, 1] by Predict; 1.0 is perturbed downwards.
    float Sigma(float sigma) => !perturbSigma ? sigma
        : sigma * (1f + options.PerturbScale) <= 1f ? sigma * (1f + options.PerturbScale) : sigma * (1f - options.PerturbScale);
    var runs = results[backendName] = new();
    var load = Stopwatch.StartNew();
    QwenImage21LoraSet lora = null;
    if (options.Lora != null)
    {
        using var gguf = new GgufFile(options.Dit);
        string prefix = gguf.Tensors.ContainsKey("img_in.weight") ? "" : "model.diffusion_model.";
        lora = QwenImage21LoraSet.Load(LoraCliFlags.Resolve(new[] { new LoraSpec(options.Lora) }), gguf, prefix, backend, 1);
    }
    var dit = new QwenImage21DiT(options.Dit, backend, null, lora);
    Log($"[{backendName}] loaded in {load.Elapsed.TotalSeconds:F1}s");
    var recipe = lora?.Recipe;
    float[] sigmas = recipe is { HasSchedule: true } ? recipe.Sigmas(recipe.DefaultSteps, sequence) : QwenImage21Sampling.Sigmas(2, sequence);
    bool bf16 = recipe?.TimestepBf16 ?? false;
    try
    {
        (float[] V, double Ms, string Path) Predict(string name, float[] x, float sigma, QwenImage21DiT.PrefixCache cache, int step)
        {
            sigma = Sigma(sigma);
            var timer = Stopwatch.StartNew();
            var v = dit.Predict(x, latent, latent, text, textSeq, sigma, slots, references, refHeights, refWidths, cache, step, bf16);
            double ms = timer.Elapsed.TotalMilliseconds;
            string path = cache?.LastPath.ToString().ToLowerInvariant() ?? "full";
            Log($"[{backendName}] {name,-12} sigma={sigma:F7} path={path,-8} {ms,10:F0} ms");
            return (v, ms, path);
        }
        runs["full0"] = Predict("full s0", Perturbed(latents0), sigmas[0], null, 0);
        if (latents1 == null)
        {
            latents1 = (float[])latents0.Clone();
            for (int i = 0; i < latents1.Length; i++) latents1[i] += (sigmas[1] - sigmas[0]) * runs["full0"].V[i];
        }
        using (var cache = QwenImage21DiT.CreatePrefixCache(text, slots, references, "1", options.CacheType))
        {
            runs["extract0"] = Predict("extract s0", Perturbed(latents0), sigmas[0], cache, 0);
            for (int rep = 0; rep < options.Reps; rep++)
                runs[rep == 0 ? "cached1" : $"cached1.{rep}"] = Predict("cached s1", Perturbed(latents1), sigmas[1], cache, 1);
            var info = cache.Info;
            Log($"[{backendName}] prefix cache: state {info.State}, {info.Tokens} tokens, {info.Bytes / 1048576.0:F1} MiB (K type {info.KeyType}, V type {info.ValueType})");
        }
        if (!options.Quick) runs["full1"] = Predict("full s1", Perturbed(latents1), sigmas[1], null, 1);
    }
    finally
    {
        dit.Dispose();
        lora?.Dispose();
    }
}

// ---- parity -------------------------------------------------------------------------------------
static (double Cosine, double RelL2, double MaxAbs) Compare(float[] a, float[] b)
{
    double dot = 0, aa = 0, bb = 0, diff = 0, max = 0;
    for (int i = 0; i < a.Length; i++)
    {
        dot += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i];
        double d = a[i] - b[i];
        diff += d * d;
        max = Math.Max(max, Math.Abs(d));
    }
    return (dot / Math.Sqrt(aa * bb), Math.Sqrt(diff / bb), max);
}

string Row(string what, float[] a, float[] b)
{
    var (cos, rel, max) = Compare(a, b);
    return $"  {what,-44} cosine {cos:F7}  relL2 {rel:E3}  maxAbs {max:E3}";
}

foreach (var (backendName, runs) in results)
{
    Log($"within {backendName}:");
    Log(Row("extract s0 vs full s0", runs["extract0"].V, runs["full0"].V));
    if (runs.ContainsKey("full1")) Log(Row("cached s1 vs full s1", runs["cached1"].V, runs["full1"].V));
}
var names = results.Keys.ToList();
for (int a = 0; a < names.Count; a++)
    for (int b = a + 1; b < names.Count; b++)
    {
        var x = results[names[a]];
        var y = results[names[b]];
        Log($"{names[a]} vs {names[b]}:");
        foreach (var key in new[] { "full0", "extract0", "cached1", "full1" })
            if (x.ContainsKey(key) && y.ContainsKey(key))
                Log(Row($"{key} ({x[key].Ms:F0} ms vs {y[key].Ms:F0} ms)", x[key].V, y[key].V));
    }
// The cpu backend must not enter native GGML code. From a repo checkout GgmlOps is always
// loadable (the resolver walks up to the repo root), so a stray call would not fail; a loaded
// module is the evidence.
bool managedOnly = options.Backends.All(b => b.StartsWith("cpu", StringComparison.Ordinal));
string native = NativeProbe.LoadedGgml();
Log($"native GGML library in the process: {native ?? "not loaded"}");
string label = options.Label ?? $"{options.Size}{(options.Lora != null ? "-lora" : "")}{(options.Edit ? "-edit" : "")}";
File.WriteAllLines(Path.Combine(options.Out, $"dit-{label}.txt"), lines);
Console.WriteLine($"Wrote {Path.Combine(options.Out, $"dit-{label}.txt")}");
if (managedOnly && native != null)
{
    if (NativeProbe.BenchUsedGgml)
        Log("  (not evidence: the bench itself encoded the conditioning on ggml_cpu; rerun with the cached conditioning)");
    else
    {
        Console.Error.WriteLine("FAIL: a cpu-only run loaded the native GGML library.");
        return 3;
    }
}
return 0;

// ---- helpers --------------------------------------------------------------------------------------
static (float[] Text, int Length) RandomConditioning(int length)
{
    var rng = new Random(123);
    var text = new float[length * 4096];
    for (int i = 0; i < text.Length; i++) text[i] = (float)(rng.NextDouble() * 2 - 1);
    return (text, length);
}

static (float[] Text, int Length) Conditioning(Options o)
{
    string key = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(o.Prompt)))[..16];
    string cachePath = Path.Combine(o.Out, $"cond-{key}.bin");
    if (File.Exists(cachePath))
    {
        using var reader = new BinaryReader(File.OpenRead(cachePath));
        int length = reader.ReadInt32();
        var data = new float[length * 4096];
        for (int i = 0; i < data.Length; i++) data[i] = reader.ReadSingle();
        Console.WriteLine($"Conditioning: {length} tokens from {cachePath}");
        return (data, length);
    }
    string dir = Path.GetDirectoryName(Path.GetFullPath(o.Dit));
    string te = Directory.EnumerateFiles(dir, "*.gguf").First(f =>
    {
        string n = Path.GetFileName(f).ToLowerInvariant();
        return (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && !n.Contains("mmproj");
    });
    var timer = Stopwatch.StartNew();
    float[] text;
    int seq;
    NativeProbe.BenchUsedGgml = true;
    using (var conditioner = new QwenImage21Conditioner(te, null, BackendType.GgmlCpu))
        (text, seq, _) = conditioner.EncodePrompt(o.Prompt, Array.Empty<RgbImage>());
    Console.WriteLine($"Conditioning: {seq} tokens encoded on ggml_cpu in {timer.Elapsed.TotalSeconds:F1}s");
    using (var writer = new BinaryWriter(File.Create(cachePath)))
    {
        writer.Write(seq);
        foreach (float v in text) writer.Write(v);
    }
    return (text, seq);
}

sealed class Options
{
    public string Dit = "C:/Works/models/qwen-image-2.1/qwen_image_2.1_Q4_K_M.gguf";
    public int Size = 256;
    public string Lora;
    public bool Edit, RandomCond, Quick, Kernels, ProbeMatmul, Pipeline;
    public int Steps;
    public string Perturb = "latents";
    public float PerturbScale = 1e-6f;
    public string[] Backends = { "cpu", "ggml_cpu" };
    public string Prompt = "A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph";
    public string Out = Path.Combine("artifacts", "cpu-perf", "qwen-dit");
    public string Label;
    public string CacheType;
    public long Seed = 42;
    public int Reps = 1;

    public static Options Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            switch (args[i])
            {
                case "--dit": o.Dit = Next(); break;
                case "--size": o.Size = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--lora": o.Lora = Next(); break;
                case "--edit": o.Edit = true; break;
                case "--random-cond": o.RandomCond = true; break;
                case "--quick": o.Quick = true; break;
                case "--kernels": o.Kernels = true; break;
                case "--probe-matmul": o.ProbeMatmul = true; break;
                case "--pipeline": o.Pipeline = true; break;
                case "--steps": o.Steps = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--perturb":
                    o.Perturb = Next();
                    if (o.Perturb is not ("latents" or "sigma")) throw new ArgumentException("--perturb takes latents or sigma");
                    break;
                case "--perturb-scale": o.PerturbScale = float.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--backends": o.Backends = Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries); break;
                case "--prompt": o.Prompt = Next(); break;
                case "--out": o.Out = Next(); break;
                case "--label": o.Label = Next(); break;
                case "--cache-type": o.CacheType = Next(); break;
                case "--seed": o.Seed = long.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--reps": o.Reps = Math.Max(1, int.Parse(Next(), CultureInfo.InvariantCulture)); break;
                default: throw new ArgumentException($"Unknown argument {args[i]}");
            }
        }
        if (o.Size <= 0 || o.Size % 32 != 0) throw new ArgumentException("--size must be a positive multiple of 32.");
        return o;
    }
}

/// <summary>Throughput of the transformer's float kernels at a resolution's sequence length:
/// the segmented attention (32 heads x 128, image-only) and the LoRA shrink/expand products
/// (rank 64 over 4096), for every kernel width the host supports.</summary>
static unsafe class KernelBench
{
    public static void Run(int size)
    {
        int tokens = (size / 16) * (size / 16) + 23, heads = 32, hd = 128, dim = heads * hd;
        var rng = new Random(1);
        float[] Rand(long n) { var v = new float[n]; for (long i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1); return v; }
        float[] q = Rand((long)tokens * dim), k = Rand((long)tokens * dim), v = Rand((long)tokens * dim), o = new float[(long)tokens * dim];
        float[] kt = new float[(long)heads * hd * QwenImage21CpuKernels.KeyStride(tokens)], vh = new float[(long)tokens * dim];
        var segments = new[]
        {
            new QwenImage21Segment { Start = 0, End = 23, IsImage = 0 },
            new QwenImage21Segment { Start = 23, End = tokens, IsImage = 1, SourceStart = 0 },
        };
        const int rank = 64;
        float[] x = Rand((long)tokens * dim), down = Rand((long)rank * dim), shrunk = new float[(long)tokens * rank];
        float[] upT = new float[QwenImage21CpuKernels.PanelFloats(rank, dim)], y = new float[(long)tokens * dim];
        fixed (float* raw = Rand((long)rank * dim), packed = upT) QwenImage21CpuKernels.PackPanels(raw, dim, rank, dim, packed);
        var widths = new List<int>();
        if (System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated && System.Runtime.Intrinsics.X86.Avx512F.IsSupported) widths.Add(16);
        if (System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated && System.Runtime.Intrinsics.X86.Avx2.IsSupported &&
            System.Runtime.Intrinsics.X86.Fma.IsSupported) widths.Add(8);
        widths.Add(1);
        double attnFlops = 0;
        foreach (var s in segments)
            for (int r = s.Start; r < s.End; r++) attnFlops += 4.0 * hd * heads * (s.IsImage != 0 ? s.End : r + 1);
        double loraFlops = 2.0 * tokens * rank * dim;
        Console.WriteLine($"Kernels at {size}x{size}: {tokens} tokens, attention {attnFlops / 1e9:F1} GFLOP, LoRA shrink/expand {loraFlops / 1e9:F2} GFLOP each");
        int saved = QwenImage21CpuKernels.Width;
        fixed (float* pq = q, pk = k, pv = v, po = o, pt = kt, ph = vh, px = x, pd = down, ps = shrunk, pu = upT, py = y)
        {
            foreach (int width in widths)
            {
                QwenImage21CpuKernels.Width = width;
                int reps = width == 1 ? 1 : 5;
                double Time(Action body)
                {
                    for (int i = 0; i < (width == 1 ? 1 : 3); i++) body();
                    var timer = Stopwatch.StartNew();
                    for (int i = 0; i < reps; i++) body();
                    return timer.Elapsed.TotalSeconds / reps;
                }
                nint aq = (nint)pq, ak = (nint)pk, av = (nint)pv, ao = (nint)po, at = (nint)pt, ah = (nint)ph;
                nint ax = (nint)px, ad = (nint)pd, asr = (nint)ps, au = (nint)pu, ay = (nint)py;
                double transpose = Time(() => QwenImage21CpuKernels.TransposeKeys((float*)ak, dim, tokens, heads, hd, (float*)at));
                double gather = Time(() => QwenImage21CpuKernels.GatherValues((float*)av, dim, tokens, heads, hd, (float*)ah));
                double attention = Time(() => QwenImage21CpuKernels.Attention((float*)aq, (float*)at, tokens, (float*)av, dim, hd, (float*)ao,
                    dim, heads, hd, segments, 0, 0, 1f / MathF.Sqrt(hd)));
                double packed = Time(() => QwenImage21CpuKernels.Attention((float*)aq, (float*)at, tokens, (float*)ah, hd, (long)tokens * hd,
                    (float*)ao, dim, heads, hd, segments, 0, 0, 1f / MathF.Sqrt(hd)));
                double shrink = Time(() => QwenImage21CpuKernels.GemmNT((float*)ax, dim, (float*)ad, dim, (float*)asr, rank, tokens, rank, dim));
                double expand = Time(() => QwenImage21CpuKernels.GemmNNAccumulate((float*)asr, rank, (float*)au, (float*)ay, dim, tokens, dim, rank));
                Console.WriteLine($"  {width * 32,3}-bit: transpose {transpose * 1e3,5:F1} ms, gather V {gather * 1e3,5:F1} ms | " +
                    $"attention V in place {attention * 1e3,7:F1} ms ({attnFlops / attention / 1e9,4:F0} GFLOPS), " +
                    $"V per head {packed * 1e3,7:F1} ms ({attnFlops / packed / 1e9,4:F0} GFLOPS) | " +
                    $"shrink {shrink * 1e3,5:F2} ms ({loraFlops / shrink / 1e9,4:F0} GFLOPS) | " +
                    $"expand {expand * 1e3,5:F2} ms ({loraFlops / expand / 1e9,4:F0} GFLOPS)");
            }
        }
        QwenImage21CpuKernels.Width = saved;
    }

    /// <summary>The F32-activation projection (dequantized weight tiles, GemmDequantNT) on real
    /// DiT weights at the resolution's row count, against the Q8_K integer-dot path.</summary>
    public static void Projections(string dit, int size)
    {
        using var gguf = new GgufFile(dit);
        int rows = (size / 16) * (size / 16) + 23;
        var rng = new Random(2);
        foreach (string name in new[] { "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_v.weight",
                     "transformer_blocks.0.img_mlp.gate_up.weight", "transformer_blocks.0.img_mlp.out.weight" })
        {
            var info = gguf.Tensors[name];
            int input = (int)info.Shape[0], output = (int)info.Shape[1];
            gguf.TryGetTensorDataPointer(info, out IntPtr data);
            long rowBytes = gguf.GetTensorByteCount(info) / output;
            var x = new float[(long)rows * input];
            for (long i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var y = new float[(long)rows * output];
            double flops = 2.0 * rows * input * output;
            fixed (float* px = x, py = y)
            {
                nint ax = (nint)px, ay = (nint)py;
                double Time(Action body, int reps)
                {
                    body();
                    var timer = Stopwatch.StartNew();
                    for (int i = 0; i < reps; i++) body();
                    return timer.Elapsed.TotalSeconds / reps;
                }
                double f32 = Time(() => QwenImage21CpuKernels.GemmDequantNT((float*)ax, input, rows, (int)info.Type, (byte*)data, rowBytes,
                    input, output, (float*)ay, output), 3);
                double q8 = Time(() => ManagedQuantizedOps.AddmmQuantizedToFloat32((int)info.Type, data, input, output, (float*)ax, input, rows,
                    (float*)ay, output), 1);
                Console.WriteLine($"  {name.Replace("transformer_blocks.0.", ""),-24} {(GgmlTensorType)info.Type} {rows}x{input}->{output}: " +
                    $"f32 {f32 * 1e3,7:F1} ms ({flops / f32 / 1e9,5:F0} GFLOPS) | q8 {q8 * 1e3,7:F1} ms ({flops / q8 / 1e9,5:F0} GFLOPS)");
            }
        }
    }
}

/// <summary>Whether native GGML code entered the process: the loaded modules, which is the
/// only reliable evidence from a repo checkout, where GgmlOps is always loadable.</summary>
static class NativeProbe
{
    /// <summary>Set when the bench itself ran something on ggml_cpu (the conditioning).</summary>
    public static bool BenchUsedGgml;

    /// <summary>The path of a loaded GgmlOps / ggml* module, or null. (The managed
    /// TensorSharp.Backends.GGML assembly does not match.)</summary>
    public static string LoadedGgml()
    {
        using var process = Process.GetCurrentProcess();
        foreach (ProcessModule module in process.Modules)
        {
            string name = module.ModuleName ?? "";
            if (name.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase) ||
                name.StartsWith("ggml", StringComparison.OrdinalIgnoreCase) || name.StartsWith("libggml", StringComparison.OrdinalIgnoreCase))
                return module.FileName;
        }
        return null;
    }
}

/// <summary>--pipeline: one text-to-image request through <see cref="QwenImageModel"/> on the
/// cpu backend (Qwen3-VL conditioner, DiT and VAE, companions resolved next to the DiT GGUF,
/// optional --lora plug-ins), then the check that no native GGML module was loaded.</summary>
static class PipelineRun
{
    public static int Run(Options o)
    {
        var timer = Stopwatch.StartNew();
        string png;
        using (var model = new QwenImageModel(o.Dit, BackendType.Cpu))
        {
            if (o.Lora != null) model.SetLoras(new[] { new LoraSpec(o.Lora) });
            Console.WriteLine($"Loaded in {timer.Elapsed.TotalSeconds:F1}s");
            var p = new QwenImageParams { Steps = o.Steps, Seed = o.Seed, Width = o.Size, Height = o.Size };
            timer.Restart();
            var image = model.GenerateImage(o.Prompt, p);
            string label = o.Label ?? $"pipeline-{o.Size}{(o.Lora != null ? "-lora" : "")}";
            png = Path.Combine(o.Out, $"{label}.png");
            ImageIO.SavePng(png, image);
            Console.WriteLine($"Generated {image.Width}x{image.Height} in {timer.Elapsed.TotalSeconds:F1}s -> {png}");
        }
        string native = NativeProbe.LoadedGgml();
        Console.WriteLine($"native GGML library in the process: {native ?? "not loaded"}");
        if (native == null) return 0;
        Console.Error.WriteLine("FAIL: the cpu pipeline loaded the native GGML library.");
        return 3;
    }
}

/// <summary>One real projection per weight type through the managed matmul and through
/// ggml-cpu's (GgmlBasicOps.AddmmQuant), both against a double-precision product of the
/// dequantized weights: separates quantized-matmul differences from the rest of the graph.</summary>
static unsafe class MatmulProbe
{
    public static void Run(string dit)
    {
        using var gguf = new GgufFile(dit);
        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
        var allocator = new GgmlAllocator(context, 0);
        var rng = new Random(5);
        foreach (string name in new[] { "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_v.weight",
                     "transformer_blocks.0.img_mlp.out.weight", "txt_in.in_layer.weight", "img_in.weight" })
        {
            var info = gguf.Tensors[name];
            int input = (int)info.Shape[0], output = Math.Min(512, (int)info.Shape[1]), rows = 32;
            gguf.TryGetTensorDataPointer(info, out IntPtr data);
            long rowBytes = gguf.GetTensorByteCount(info) / (long)info.Shape[1];
            // Hidden-state-like activations: unit Gaussian with a few large channels.
            var x = new float[rows * input];
            for (int i = 0; i < x.Length; i++)
            {
                double u1 = 1 - rng.NextDouble(), u2 = rng.NextDouble();
                x[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2)) * (i % input < 8 ? 20f : 1f);
            }
            var managed = new float[rows * output];
            fixed (float* px = x, py = managed)
            {
                var type = (GgmlTensorType)info.Type;
                if (type is GgmlTensorType.BF16 or GgmlTensorType.F16 or GgmlTensorType.F32 || !ManagedQuantizedOps.TryGetActivationPlan(type, input, out _))
                    QwenImage21CpuKernels.GemmDequantNT(px, input, rows, (int)info.Type, (byte*)data, rowBytes, input, output, py, output);
                else
                    ManagedQuantizedOps.AddmmQuantizedToFloat32((int)info.Type, data, input, output, px, input, rows, py, output);
            }
            var exact = new double[rows * output];
            var w = new float[input];
            for (int o = 0; o < output; o++)
            {
                fixed (float* pw = w) ManagedQuantizedOps.DequantizeRowToFloat32((int)info.Type, data + (nint)(o * rowBytes), pw, input);
                for (int r = 0; r < rows; r++)
                {
                    double sum = 0;
                    for (int i = 0; i < input; i++) sum += (double)w[i] * x[r * input + i];
                    exact[r * output + o] = sum;
                }
            }
            float[] native;
            using (var m1 = new TensorSharp.Tensor(allocator, TensorSharp.DType.Float32, rows, input))
            using (var result = new TensorSharp.Tensor(allocator, TensorSharp.DType.Float32, rows, output))
            {
                m1.SetElementsAsFloat(x);
                GgmlBasicOps.AddmmQuant(result, m1, data, (int)info.Type, input, output, rowBytes * output);
                native = result.GetElementsAsFloat(rows * output);
            }
            double Rel(Func<int, double> a, Func<int, double> b)
            {
                double diff = 0, norm = 0;
                for (int i = 0; i < rows * output; i++) { diff += Math.Pow(a(i) - b(i), 2); norm += b(i) * b(i); }
                return Math.Sqrt(diff / norm);
            }
            Console.WriteLine($"{name} ({(GgmlTensorType)info.Type}, {input} -> {output} of {info.Shape[1]}): " +
                $"managed vs exact {Rel(i => managed[i], i => exact[i]):E2}, ggml vs exact {Rel(i => native[i], i => exact[i]):E2}, " +
                $"managed vs ggml {Rel(i => managed[i], i => native[i]):E2}");
        }
    }
}
