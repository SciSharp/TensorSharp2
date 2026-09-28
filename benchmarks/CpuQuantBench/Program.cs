// Microbenchmark for the pure-C# managed quantized matmul (ManagedQuantizedOps).
//
//   dotnet run -c Release --project benchmarks/CpuQuantBench                 # GEMM table, all types
//   dotnet run -c Release --project benchmarks/CpuQuantBench gemm q4_k       # one type
//   dotnet run -c Release --project benchmarks/CpuQuantBench gemm all 256    # shapes with M <= 256
//   dotnet run -c Release --project benchmarks/CpuQuantBench gemm decode     # DRAM-bound M = 1 matvecs
//   dotnet run -c Release --project benchmarks/CpuQuantBench batch           # MoE: 128 jobs in one call
//   dotnet run -c Release --project benchmarks/CpuQuantBench legacy [quant]  # decode GB/s table
//
// "gemm" runs every (M, K, N) shape through the old per-row path (Legacy) and
// the multi-row GEMM (AVX-512 and AVX2 kernels) in ONE process, checks the new
// results against the old ones, and reports GOPS = 2*M*N*K / s (plus weight
// GB/s at M = 1, where the matmul is bandwidth-bound).
//
// "legacy" is the original table: effective weight-read bandwidth per quant
// type for rowCount 1 and 4. Decode tok/s for a model is ~ (bytes read per
// token) / (GB/s here), so that GB/s number is the lever for decode.
using System.Diagnostics;
using TensorSharp.Models;
using TensorSharp.Runtime;

NativeDequant.PreferManaged = true;   // reference dequant stays in managed code
string mode = args.Length > 0 ? args[0].ToLowerInvariant() : "gemm";
if (mode == "legacy")
{
    RunLegacyTable(args.Length > 1 ? args[1].ToLowerInvariant() : null);
    return;
}
if (mode == "batch")
{
    RunBatchTable();
    return;
}
string typeFilter = args.Length > 1 ? args[1].ToLowerInvariant() : "all";
int maxM = args.Length > 2 ? int.Parse(args[2]) : int.MaxValue;
bool skipLegacyBig = Environment.GetEnvironmentVariable("QBENCH_SKIP_LEGACY_BIG") == "1";
RunGemmTable(typeFilter, maxM, skipLegacyBig);

static void RunGemmTable(string typeFilter, int maxM, bool skipLegacyBig)
{
    var dg = new (int m, int k, int n)[] { (1, 2816, 1408), (2, 2816, 1408), (4, 2816, 1408), (70, 2816, 2112), (70, 2816, 4096) };
    var dgDown = new (int m, int k, int n)[] { (1, 704, 2816), (2, 704, 2816), (8, 704, 2816), (70, 2112, 2816) };
    var dit = new (int m, int k, int n)[]
    {
        (256, 4096, 4096), (256, 4096, 24576), (256, 12288, 4096), (1024, 4096, 4096), (4096, 4096, 4096),
    };
    var plan = new List<(GgmlTensorType type, (int m, int k, int n)[] shapes)>
    {
        (GgmlTensorType.Q4_K, dg.Concat(dit).ToArray()),
        (GgmlTensorType.Q6_K, dg.Concat(new[] { (256, 4096, 4096), (1024, 4096, 4096) }).ToArray()),
        (GgmlTensorType.Q5_K, new[] { (1, 2816, 1408), (4, 2816, 1408), (70, 2816, 2112), (256, 4096, 4096) }),
        (GgmlTensorType.Q8_0, dgDown.Concat(new[] { (256, 4096, 4096), (1024, 4096, 4096) }).ToArray()),
        (GgmlTensorType.Q5_0, dgDown.Concat(new[] { (256, 4096, 4096) }).ToArray()),
        (GgmlTensorType.Q4_0, new[] { (1, 4096, 4096), (8, 704, 2816), (256, 4096, 4096) }),
        (GgmlTensorType.BF16, new[] { (1, 4096, 4096), (256, 4096, 4096), (256, 64, 4096) }),
        (GgmlTensorType.F16, new[] { (256, 4096, 4096) }),
    };
    // "decode": single-row shapes whose weights (33-60 MB) do not fit the L3,
    // so the matvec streams from DRAM the way a decoded token does.
    if (typeFilter == "decode")
    {
        var big = new[] { (1, 4096, 14336), (1, 14336, 4096) };
        plan = new List<(GgmlTensorType type, (int m, int k, int n)[] shapes)>
        {
            (GgmlTensorType.Q4_K, big), (GgmlTensorType.Q6_K, big), (GgmlTensorType.Q5_K, big),
            (GgmlTensorType.Q8_0, big), (GgmlTensorType.Q5_0, big), (GgmlTensorType.Q4_0, big),
        };
        typeFilter = "all";
    }

    Console.WriteLine($"cores={Environment.ProcessorCount} avx512={ManagedQuantizedOps.QGemmAvx512Supported} " +
                      $"avx2={ManagedQuantizedOps.QGemmAvx2Supported}");
    Console.WriteLine($"{"type",-5} {"M",5} {"K",6} {"N",6} | {"legacy ms",10} {"GOPS",7} | {"avx512 ms",10} {"GOPS",7} {"x",6} | " +
                      $"{"avx2 ms",10} {"GOPS",7} {"x",6} | {"relErr512",9} {"relErr2",9} {"GB/s 512/legacy",15}");
    foreach (var (type, shapes) in plan)
    {
        if (typeFilter != "all" && !type.ToString().Equals(typeFilter.Replace("_", ""), StringComparison.OrdinalIgnoreCase)
            && !type.ToString().Equals(typeFilter, StringComparison.OrdinalIgnoreCase))
            continue;
        foreach (var (m, k, n) in shapes)
        {
            if (m > maxM) continue;
            RunShape(type, m, k, n, skipLegacyBig && (long)m * n * k > 20_000_000_000L);
        }
    }
}

static unsafe void RunShape(GgmlTensorType type, int m, int k, int n, bool skipLegacy)
{
    var rng = new Random(1234 + (int)type * 17 + k + n);
    byte[] weights = BuildRandom(rng, type, n, k);
    float[] input = new float[(long)m * k];
    for (long i = 0; i < input.Length; i++)
        input[i] = 0.08f * MathF.Sin(i * 0.011f) + 0.02f * (float)(rng.NextDouble() - 0.5);
    float[] outLegacy = new float[(long)m * n];
    float[] out512 = new float[(long)m * n];
    float[] out2 = new float[(long)m * n];
    double flops = 2.0 * m * n * k;
    long weightBytes = NativeDequant.RowSize((int)type, k) * n;

    fixed (byte* w = weights)
    fixed (float* x = input)
    fixed (float* oL = outLegacy)
    fixed (float* o5 = out512)
    fixed (float* o2 = out2)
    {
        byte* wp = w;
        float* xp = x;
        void Run(float* o, ManagedQuantizedOps.QGemmIsa isa) =>
            ManagedQuantizedOps.AddmmQuantizedToFloat32((int)type, (IntPtr)wp, k, n, xp, k, m, o, n, null, isa);

        float* oLp = oL, o5p = o5, o2p = o2;
        double tL = skipLegacy ? double.NaN : Time(() => Run(oLp, ManagedQuantizedOps.QGemmIsa.Legacy), flops);
        double t5 = Time(() => Run(o5p, ManagedQuantizedOps.QGemmIsa.Avx512), flops);
        double t2 = Time(() => Run(o2p, ManagedQuantizedOps.QGemmIsa.Avx2), flops);
        if (skipLegacy)
            outLegacy.AsSpan().Clear();
        float e5 = skipLegacy ? float.NaN : RelErr(outLegacy, out512);
        float e2 = RelErr(skipLegacy ? out512 : outLegacy, out2);
        string gbs = m == 1 ? $"{weightBytes / t5 / 1e9,7:F1} /{weightBytes / tL / 1e9,6:F1}" : "";
        Console.WriteLine($"{Name(type),-5} {m,5} {k,6} {n,6} | {tL * 1e3,10:F3} {flops / tL / 1e9,7:F1} | " +
                          $"{t5 * 1e3,10:F3} {flops / t5 / 1e9,7:F1} {tL / t5,6:F2} | " +
                          $"{t2 * 1e3,10:F3} {flops / t2 / 1e9,7:F1} {tL / t2,6:F2} | {e5,9:E2} {e2,9:E2} {gbs}");
    }
}

static string Name(GgmlTensorType t) => t.ToString().ToLowerInvariant();

// "batch": TryAddmmQuantizedBatch the way an MoE layer uses it - 128 expert jobs in
// one call, each with 1..16 routed rows (DiffusionGemma's shapes: gate_up
// 2816 -> 1408 Q4_K, down 704 -> 2816 Q5_0 / Q8_0). Old per-row batch vs GEMM batch.
static unsafe void RunBatchTable()
{
    Console.WriteLine($"{"type",-5} {"K",6} {"N",6} {"jobs",5} {"rows",6} | {"legacy ms",10} | {"avx512 ms",10} {"x",6} | " +
                      $"{"avx2 ms",10} {"x",6} | {"relErr512",9}");
    foreach (var (type, k, n) in new[] { (GgmlTensorType.Q4_K, 2816, 1408), (GgmlTensorType.Q5_0, 704, 2816), (GgmlTensorType.Q8_0, 704, 2816) })
    {
        const int jobs = 128;
        var rng = new Random(7 + (int)type);
        int[] rows = new int[jobs];
        int totalRows = 0;
        for (int j = 0; j < jobs; j++) { rows[j] = 1 + rng.Next(0, 16); totalRows += rows[j]; }
        var weights = new byte[jobs][];
        var inputs = new float[jobs][];
        var outL = new float[jobs][];
        var out5 = new float[jobs][];
        var out2 = new float[jobs][];
        for (int j = 0; j < jobs; j++)
        {
            weights[j] = BuildRandom(rng, type, n, k);
            inputs[j] = new float[rows[j] * k];
            for (int i = 0; i < inputs[j].Length; i++) inputs[j][i] = (float)(rng.NextDouble() - 0.5) * 0.3f;
            outL[j] = new float[rows[j] * n];
            out5[j] = new float[rows[j] * n];
            out2[j] = new float[rows[j] * n];
        }
        var handles = new List<System.Runtime.InteropServices.GCHandle>();
        IntPtr Pin(object o)
        {
            var h = System.Runtime.InteropServices.GCHandle.Alloc(o, System.Runtime.InteropServices.GCHandleType.Pinned);
            handles.Add(h);
            return h.AddrOfPinnedObject();
        }
        ManagedQuantizedOps.QuantMatMulJob[] Jobs(float[][] outputs)
        {
            var list = new ManagedQuantizedOps.QuantMatMulJob[jobs];
            for (int j = 0; j < jobs; j++)
                list[j] = new ManagedQuantizedOps.QuantMatMulJob(Pin(weights[j]), Pin(inputs[j]), Pin(outputs[j]), n, rows[j], n);
            return list;
        }
        var jl = Jobs(outL); var j5 = Jobs(out5); var j2 = Jobs(out2);
        double flops = 2.0 * totalRows * n * k;
        double tL = Time(() => ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, jl, null, ManagedQuantizedOps.QGemmIsa.Legacy), flops);
        double t5 = Time(() => ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, j5, null, ManagedQuantizedOps.QGemmIsa.Avx512), flops);
        double t2 = Time(() => ManagedQuantizedOps.TryAddmmQuantizedBatch((int)type, k, k, j2, null, ManagedQuantizedOps.QGemmIsa.Avx2), flops);
        float err = RelErr(outL.SelectMany(a => a).ToArray(), out5.SelectMany(a => a).ToArray());
        Console.WriteLine($"{Name(type),-5} {k,6} {n,6} {jobs,5} {totalRows,6} | {tL * 1e3,10:F3} | {t5 * 1e3,10:F3} {tL / t5,6:F2} | " +
                          $"{t2 * 1e3,10:F3} {tL / t2,6:F2} | {err,9:E2}");
        foreach (var h in handles) h.Free();
    }
}

// Median-of-reps wall time: at least 3 reps and ~0.6 s of runtime, after
// ~0.3 s of warmup (enough calls for tiered JIT to reach its optimized code).
static double Time(Action run, double flops)
{
    var warm = Stopwatch.StartNew();
    do run(); while (warm.Elapsed.TotalSeconds < 0.3);
    var times = new List<double>();
    var total = Stopwatch.StartNew();
    int minReps = flops > 5e10 ? 2 : 3;
    while (times.Count < minReps || (total.Elapsed.TotalSeconds < 0.6 && times.Count < 200))
    {
        var sw = Stopwatch.StartNew();
        run();
        times.Add(sw.Elapsed.TotalSeconds);
    }
    times.Sort();
    return times[times.Count / 2];
}

// max |a - b| / max |a| over the whole output.
static float RelErr(float[] reference, float[] actual)
{
    float maxRef = 1e-20f, maxDiff = 0f;
    for (long i = 0; i < reference.Length; i++)
    {
        maxRef = MathF.Max(maxRef, MathF.Abs(reference[i]));
        maxDiff = MathF.Max(maxDiff, MathF.Abs(reference[i] - actual[i]));
    }
    return maxDiff / maxRef;
}

static byte[] BuildRandom(Random rng, GgmlTensorType type, int outDim, int inDim)
{
    int blockBytes = (int)GgufFile.GetTypeSize(type);
    int blockSize = (int)GgufFile.GetBlockSize(type);
    int blocksPerRow = inDim / blockSize;
    byte[] raw = new byte[(long)outDim * blocksPerRow * blockBytes];
    long o = 0;
    for (int r = 0; r < outDim; r++)
    {
        for (int b = 0; b < blocksPerRow; b++)
        {
            long sb = o;
            switch (type)
            {
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q8_0:
                case GgmlTensorType.Q5_0:
                    WriteHalf(raw, sb, 0.02f + 0.03f * (float)rng.NextDouble());
                    for (int i = 0; i < blockBytes - 2; i++) raw[sb + 2 + i] = (byte)rng.Next(0, 256);
                    break;
                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                    WriteHalf(raw, sb, 0.02f + 0.03f * (float)rng.NextDouble());
                    WriteHalf(raw, sb + 2, 0.01f + 0.02f * (float)rng.NextDouble());
                    for (int i = 0; i < blockBytes - 4; i++) raw[sb + 4 + i] = (byte)rng.Next(0, 256);
                    break;
                case GgmlTensorType.Q6_K:
                    for (int i = 0; i < blockBytes - 2; i++) raw[sb + i] = (byte)rng.Next(0, 256);
                    WriteHalf(raw, sb + blockBytes - 2, 0.01f + 0.02f * (float)rng.NextDouble());
                    break;
                case GgmlTensorType.F16:
                    WriteHalf(raw, sb, 0.05f * (float)(rng.NextDouble() - 0.5));
                    break;
                case GgmlTensorType.BF16:
                {
                    uint bits = BitConverter.SingleToUInt32Bits(0.05f * (float)(rng.NextDouble() - 0.5));
                    raw[sb] = (byte)(bits >> 16); raw[sb + 1] = (byte)(bits >> 24);
                    break;
                }
                default:
                    throw new NotSupportedException(type.ToString());
            }
            o += blockBytes;
        }
    }
    return raw;

    static void WriteHalf(byte[] buf, long off, float val)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)val);
        buf[off] = (byte)bits; buf[off + 1] = (byte)(bits >> 8);
    }
}

static void RunLegacyTable(string filter)
{
    var quants = new (string name, GgmlTensorType type)[]
    {
        ("q4_0", GgmlTensorType.Q4_0),
        ("q8_0", GgmlTensorType.Q8_0),
        ("q5_0", GgmlTensorType.Q5_0),
        ("q4_k", GgmlTensorType.Q4_K),
        ("q5_k", GgmlTensorType.Q5_K),
        ("q6_k", GgmlTensorType.Q6_K),
    };

    int inDim = 4096, outDim = 4096, iters = 200;
    Console.WriteLine($"cores={Environment.ProcessorCount}  matmul={inDim}x{outDim}");
    Console.WriteLine($"{"quant",-6} {"rows",4}  {"GB/s",8}  {"ms/call",9}  {"relErr",9}");
    foreach (var (name, type) in quants)
    {
        if (filter != null && name != filter) continue;
        foreach (int rows in new[] { 1, 4 })
        {
            var (gbps, ms, err) = BenchLegacy(type, inDim, outDim, rows, iters);
            Console.WriteLine($"{name,-6} {rows,4}  {gbps,8:F1}  {ms,9:F3}  {err,9:E2}");
        }
    }
}

static (double gbps, double msPerCall, float maxRelErr) BenchLegacy(
    GgmlTensorType type, int inDim, int outDim, int rows, int iters)
{
    var rng = new Random(12345 + (int)type);
    byte[] weights = BuildRandom(rng, type, outDim, inDim);
    float[] input = new float[rows * inDim];
    for (int i = 0; i < input.Length; i++) input[i] = 0.08f * MathF.Sin(i * 0.011f);
    float[] output = new float[rows * outDim];

    long weightBytes = (long)NativeDequant.RowSize((int)type, inDim) * outDim;

    // correctness on a few columns against dequantize-then-dot
    float[] wrow = new float[inDim];
    float maxRel = 0f, refMag = 1e-6f;
    bool ok = ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
        (int)type, weights, 0, inDim, outDim, input, 0, inDim, rows, output, 0, outDim);
    if (!ok) throw new Exception($"{type}: TryAddmm returned false");
    int[] checkCols = { 0, outDim / 3, outDim / 2, outDim - 1 };
    long rowBytes = NativeDequant.RowSize((int)type, inDim);
    foreach (int c in checkCols)
    {
        NativeDequant.DequantizeToFloat32((int)type, weights, (int)(c * rowBytes), wrow, 0, inDim);
        float exp = 0f;
        for (int i = 0; i < inDim; i++) exp += wrow[i] * input[i];
        refMag = MathF.Max(refMag, MathF.Abs(exp));
        maxRel = MathF.Max(maxRel, MathF.Abs(exp - output[c]));
    }
    maxRel /= refMag;

    for (int i = 0; i < 3; i++)
        ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, weights, 0, inDim, outDim, input, 0, inDim, rows, output, 0, outDim);

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++)
        ManagedQuantizedOps.TryAddmmQuantizedToFloat32(
            (int)type, weights, 0, inDim, outDim, input, 0, inDim, rows, output, 0, outDim);
    sw.Stop();

    double seconds = sw.Elapsed.TotalSeconds;
    double gbps = (double)weightBytes * iters / seconds / (1024.0 * 1024 * 1024);
    double msPerCall = sw.Elapsed.TotalMilliseconds / iters;
    return (gbps, msPerCall, maxRel);
}
