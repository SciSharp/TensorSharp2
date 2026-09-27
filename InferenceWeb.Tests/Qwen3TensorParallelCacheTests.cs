using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class Qwen3TensorParallelCacheTests
{
    [Fact]
    public void CacheGrowsOnEveryRankPreservesTokensAndEnforcesConfiguredLimit()
    {
        // Exercise the actual cache allocator/growth code with two managed ranks;
        // no weight checkpoint or GPU is needed for this cache ownership contract.
        var model = (Qwen3Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen3Model));
        Set(typeof(ModelBase), model, "<Config>k__BackingField", new ModelConfig
            { NumLayers = 1, NumHeads = 4, NumKVHeads = 4, HiddenSize = 16 });
        Set(typeof(ModelBase), model, "_tpGroup", new ManagedRanks());
        Set(typeof(ModelBase), model, "_backend", BackendType.Cpu);
        try
        {
            Invoke(model, "InitTpKVCache", 2, 8);
            Assert.Equal(8, model.MaxContextLength);
            Set(typeof(ModelBase), model, "_cacheSeqLen", 2);
            string[] fields = { "_tpKvCacheK", "_tpKvCacheV" };
            for (int kind = 0; kind < fields.Length; kind++)
                for (int rank = 0; rank < 2; rank++)
                    GetCaches(model, fields[kind])[0][rank].SetElementsAsFloat(Expected(kind, rank));

            Invoke(model, "EnsureTpCacheCapacity", 3);
            VerifyCaches(4);
            Invoke(model, "EnsureTpCacheCapacity", 8);
            VerifyCaches(8);
            var error = Assert.Throws<TargetInvocationException>(() => Invoke(model, "EnsureTpCacheCapacity", 9));
            Assert.IsType<InvalidOperationException>(error.InnerException);
            Assert.Contains("max context 8", error.InnerException!.Message);

            // Each rank has two heads. Their stride changes when capacity grows,
            // so a flat prefix copy would preserve head 0 but corrupt head 1.
            void VerifyCaches(int capacity)
            {
                for (int kind = 0; kind < fields.Length; kind++)
                    for (int rank = 0; rank < 2; rank++)
                    {
                        Tensor cache = GetCaches(model, fields[kind])[0][rank];
                        Assert.Equal(capacity, cache.Sizes[1]);
                        var expected = new float[2 * capacity * 4];
                        float[] occupied = Expected(kind, rank);
                        Array.Copy(occupied, 0, expected, 0, 8);
                        Array.Copy(occupied, 8, expected, capacity * 4, 8);
                        Assert.Equal(expected, cache.GetElementsAsFloat(expected.Length));
                    }
            }
        }
        finally
        {
            foreach (string field in new[] { "_tpKvCacheK", "_tpKvCacheV" })
                foreach (var layer in GetCaches(model, field) ?? Array.Empty<Tensor[]>())
                    foreach (Tensor cache in layer ?? Array.Empty<Tensor>()) cache?.Dispose();
        }
    }

    private static float[] Expected(int kind, int rank)
        => Enumerable.Range(1, 16).Select(value => (float)(kind * 1000 + rank * 100 + value)).ToArray();

    private static void Invoke(Qwen3Model model, string name, params object[] values)
        => typeof(Qwen3Model).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, values);
    private static void Set(Type type, object model, string name, object value)
        => type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, value);
    private static Tensor[][] GetCaches(Qwen3Model model, string name)
        => (Tensor[][])typeof(Qwen3Model).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private sealed class ManagedRanks : ITensorParallelGroup
    {
        private readonly IAllocator _allocator = new CpuAllocator(BlasEnum.DotNet);
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => _allocator;
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
