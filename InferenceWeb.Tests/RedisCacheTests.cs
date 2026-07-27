// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Redis;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

/// <summary>
/// In-memory fake of <see cref="IRedisKeyValueStore"/> for unit-testing the
/// Redis-backed tiers without a live Redis server.
/// </summary>
internal sealed class FakeRedisKeyValueStore : IRedisKeyValueStore
{
    private readonly Dictionary<string, byte[]> _store = new();
    private readonly object _gate = new();

    public bool IsConnected => true;
    public int SetCount { get; private set; }
    public int GetCount { get; private set; }

    public bool StringSet(string key, byte[] value, TimeSpan? expiry)
    {
        lock (_gate)
        {
            _store[key] = value;
            SetCount++;
            return true;
        }
    }

    public byte[] StringGet(string key)
    {
        lock (_gate)
        {
            GetCount++;
            return _store.TryGetValue(key, out var value) ? value : null;
        }
    }

    public void KeyDelete(string[] keys)
    {
        lock (_gate)
        {
            foreach (var key in keys)
                _store.Remove(key);
        }
    }

    public string[] ScanKeys(string pattern)
    {
        lock (_gate)
        {
            // Simple prefix match: strip trailing '*'
            string prefix = pattern.TrimEnd('*');
            return _store.Keys.Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).ToArray();
        }
    }

    public int Count
    {
        get { lock (_gate) return _store.Count; }
    }

    public void Dispose() { }
}

public class RedisResponsesStoreTests
{
    [Fact]
    public void Store_And_TryGet_RoundTrips()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        var response = new StoredResponse
        {
            Id = "resp_abc123",
            Json = """{"id":"resp_abc123","output":[]}""",
        };
        store.Store(response);

        Assert.True(store.TryGet("resp_abc123", out var retrieved));
        Assert.Equal("resp_abc123", retrieved.Id);
        Assert.Equal(response.Json, retrieved.Json);
    }

    [Fact]
    public void TryGet_MissingKey_ReturnsFalse()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        Assert.False(store.TryGet("resp_nonexistent", out var retrieved));
        Assert.Null(retrieved);
    }

    [Fact]
    public void Store_OverwritesExistingEntry()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        store.Store(new StoredResponse { Id = "resp_1", Json = """{"v":1}""" });
        store.Store(new StoredResponse { Id = "resp_1", Json = """{"v":2}""" });

        Assert.True(store.TryGet("resp_1", out var retrieved));
        Assert.Equal("""{"v":2}""", retrieved.Json);
    }

    [Fact]
    public void Store_UsesCorrectKeyPrefix()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        store.Store(new StoredResponse { Id = "resp_xyz", Json = "{}" });

        Assert.Equal(1, fake.Count);
        // The key should be prefixed with "tsresp:"
        Assert.True(fake.StringGet("tsresp:resp_xyz") != null);
    }

    [Fact]
    public void Store_MultipleEntries_AllRetrievable()
    {
        var fake = new FakeRedisKeyValueStore();
        using var store = new RedisResponsesStore(fake, TimeSpan.FromMinutes(30));

        for (int i = 0; i < 10; i++)
            store.Store(new StoredResponse { Id = $"resp_{i}", Json = $$"""{"i":{{i}}}""" });

        for (int i = 0; i < 10; i++)
        {
            Assert.True(store.TryGet($"resp_{i}", out var retrieved));
            Assert.Equal($$"""{"i":{{i}}}""", retrieved.Json);
        }
    }
}

public class RedisKvBlockTierTests
{
    private static KvBlockHash MakeHash(ulong lo, ulong hi) => new(lo, hi);

    [Fact]
    public void WriteAndRead_RoundTrips()
    {
        var fake = new FakeRedisKeyValueStore();
        using var tier = new RedisKvBlockTier(fake, "test-fp", ttl: null);

        var hash = MakeHash(0x1234, 0x5678);
        byte[] payload = [1, 2, 3, 4, 5, 6, 7, 8];

        tier.EnqueueWrite(hash, payload);
        // Wait for the background writer to flush
        WaitForWriter(tier);

        Assert.True(tier.TryRead(hash, out var readBack));
        Assert.Equal(payload, readBack);
    }

    [Fact]
    public void TryRead_MissingKey_ReturnsFalse()
    {
        var fake = new FakeRedisKeyValueStore();
        using var tier = new RedisKvBlockTier(fake, "test-fp", ttl: null);

        Assert.False(tier.TryRead(MakeHash(0xDEAD, 0xBEEF), out var payload));
        Assert.Null(payload);
    }

    [Fact]
    public void TryRead_WrongFingerprint_ReturnsFalse()
    {
        var fake = new FakeRedisKeyValueStore();
        // Write with one fingerprint
        using (var writer = new RedisKvBlockTier(fake, "model-A", ttl: null))
        {
            writer.EnqueueWrite(MakeHash(1, 2), [10, 20, 30]);
            WaitForWriter(writer);
        }

        // Read with a different fingerprint
        using var reader = new RedisKvBlockTier(fake, "model-B", ttl: null);
        // The key prefix includes the fingerprint hash, so a different
        // fingerprint produces a different key and the read misses.
        Assert.False(reader.TryRead(MakeHash(1, 2), out _));
    }

    [Fact]
    public void Clear_RemovesAllEntries()
    {
        var fake = new FakeRedisKeyValueStore();
        using var tier = new RedisKvBlockTier(fake, "test-fp", ttl: null);

        tier.EnqueueWrite(MakeHash(1, 1), [1]);
        tier.EnqueueWrite(MakeHash(2, 2), [2]);
        WaitForWriter(tier);

        Assert.Equal(2, fake.Count);
        tier.Clear();
        Assert.Equal(0, fake.Count);
    }

    [Fact]
    public void EnqueueWrite_NullPayload_IsIgnored()
    {
        var fake = new FakeRedisKeyValueStore();
        using var tier = new RedisKvBlockTier(fake, "test-fp", ttl: null);

        tier.EnqueueWrite(MakeHash(1, 1), null);
        // Give the writer thread a moment to process (it should skip the null)
        Thread.Sleep(100);

        Assert.Equal(0, fake.Count);
    }

    [Fact]
    public void EnqueueWrite_EmptyPayload_IsIgnored()
    {
        var fake = new FakeRedisKeyValueStore();
        using var tier = new RedisKvBlockTier(fake, "test-fp", ttl: null);

        tier.EnqueueWrite(MakeHash(1, 1), []);
        // Give the writer thread a moment to process (it should skip the empty)
        Thread.Sleep(100);

        Assert.Equal(0, fake.Count);
    }

    [Fact]
    public void LargePayload_RoundTrips()
    {
        var fake = new FakeRedisKeyValueStore();
        using var tier = new RedisKvBlockTier(fake, "test-fp", ttl: null);

        var hash = MakeHash(0xABCD, 0xEF01);
        byte[] payload = new byte[256 * 1024]; // 256 KB block
        new Random(42).NextBytes(payload);

        tier.EnqueueWrite(hash, payload);
        WaitForWriter(tier);

        Assert.True(tier.TryRead(hash, out var readBack));
        Assert.Equal(payload, readBack);
    }

    private static void WaitForWriter(RedisKvBlockTier tier)
    {
        // The background writer thread processes the queue asynchronously.
        // Poll until the write has been flushed (up to 2 seconds).
        for (int i = 0; i < 200; i++)
        {
            if (tier.Count > 0 || tier.ResidentBytes > 0)
                return;
            Thread.Sleep(10);
        }
    }
}

public class PagedKvCacheConfigRedisTests
{
    [Fact]
    public void FromEnvironment_NoRedisVars_RedisUrlIsNull()
    {
        // Ensure the env vars are not set
        Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_URL", null);
        Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_TTL_MINUTES", null);

        var config = PagedKvCacheConfig.FromEnvironment();
        Assert.Null(config.RedisUrl);
    }

    [Fact]
    public void FromEnvironment_RedisUrlSet_IsPopulated()
    {
        Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_URL", "localhost:6379");
        try
        {
            var config = PagedKvCacheConfig.FromEnvironment();
            Assert.Equal("localhost:6379", config.RedisUrl);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_URL", null);
        }
    }

    [Fact]
    public void FromEnvironment_RedisTtlDefault_Is24Hours()
    {
        Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_TTL_MINUTES", null);
        var config = PagedKvCacheConfig.FromEnvironment();
        Assert.Equal(TimeSpan.FromHours(24), config.RedisTtl);
    }

    [Fact]
    public void FromEnvironment_RedisTtlZero_IsNull()
    {
        Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_TTL_MINUTES", "0");
        try
        {
            var config = PagedKvCacheConfig.FromEnvironment();
            Assert.Null(config.RedisTtl);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_TTL_MINUTES", null);
        }
    }

    [Fact]
    public void FromEnvironment_RedisTtlCustom_IsParsed()
    {
        Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_TTL_MINUTES", "120");
        try
        {
            var config = PagedKvCacheConfig.FromEnvironment();
            Assert.Equal(TimeSpan.FromMinutes(120), config.RedisTtl);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_KV_CACHE_REDIS_TTL_MINUTES", null);
        }
    }
}
