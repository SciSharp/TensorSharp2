// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Redis;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Redis-backed third tier for the paged KV cache. Blocks evicted from the
    /// in-memory tier flow here (asynchronously, on a background writer thread)
    /// and are re-read synchronously on lookup miss. Redis provides a shared
    /// cache across multiple server instances and survives process restarts
    /// (subject to Redis persistence configuration).
    ///
    /// Key format: <c>tskv:{fingerprint_hash_hex}:{block_hash_hex}</c>
    /// Value format (little-endian):
    ///   bytes 0..3   : magic 0x544B5652 ("TKVR" - TensorSharp KV-block Redis)
    ///   bytes 4..7   : format version (currently 1)
    ///   bytes 8..15  : payload byte length
    ///   bytes 16..23 : 8-byte fingerprint hash (for cross-model collision rejection)
    ///   bytes 24..   : raw payload
    /// </summary>
    internal sealed class RedisKvBlockTier : IDisposable
    {
        private const uint Magic = 0x544B5652u; // "TKVR"
        private const int FormatVersion = 1;
        private const int HeaderSize = 24;

        private readonly IRedisKeyValueStore _redis;
        private readonly string _keyPrefix;
        private readonly ulong _fingerprintHash;
        private readonly TimeSpan? _ttl;
        private readonly ILogger _logger;
        private readonly BlockingCollection<WriteJob> _writeQueue;
        private readonly Thread _writerThread;
        private volatile bool _disposed;
        private long _residentBytes;
        private int _count;

        public RedisKvBlockTier(IRedisKeyValueStore redis, string fingerprint, TimeSpan? ttl, ILogger logger = null)
        {
            _redis = redis ?? throw new ArgumentNullException(nameof(redis));
            _fingerprintHash = StableFingerprintHash(fingerprint ?? string.Empty);
            _keyPrefix = $"tskv:{_fingerprintHash:x16}:";
            _ttl = ttl;
            _logger = logger ?? NullLogger.Instance;

            _writeQueue = new BlockingCollection<WriteJob>(boundedCapacity: 256);
            _writerThread = new Thread(WriterLoop)
            {
                Name = "TensorSharp KV Redis writer",
                IsBackground = true,
            };
            _writerThread.Start();
        }

        public long ResidentBytes => Interlocked.Read(ref _residentBytes);
        public int Count => Volatile.Read(ref _count);

        public bool TryRead(KvBlockHash hash, out byte[] payload)
        {
            string key = KeyFor(hash);
            byte[] raw = _redis.StringGet(key);
            if (raw == null || raw.Length < HeaderSize)
            {
                payload = null;
                return false;
            }

            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(raw);
            int version = BinaryPrimitives.ReadInt32LittleEndian(raw.AsSpan(4));
            long payloadLength = BinaryPrimitives.ReadInt64LittleEndian(raw.AsSpan(8));
            ulong fingerprint = BinaryPrimitives.ReadUInt64LittleEndian(raw.AsSpan(16));

            if (magic != Magic || version != FormatVersion || fingerprint != _fingerprintHash
                || payloadLength <= 0 || payloadLength > int.MaxValue
                || raw.Length < HeaderSize + payloadLength)
            {
                payload = null;
                return false;
            }

            payload = new byte[(int)payloadLength];
            Array.Copy(raw, HeaderSize, payload, 0, (int)payloadLength);
            return true;
        }

        public void EnqueueWrite(KvBlockHash hash, byte[] payload)
        {
            if (payload == null || payload.Length == 0)
                return;
            if (_disposed)
                return;
            try
            {
                _writeQueue.Add(new WriteJob(hash, payload));
            }
            catch (InvalidOperationException)
            {
                // Queue completed during shutdown - drop write silently.
            }
        }

        public void Clear()
        {
            try
            {
                string[] keys = _redis.ScanKeys(_keyPrefix + "*");
                if (keys.Length > 0)
                    _redis.KeyDelete(keys);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to clear Redis KV cache tier");
            }
            Interlocked.Exchange(ref _residentBytes, 0);
            Interlocked.Exchange(ref _count, 0);
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            try { _writeQueue.CompleteAdding(); }
            catch { /* already completed */ }
            try { _writerThread.Join(TimeSpan.FromSeconds(5)); }
            catch { /* best effort */ }
            _writeQueue.Dispose();
            _redis.Dispose();
        }

        private void WriterLoop()
        {
            try
            {
                foreach (var job in _writeQueue.GetConsumingEnumerable())
                {
                    try
                    {
                        WriteBlock(job.Hash, job.Payload);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to write KV block {Hash} to Redis tier", job.Hash);
                    }
                }
            }
            catch (ObjectDisposedException)
            {
                // Queue disposed during shutdown.
            }
        }

        private void WriteBlock(KvBlockHash hash, byte[] payload)
        {
            byte[] raw = new byte[HeaderSize + payload.Length];
            BinaryPrimitives.WriteUInt32LittleEndian(raw, Magic);
            BinaryPrimitives.WriteInt32LittleEndian(raw.AsSpan(4), FormatVersion);
            BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(8), payload.LongLength);
            BinaryPrimitives.WriteUInt64LittleEndian(raw.AsSpan(16), _fingerprintHash);
            Array.Copy(payload, 0, raw, HeaderSize, payload.Length);

            string key = KeyFor(hash);
            if (_redis.StringSet(key, raw, _ttl))
            {
                Interlocked.Add(ref _residentBytes, raw.Length);
                Interlocked.Increment(ref _count);
            }
        }

        private string KeyFor(KvBlockHash hash) => _keyPrefix + hash.ToHexString();

        private static ulong StableFingerprintHash(string fingerprint)
        {
            const ulong offsetBasis = 14695981039346656037UL;
            const ulong prime = 1099511628211UL;
            ulong hash = offsetBasis;
            foreach (char c in fingerprint)
            {
                hash ^= c;
                hash *= prime;
            }
            return hash;
        }

        private readonly record struct WriteJob(KvBlockHash Hash, byte[] Payload);
    }
}
