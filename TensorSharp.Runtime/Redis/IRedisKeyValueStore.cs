// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;

namespace TensorSharp.Runtime.Redis
{
    /// <summary>
    /// Minimal key/value contract over a Redis connection. The paged KV cache
    /// tier and the Responses API store both need only string-keyed binary
    /// values with optional TTL, so this narrow surface keeps the two consumers
    /// testable without pulling the full <c>IDatabase</c> surface into mocks.
    /// </summary>
    public interface IRedisKeyValueStore : IDisposable
    {
        bool IsConnected { get; }

        bool StringSet(string key, byte[] value, TimeSpan? expiry);

        byte[] StringGet(string key);

        void KeyDelete(string[] keys);

        string[] ScanKeys(string pattern);
    }
}
