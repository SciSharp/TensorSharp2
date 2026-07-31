// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Random-access byte source the DSV4 engine streams weights from during
    /// load. Lets a multi-hundred-GB model go GGUF shard -> pinned chunk ->
    /// VRAM without ever staging the whole thing in host RAM.
    /// </summary>
    /// <remarks>
    /// Implementations are called concurrently from every loader thread and
    /// must be thread-safe (positional reads, no shared file cursor).
    /// </remarks>
    public interface IDsv4WeightSource
    {
        /// <summary>
        /// Reads exactly <paramref name="bytes"/> bytes at
        /// <paramref name="offset"/> into <paramref name="dst"/>, or throws.
        /// </summary>
        void Read(long offset, IntPtr dst, long bytes);
    }
}
