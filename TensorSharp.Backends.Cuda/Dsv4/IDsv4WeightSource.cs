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

    /// <summary>
    /// Optional capability of an <see cref="IDsv4WeightSource"/>: hand out a
    /// stable read-only pointer into a file mapping of the shard. Weights that
    /// stay in host RAM for the engine's lifetime (the <c>--n-cpu-moe</c>
    /// routed experts) borrow the mapping instead of taking a private copy:
    /// file-backed pages are page cache the kernel can evict and re-read, so
    /// experts that outweigh the host's memory allowance (the cgroup limit on
    /// containers, where a private 137 GiB copy is a silent OOM SIGKILL)
    /// degrade to storage speed instead of killing the process.
    /// </summary>
    public interface IDsv4MappedWeightSource : IDsv4WeightSource
    {
        /// <summary>
        /// Maps <c>[offset, offset+bytes)</c> of the shard read-only. The
        /// pointer stays valid until the source is disposed, which therefore
        /// must not happen before the engine that borrowed it. False when the
        /// platform or filesystem cannot map; callers fall back to a copy.
        /// </summary>
        bool TryMapRange(long offset, long bytes, out IntPtr ptr);
    }
}
