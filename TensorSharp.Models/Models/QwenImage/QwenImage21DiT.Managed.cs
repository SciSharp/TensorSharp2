// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage;

// The pure-C# (cpu backend) half of QwenImage21DiT: the same descriptors the native graph
// reads (_nativeWeights, _blocks, the LoRA adapter) drive QwenImage21ManagedDiT instead.
internal sealed partial class QwenImage21DiT
{
    // Fixed at construction: predictions route on it, so nothing that happens later (Dispose
    // clears _managed) can send a cpu transformer to the native graph.
    private readonly bool _pureCpu;
    private QwenImage21ManagedDiT _managed;
    private volatile bool _disposed;

    private void PredictManaged(float[] images, float[] textCond, int textSeq, float[] time,
        (QwenImage21Segment[] Segments, float[] Cos, float[] Sin, int Prefix) layout, PrefixCache prefixCache, float[] output)
    {
        // Disposed concurrently after Predict's check: the managed forward also refuses once
        // disposed, under the lock Dispose takes before the weights are unmapped.
        var managed = _managed ?? throw new ObjectDisposedException(nameof(QwenImage21DiT));
        QwenImage21ManagedPrefix store = null;
        if (prefixCache != null) store = prefixCache.Managed ??= new QwenImage21ManagedPrefix(prefixCache.Type);
        var path = managed.Forward(images, images.Length / Channels, textCond, textSeq, time, layout.Cos, layout.Sin,
            layout.Segments, layout.Prefix, layout.Cos.Length / (HeadDim / 2), _lora?.AdapterFor(0) ?? IntPtr.Zero,
            store, output);
        if (prefixCache != null) prefixCache.LastPath = path;
    }

    /// <summary>Releases the pure-C# forward's activation scratch and F32 LoRA factors between
    /// requests (the cpu counterpart of releasing the GGML compute buffers); a no-op on GGML
    /// backends.</summary>
    internal void ReleaseScratch() => _managed?.ReleaseScratch();

    /// <summary>True when predictions run in pure C# (the cpu backend).</summary>
    internal bool IsManaged => _pureCpu;
}
