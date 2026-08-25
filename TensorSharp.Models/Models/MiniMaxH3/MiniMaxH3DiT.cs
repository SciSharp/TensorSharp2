// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 diffusion transformer driver.
//
// The heavy lifting is one native ggml graph per step; this class owns the weight
// binding and the host-side pieces that are cheaper (or clearer) to compute in
// managed code: the patchify/unpatchify reshuffles, the RoPE tables over the
// packed sequence's continuous float positions, and the AdaLN curve-table lookup
// that stands in for a timestep MLP.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>The MiniMax-H3 packed audio-video diffusion transformer.</summary>
    public sealed class MiniMaxH3DiT : IDisposable
    {
        private readonly GgufFile _gguf;
        private readonly List<GCHandle> _pins = new();
        // Weights bound resident on the device are cached by host pointer; the cache
        // has to be told when those pointers stop being valid, or the backend still
        // holds their buffers at teardown.
        private readonly List<IntPtr> _bound = new();
        private readonly H3DitBlockW[] _blocks;
        private readonly H3RefinerBlockW[] _refiner;
        private GCHandle _blocksPin, _refinerPin;
        private H3Lin _videoPatch, _audioPatch, _conditionProj;
        private H3Lin _finalAdaLn, _finalVideoOut, _finalAudioOut;
        private IntPtr _refinerFinalNorm, _finalNorm;
        private readonly float[] _curveTable;   // [grid][timeEmbedDim]
        private bool _disposed;

        public MiniMaxH3Config Config { get; }

        public MiniMaxH3DiT(string ggufPath)
        {
            _gguf = new GgufFile(ggufPath);
            Config = MiniMaxH3Config.Detect(_gguf.Tensors);
            string p = Config.Prefix;

            _videoPatch = Lin(p + "video_patch_proj.weight", p + "video_patch_proj.bias");
            _audioPatch = Lin(p + "audio_patch_proj.weight", p + "audio_patch_proj.bias");
            _conditionProj = Lin(p + "condition_proj.weight", p + "condition_proj.bias");
            _finalAdaLn = Lin(p + "final_layer.adaln_proj.linear.weight",
                              p + "final_layer.adaln_proj.linear.bias");
            _finalVideoOut = Lin(p + "final_layer.video_out.weight", p + "final_layer.video_out.bias");
            _finalAudioOut = Lin(p + "final_layer.audio_out.weight", p + "final_layer.audio_out.bias");
            _finalNorm = PinF32(p + "final_layer.norm.weight");
            _refinerFinalNorm = PinF32(p + "token_refiner.final_norm.weight");

            _refiner = new H3RefinerBlockW[Config.TokenRefinerLayers];
            for (int i = 0; i < _refiner.Length; i++)
            {
                string b = $"{p}token_refiner.blocks.{i}.";
                _refiner[i] = new H3RefinerBlockW
                {
                    Norm1 = PinF32(b + "norm1.weight"),
                    Norm2 = PinF32(b + "norm2.weight"),
                    QNorm = PinF32(b + "attn.q_norm.weight"),
                    KNorm = PinF32(b + "attn.k_norm.weight"),
                    Qkv = Lin(b + "attn.qkv_proj.weight", null),
                    Out = Lin(b + "attn.out_proj.weight", null),
                    Fc1 = Lin(b + "mlp.fc1.weight", null),
                    Fc2 = Lin(b + "mlp.fc2.weight", null),
                };
            }
            _refinerPin = GCHandle.Alloc(_refiner, GCHandleType.Pinned);

            _blocks = new H3DitBlockW[Config.NumLayers];
            for (int i = 0; i < _blocks.Length; i++)
            {
                string b = $"{p}blocks.{i}.";
                _blocks[i] = new H3DitBlockW
                {
                    Norm1 = PinF32(b + "norm1.weight"),
                    Norm2 = PinF32(b + "norm2.weight"),
                    QNorm = PinF32(b + "attn.q_norm.weight"),
                    KNorm = PinF32(b + "attn.k_norm.weight"),
                    AdaLn = Lin(b + "adaln_proj.linear.weight", b + "adaln_proj.linear.bias"),
                    Qkv = Lin(b + "attn.qkv_proj.weight", null),
                    Out = Lin(b + "attn.out_proj.weight", null),
                    Fc1 = Lin(b + "mlp.fc1.weight", null),
                    Fc2 = Lin(b + "mlp.fc2.weight", null),
                };
            }
            _blocksPin = GCHandle.Alloc(_blocks, GCHandleType.Pinned);

            if (Config.TimeEmbedding != MiniMaxH3TimeEmbedding.CurveTable)
                throw new NotSupportedException(
                    "this MiniMax-H3 checkpoint uses the sinusoidal time-embedder variant, " +
                    "which is not implemented; the released checkpoints ship the curve table.");
            // Stored as [timeEmbedDim, grid] in ne order -> [grid][dim] here.
            _curveTable = ReadVectorF32(p + "adaln_t_table");
        }

        // ---- weights --------------------------------------------------------

        private H3Lin Lin(string weight, string bias)
        {
            if (!_gguf.Tensors.TryGetValue(weight, out var info))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            _bound.Add(ptr);
            return new H3Lin
            {
                W = ptr,
                B = bias != null ? PinF32(bias) : IntPtr.Zero,
                Ne0 = (long)info.Shape[0],
                Ne1 = (long)info.Shape[1],
                Bytes = 0,                 // the native side asks ggml for the real size
                Type = (int)info.Type,
            };
        }

        private IntPtr PinF32(string name)
        {
            if (!_gguf.Tensors.ContainsKey(name)) return IntPtr.Zero;
            var h = GCHandle.Alloc(ReadVectorF32(name), GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        /// <summary>Read a small unquantized tensor as F32, converting by dtype.
        /// The norms here are BF16 and the tables F32/F16.</summary>
        private unsafe float[] ReadVectorF32(string name)
        {
            var info = _gguf.Tensors[name];
            long n = info.NumElements;
            var data = new float[n];
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{name}'.");
            switch (info.Type)
            {
                case GgmlTensorType.F32:
                {
                    float* src = (float*)ptr;
                    for (long i = 0; i < n; i++) data[i] = src[i];
                    break;
                }
                case GgmlTensorType.F16:
                {
                    ushort* src = (ushort*)ptr;
                    for (long i = 0; i < n; i++) data[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
                    break;
                }
                case GgmlTensorType.BF16:
                {
                    ushort* src = (ushort*)ptr;
                    for (long i = 0; i < n; i++)
                        data[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
                    break;
                }
                default:
                    throw new NotSupportedException($"'{name}' has unexpected type {info.Type}.");
            }
            return data;
        }

        // ---- host-side helpers ----------------------------------------------

        /// <summary>Interpolate the learned AdaLN curve table at each timestep.
        /// This replaces a sinusoidal timestep MLP entirely.</summary>
        public float[] TimeEmbedding(IReadOnlyList<float> timesteps)
        {
            int dim = Config.TimeEmbedDim, grid = Config.AdaLnCurveGrid;
            var outp = new float[(long)dim * timesteps.Count];
            for (int t = 0; t < timesteps.Count; t++)
            {
                var (lo, hi, frac) = MiniMaxH3Layout.CurveIndex(timesteps[t], grid);
                long dst = (long)t * dim;
                for (int c = 0; c < dim; c++)
                {
                    // ne order [dim, grid] -> row-major [grid][dim] is a transpose.
                    float a = _curveTable[(long)lo * dim + c];
                    float b = _curveTable[(long)hi * dim + c];
                    outp[dst + c] = a + (b - a) * frac;
                }
            }
            return outp;
        }

        /// <summary>Build the 3-axis partial-RoPE tables from the layout's continuous
        /// float positions. 3 axes x <c>RopeInvFreqLength</c> frequencies give
        /// <c>RopeRotaryDim</c>/2 angles, duplicated so pairs (j, j+half) share one.</summary>
        public (float[] Cos, float[] Sin) BuildRope(float[] positions)
        {
            int nf = Config.RopeInvFreqLength;
            int rot = Config.RopeRotaryDim;          // 3 * nf * 2
            int half = rot / 2;                       // 3 * nf
            int tokens = positions.Length / 3;
            var invFreq = ReadVectorF32(Config.Prefix + "rope.inv_freq");
            var cos = new float[(long)rot * tokens];
            var sin = new float[(long)rot * tokens];
            for (int t = 0; t < tokens; t++)
            {
                long b = (long)t * rot;
                for (int axis = 0; axis < 3; axis++)
                {
                    float pos = positions[t * 3 + axis];
                    for (int k = 0; k < nf; k++)
                    {
                        double a = (double)pos * invFreq[k];
                        int j = axis * nf + k;
                        float c = (float)Math.Cos(a), s = (float)Math.Sin(a);
                        cos[b + j] = c; cos[b + j + half] = c;
                        sin[b + j] = s; sin[b + j + half] = s;
                    }
                }
            }
            return (cos, sin);
        }

        /// <summary>Patchify a video latent into transformer tokens.
        /// <paramref name="latent"/> is [T][H][W][C] with the channel axis contiguous;
        /// the result is [tokens][C * patchH * patchW] over the (t, h/2, w/2) grid.
        /// <para>Within a token the 96 values are CHANNEL-MAJOR, patch-minor:
        /// <c>c * (patchH*patchW) + ky * patchW + kx</c>. That is the reference's
        /// "patch_last" layout. Getting it the other way round leaves the RMS
        /// unchanged and silently scrambles every token, which the patch projection
        /// then turns into noise.</para></summary>
        public static float[] PatchifyVideo(float[] latent, int latentT, int latentH, int latentW, int channels)
        {
            int ph = MiniMaxH3Geometry.PatchH, pw = MiniMaxH3Geometry.PatchW;
            int gh = latentH / ph, gw = latentW / pw;
            int patchDim = channels * ph * pw;
            var outp = new float[(long)latentT * gh * gw * patchDim];
            for (int t = 0; t < latentT; t++)
                for (int y = 0; y < gh; y++)
                    for (int xx = 0; xx < gw; xx++)
                    {
                        long dst = (((long)t * gh + y) * gw + xx) * patchDim;
                        for (int ky = 0; ky < ph; ky++)
                            for (int kx = 0; kx < pw; kx++)
                            {
                                long src = (((long)t * latentH + (y * ph + ky)) * latentW
                                            + (xx * pw + kx)) * channels;
                                long spatial = (long)ky * pw + kx;
                                for (int c = 0; c < channels; c++)
                                    outp[dst + (long)c * ph * pw + spatial] = latent[src + c];
                            }
                    }
            return outp;
        }

        /// <summary>Inverse of <see cref="PatchifyVideo"/>.</summary>
        public static float[] UnpatchifyVideo(float[] tokens, int latentT, int latentH, int latentW, int channels)
        {
            int ph = MiniMaxH3Geometry.PatchH, pw = MiniMaxH3Geometry.PatchW;
            int gh = latentH / ph, gw = latentW / pw;
            int patchDim = channels * ph * pw;
            var outp = new float[(long)latentT * latentH * latentW * channels];
            for (int t = 0; t < latentT; t++)
                for (int y = 0; y < gh; y++)
                    for (int xx = 0; xx < gw; xx++)
                    {
                        long src = (((long)t * gh + y) * gw + xx) * patchDim;
                        for (int ky = 0; ky < ph; ky++)
                            for (int kx = 0; kx < pw; kx++)
                            {
                                long dst = (((long)t * latentH + (y * ph + ky)) * latentW
                                            + (xx * pw + kx)) * channels;
                                long spatial = (long)ky * pw + kx;
                                for (int c = 0; c < channels; c++)
                                    outp[dst + c] = tokens[src + (long)c * ph * pw + spatial];
                            }
                    }
            return outp;
        }

        // ---- forward --------------------------------------------------------

        private static H3CondChunk[] ToArray(IReadOnlyList<H3CondChunk> chunks)
        {
            var array = new H3CondChunk[chunks.Count];
            for (int i = 0; i < chunks.Count; i++) array[i] = chunks[i];
            return array;
        }

        /// <summary>One diffusion step. Latents are in token order; the returned
        /// velocities are already sign- and slope-corrected, so a single Euler update
        /// on the video schedule advances both streams.</summary>
        public (float[] Video, float[] Audio) Forward(
            float[] videoTokens, int videoCount,
            float[] audioLatent, int audioCount,
            float[] textHidden, int textCount,
            MiniMaxH3PackedLayout layout,
            float sigma,
            float[] conditionTokens = null, int conditionCount = 0,
            float[] conditionAudio = null, int conditionAudioCount = 0,
            IReadOnlyList<H3CondChunk> conditionChunks = null,
            float videoShift = MiniMaxH3Scheduler.VideoShift,
            float audioShift = MiniMaxH3Scheduler.AudioShift,
            int blockLimit = 0)
        {
            if (layout is null) throw new ArgumentNullException(nameof(layout));

            var segments = new List<H3DitSegment>(layout.ModulationSpans.Count);
            foreach (var span in layout.ModulationSpans)
            {
                // The block projection is viewed as [hidden, 18*nTimesteps]; a row
                // timestep*3+modality maps to column base timestep*18 + modality*6.
                int ts = span.Row / MiniMaxH3Modality.Count;
                int modality = span.Row % MiniMaxH3Modality.Count;
                segments.Add(new H3DitSegment
                {
                    Start = span.Start,
                    End = span.End,
                    Col = ts * (MiniMaxH3Modality.VectorsPerRow * MiniMaxH3Modality.Count)
                          + modality * MiniMaxH3Modality.VectorsPerRow,
                });
            }
            var segArray = segments.ToArray();

            // Conditioning frames share the target's patch projection, so they are
            // handed over as one buffer with the conditions first.
            float[] videoInput = videoTokens;
            if (conditionCount > 0)
            {
                if (conditionTokens is null)
                    throw new ArgumentNullException(nameof(conditionTokens));
                videoInput = new float[conditionTokens.Length + videoTokens.Length];
                Array.Copy(conditionTokens, videoInput, conditionTokens.Length);
                Array.Copy(videoTokens, 0, videoInput, conditionTokens.Length, videoTokens.Length);
            }

            float[] timeEmbed = TimeEmbedding(layout.Timesteps);
            var (cos, sin) = BuildRope(layout.Positions);
            var videoOut = new float[(long)videoCount * Config.VideoPatchDim];
            var audioOut = new float[(long)audioCount * Config.AudioLatentChannels];

            var pins = new List<GCHandle>();
            IntPtr Pin(Array a)
            {
                var h = GCHandle.Alloc(a, GCHandleType.Pinned);
                pins.Add(h);
                return h.AddrOfPinnedObject();
            }
            try
            {
                var args = new H3DitForwardArgs
                {
                    StructBytes = Marshal.SizeOf<H3DitForwardArgs>(),
                    NumBlocks = blockLimit > 0 ? Math.Min(blockLimit, _blocks.Length) : _blocks.Length,
                    NumRefinerBlocks = textCount > 0 ? _refiner.Length : 0,
                    NumSegments = segArray.Length,
                    VideoTokens = Pin(videoInput),
                    AudioTokens = Pin(audioLatent),
                    TextHidden = textCount > 0 ? Pin(textHidden) : IntPtr.Zero,
                    TimeEmbed = Pin(timeEmbed),
                    Cos = Pin(cos),
                    Sin = Pin(sin),
                    VideoOut = Pin(videoOut),
                    AudioOut = Pin(audioOut),
                    VideoPatchProj = _videoPatch,
                    AudioPatchProj = _audioPatch,
                    ConditionProj = _conditionProj,
                    Refiner = _refinerPin.AddrOfPinnedObject(),
                    RefinerFinalNorm = _refinerFinalNorm,
                    Blocks = _blocksPin.AddrOfPinnedObject(),
                    Segments = Pin(segArray),
                    FinalNorm = _finalNorm,
                    FinalAdaLn = _finalAdaLn,
                    FinalVideoOut = _finalVideoOut,
                    FinalAudioOut = _finalAudioOut,
                    NTok = layout.TokenCount,
                    TextCount = textCount,
                    ConditionCount = conditionCount,
                    ConditionAudio = conditionAudioCount > 0 && conditionAudio != null
                        ? Pin(conditionAudio) : IntPtr.Zero,
                    ConditionAudioCount = conditionAudioCount,
                    CondChunks = conditionChunks is { Count: > 0 }
                        ? Pin(ToArray(conditionChunks)) : IntPtr.Zero,
                    NumCondChunks = conditionChunks?.Count ?? 0,
                    AudioStart = layout.AudioSpan.Start,
                    AudioCount = audioCount,
                    VideoStart = layout.VideoSpan.Start,
                    VideoCount = videoCount,
                    // The final layer's projection emits only shift+scale and is
                    // indexed by TIMESTEP row, not timestep*3+modality.
                    AudioCol = layout.AudioSpan.Row * 2,
                    VideoCol = layout.VideoSpan.Row * 2,
                    Hidden = Config.HiddenSize,
                    Heads = Config.NumAttentionHeads,
                    HeadDim = Config.AttentionHeadDim,
                    Inner = Config.AttentionInnerDim,
                    Ffn = Config.FfnHiddenSize,
                    RotDim = Config.RopeRotaryDim,
                    TimeEmbedDim = Config.TimeEmbedDim,
                    NTimesteps = layout.Timesteps.Length,
                    VideoPatchDim = Config.VideoPatchDim,
                    AudioChannels = Config.AudioLatentChannels,
                    TextDim = Config.TextDim,
                    Eps = Config.NormEps,
                    VideoScale = -1f,
                    AudioScale = -MiniMaxH3Scheduler.ShiftSlope(
                        MiniMaxH3Scheduler.ClampSigma(sigma), videoShift, audioShift),
                };
                if (!GgmlBasicOps.TryMiniMaxH3DitForward(in args))
                    throw new InvalidOperationException("MiniMax-H3 DiT forward failed.");
            }
            finally
            {
                foreach (var h in pins) if (h.IsAllocated) h.Free();
            }
            return (videoOut, audioOut);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_blocksPin.IsAllocated) _blocksPin.Free();
            if (_refinerPin.IsAllocated) _refinerPin.Free();
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
            _bound.Clear();
            foreach (var h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
            _gguf?.Dispose();
        }
    }
}
