// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 generation pipeline: prompt -> frames (+ audio latents).
//
// Three networks run in sequence, and they are deliberately not all resident at
// once: the Qwen3-VL encoder is 17 GB and runs exactly once, so it is disposed
// before the 11 GB denoiser starts stepping. That ordering is what makes a 22-frame
// clip fit alongside the VAE on a 48 GB machine.
//
// The denoise loop keeps the video latent in PATCH-token space throughout — the
// model's velocity comes back in that space — and unpatchifies only for the VAE.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Drives a MiniMax-H3 generation request end to end.</summary>
    internal sealed class MiniMaxH3Pipeline : IDisposable
    {
        private readonly MiniMaxH3Model _model;
        private MiniMaxH3DiT _dit;
        private MiniMaxH3VideoVae _vae;
        private MiniMaxH3AudioVae _audioVae;
        private bool _disposed;

        /// <summary>Ceiling on the latent voxels handed to one VAE decode call.
        /// Flash attention means the score matrix is never materialized, so this is a
        /// memory-headroom guard rather than the hard O(n^2) wall it would otherwise
        /// be — the reference tiles at 256 px, we do not need to.</summary>
        private const int MaxVaeTokensPerCall = 16384;

        /// <summary>TS_H3_PHASE=1 prints a per-stage breakdown - encoder open vs trunk,
        /// every denoise step, and VAE open vs decode. The existing one-line summaries
        /// tell you a phase was slow; this tells you which half of it.</summary>
        private static bool PhaseTiming =>
            Environment.GetEnvironmentVariable("TS_H3_PHASE") == "1";

        /// <summary>Sequentially read the denoiser file so its first upload is a copy out of
        /// cache rather than a fault storm. TS_H3_PREFAULT: 0 off, 1 serial, 2 overlapped
        /// with text conditioning, 3 (default) pipelined with the upload itself.
        ///
        /// <para>Measured at 640x384, best and median of three paired runs each:
        /// off 67.2 s, serial 64.4 / 65.8 s, pipelined 63.3 / 63.7 s. Pipelined wins because
        /// the read and the upload walk the file in the same order and the read is faster
        /// (2.7 GB/s against ~2.4 GB/s), so it stays ahead and the upload lands on pages that
        /// are already there; joining first would serialize 4.2 s of read before 4.3 s of
        /// copy.</para>
        ///
        /// <para>Serial is the default because overlapping measured WORSE: 63.7 s against
        /// 60.4 s at 640x384. The prefault itself finishes faster when overlapped (2.2 s
        /// against 4.0 s, since it shares the run with GPU work), but the text encoder it
        /// overlaps with streams its own 17 GB through the same page cache and evicts every
        /// page the prefault just placed - the first denoiser step went back to 13.7 s,
        /// against 7.3 s when the read happens after the encoder has been released. Mode 2
        /// is kept because a host with enough RAM to hold both files would flip that result.</para></summary>
        private static int PrefaultMode
        {
            get
            {
                string v = Environment.GetEnvironmentVariable("TS_H3_PREFAULT");
                if (string.IsNullOrWhiteSpace(v)) return 3;
                return int.TryParse(v, out int m) && m >= 0 && m <= 3 ? m : 3;
            }
        }

        /// <summary>Physical memory free right now, or 0 when it cannot be read. Used only
        /// to decide whether caching a weight file is worth attempting.</summary>
        private static long AvailablePhysicalBytes()
        {
            try
            {
                var info = GC.GetGCMemoryInfo();
                long avail = info.TotalAvailableMemoryBytes - info.MemoryLoadBytes;
                return avail > 0 ? avail : 0;
            }
            catch (PlatformNotSupportedException) { return 0; }
        }

        public MiniMaxH3Pipeline(MiniMaxH3Model model) { _model = model; }

        public GeneratedVideo Generate(string prompt, VideoGenerationParams p)
        {
            p ??= new VideoGenerationParams();
            var total = Stopwatch.StartNew();

            // Resolve the canvas. A 256x256 default is far too small for faces to hold
            // together, so default to the model's own operating area and, when a
            // conditioning image is supplied, take the aspect ratio FROM THE IMAGE —
            // otherwise the frame it is meant to become gets stretched into it.
            var (width, height) = ResolveCanvas(p);
            var shape = MiniMaxH3Geometry.Resolve(
                width, height,
                p.Frames > 0 ? p.Frames : 22,
                p.Fps > 0 ? p.Fps : MiniMaxH3Geometry.Fps);

            // 20 is the reference implementation's default. H3 is step-distilled and
            // runs at 4-8, but measured on a 640x384 image-to-video the low end leaves
            // visible chromatic fringing around moving subjects that is gone by ~20.
            int steps = p.Steps > 0 ? p.Steps : 20;
            float videoShift = p.FlowShift > 0 ? p.FlowShift : MiniMaxH3Scheduler.VideoShift;
            long seed = p.Seed >= 0 ? p.Seed : Random.Shared.NextInt64() & 0x7fffffff;

            if (p.CfgScale > 1.0f)
                throw new ArgumentException(
                    "MiniMax-H3 is CFG-distilled: guidance above 1.0 degrades it badly. " +
                    "Use --cfg 1.0 (the default).", nameof(p));

            var mode = MiniMaxH3ModeResolver.Resolve(p, _model.Partition);
            Report(p, "text-encode", 0, steps, total);

            // Ref2VA reference images are loaded once and used twice: the vision tower
            // reads them for the prompt, and the video VAE encodes them into
            // conditioning latents. Both must see the same pixels.
            List<MiniMaxH3Reference> references = mode == MiniMaxH3Mode.Reference
                ? LoadReferences(p, width, height, shape.Frames)
                : null;

            // A KEYFRAME is conditioned twice, and both halves matter.
            //
            // The VAE-encoded latent pins the frame it is anchored to. On its own that
            // fades: the clip starts on the image and wanders off within a second or
            // two, which is invisible at 22 frames and obvious at 124. The second half
            // is this — the same picture goes through the Qwen3-VL vision tower into
            // the PROMPT, where it describes the scene for the whole clip rather than
            // for one frame. The reference does this for keyframes exactly as it does
            // for Ref2VA references; leaving it out is what made long clips drift.
            var keyframes = new List<RgbImage>();
            var keyframeAtEndFlags = new List<bool>();
            if (mode is MiniMaxH3Mode.ImageToVideo or MiniMaxH3Mode.FirstLastFrame)
            {
                // Fitted to the generation canvas ONCE, here: the same pixels go to the
                // VAE and to the vision tower, which is what the reference presents. The
                // original photo would give the tower a far finer patch grid — a 1024x768
                // source is 768 merged tokens against 108 for a 384x288 canvas — and that
                // is a different conditioning signal, not a better one.
                ResolveKeyframes(p, shape, keyframes, keyframeAtEndFlags);
            }

            // Warming the denoiser file is pure disk work and the text encoder that runs
            // next is pure GPU work, so mode 2 starts it here and collects it just before
            // the first upload. Mode 1 does the same read serially after the encoder is
            // released, which is safer on a host whose page cache cannot hold both files.
            System.Threading.Tasks.Task<double> prefault = null;
            if (PrefaultMode == 2)
            {
                string ditPath = _model.DiTPath;
                long avail = AvailablePhysicalBytes();
                prefault = System.Threading.Tasks.Task.Run(
                    () => MiniMaxH3Model.PrefaultWeightFile(ditPath, avail));
            }

            // ---- text conditioning (then give the 17 GB encoder back) ----
            float[] textHidden;
            int textLength;
            // Per-token modality for the AdaLN stream: plain text is modulated as text,
            // but the tokens standing in for a reference image are modulated as VISUAL.
            List<int> textModalities = null;
            var textClock = Stopwatch.StartNew();
            var teOpenClock = Stopwatch.StartNew();
            double teRunSeconds = 0;
            using (var te = _model.CreateTextEncoder())
            {
                teOpenClock.Stop();
                string presented = prompt ?? string.Empty;
                List<MiniMaxH3PromptImage> promptImages = null;
                if (keyframes.Count > 0)
                {
                    if (!te.HasVision)
                        throw new InvalidOperationException(
                            "image conditioning needs the Qwen3-VL vision tower, and this " +
                            "text-encoder checkpoint does not carry one. Point --video-text-encoder " +
                            "at the full qwen3vl_32b_minimax_h3 GGUF.");
                    var shown = new List<MiniMaxH3Reference>(keyframes.Count);
                    foreach (RgbImage frame in keyframes)
                        shown.Add(new MiniMaxH3Reference
                        {
                            Kind = MiniMaxH3ReferenceKind.Image,
                            Frames = new List<RgbImage> { frame },
                            Presented = new List<RgbImage> { frame },
                            Timestamps = new List<float> { 0f },
                        });
                    (presented, promptImages) = BuildReferencePrompt(te, shown, prompt);
                    Console.WriteLine($"  [h3] {keyframes.Count} keyframe(s) presented to the " +
                                      $"vision tower -> {promptImages.Count} vision span(s)");
                }
                else if (references is { Count: > 0 })
                {
                    if (!te.HasVision)
                        throw new InvalidOperationException(
                            "reference conditioning needs the Qwen3-VL vision tower, and this " +
                            "text-encoder checkpoint does not carry one. Point --video-text-encoder " +
                            "at the full qwen3vl_32b_minimax_h3 GGUF.");
                    (presented, promptImages) = BuildReferencePrompt(te, references, prompt);
                    Console.WriteLine($"  [h3] ref2va: {references.Count} reference(s) -> " +
                                      $"{promptImages.Count} vision span(s)");
                }
                var ids = te.Tokenize(presented);
                if (ids.Count == 0) ids.Add(te.Tokenizer.BosTokenId);
                textLength = ids.Count;
                var teRunClock = Stopwatch.StartNew();
                textHidden = te.Encode(ids, promptImages);
                teRunClock.Stop();
                teRunSeconds = teRunClock.Elapsed.TotalSeconds;
                // The trunk is finished with, so the denoiser read can start now and run
                // alongside the encoder's teardown. Releasing 551 per-tensor device
                // buffers is ~2 s of driver work (cudaFree measures 1.15 ms apiece on this
                // card) and the read is ~4 s of disk; they contend for nothing, so
                // whichever is shorter comes for free. The denoiser cannot upload until
                // BOTH are done - it needs the VRAM the teardown returns and the pages the
                // read places - which is exactly the join below.
                if (PrefaultMode == 1 || PrefaultMode == 3)
                {
                    string ditPath = _model.DiTPath;
                    long avail = AvailablePhysicalBytes();
                    prefault = System.Threading.Tasks.Task.Run(
                        () => MiniMaxH3Model.PrefaultWeightFile(ditPath, avail));
                }
                if (PhaseTiming)
                    Console.WriteLine($"  [h3] .. encoder open {teOpenClock.Elapsed.TotalSeconds:F1}s, "
                                      + $"trunk {teRunClock.Elapsed.TotalSeconds:F1}s");
                if (promptImages is { Count: > 0 })
                    textModalities = BuildTextModalities(textLength, promptImages);
            }

            textClock.Stop();
            if (PhaseTiming)
                Console.WriteLine($"  [h3] .. encoder teardown "
                                  + $"{textClock.Elapsed.TotalSeconds - teOpenClock.Elapsed.TotalSeconds - teRunSeconds:F1}s");
            Console.WriteLine($"  [h3] text conditioning: {textLength} tokens in " +
                              $"{textClock.Elapsed.TotalSeconds:F1}s");

            // The encoder has just handed its ~17 GB back, so this is the moment the host has
            // room to cache the denoiser. Sequential read now beats scattered fault-in during
            // the upload; see MiniMaxH3Model.PrefaultWeightFile.
            // Mode 3 deliberately does NOT join here. The read walks the file in the same
            // order the upload walks tensors, and it moves faster (2.7 GB/s sequential
            // against ~2.4 GB/s for the copy), so leaving it running lets it stay ahead
            // and the upload finds pages already placed. Joining first serializes the two
            // - 4.2 s of read THEN 4.3 s of copy - where running them together costs
            // about the longer of the pair.
            if (prefault != null && PrefaultMode != 3)
            {
                var wait = Stopwatch.StartNew();
                double pf = prefault.GetAwaiter().GetResult();
                wait.Stop();
                if (PhaseTiming)
                    Console.WriteLine(pf >= 0
                        ? $"  [h3] .. denoiser prefault {pf:F1}s overlapped, "
                          + $"{wait.Elapsed.TotalSeconds:F1}s still to wait"
                        : "  [h3] .. denoiser prefault skipped (not enough free RAM)");
            }


            var ditClock = Stopwatch.StartNew();
            _dit ??= _model.CreateDiT();
            ditClock.Stop();
            Console.WriteLine($"  [h3] denoiser loaded in {ditClock.Elapsed.TotalSeconds:F1}s");

            // ---- conditioning frames ----
            // A keyframe is NOT a mask: it becomes extra sequence tokens at a
            // near-clean timestep, pinned to the start or end of the timeline. The
            // frames are resized to the generation canvas so they share the target's
            // spatial axes.
            float[] conditionTokens = null;
            int conditionCount = 0;
            var conditionFrames = new List<int>();
            var keyframeAtEnd = new List<bool>();
            if (mode is MiniMaxH3Mode.ImageToVideo or MiniMaxH3Mode.FirstLastFrame)
            {
                _vae ??= _model.CreateVideoVae();
                var parts = new List<float[]>();
                void AddKeyframe(RgbImage scaled, bool atEnd)
                {
                    float[] latent = _vae.EncodeFrame(scaled);
                    _vae.NormalizeLatent(latent);
                    AddVisualNoise(latent, seed);
                    parts.Add(MiniMaxH3DiT.PatchifyVideo(latent, 1, shape.LatentHeight,
                        shape.LatentWidth, MiniMaxH3Geometry.VideoLatentChannels));
                    conditionFrames.Add(1);
                    keyframeAtEnd.Add(atEnd);
                }

                for (int i = 0; i < keyframes.Count; i++)
                    AddKeyframe(keyframes[i], keyframeAtEndFlags[i]);

                int perFrame = shape.TokenGridWidth * shape.TokenGridHeight;
                conditionCount = parts.Count * perFrame;
                conditionTokens = new float[(long)conditionCount * _dit.Config.VideoPatchDim];
                long off = 0;
                foreach (var part in parts) { Array.Copy(part, 0, conditionTokens, off, part.Length); off += part.Length; }
                Console.WriteLine($"  [h3] {mode}: {parts.Count} keyframe(s) -> {conditionCount} conditioning tokens");
            }

            // ---- Ref2VA reference latents ----
            // A reference is not a frame the clip must reproduce: it takes its own
            // stretch of the shared timeline before the generated video, keeps its own
            // aspect ratio, and is noised to the same near-clean timestep as a keyframe.
            List<MiniMaxH3ReferenceBlock> referenceBlocks = null;
            float[] conditionAudio = null;
            int conditionAudioCount = 0;
            List<H3CondChunk> condChunks = null;
            if (references is { Count: > 0 })
            {
                referenceBlocks = new List<MiniMaxH3ReferenceBlock>(references.Count);
                condChunks = new List<H3CondChunk>();
                var videoParts = new List<float[]>();
                var audioParts = new List<float[]>();
                int videoIndex = 0, audioIndex = 0;

                foreach (var reference in references)
                {
                    int thisAudio = -1, thisVideo = -1;
                    int audioLatentFrames = 0, latentH = 0, latentW = 0, latentT = 0;

                    if (reference.Audio != null)
                    {
                        _audioVae ??= _model.CreateAudioVae();
                        if (_audioVae == null || !_audioVae.HasEncoder)
                            throw new InvalidOperationException(
                                "reference audio needs the audio VAE and its encoder; point " +
                                "--audio-vae at minimax_h3_audio_vae_fp32.safetensors.");
                        audioLatentFrames = MiniMaxH3AudioVae.FramesFor(reference.Audio[0].Length);
                        var planes = new List<float[]>(reference.Audio.Length);
                        foreach (float[] channel in reference.Audio)
                        {
                            float[] plane = _audioVae.EncodeMono(channel);
                            _audioVae.NormalizePlane(plane, audioLatentFrames);
                            planes.Add(plane);
                        }
                        // The DiT wants [channel][t][32]; the encoder emits [32][t].
                        var tokens = new float[(long)planes.Count * audioLatentFrames
                                               * MiniMaxH3AudioVae.LatentChannels];
                        for (int ch = 0; ch < planes.Count; ch++)
                            for (int t = 0; t < audioLatentFrames; t++)
                                for (int c = 0; c < MiniMaxH3AudioVae.LatentChannels; c++)
                                    tokens[((long)ch * audioLatentFrames + t)
                                           * MiniMaxH3AudioVae.LatentChannels + c] =
                                        planes[ch][(long)c * audioLatentFrames + t];
                        // No visual noise here: only the PICTURE conditioning is nudged
                        // off the clean manifold, and the reference implementation
                        // leaves an encoded soundtrack exactly as it came out.
                        audioParts.Add(tokens);
                        thisAudio = audioIndex++;
                        conditionAudioCount += audioLatentFrames * MiniMaxH3Geometry.AudioChannels;
                        condChunks.Add(new H3CondChunk
                        {
                            Kind = 1,
                            Count = audioLatentFrames * MiniMaxH3Geometry.AudioChannels,
                        });
                    }

                    if (reference.Kind != MiniMaxH3ReferenceKind.Audio)
                    {
                        var encodeClock = Stopwatch.StartNew();
                        _vae ??= _model.CreateVideoVae();
                        RgbImage first = reference.Frames[0];
                        latentH = first.Height / MiniMaxH3Geometry.VaeSpatialRatio;
                        latentW = first.Width / MiniMaxH3Geometry.VaeSpatialRatio;
                        float[] refLatent = reference.Frames.Count == 1
                            ? EncodeSingle(first, out latentT)
                            : _vae.EncodeClip(reference.Frames, out latentT);
                        _vae.NormalizeLatent(refLatent);
                        AddVisualNoise(refLatent, seed);
                        videoParts.Add(MiniMaxH3DiT.PatchifyVideo(refLatent, latentT, latentH, latentW,
                            MiniMaxH3Geometry.VideoLatentChannels));
                        thisVideo = videoIndex++;
                        encodeClock.Stop();
                        Console.WriteLine($"  [h3] encoded reference {thisVideo}: " +
                                          $"{reference.Frames.Count}x{first.Width}x{first.Height} -> " +
                                          $"{latentT} latent frame(s) in " +
                                          $"{encodeClock.Elapsed.TotalSeconds:F1}s");
                        condChunks.Add(new H3CondChunk
                        {
                            Kind = 0,
                            Count = latentT * (latentH / MiniMaxH3Geometry.PatchH)
                                            * (latentW / MiniMaxH3Geometry.PatchW),
                        });
                    }

                    referenceBlocks.Add(new MiniMaxH3ReferenceBlock
                    {
                        Kind = reference.Kind,
                        VideoIndex = thisVideo,
                        AudioIndex = thisAudio,
                        LatentFrames = Math.Max(1, latentT),
                        LatentHeight = latentH,
                        LatentWidth = latentW,
                        AudioLatentFrames = audioLatentFrames,
                    });
                }

                conditionTokens = Concat(videoParts);
                conditionCount = conditionTokens.Length == 0
                    ? 0 : conditionTokens.Length / _dit.Config.VideoPatchDim;
                conditionAudio = Concat(audioParts);
                if (conditionAudio.Length == 0) { conditionAudio = null; conditionAudioCount = 0; }
                Console.WriteLine($"  [h3] {mode}: {references.Count} reference(s) -> " +
                                  $"{conditionCount} visual + {conditionAudioCount} audio " +
                                  "conditioning tokens");
            }

            // ---- initial noise ----
            // sigma[0] is 1.0, so noise_scaling(init=0) reduces to pure noise.
            var rng = new PhiloxLikeRng(seed);
            int videoCount = shape.VideoTokenCount;
            int audioCount = shape.AudioTokenCount;
            var videoPatches = new float[(long)videoCount * _dit.Config.VideoPatchDim];
            var audioLatent = new float[(long)audioCount * _dit.Config.AudioLatentChannels];
            for (long i = 0; i < videoPatches.LongLength; i++) videoPatches[i] = rng.NextNormal();
            for (long i = 0; i < audioLatent.LongLength; i++) audioLatent[i] = rng.NextNormal();

            float[] sigmas = MiniMaxH3Scheduler.BuildSigmas(steps, videoShift);

            for (int step = 0; step < steps; step++)
            {
                float sigma = sigmas[step];
                var layout = MiniMaxH3Layout.Build(textLength, shape, sigma,
                    conditionFrames: conditionFrames.Count > 0 ? conditionFrames : null,
                    keyframeAtEnd: keyframeAtEnd.Count > 0 ? keyframeAtEnd : null,
                    textTokenModalities: textModalities,
                    videoShift: videoShift, audioShift: MiniMaxH3Scheduler.AudioShift,
                    references: referenceBlocks);

                if (step == 0) RequireChunksMatchLayout(condChunks, layout);

                var stepClock = Stopwatch.StartNew();
                var (vVel, aVel) = _dit.Forward(
                    videoPatches, videoCount, audioLatent, audioCount,
                    textHidden, textLength, layout, sigma,
                    conditionTokens, conditionCount,
                    conditionAudio, conditionAudioCount, condChunks, videoShift);

                // Both streams share one trunk, so one bad number takes the whole
                // clip down; check them before it is integrated into the latent.
                RequireFinite(vVel, "video velocity", step, steps, layout.TokenCount);
                RequireFinite(aVel, "audio velocity", step, steps, layout.TokenCount);

                MiniMaxH3Scheduler.EulerStep(videoPatches, vVel, sigma, sigmas[step + 1]);
                MiniMaxH3Scheduler.EulerStep(audioLatent, aVel, sigma, sigmas[step + 1]);

                p.OnStep?.Invoke(step + 1, steps);
                Report(p, "denoise", step + 1, steps, total);
                if (step == 0)
                    Console.WriteLine($"  [h3] {layout.TokenCount} packed tokens; " +
                                      $"first step in {total.Elapsed.TotalSeconds - textClock.Elapsed.TotalSeconds:F1}s");
                if (PhaseTiming)
                    Console.WriteLine($"  [h3] .. step {step + 1}/{steps} {stepClock.Elapsed.TotalSeconds:F2}s");
                if (TraceEnabled)
                    Console.WriteLine(
                        $"  [h3] step {step + 1}/{steps} sigma={sigma:F4} " +
                        $"| video latent rms={Rms(videoPatches):F3} absmax={AbsMax(videoPatches):F2} " +
                        $"velocity rms={Rms(vVel):F3} absmax={AbsMax(vVel):F2} " +
                        $"| audio latent rms={Rms(audioLatent):F3} velocity rms={Rms(aVel):F3}");
            }

            // The pipelined read (mode 3) is collected here: the denoiser has finished
            // with the file either way, and a Task holding a FileStream must never be
            // left dangling.
            if (prefault != null && PrefaultMode == 3)
            {
                double pf3 = prefault.GetAwaiter().GetResult();
                if (PhaseTiming)
                    Console.WriteLine(pf3 >= 0
                        ? $"  [h3] .. denoiser prefault {pf3:F1}s, pipelined with the upload"
                        : "  [h3] .. denoiser prefault skipped (not enough free RAM)");
            }

            // ---- decode video ----
            Report(p, "vae-decode", steps, steps, total);
            ReleaseDenoiserIfVaeWouldNotFit();
            // The VAE is deliberately NOT prefaulted. Measured at 640x384 the read costs
            // 1.9 s and the decode does not move (9.4 s against 9.1 s), so it is a straight
            // loss: at 4.85 GB the VAE is small enough that its upload is not fault-bound the
            // way the 10.6 GB denoiser is, and its decode is compute-bound rather than
            // transfer-bound. Prefaulting pays only where a large file meets a first-touch
            // upload.
            var vaeOpenClock = Stopwatch.StartNew();
            _vae ??= _model.CreateVideoVae();
            vaeOpenClock.Stop();

            float[] latent = MiniMaxH3DiT.UnpatchifyVideo(
                videoPatches, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth,
                MiniMaxH3Geometry.VideoLatentChannels);
            _vae.DenormalizeLatent(latent);

            var vaeRunClock = Stopwatch.StartNew();
            float[] pixels = _vae.Decode(
                latent, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth,
                (tile, tiles) => Report(p, "vae-decode", steps, steps, total));
            vaeRunClock.Stop();
            if (PhaseTiming)
                Console.WriteLine($"  [h3] .. video VAE open {vaeOpenClock.Elapsed.TotalSeconds:F1}s, "
                                  + $"decode {vaeRunClock.Elapsed.TotalSeconds:F1}s");

            // The decoder now hands back exactly the clip's frames, warm-up and
            // chunk overlaps already resolved.
            int decodedFrames = MiniMaxH3VideoVae.DecodedFrameCount(shape.LatentFrames);
            MiniMaxH3VideoVae.ToRgb(pixels, decodedFrames, shape.Height, shape.Width);
            RgbImage[] frames = TakeFrames(pixels, decodedFrames, shape);

            // ---- decode the joint audio track ----
            // H3 denoises audio alongside video whether or not we decode it, so this
            // is purely the vocoder step; without the audio VAE the clip is simply
            // silent rather than wrong.
            GeneratedVideoAudio audio = null;
            if (p.GenerateAudio)
            {
                _audioVae ??= _model.CreateAudioVae();
                if (_audioVae != null)
                {
                    Report(p, "audio-decode", steps, steps, total);
                    audio = _audioVae.DecodeStereo(audioLatent, shape.AudioLatentFrames);
                    // The audio grid is 40 Hz and the video grid 24 fps, so the tracks
                    // rarely land on the same length; trim to the video's duration.
                    int wanted = (int)Math.Round(shape.Frames / (double)shape.Fps * audio.SampleRate);
                    if (audio.SampleCount > wanted && wanted > 0)
                    {
                        var trimmed = new float[audio.ChannelCount][];
                        for (int c = 0; c < audio.ChannelCount; c++)
                        {
                            trimmed[c] = new float[wanted];
                            Array.Copy(audio.Channels[c], trimmed[c], wanted);
                        }
                        audio = new GeneratedVideoAudio
                        { Channels = trimmed, SampleRate = audio.SampleRate };
                    }
                }
            }

            total.Stop();
            Report(p, "done", steps, steps, total);
            return new GeneratedVideo
            {
                Frames = frames,
                Fps = shape.Fps,
                Seed = seed,
                Audio = audio,
            };
        }

        /// <summary>One resolved Ref2VA reference: the media it carries, how it is
        /// presented to the language model, and (once encoded) the shape of its
        /// latents.</summary>
        internal sealed class MiniMaxH3Reference
        {
            public MiniMaxH3ReferenceKind Kind { get; set; }
            /// <summary>Frames at the VAE's canvas — one for an image, a clip for a video.</summary>
            public List<RgbImage> Frames { get; init; }
            /// <summary>The subset shown to the vision tower: every frame for an image,
            /// every 12th (i.e. 2 fps) for a video.</summary>
            public List<RgbImage> Presented { get; init; }
            /// <summary>Seconds into the clip for each presented frame.</summary>
            public List<float> Timestamps { get; init; }
            /// <summary>Soundtrack as channel planes at the audio VAE's rate.</summary>
            public float[][] Audio { get; init; }
        }

        /// <summary>Pixel frames between presented frames: 24 fps / 12 = the 2 fps the
        /// language model sees a reference clip at.</summary>
        private const int PresentationStride = 12;

        /// <summary>Resolve every reference input into media the encoders can consume.</summary>
        internal static List<MiniMaxH3Reference> LoadReferences(
            VideoGenerationParams p, int width, int height, int maxFrames)
        {
            var refs = new List<MiniMaxH3Reference>();

            foreach (RgbImage image in LoadReferenceImages(p, width, height)
                                       ?? new List<RgbImage>())
            {
                refs.Add(new MiniMaxH3Reference
                {
                    Kind = MiniMaxH3ReferenceKind.Image,
                    Frames = new List<RgbImage> { image },
                    Presented = new List<RgbImage> { image },
                    Timestamps = new List<float> { 0f },
                });
            }

            var videos = p.ReferenceVideoPaths;
            var videoAudios = p.ReferenceVideoAudioPaths;
            for (int i = 0; videos != null && i < videos.Count; i++)
            {
                var clip = LoadReferenceVideo(videos[i], maxFrames);
                // A soundtrack pairs with its clip BY INDEX, which is why they are two
                // lists rather than one: a video file's own audio track is not readable
                // through the frame decoder.
                float[][] audio = null;
                if (videoAudios != null && i < videoAudios.Count &&
                    !string.IsNullOrWhiteSpace(videoAudios[i]))
                {
                    audio = LoadReferenceAudio(videoAudios[i], clip.Count, MiniMaxH3Geometry.Fps);
                }
                refs.Add(new MiniMaxH3Reference
                {
                    Kind = audio != null ? MiniMaxH3ReferenceKind.VideoAudio : MiniMaxH3ReferenceKind.Video,
                    Frames = clip,
                    Presented = Subsample(clip, PresentationStride),
                    Timestamps = Stamps(clip.Count, PresentationStride),
                    Audio = audio,
                });
            }

            var audios = p.ReferenceAudioPaths;
            for (int i = 0; audios != null && i < audios.Count; i++)
            {
                refs.Add(new MiniMaxH3Reference
                {
                    Kind = MiniMaxH3ReferenceKind.Audio,
                    Audio = LoadReferenceAudio(audios[i], maxFrames, MiniMaxH3Geometry.Fps),
                });
            }
            return refs;
        }

        private static List<RgbImage> Subsample(List<RgbImage> frames, int stride)
        {
            var outp = new List<RgbImage>();
            for (int i = 0; i < frames.Count; i += stride) outp.Add(frames[i]);
            return outp;
        }

        private static List<float> Stamps(int frameCount, int stride)
        {
            var outp = new List<float>();
            for (int i = 0; i < frameCount; i += stride)
                outp.Add(i / (float)MiniMaxH3Geometry.Fps);
            return outp;
        }

        /// <summary>Load a reference clip onto H3's own 24 fps timeline and canvas.
        ///
        /// <para>The canvas is derived from the aspect ratio around a nominal 768, then
        /// capped by area and snapped to 32 — the reference's own rule, which does NOT
        /// take the generated clip's size. The frame count is pulled down onto the
        /// 17k+5 grid because that is the only length the causal encoder maps cleanly
        /// onto latent frames.</para></summary>
        private static List<RgbImage> LoadReferenceVideo(string path, int maxFrames)
        {
            var (files, sourceFps) = MediaHelper.ExtractFramesAtRate(
                path, MiniMaxH3Geometry.Fps, maxFrames: 0);

            int frames = Math.Min(files.Count, Math.Max(maxFrames, MiniMaxH3Geometry.MinFrames));
            if (frames < MiniMaxH3Geometry.MinFrames)
                throw new ArgumentException(
                    $"'{path}' gives {frames} frames at 24 fps; a reference clip needs at least " +
                    $"{MiniMaxH3Geometry.MinFrames}.", nameof(path));
            // Down onto the grid: 5, 22, 39, ... Anything between rungs cannot become a
            // whole number of latent frames.
            while (frames % 17 != 5) frames--;

            RgbImage first = ImageIO.Load(files[0]);
            var (w, h) = ReferenceVideoCanvas(first.Width, first.Height);
            var clip = new List<RgbImage>(frames);
            for (int i = 0; i < frames; i++)
            {
                RgbImage src = i == 0 ? first : ImageIO.Load(files[i]);
                clip.Add(src.Width == w && src.Height == h ? src : ImageIO.Resize(src, w, h));
            }
            Console.WriteLine($"  [h3] reference video: {files.Count} frames at 24 fps " +
                              $"(source {sourceFps:F2} fps) -> {frames} frames at {w}x{h}");
            return clip;
        }

        /// <summary>The reference-clip canvas: aspect ratio around a nominal 768, capped
        /// at 768x1344 of area, snapped to the VAE's 32-pixel grid.
        ///
        /// <para>A source SMALLER than that keeps its own size instead of being blown
        /// up to it. Upscaling invents no detail for the model to reference and the
        /// cost is steep — a 448x320 clip would otherwise become 1088x768, which is
        /// 5712 conditioning tokens against 1680 for the clip being generated.</para></summary>
        internal static (int Width, int Height) ReferenceVideoCanvas(int sourceWidth, int sourceHeight)
        {
            const double nominal = 768.0, maxArea = 768.0 * 1344.0;
            double ratio = sourceWidth / (double)sourceHeight;
            double w = ratio >= 1.0 ? nominal * ratio : nominal;
            double h = ratio >= 1.0 ? nominal : nominal / ratio;
            if (w * h > maxArea)
            {
                double scale = Math.Sqrt(maxArea / (w * h));
                w *= scale; h *= scale;
            }
            if ((double)sourceWidth * sourceHeight < w * h)
                return (SnapToVaeGrid(sourceWidth), SnapToVaeGrid(sourceHeight));
            return (SnapToVaeGrid(w), SnapToVaeGrid(h));
        }

        /// <summary>Decode a reference soundtrack to the audio VAE's stereo 32 kHz, and
        /// trim it to the duration the clip covers — a longer reference is truncated,
        /// not stretched.</summary>
        private static float[][] LoadReferenceAudio(string path, int frames, int fps)
        {
            var decoded = AudioIO.Decode(path, MiniMaxH3AudioVae.SampleRate,
                                         MiniMaxH3Geometry.AudioChannels);
            int wanted = (int)Math.Round(frames / (double)fps * MiniMaxH3AudioVae.SampleRate);
            // Whole latent frames only: the encoder's strided convolutions land on a
            // frame boundary or not at all.
            wanted = Math.Max(MiniMaxH3AudioVae.TotalUpsample,
                              wanted / MiniMaxH3AudioVae.TotalUpsample * MiniMaxH3AudioVae.TotalUpsample);
            var planes = new float[decoded.ChannelCount][];
            for (int c = 0; c < planes.Length; c++)
            {
                planes[c] = new float[wanted];
                Array.Copy(decoded.Channels[c], planes[c],
                           Math.Min(wanted, decoded.Channels[c].Length));
            }
            return planes;
        }

        /// <summary>Load the Ref2VA reference images and fit each to the grid the video
        /// VAE needs.
        ///
        /// <para>References are only ever scaled DOWN, to the generated clip's area,
        /// and keep their own aspect ratio: a reference exists to carry identity and
        /// appearance, so distorting it to the output canvas would work against the one
        /// thing it is there for.</para></summary>
        internal static List<RgbImage> LoadReferenceImages(VideoGenerationParams p, int width, int height)
        {
            double targetArea = (double)width * height;
            RgbImage Fit(RgbImage image)
            {
                double scale = Math.Min(1.0, Math.Sqrt(targetArea / ((double)image.Width * image.Height)));
                int w = SnapToVaeGrid(image.Width * scale);
                int h = SnapToVaeGrid(image.Height * scale);
                return w == image.Width && h == image.Height ? image : ImageIO.Resize(image, w, h);
            }

            var paths = p.ReferenceImagePaths;
            if (paths is not { Count: > 0 })
            {
                // A client that only knows how to attach "an image" still gets a
                // reference when the Ref2VA checkpoint is loaded; the mode resolver
                // has already decided that is what the image means. A request that
                // named clips or soundtracks explicitly is not that client, so a
                // stray --image does not silently become an extra reference.
                bool namedOthers = (p.ReferenceVideoPaths?.Count ?? 0) > 0
                                   || (p.ReferenceAudioPaths?.Count ?? 0) > 0;
                RgbImage plain = namedOthers ? null : p.ResolveImage();
                return plain == null ? null : new List<RgbImage> { Fit(plain) };
            }

            // The published Ref2VA takes at most nine references; more is a mistake
            // worth naming rather than a sequence quietly ballooning past what the
            // model was trained on.
            if (paths.Count > MiniMaxH3Geometry.MaxReferences)
                throw new ArgumentException(
                    $"MiniMax-H3 Ref2VA accepts at most {MiniMaxH3Geometry.MaxReferences} " +
                    $"reference images; {paths.Count} were supplied.", nameof(p));

            var images = new List<RgbImage>(paths.Count);
            foreach (string path in paths) images.Add(Fit(ImageIO.Load(path)));
            return images;
        }

        /// <summary>Snap a reference dimension onto the VAE's 32-pixel grid, breaking ties
        /// the way the reference implementation does.
        ///
        /// <para>The tie-break is asked for explicitly because the two languages disagree:
        /// the reference computes <c>(int)std::round(size * scale / 32.f) * 32</c>, and
        /// <c>std::round</c> goes half AWAY FROM ZERO, while .NET's <see cref="Math.Round(double)"/>
        /// goes half to EVEN. A reference whose scaled height lands on 2.5 grid units
        /// becomes 96 px there and 64 px here — a whole latent row's difference in the
        /// conditioning, from a default nobody chose. Same reasoning as
        /// <see cref="FormatSeconds"/>, opposite direction.</para></summary>
        private static int SnapToVaeGrid(double value) =>
            Math.Max(MiniMaxH3Geometry.SpatialMultiple,
                     (int)Math.Round(value / MiniMaxH3Geometry.SpatialMultiple,
                                     MidpointRounding.AwayFromZero)
                        * MiniMaxH3Geometry.SpatialMultiple);

        /// <summary>Present the references to the language model and run the vision
        /// tower over each.
        ///
        /// <para>An image becomes a numbered picture; a clip becomes a numbered video
        /// whose frames are shown two at a time, each pair stamped with the time it
        /// sits at; a soundtrack contributes only its label, because the language model
        /// never hears it — the audio reaches the denoiser as conditioning latents
        /// instead. Every visual block is followed by one placeholder token per merged
        /// patch, and those placeholders are what the tower's output later replaces.</para></summary>
        internal static (string Prompt, List<MiniMaxH3PromptImage> Images) BuildReferencePrompt(
            MiniMaxH3TextEncoder te, IReadOnlyList<MiniMaxH3Reference> references, string text)
        {
            const string visionStart = "<|vision_start|>";
            const string visionEnd = "<|vision_end|>";
            const string placeholder = "<|image_pad|>";

            var builder = new System.Text.StringBuilder();
            var images = new List<MiniMaxH3PromptImage>();

            void AddBlock(RgbImage first, RgbImage second)
            {
                MiniMaxH3VisionOutput vision = te.Vision.Encode(first, second);
                builder.Append(visionStart);
                images.Add(new MiniMaxH3PromptImage
                {
                    TokenIndex = te.Tokenize(builder.ToString()).Count,
                    Vision = vision,
                });
                for (int t = 0; t < vision.TokenCount; t++) builder.Append(placeholder);
                builder.Append(visionEnd);
            }

            int pictures = 0, videos = 0, sounds = 0;
            foreach (var reference in references)
            {
                switch (reference.Kind)
                {
                    case MiniMaxH3ReferenceKind.Audio:
                        builder.Append("<Audio ").Append(++sounds).Append(">: ");
                        break;

                    case MiniMaxH3ReferenceKind.Image:
                        builder.Append("<Picture ").Append(++pictures).Append(">: ");
                        AddBlock(reference.Presented[0], null);
                        break;

                    default:
                        // A clip that came with its own soundtrack is presented as TWO
                        // things, not one: the reference pushes an AUDIO item and then a
                        // VIDEO item, so the prompt reads "<Audio 1>: <Video 1>: ...".
                        // Emitting only the video label also stopped the audio counter,
                        // which then renumbered every later standalone soundtrack — a
                        // divergence that cannot appear with a single reference and grows
                        // with each extra one.
                        if (reference.Kind == MiniMaxH3ReferenceKind.VideoAudio)
                            builder.Append("<Audio ").Append(++sounds).Append(">: ");
                        builder.Append("<Video ").Append(++videos).Append(">: ");
                        var shown = reference.Presented;
                        for (int i = 0; i < shown.Count; i += 2)
                        {
                            int next = Math.Min(i + 1, shown.Count - 1);
                            float t0 = reference.Timestamps[i];
                            float t1 = reference.Timestamps[next];
                            builder.Append('<').Append(FormatSeconds((t0 + t1) * 0.5f))
                                   .Append(" seconds>");
                            AddBlock(shown[i], i == next ? null : shown[next]);
                        }
                        break;
                }
            }
            builder.Append(text ?? string.Empty);

            string prompt = builder.ToString();
            int length = te.Tokenize(prompt).Count;
            foreach (var image in images)
            {
                if (image.TokenIndex + image.Vision.TokenCount > length)
                    throw new InvalidOperationException(
                        "the reference placeholders did not tokenize one-to-one; " +
                        "the tokenizer is missing the <|image_pad|> special token.");
            }
            return (prompt, images);
        }

        /// <summary>Format a timestamp exactly as the reference's C++ stream does.
        ///
        /// <para>Presented frames sit 12 apart at 24 fps, so a pair's midpoint is always
        /// an exact quarter-second — every value lands precisely on a rounding tie.
        /// C++ streams break ties to EVEN, and .NET's default breaks them away from
        /// zero, so "0.25" would become "0.3" here and "0.2" there. One character of
        /// prompt is one different token, so the tie-break is asked for explicitly.</para></summary>
        internal static string FormatSeconds(float seconds) =>
            Math.Round(seconds, 1, MidpointRounding.ToEven)
                .ToString("F1", System.Globalization.CultureInfo.InvariantCulture);

        /// <summary>Fit the request's keyframes to the generation canvas, in timeline
        /// order (start then end).
        ///
        /// <para>Fitted ONCE, here: the same pixels go to the VAE and to the vision
        /// tower, which is what the reference presents. Handing the tower the original
        /// photo instead gives it a far finer patch grid — a 1024x768 source is 768
        /// merged tokens against 108 for a 384x288 canvas — and that is a different
        /// conditioning signal, not a better one.</para></summary>
        internal static void ResolveKeyframes(VideoGenerationParams p, MiniMaxH3Shape shape,
                                              List<RgbImage> frames, List<bool> atEnd)
        {
            RgbImage first = p.ResolveImage();
            RgbImage last = p.ResolveEndImage();
            if (first != null)
            {
                WarnOnAspectCrop(first, shape);
                frames.Add(ImageIO.ResizeCover(first, shape.Width, shape.Height));
                atEnd.Add(false);
            }
            if (last != null)
            {
                frames.Add(ImageIO.ResizeCover(last, shape.Width, shape.Height));
                atEnd.Add(true);
            }
        }

        /// <summary>Say so when the canvas forces a crop. Centre-cropping is the right
        /// trade against stretching — a squeezed face is baked into every frame — but it
        /// silently loses the edges of the shot, so it should never be a surprise.</summary>
        private static void WarnOnAspectCrop(RgbImage img, MiniMaxH3Shape shape)
        {
            if (img.Width == shape.Width && img.Height == shape.Height) return;
            double want = shape.Width / (double)shape.Height;
            double have = img.Width / (double)img.Height;
            if (Math.Abs(want - have) / have <= 0.02) return;
            Console.WriteLine(
                $"  [h3] conditioning image is {img.Width}x{img.Height} ({have:F2}:1) but the " +
                $"canvas is {shape.Width}x{shape.Height} ({want:F2}:1); centre-cropping to fit. " +
                "Leave --width/--height unset to take the canvas from the image instead.");
        }

        /// <summary>Mark which prompt tokens are modulated on the visual stream.
        ///
        /// <para>The span covers the placeholders <i>and</i> the surrounding
        /// <c>&lt;|vision_start|&gt;</c> / <c>&lt;|vision_end|&gt;</c> markers, so the
        /// AdaLN switch happens one token early and ends one token late.</para></summary>
        internal static List<int> BuildTextModalities(
            int textLength, IReadOnlyList<MiniMaxH3PromptImage> images)
        {
            var modalities = new List<int>(textLength);
            for (int i = 0; i < textLength; i++) modalities.Add(MiniMaxH3Modality.Text);
            foreach (var image in images)
            {
                int begin = Math.Max(0, image.TokenIndex - 1);
                int end = Math.Min(textLength, image.TokenIndex + image.Vision.TokenCount + 1);
                for (int i = begin; i < end; i++) modalities[i] = MiniMaxH3Modality.Visual;
            }
            return modalities;
        }

        private static float[] Concat(List<float[]> parts)
        {
            long total = 0;
            foreach (var part in parts) total += part.LongLength;
            var all = new float[total];
            long off = 0;
            foreach (var part in parts) { Array.Copy(part, 0, all, off, part.Length); off += part.Length; }
            return all;
        }

        // A lone reference frame goes through the cheap 2-D reduction of the encoder
        // rather than the full 3-D graph, which is exact for a single frame.
        private float[] EncodeSingle(RgbImage image, out int latentFrames)
        {
            latentFrames = 1;
            return _vae.EncodeFrame(image);
        }

        /// <summary>Check that the conditioning chunk list describes the same runs, in the
        /// same order, that the layout put in the sequence.
        ///
        /// <para>These are built by two independent loops — this file fills
        /// <c>condChunks</c> while encoding, and <see cref="MiniMaxH3Layout.Build"/> emits
        /// segments — and the native side reassembles the conditioning rows POSITIONALLY
        /// from the chunk list, discarding the segment's source index. So they agree by
        /// construction rather than by contract, and if they ever stopped agreeing the
        /// symptom would be reference 2's tokens sitting where reference 1's belong: a
        /// plausible clip conditioned on the wrong pictures, with nothing reported. That
        /// cannot happen with a single reference, which is exactly why it is worth
        /// asserting now that several are the point.</para>
        ///
        /// <para>Runs once per request, over a handful of chunks.</para></summary>
        private static void RequireChunksMatchLayout(
            IReadOnlyList<H3CondChunk> chunks, MiniMaxH3PackedLayout layout)
        {
            if (chunks is not { Count: > 0 }) return;

            var expected = new List<(int Kind, int Count)>();
            foreach (var segment in layout.Segments)
            {
                if (segment.Kind == MiniMaxH3SegmentKind.ConditionVideo)
                    expected.Add((0, segment.Length));
                else if (segment.Kind == MiniMaxH3SegmentKind.ConditionAudio)
                    expected.Add((1, segment.Length));
            }

            string Describe(IEnumerable<(int Kind, int Count)> runs) =>
                string.Join(", ", runs.Select(r => $"{(r.Kind == 0 ? "video" : "audio")}x{r.Count}"));

            var actual = chunks.Select(c => (Kind: c.Kind, Count: c.Count)).ToList();
            if (actual.Count != expected.Count || !actual.SequenceEqual(expected))
                throw new InvalidOperationException(
                    "MiniMax-H3 conditioning is inconsistent: the encoder produced [" +
                    Describe(actual) + "] but the packed layout expects [" + Describe(expected) +
                    "]. The denoiser rebuilds these rows by position, so continuing would " +
                    "condition the clip on the wrong references rather than fail.");
        }

        /// <summary>Per-step magnitude trace, off unless <c>TS_H3_TRACE=1</c>. A
        /// diverging sample is visible in the latent's absmax several steps before it
        /// reaches infinity, and without this the only symptom is the finished file.</summary>
        private static readonly bool TraceEnabled =
            Environment.GetEnvironmentVariable("TS_H3_TRACE") == "1";

        private static double Rms(float[] v)
        {
            double sum = 0; long n = 0;
            foreach (float x in v) if (float.IsFinite(x)) { sum += (double)x * x; n++; }
            return n == 0 ? double.NaN : Math.Sqrt(sum / n);
        }

        private static float AbsMax(float[] v)
        {
            float m = 0;
            foreach (float x in v) if (float.IsFinite(x) && Math.Abs(x) > m) m = Math.Abs(x);
            return m;
        }

        /// <summary>Stop a diverged sample rather than writing it out as a black clip.
        ///
        /// <para>This failure mode is silent by construction: a NaN or -Inf video latent
        /// decodes to a pixel the RGB clamp pins at 0 and an audio latent the WAV writer
        /// clamps to -1, so the user gets a plausible-looking file — right length, right
        /// frame rate, right soundtrack duration — that is uniformly black with a
        /// saturated buzz, and nothing anywhere reported an error. A flow-matching
        /// velocity is O(1) by construction, so a non-finite one is a numeric failure
        /// upstream and there is nothing left to salvage from the run.</para></summary>
        private static void RequireFinite(float[] values, string what, int step, int steps, int tokens)
        {
            for (long i = 0; i < values.LongLength; i++)
            {
                if (float.IsFinite(values[i])) continue;
                throw new InvalidOperationException(
                    $"MiniMax-H3 denoising diverged: the {what} is " +
                    $"{(float.IsNaN(values[i]) ? "NaN" : "infinite")} at step {step + 1}/{steps} " +
                    $"(element {i} of {values.LongLength}, {tokens} packed tokens). Every frame " +
                    "would decode to black and the soundtrack to a saturated buzz, so the run is " +
                    "stopped instead. Re-run with TS_H3_TRACE=1 to see where the magnitudes left " +
                    "their normal range, and with a shorter clip to confirm the length is what " +
                    "triggers it.");
            }
        }

        /// <summary>Nudge a conditioning latent off the clean manifold.
        ///
        /// The model never sees a perfectly clean latent during training, so a
        /// reference or keyframe is mixed with a sliver of noise to match the
        /// near-clean timestep its tokens are modulated at.</summary>
        private static void AddVisualNoise(float[] latent, long seed)
        {
            // The reference reseeds per latent, so every conditioning frame in a
            // request gets the same noise draw rather than consecutive slices of one.
            var rng = new PhiloxLikeRng(seed);
            const float t = MiniMaxH3Scheduler.VisualConditionTimestep;
            for (long i = 0; i < latent.LongLength; i++)
                latent[i] = latent[i] * t + rng.NextNormal() * (1f - t);
        }

        /// <summary>Default generation area when the request does not give a size.
        /// 640x384 is the model card's own working point and the smallest size at
        /// which faces stay coherent.</summary>
        private const long DefaultArea = 640L * 384L;

        /// <summary>Resolve the output canvas. An explicit width/height always wins;
        /// otherwise a conditioning image sets the aspect ratio and the default area
        /// sets the size, so "animate this photo" does not silently letterbox or
        /// stretch it.</summary>
        internal static (int Width, int Height) ResolveCanvas(VideoGenerationParams p)
        {
            if (p.Width > 0 && p.Height > 0) return (p.Width, p.Height);

            double aspect = 0;
            RgbImage reference = p.ResolveImage() ?? p.ResolveEndImage();
            if (reference is { Width: > 0, Height: > 0 })
                aspect = reference.Width / (double)reference.Height;

            if (p.Width > 0 && aspect > 0) return (p.Width, (int)Math.Round(p.Width / aspect));
            if (p.Height > 0 && aspect > 0) return ((int)Math.Round(p.Height * aspect), p.Height);
            if (p.Width > 0) return (p.Width, p.Width);
            if (p.Height > 0) return (p.Height, p.Height);

            if (aspect <= 0) return (640, 384);
            int w = (int)Math.Round(Math.Sqrt(DefaultArea * aspect));
            int h = (int)Math.Round(Math.Sqrt(DefaultArea / aspect));
            return (Math.Max(w, 32), Math.Max(h, 32));
        }

        /// <summary>Slice the decoded clip into frames.
        ///
        /// <para>The temporal chunking in the VAE already dropped the decoder's warm-up
        /// frames and cross-faded the chunk seams, so this is a straight take of the
        /// requested count — the frame arithmetic lives in one place now instead of
        /// being split between a decode heuristic here and the chunk loop there.</para></summary>
        private static RgbImage[] TakeFrames(float[] pixels, int decodedFrames, MiniMaxH3Shape shape)
        {
            int count = Math.Min(shape.Frames, decodedFrames);
            var frames = new RgbImage[count];
            long plane = (long)decodedFrames * shape.Height * shape.Width;
            int hw = shape.Height * shape.Width;
            for (int f = 0; f < count; f++)
            {
                // The decoder is channel-planar over the whole clip; slice one frame
                // out of each channel plane into the CHW layout RgbImage expects.
                var chw = new float[3 * hw];
                long frameBase = (long)f * hw;
                for (int c = 0; c < 3; c++)
                    Array.Copy(pixels, c * plane + frameBase, chw, c * hw, hw);
                frames[f] = RgbImage.FromPlanarChw(shape.Width, shape.Height, chw);
            }
            return frames;
        }

        private static void Report(VideoGenerationParams p, string phase, int step, int steps,
                                  Stopwatch sw)
        {
            if (p.OnProgress == null) return;
            double elapsed = sw.Elapsed.TotalSeconds;
            double eta = step > 0 && step < steps ? elapsed / step * (steps - step) : -1;
            p.OnProgress(new VideoGenerationProgress
            {
                Phase = phase,
                Step = step,
                TotalSteps = steps,
                ElapsedSeconds = elapsed,
                EtaSeconds = eta,
            });
        }

        /// <summary>Hand the finished denoiser's VRAM back before the video VAE loads, when
        /// the card does not have room for both.
        ///
        /// <para>The DiT is ~10.6 GB at Q4_K and the video VAE another ~5.2 GB. Held together on
        /// a 16 GB card that is 15.8 GB of weights before a single activation, and Windows/WDDM
        /// does not fail that allocation - it silently backs the overflow with shared host memory,
        /// so the decode runs at PCIe speed. Measured on a 16 GB RTX 3080 Laptop at 640x384 x 22
        /// frames, the decode window sat pinned at 16.0 GB of a 16.4 GB card for its whole
        /// duration.</para>
        ///
        /// <para>This is a cap, not a policy change: when the spare VRAM already fits the VAE the
        /// denoiser stays resident and behaviour is exactly as before, so cards with room pay
        /// nothing. When it does not, the release costs a re-upload of the DiT on the NEXT
        /// request only - the weights are mmapped GGUF pages that are still in RAM - and buys
        /// back a decode that is not crossing the bus. Non-CUDA GGML backends are left alone:
        /// unified-memory devices have nothing to hand back.</para></summary>
        private void ReleaseDenoiserIfVaeWouldNotFit()
        {
            if (_dit == null || _model.Backend != BackendType.GgmlCuda)
                return;
            if (!GpuMemoryBudget.TryGetSpareBytes(_model.Backend, out long spare))
                return;   // no reading of the device: leave today's behaviour alone
            long need = _model.VideoVaeResidentBytesEstimate();
            if (need <= 0 || spare >= need)
                return;   // it fits alongside the denoiser; keep it resident
            _dit.ReleaseDeviceResidency();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _dit?.Dispose();
            _vae?.Dispose();
            _audioVae?.Dispose();
        }

        /// <summary>A small counter-based normal generator. Deterministic across
        /// platforms and independent of System.Random's implementation, so a seed
        /// reproduces the same clip anywhere.</summary>
        private sealed class PhiloxLikeRng
        {
            private ulong _state;
            private float _spare;
            private bool _hasSpare;

            public PhiloxLikeRng(long seed) { _state = (ulong)seed * 6364136223846793005UL + 1442695040888963407UL; }

            private uint NextUInt()
            {
                _state = _state * 6364136223846793005UL + 1442695040888963407UL;
                ulong x = _state;
                x ^= x >> 33; x *= 0xff51afd7ed558ccdUL;
                x ^= x >> 33;
                return (uint)(x >> 32);
            }

            private double NextDouble() => (NextUInt() + 0.5) / 4294967296.0;

            /// <summary>Box-Muller, keeping the second deviate so pairs are not wasted.</summary>
            public float NextNormal()
            {
                if (_hasSpare) { _hasSpare = false; return _spare; }
                double u1 = NextDouble(), u2 = NextDouble();
                double r = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;
                _spare = (float)(r * Math.Sin(theta));
                _hasSpare = true;
                return (float)(r * Math.Cos(theta));
            }
        }
    }
}
