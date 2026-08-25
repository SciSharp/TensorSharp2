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

            // ---- text conditioning (then give the 17 GB encoder back) ----
            float[] textHidden;
            int textLength;
            // Per-token modality for the AdaLN stream: plain text is modulated as text,
            // but the tokens standing in for a reference image are modulated as VISUAL.
            List<int> textModalities = null;
            var textClock = Stopwatch.StartNew();
            using (var te = _model.CreateTextEncoder())
            {
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
                textHidden = te.Encode(ids, promptImages);
                if (promptImages is { Count: > 0 })
                    textModalities = BuildTextModalities(textLength, promptImages);
            }

            textClock.Stop();
            Console.WriteLine($"  [h3] text conditioning: {textLength} tokens in " +
                              $"{textClock.Elapsed.TotalSeconds:F1}s");

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

                var (vVel, aVel) = _dit.Forward(
                    videoPatches, videoCount, audioLatent, audioCount,
                    textHidden, textLength, layout, sigma,
                    conditionTokens, conditionCount,
                    conditionAudio, conditionAudioCount, condChunks, videoShift);

                MiniMaxH3Scheduler.EulerStep(videoPatches, vVel, sigma, sigmas[step + 1]);
                MiniMaxH3Scheduler.EulerStep(audioLatent, aVel, sigma, sigmas[step + 1]);

                p.OnStep?.Invoke(step + 1, steps);
                Report(p, "denoise", step + 1, steps, total);
                if (step == 0)
                    Console.WriteLine($"  [h3] {layout.TokenCount} packed tokens; " +
                                      $"first step in {total.Elapsed.TotalSeconds - textClock.Elapsed.TotalSeconds:F1}s");
            }

            // ---- decode video ----
            Report(p, "vae-decode", steps, steps, total);
            _vae ??= _model.CreateVideoVae();

            float[] latent = MiniMaxH3DiT.UnpatchifyVideo(
                videoPatches, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth,
                MiniMaxH3Geometry.VideoLatentChannels);
            _vae.DenormalizeLatent(latent);

            float[] pixels = _vae.Decode(
                latent, shape.LatentFrames, shape.LatentHeight, shape.LatentWidth,
                (tile, tiles) => Report(p, "vae-decode", steps, steps, total));

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
            const int maxReferences = 9;
            if (paths.Count > maxReferences)
                throw new ArgumentException(
                    $"MiniMax-H3 Ref2VA accepts at most {maxReferences} reference images; " +
                    $"{paths.Count} were supplied.", nameof(p));

            var images = new List<RgbImage>(paths.Count);
            foreach (string path in paths) images.Add(Fit(ImageIO.Load(path)));
            return images;
        }

        private static int SnapToVaeGrid(double value) =>
            Math.Max(MiniMaxH3Geometry.SpatialMultiple,
                     (int)Math.Round(value / MiniMaxH3Geometry.SpatialMultiple)
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
