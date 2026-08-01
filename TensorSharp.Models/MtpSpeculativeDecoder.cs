// Copyright (c) Zhongkai Fu. All rights reserved.
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
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    /// <summary>
    /// Greedy speculative decoding driven by a Qwen3.5/3.6 NextN/MTP draft head
    /// (llama.cpp's `--spec-type draft-mtp`, vLLM's qwen3_5_mtp speculator).
    ///
    /// Thin standalone wrapper over <see cref="MtpSpeculativeExecution"/> (the
    /// shared draft/verify/rollback/catch-up core also driven by the engine's
    /// <c>BatchExecutor</c>): owns the whole generate loop — KV reset, chunked
    /// prompt prefill, greedy argmax verification — for tests and offline use.
    ///
    /// Greedy verification makes the output token stream identical to plain
    /// greedy decoding up to batched-vs-sequential floating point drift.
    /// Single-sequence; the caller owns the model's KV cache lifecycle.
    /// </summary>
    public sealed class MtpSpeculativeDecoder
    {
        private readonly IMtpSpeculativeModel _model;
        private readonly MtpSpeculativeExecution _exec;
        private readonly int _vocab;
        private readonly List<int> _acceptedScratch = new();

        /// <summary>Maximum tokens drafted per speculative step (llama.cpp n_max).</summary>
        public int MaxDraftTokens => _exec.MaxDraftTokens;

        /// <summary>
        /// Minimum draft confidence (top-1 probability over the draft head's
        /// top-10 logits, matching llama.cpp's top-k(10) draft sampler) for a
        /// drafted token to be kept. Drafting stops at the first low-confidence
        /// token.
        /// </summary>
        public float MinDraftProb
        {
            get => _exec.MinDraftProb;
            set => _exec.MinDraftProb = value;
        }

        /// <summary>Prompt prefill chunk size (bounds the h-capture buffers).</summary>
        public int PrefillChunkSize { get; set; } = 512;

        // Statistics (reset by Reset()).
        public long TokensDrafted => _exec.Stats.TokensDrafted;
        public long TokensAccepted => _exec.Stats.TokensAccepted;
        public long VerifySteps => _exec.Stats.VerifySteps;
        public long PlainSteps => _exec.Stats.PlainSteps;
        public long RollbackSteps => _exec.Stats.RollbackSteps;
        public double AcceptanceRate => _exec.Stats.AcceptanceRate;

        /// <summary>Wall-clock phase breakdown of the speculative decode loop
        /// (draft / verify / snapshot / rollback / catch-up / plain).</summary>
        public MtpSpecStats Stats => _exec.Stats;

        /// <summary>Wall-clock seconds the last GenerateGreedy call spent in prompt prefill.</summary>
        public double LastPrefillSeconds { get; private set; }

        /// <summary>Wall-clock seconds the last GenerateGreedy call spent past prefill.</summary>
        public double LastDecodeSeconds { get; private set; }

        // Default draft window: 8. The MinDraftProb gate stops drafting at the
        // first low-confidence token, so a longer window only extends confident
        // streaks — measured 1.21x vs 1.08x (window 4) on Qwen3.6-35B-A3B
        // ggml_cpu at unchanged 86% acceptance; neutral on 27B ggml_cuda.
        public MtpSpeculativeDecoder(IMtpSpeculativeModel model, int maxDraftTokens = 8)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _exec = new MtpSpeculativeExecution(model, maxDraftTokens);
            _vocab = model.Config.VocabSize;
        }

        /// <summary>Reset speculative state and statistics. Does NOT touch the model's KV cache.</summary>
        public void Reset() => _exec.Reset();

        /// <summary>
        /// Prefill the prompt through the trunk (capturing h_nextn) and replay it
        /// through the MTP block so its KV cache covers the whole prompt.
        /// Returns the last-position logits (vocab floats, caller-owned copy).
        /// </summary>
        public float[] Prefill(int[] promptTokens)
        {
            if (promptTokens == null || promptTokens.Length == 0)
                throw new ArgumentException("Prompt must not be empty.", nameof(promptTokens));

            // A trunk that keeps its draft head in sync internally wants the whole
            // prompt in one call: chunking it only adds a host round trip per
            // chunk, which stalls the trunk's own micro-batch pipelining.
            int chunkSize = _model.MtpPrefillSelfCatchUp
                ? Math.Max(1, promptTokens.Length)
                : Math.Max(1, PrefillChunkSize);
            float[] logits = null;
            int offset = 0;
            while (offset < promptTokens.Length)
            {
                int n = Math.Min(chunkSize, promptTokens.Length - offset);
                int[] chunk = new int[n];
                Array.Copy(promptTokens, offset, chunk, 0, n);
                // Linear trunk: the chunk's start position is the model's
                // current cache length (contiguous from the ResetKVCache).
                logits = _exec.PrefillStep(chunk, _model.CacheSeqLen);
                offset += n;
            }
            return logits;
        }

        /// <summary>
        /// Greedy generation with MTP speculative decoding. Resets the model KV
        /// cache, prefills <paramref name="promptTokens"/>, then emits up to
        /// <paramref name="maxNewTokens"/> tokens (the stop token, when hit, is
        /// included as the final element). <paramref name="onToken"/>, when
        /// given, receives every emitted token as it is produced and returns
        /// false to stop generating.
        /// </summary>
        public List<int> GenerateGreedy(int[] promptTokens, int maxNewTokens, Func<int, bool> isStopToken = null,
            Func<int, bool> onToken = null)
        {
            _model.ResetKVCache();
            _exec.Reset();

            var output = new List<int>();
            if (maxNewTokens <= 0)
                return output;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] logits = Prefill(promptTokens);
            LastPrefillSeconds = sw.Elapsed.TotalSeconds;
            GenerateFrom(logits, promptTokens.Length, maxNewTokens, isStopToken, onToken, output);
            return output;
        }

        /// <summary>
        /// Greedy speculative generation continuing from <paramref name="promptLogits"/>,
        /// the next-token logits at trunk position <paramref name="position"/>. Unlike
        /// <see cref="GenerateGreedy"/> this touches neither the KV cache nor the
        /// speculative state, so a caller that owns prompt prefill (a chat turn reusing
        /// the prefix its predecessor left in the cache) can drive the same draft/verify
        /// loop. <paramref name="onToken"/> receives each token as it is emitted and
        /// returns false to stop; the stop token is appended to the result but is not
        /// streamed.
        ///
        /// On return the trunk holds exactly the first
        /// <c>CacheSeqLen - position</c> tokens of the result, so a caller tracking the
        /// cache can mirror it without guessing where a stopped step left off.
        /// </summary>
        public List<int> GenerateGreedyFrom(float[] promptLogits, int position, int maxNewTokens,
            Func<int, bool> isStopToken = null, Func<int, bool> onToken = null)
        {
            if (promptLogits == null)
                throw new ArgumentNullException(nameof(promptLogits));

            var output = new List<int>();
            if (maxNewTokens > 0)
                GenerateFrom(promptLogits, position, maxNewTokens, isStopToken, onToken, output);
            return output;
        }

        private void GenerateFrom(float[] promptLogits, int position, int maxNewTokens,
            Func<int, bool> isStopToken, Func<int, bool> onToken, List<int> output)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                Decode();

                // A verify batch commits the whole accepted prefix at once, so a run
                // that ends part-way through emitting one (a stop token, a consumer
                // that asked to stop) leaves the trunk holding tokens that never
                // reached the result. Drop them, so the trunk always holds exactly a
                // prefix of what was returned and a caller mirroring the cache does
                // not have to reason about where the last step stopped.
                int emitted = position + output.Count;
                if (_model.CacheSeqLen > emitted)
                    _model.MtpRewindCache(emitted);
            }
            finally
            {
                LastDecodeSeconds = sw.Elapsed.TotalSeconds;
            }

            void Decode()
            {
                // Appends a token to the result and reports whether generation
                // should continue: a stop token ends the run without being
                // streamed, and the consumer can end it too (cancellation, a
                // stop sequence).
                bool Emit(int t)
                {
                    output.Add(t);
                    if (isStopToken != null && isStopToken(t))
                        return false;
                    return onToken == null || onToken(t);
                }

                int tLast = Argmax(promptLogits, _vocab);
                if (!Emit(tLast))
                    return;

                int n = position;

                while (output.Count < maxNewTokens)
                {
                    // Never draft more than we can still emit; the core also
                    // clamps to MaxDraftTokens and the context headroom.
                    int kMax = maxNewTokens - output.Count;

                    _acceptedScratch.Clear();
                    MtpDecodeOutcome outcome = _exec.DecodeStep(
                        tLast, n, kMax,
                        drawNext: row => Argmax(row, _vocab),
                        adjustDraftLogits: null,
                        onDraftAccepted: _acceptedScratch.Add);

                    if (!outcome.UsedSpeculation)
                    {
                        // Plain decode step.
                        int next = Argmax(outcome.NextLogits, _vocab);
                        n += 1;
                        tLast = next;
                        if (!Emit(next))
                            return;
                        continue;
                    }

                    // The trunk has already committed the accepted prefix plus the
                    // corrected/bonus token, so advance the position before emitting:
                    // a consumer that stops mid-step must still see a position that
                    // matches the cache.
                    n += outcome.AcceptedCount + 1;
                    tLast = outcome.NextToken;

                    // Emit accepted drafts then the corrected/bonus token.
                    int emittedThisStep = 0;
                    for (int i = 0; i < _acceptedScratch.Count && output.Count < maxNewTokens; i++)
                    {
                        emittedThisStep++;
                        if (!Emit(_acceptedScratch[i]))
                            return;
                    }
                    if (output.Count < maxNewTokens)
                    {
                        emittedThisStep++;
                        if (!Emit(outcome.NextToken))
                            return;
                    }

                    if (emittedThisStep == 0)
                        break; // budget exhausted mid-step
                }
            }
        }

        private static int Argmax(float[] logits, int vocab)
        {
            int best = 0;
            float bestVal = logits[0];
            for (int i = 1; i < vocab; i++)
            {
                float v = logits[i];
                if (v > bestVal)
                {
                    bestVal = v;
                    best = i;
                }
            }
            return best;
        }
    }
}
