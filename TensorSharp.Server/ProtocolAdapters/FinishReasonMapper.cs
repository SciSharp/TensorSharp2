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

namespace TensorSharp.Server.ProtocolAdapters
{
    /// <summary>
    /// Translates <see cref="ChatStreamUpdate.FinishReason"/> — the generation
    /// pipeline's own vocabulary — into the value each wire protocol defines for
    /// "why did generation stop".
    ///
    /// <para>The pipeline reports one of:</para>
    /// <list type="bullet">
    ///   <item><c>max_tokens</c> — the token budget was exhausted (the answer is cut off).</item>
    ///   <item><c>stop_sequence</c> — a caller-supplied stop string matched.</item>
    ///   <item><c>cancelled</c> — the client went away and the request was aborted.</item>
    ///   <item><c>eos</c> — the model emitted end-of-sequence, i.e. it finished on its own.</item>
    ///   <item><c>aborted</c> / <c>error</c> — the engine terminated the sequence.</item>
    ///   <item><c>stop</c> — the diffusion sampler finished its planned blocks.</item>
    /// </list>
    ///
    /// <para>Clients key off these values to decide whether to continue a truncated
    /// answer, so the truncation case has to be distinguishable from every other
    /// case. Everything that is *not* truncation collapses to the protocol's
    /// "finished normally" value — see the per-method remarks for why.</para>
    /// </summary>
    internal static class FinishReasonMapper
    {
        // ---- Pipeline vocabulary (inputs) -------------------------------------

        /// <summary>The pipeline's reason for "ran out of token budget". This is the
        /// only reason that must survive translation intact on every protocol.</summary>
        public const string PipelineMaxTokens = "max_tokens";

        // ---- OpenAI /v1/chat/completions --------------------------------------

        /// <summary>Truncated by the token budget.</summary>
        public const string OpenAiLength = "length";

        /// <summary>Finished normally (natural EOS, stop sequence, or anything else non-truncating).</summary>
        public const string OpenAiStop = "stop";

        /// <summary>Stopped because the model emitted tool calls the caller must run.</summary>
        public const string OpenAiToolCalls = "tool_calls";

        /// <summary>
        /// Map to the <c>finish_reason</c> of an OpenAI chat completion (streaming
        /// and non-streaming alike).
        /// </summary>
        /// <remarks>
        /// The schema defines exactly <c>stop</c>, <c>length</c>, <c>tool_calls</c>,
        /// <c>content_filter</c> and <c>function_call</c> here, so anything the
        /// pipeline invents has to land on one of those.
        /// <para>
        /// Truncation outranks tool calls: a run that hit the budget may have cut a
        /// tool call in half, and a client that saw <c>tool_calls</c> would dispatch
        /// the fragment instead of noticing the answer is incomplete.
        /// </para>
        /// <para>
        /// <c>cancelled</c> / <c>aborted</c> map to <c>stop</c>. There is no
        /// "the client hung up" value in the schema, and the distinction is
        /// unobservable anyway — the socket that would carry the answer is the one
        /// that just closed. What matters is that they are NOT reported as
        /// <c>length</c>, because the caller has no truncated answer to resume.
        /// </para>
        /// </remarks>
        /// <param name="pipelineReason">A <see cref="ChatStreamUpdate.FinishReason"/> value.</param>
        /// <param name="hasToolCalls">True when the parsed output carries tool calls.</param>
        public static string ToOpenAIChat(string pipelineReason, bool hasToolCalls)
        {
            if (IsTruncated(pipelineReason))
                return OpenAiLength;
            return hasToolCalls ? OpenAiToolCalls : OpenAiStop;
        }

        // ---- Ollama /api/chat, /api/generate ----------------------------------

        /// <summary>Ollama's <c>done_reason</c> for a budget-truncated response.</summary>
        public const string OllamaLength = "length";

        /// <summary>Ollama's <c>done_reason</c> for a response that finished normally.</summary>
        public const string OllamaStop = "stop";

        /// <summary>Ollama's <c>done_reason</c> for a response that ended in tool calls.</summary>
        public const string OllamaToolCalls = "tool_calls";

        /// <summary>
        /// Map to Ollama's <c>done_reason</c>. Same three-way split as
        /// <see cref="ToOpenAIChat"/> — Ollama reports <c>length</c> on truncation
        /// and <c>stop</c> otherwise, and this server has always additionally
        /// reported <c>tool_calls</c> when the turn ended in a tool call.
        /// </summary>
        public static string ToOllamaDoneReason(string pipelineReason, bool hasToolCalls)
        {
            if (IsTruncated(pipelineReason))
                return OllamaLength;
            return hasToolCalls ? OllamaToolCalls : OllamaStop;
        }

        // ---- OpenAI /v1/responses ---------------------------------------------

        /// <summary>Responses-API status for a response that ran to completion.</summary>
        public const string ResponsesCompleted = "completed";

        /// <summary>Responses-API status for a response that stopped before it was done.</summary>
        public const string ResponsesIncomplete = "incomplete";

        /// <summary>Responses-API <c>incomplete_details.reason</c> for budget truncation.</summary>
        public const string ResponsesMaxOutputTokens = "max_output_tokens";

        /// <summary>
        /// Map to the Responses API's terminal <c>status</c> and its companion
        /// <c>incomplete_details.reason</c>.
        /// </summary>
        /// <remarks>
        /// <c>/v1/responses</c> has no <c>finish_reason</c>: truncation shows up as
        /// <c>status: "incomplete"</c> plus <c>incomplete_details: { "reason":
        /// "max_output_tokens" }</c>. The only other reason that field defines is
        /// <c>content_filter</c>, which this server never produces, so every
        /// non-truncating outcome is a plain <c>completed</c> with no details.
        /// </remarks>
        /// <returns>The <c>status</c>, and the <c>incomplete_details.reason</c>
        /// (null when the response completed).</returns>
        public static (string Status, string IncompleteReason) ToResponsesStatus(string pipelineReason)
        {
            return IsTruncated(pipelineReason)
                ? (ResponsesIncomplete, ResponsesMaxOutputTokens)
                : (ResponsesCompleted, null);
        }

        // ---- Shared ------------------------------------------------------------

        /// <summary>
        /// True when generation stopped because it ran out of token budget, i.e.
        /// the answer is cut off mid-thought and the client may want to continue it.
        /// An unrecognised or missing reason is deliberately NOT treated as
        /// truncation: falsely claiming a complete answer was cut off makes clients
        /// re-request work that was already finished.
        /// </summary>
        public static bool IsTruncated(string pipelineReason) =>
            string.Equals(pipelineReason, PipelineMaxTokens, StringComparison.Ordinal);
    }
}
