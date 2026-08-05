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
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Grammar
{
    /// <summary>
    /// Process-wide cache of compiled grammars and their warmed token-mask
    /// caches, keyed by grammar source and tokenizer.
    /// </summary>
    /// <remarks>
    /// Compiling a schema is cheap; warming its masks is not. The first visit to
    /// a permissive state — "inside a JSON string", where almost the whole
    /// vocabulary is legal — walks the entire token trie. That cost is per
    /// <i>state</i>, not per request, so it must be paid once per (grammar,
    /// model) and then shared, otherwise every request re-pays it and structured
    /// output looks slow for no reason. Entries are held weakly against the
    /// tokenizer so they die with the model.
    /// </remarks>
    public static class GrammarLibrary
    {
        private sealed class PerTokenizer
        {
            public readonly Dictionary<string, GrammarMaskCache> ByGrammarSource = new(StringComparer.Ordinal);
        }

        private static readonly ConditionalWeakTable<ITokenizer, PerTokenizer> Cache = new();

        /// <summary>
        /// Get the shared mask cache for <paramref name="gbnf"/> under
        /// <paramref name="tokenizer"/>, compiling and caching on first use.
        /// </summary>
        public static GrammarMaskCache ForGbnf(string gbnf, ITokenizer tokenizer)
        {
            if (gbnf == null) throw new ArgumentNullException(nameof(gbnf));
            if (tokenizer == null) throw new ArgumentNullException(nameof(tokenizer));

            PerTokenizer per = Cache.GetValue(tokenizer, _ => new PerTokenizer());
            lock (per)
            {
                if (per.ByGrammarSource.TryGetValue(gbnf, out GrammarMaskCache? hit))
                    return hit;
                var grammar = Grammar.Parse(gbnf);
                var built = new GrammarMaskCache(grammar, GrammarTokenVocabulary.ForTokenizer(tokenizer));
                per.ByGrammarSource[gbnf] = built;
                return built;
            }
        }

        /// <summary>
        /// Shared mask cache for a JSON Schema. A null or empty schema yields the
        /// any-JSON-value grammar, which is what <c>response_format:
        /// {"type":"json_object"}</c> means when no schema is supplied.
        /// </summary>
        public static GrammarMaskCache ForJsonSchema(string? schemaJson, ITokenizer tokenizer) =>
            ForGbnf(JsonSchemaGrammarCompiler.Compile(schemaJson), tokenizer);

        /// <summary>Shared mask cache for "any single JSON object".</summary>
        public static GrammarMaskCache ForJsonObject(ITokenizer tokenizer) =>
            ForGbnf(JsonSchemaGrammarCompiler.JsonObjectGrammar, tokenizer);

        /// <summary>
        /// Create a fresh per-request constraint over a shared cache. Every
        /// sequence needs its own, because the constraint carries the live parse
        /// position.
        /// </summary>
        public static GrammarConstraint NewConstraint(GrammarMaskCache cache, ITokenizer tokenizer) =>
            new GrammarConstraint(cache, tokenizer);
    }
}
