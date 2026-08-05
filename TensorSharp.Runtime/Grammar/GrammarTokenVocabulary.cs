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
    /// A byte trie over every token's exact bytes, built once per tokenizer and
    /// shared by every request.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is what makes constrained decoding cheap. The obvious way to build a
    /// token mask — llama.cpp's way — is to test all V tokens against the grammar
    /// independently, which at a 256k vocabulary is 256k grammar walks <i>per
    /// generated token</i>. Sharing prefixes instead means a token's first byte
    /// is tested once for all tokens starting with it, and when a prefix is
    /// rejected the entire subtree below it is skipped. This is the same
    /// structure xgrammar uses, and the reason it and outlines are usable at
    /// production vocab sizes.
    /// </para>
    /// <para>
    /// Nodes are stored in flat arrays rather than as objects: children live in
    /// one contiguous span per node, sorted by byte, so a walk is a couple of
    /// array reads with no pointer chasing or dictionary hashing on the hot path.
    /// </para>
    /// </remarks>
    public sealed class GrammarTokenVocabulary
    {
        // Children of node i occupy [_childStart[i], _childStart[i+1]) of the
        // _childByte/_childNode pair of arrays, sorted ascending by byte.
        private readonly int[] _childStart;
        private readonly byte[] _childByte;
        private readonly int[] _childNode;

        /// <summary>Token that ends at each node, or -1.</summary>
        private readonly int[] _nodeToken;

        private static readonly ConditionalWeakTable<ITokenizer, GrammarTokenVocabulary> Cache = new();

        public int VocabSize { get; }
        public int NodeCount => _nodeToken.Length;

        /// <summary>Ids excluded from grammar matching (control tokens, EOS).</summary>
        public IReadOnlyCollection<int> SpecialTokenIds { get; }

        /// <summary>Number of ulong words a full-vocabulary bitmask needs.</summary>
        public int MaskWords => (VocabSize + 63) >> 6;

        private GrammarTokenVocabulary(
            int[] childStart, byte[] childByte, int[] childNode, int[] nodeToken,
            int vocabSize, IReadOnlyCollection<int> special)
        {
            _childStart = childStart;
            _childByte = childByte;
            _childNode = childNode;
            _nodeToken = nodeToken;
            VocabSize = vocabSize;
            SpecialTokenIds = special;
        }

        /// <summary>
        /// Get (building on first use) the trie for <paramref name="tokenizer"/>.
        /// Cached weakly against the tokenizer, so it is built once per model and
        /// released with it.
        /// </summary>
        public static GrammarTokenVocabulary ForTokenizer(ITokenizer tokenizer)
        {
            if (tokenizer == null) throw new ArgumentNullException(nameof(tokenizer));
            return Cache.GetValue(tokenizer, Build);
        }

        private static GrammarTokenVocabulary Build(ITokenizer tokenizer)
        {
            int vocabSize = tokenizer.VocabSize;

            var special = new HashSet<int>();
            if (tokenizer is ISpecialTokenVocabulary sv)
            {
                foreach (int id in sv.SpecialTokenIds) special.Add(id);
            }
            else
            {
                foreach (int id in tokenizer.EosTokenIds) special.Add(id);
            }

            // Mutable build representation; flattened below.
            var childMaps = new List<Dictionary<byte, int>> { new() };
            var tokenAt = new List<int> { -1 };

            var buffer = new List<byte>(64);
            for (int id = 0; id < vocabSize; id++)
            {
                if (special.Contains(id)) continue;

                buffer.Clear();
                try
                {
                    tokenizer.AppendTokenBytes(id, buffer);
                }
                catch
                {
                    // A vocabulary entry we cannot render as bytes cannot be
                    // matched against a grammar either; leaving it out of the
                    // trie means it is never unmasked, which is the safe side.
                    continue;
                }
                if (buffer.Count == 0) continue;

                int node = 0;
                for (int k = 0; k < buffer.Count; k++)
                {
                    byte b = buffer[k];
                    if (!childMaps[node].TryGetValue(b, out int next))
                    {
                        next = childMaps.Count;
                        childMaps.Add(new Dictionary<byte, int>());
                        tokenAt.Add(-1);
                        childMaps[node][b] = next;
                    }
                    node = next;
                }
                // Two ids with identical bytes: keep the first. Either is a
                // correct decode and the grammar cannot tell them apart.
                if (tokenAt[node] < 0) tokenAt[node] = id;
            }

            int nodeCount = childMaps.Count;
            var childStart = new int[nodeCount + 1];
            int totalEdges = 0;
            for (int i = 0; i < nodeCount; i++)
            {
                childStart[i] = totalEdges;
                totalEdges += childMaps[i].Count;
            }
            childStart[nodeCount] = totalEdges;

            var childByte = new byte[totalEdges];
            var childNode = new int[totalEdges];
            var scratch = new List<byte>(256);
            for (int i = 0; i < nodeCount; i++)
            {
                scratch.Clear();
                foreach (byte b in childMaps[i].Keys) scratch.Add(b);
                scratch.Sort();
                int at = childStart[i];
                foreach (byte b in scratch)
                {
                    childByte[at] = b;
                    childNode[at] = childMaps[i][b];
                    at++;
                }
            }

            return new GrammarTokenVocabulary(
                childStart, childByte, childNode, tokenAt.ToArray(), vocabSize, special);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int ChildStart(int node) => _childStart[node];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int ChildEnd(int node) => _childStart[node + 1];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal byte EdgeByte(int edge) => _childByte[edge];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int EdgeNode(int edge) => _childNode[edge];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal int TokenAt(int node) => _nodeToken[node];
    }
}
