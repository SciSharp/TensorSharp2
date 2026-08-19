// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// glm-dsa tensor parallelism.
//
// The split follows the shape of the block rather than a uniform "shard every
// matrix" rule, because GLM-5.x's bytes are not where a dense model's are:
//
//   * ROUTED EXPERTS (~92% of the checkpoint) are split EXPERT-wise, not
//     tensor-wise. Each rank owns a contiguous slice of the 256 experts and
//     computes only the selected experts that fall in its slice, so no rank
//     ever holds another rank's expert bytes. Expert counts need not divide the
//     rank count: the remainder is spread over the leading ranks.
//   * ATTENTION is split HEAD-wise across ranks (q_b / wk_b / wv_b column-wise,
//     attn_output row-wise). The MLA cache itself is replicated because it is a
//     single 576-wide row per token shared by every head — 576 floats is
//     cheaper to keep on each rank than to gather every step.
//   * Everything small (norms, q_a, kv_a_mqa, the router, the indexer, the
//     shared expert, embeddings, lm_head) is replicated. Together they are a
//     few percent of the model and replicating them removes a collective from
//     the critical path.
//
// One AllReduce per block closes the attention output and one closes the MoE
// output, matching the two row-parallel projections.
using System;
using System.Collections.Generic;
using TensorSharp;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        /// <summary>First expert index owned by <paramref name="rank"/> of <paramref name="degree"/>.</summary>
        internal static int ExpertShardStart(int numExperts, int degree, int rank)
        {
            int baseCount = numExperts / degree;
            int remainder = numExperts % degree;
            return rank * baseCount + Math.Min(rank, remainder);
        }

        /// <summary>Expert count owned by <paramref name="rank"/>; the remainder goes to the leading ranks.</summary>
        internal static int ExpertShardCount(int numExperts, int degree, int rank)
            => ExpertShardStart(numExperts, degree, rank + 1) - ExpertShardStart(numExperts, degree, rank);

        private void ShardGlmDsaWeightsForTP()
        {
            throw new NotSupportedException(
                "glm-dsa tensor parallelism is not enabled in this build yet; run without --tp.");
        }
    }
}
