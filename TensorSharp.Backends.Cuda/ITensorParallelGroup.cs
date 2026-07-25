using System;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Abstraction over a tensor-parallel group that may span local GPUs
    /// (single process) or multiple nodes connected by a network.
    /// Model code uses this interface to place per-rank tensors and
    /// invoke collective operations without knowing the physical topology.
    /// </summary>
    public interface ITensorParallelGroup : IDisposable
    {
        /// <summary>Number of GPUs in this parallel group (local to this process).</summary>
        int Degree { get; }

        /// <summary>True when TP is active (degree > 1).</summary>
        bool IsActive { get; }

        /// <summary>
        /// Total number of GPUs across all nodes. Equals <see cref="Degree"/>
        /// for single-node groups; larger for multi-node distributed groups.
        /// Weight sharding uses this value to compute shard sizes.
        /// </summary>
        int GlobalDegree { get; }

        /// <summary>
        /// The global rank offset for this node. Local rank <c>r</c> maps to
        /// global rank <c>GlobalRankOffset + r</c>. Zero for single-node groups.
        /// </summary>
        int GlobalRankOffset { get; }

        /// <summary>Number of nodes in the distributed group (1 for local-only).</summary>
        int NodeCount { get; }

        CudaAllocator GetAllocator(int rank);

        /// <summary>
        /// In-place AllReduce (element-wise sum) across all GPUs.
        /// tensors[i] must be a F32 tensor on GPU i, all the same shape.
        /// After this call every tensor holds the global sum.
        /// For multi-node groups this performs a hierarchical reduce:
        /// local P2P first, then network exchange, then local broadcast.
        /// </summary>
        void AllReduce(Tensor[] tensors);

        /// <summary>Synchronize all GPU streams.</summary>
        void Synchronize();

        /// <summary>
        /// Block until every node in the group has reached this point.
        /// Single-node groups return immediately.
        /// </summary>
        void Barrier();

        /// <summary>
        /// Driver→worker control channel for multi-node lockstep execution.
        /// The driver node (global rank offset 0) calls this to broadcast an
        /// op code plus an int payload (e.g. the tokens for a forward pass) to
        /// every worker node before running the op locally. Single-node groups
        /// have no workers and do not implement this.
        /// </summary>
        void BroadcastControl(int op, int[] payload);

        /// <summary>
        /// Worker-side counterpart to <see cref="BroadcastControl"/>: block until
        /// the driver broadcasts the next control message, and return it.
        /// </summary>
        (int op, int[] payload) ReceiveControl();
    }
}
