using System;
using System.Net;
using TensorSharp.Cuda;

namespace TensorSharp.Distributed
{
    /// <summary>
    /// Multi-node tensor-parallel group that combines local CUDA P2P
    /// communication (within a node) with TCP communication (across nodes).
    ///
    /// AllReduce is hierarchical:
    ///   1. Local P2P AllReduce across GPUs within this node
    ///   2. Copy the rank-0 result to a host buffer
    ///   3. TCP AllReduce across node representatives
    ///   4. Broadcast the reduced result back to all local GPUs
    ///
    /// This minimises network traffic: only one buffer per AllReduce
    /// crosses the network, regardless of how many local GPUs participate.
    /// </summary>
    public sealed class DistributedTensorParallelGroup : ITensorParallelGroup
    {
        private readonly TensorParallelGroup _localGroup;
        private readonly TcpCommunicator _tcp;
        private readonly int _nodeId;
        private readonly int _nodeCount;
        private bool _disposed;

        // Reusable host buffer for GPU↔network transfers.
        private float[] _hostBuffer = Array.Empty<float>();

        /// <param name="localDegree">Number of GPUs on this node.</param>
        /// <param name="nodeId">This node's ID (0..nodeCount-1).</param>
        /// <param name="peerEndpoints">
        /// TCP endpoints for every node in the group, indexed by node ID.
        /// </param>
        public DistributedTensorParallelGroup(int localDegree, int nodeId, IPEndPoint[] peerEndpoints)
        {
            _nodeId = nodeId;
            _nodeCount = peerEndpoints.Length;

            if (localDegree < 1)
                throw new ArgumentOutOfRangeException(nameof(localDegree));
            if (_nodeCount < 2)
                throw new ArgumentException("Distributed TP requires at least 2 nodes.", nameof(peerEndpoints));

            _localGroup = new TensorParallelGroup(localDegree);
            _tcp = new TcpCommunicator(nodeId, peerEndpoints);

            Console.WriteLine($"Distributed tensor parallelism: node {nodeId}/{_nodeCount}, " +
                $"{localDegree} local GPU(s), {_nodeCount * localDegree} total across cluster.");
        }

        /// <summary>Number of local GPUs on this node.</summary>
        public int Degree => _localGroup.Degree;

        /// <summary>True when TP is active (always true for distributed groups).</summary>
        public bool IsActive => true;

        /// <summary>Total GPUs across all nodes.</summary>
        public int GlobalDegree => Degree * _nodeCount;

        /// <summary>First global rank on this node.</summary>
        public int GlobalRankOffset => _nodeId * Degree;

        /// <summary>Number of nodes in the cluster.</summary>
        public int NodeCount => _nodeCount;

        public CudaAllocator GetAllocator(int rank) => _localGroup.GetAllocator(rank);

        /// <summary>
        /// Hierarchical AllReduce: local P2P reduce → TCP reduce → local broadcast.
        /// <paramref name="tensors"/> has one tensor per local GPU.
        /// After this call every tensor on every node holds the global sum.
        /// </summary>
        public void AllReduce(Tensor[] tensors)
        {
            if (tensors == null || tensors.Length != Degree)
                throw new ArgumentException($"Expected {Degree} tensors, got {tensors?.Length ?? 0}.");

            // Phase 1: Local P2P AllReduce across GPUs within this node.
            _localGroup.AllReduce(tensors);

            // Phase 2: Copy rank-0 GPU data to host buffer.
            int elementCount = (int)tensors[0].Storage.ElementCount;
            EnsureHostBuffer(elementCount);

            var hostData = tensors[0].GetElementsAsFloat(elementCount);
            Array.Copy(hostData, _hostBuffer, elementCount);

            // Phase 3: TCP AllReduce across nodes (modifies _hostBuffer in-place).
            _tcp.AllReduce(_hostBuffer.AsSpan(0, elementCount));

            // Phase 4: Broadcast the reduced result from host to all local GPUs.
            var resultSlice = new float[elementCount];
            Array.Copy(_hostBuffer, resultSlice, elementCount);

            for (int r = 0; r < Degree; r++)
            {
                tensors[r].SetElementsAsFloat(resultSlice);
                // SetElementsAsFloat only marks the host buffer dirty; force the
                // upload now so downstream GPU kernels (including fused kernels
                // that read the raw device pointer) see the reduced values
                // rather than the stale pre-reduction partial. Without this the
                // multi-node result never reaches the device and generation is
                // garbage, even though single-node (device-to-device) works.
                tensors[r].EnsureDeviceCurrent();
            }

            // Sync all local GPUs so the result is visible.
            _localGroup.Synchronize();
        }

        public void Synchronize()
        {
            _localGroup.Synchronize();
        }

        /// <summary>Block until every node has reached this point.</summary>
        public void Barrier()
        {
            _localGroup.Synchronize();
            _tcp.Barrier();
        }

        /// <summary>Driver (node 0) broadcasts a control op + payload to worker nodes.</summary>
        public void BroadcastControl(int op, int[] payload) => _tcp.BroadcastControl(op, payload);

        /// <summary>Worker node blocks for the next control message from the driver.</summary>
        public (int op, int[] payload) ReceiveControl() => _tcp.ReceiveControl();

        private void EnsureHostBuffer(int elementCount)
        {
            if (_hostBuffer.Length >= elementCount)
                return;
            _hostBuffer = new float[elementCount];
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _tcp?.Dispose();
            _localGroup?.Dispose();
        }
    }
}
