using System;
using System.Linq;
using System.Net;

namespace TensorSharp.Distributed
{
    /// <summary>
    /// Configuration for a distributed tensor-parallel group.
    /// Parsed from CLI arguments or environment variables.
    /// </summary>
    public sealed class DistributedTpConfig
    {
        /// <summary>This node's ID (0-based).</summary>
        public int NodeId { get; }

        /// <summary>Number of local GPUs on this node.</summary>
        public int LocalDegree { get; }

        /// <summary>TCP endpoints for every node, indexed by node ID.</summary>
        public IPEndPoint[] PeerEndpoints { get; }

        /// <summary>Total GPUs across all nodes.</summary>
        public int GlobalDegree => LocalDegree * PeerEndpoints.Length;

        public DistributedTpConfig(int nodeId, int localDegree, IPEndPoint[] peerEndpoints)
        {
            NodeId = nodeId;
            LocalDegree = localDegree;
            PeerEndpoints = peerEndpoints ?? throw new ArgumentNullException(nameof(peerEndpoints));

            if ((uint)nodeId >= (uint)peerEndpoints.Length)
                throw new ArgumentOutOfRangeException(nameof(nodeId),
                    $"Node ID {nodeId} is out of range for {peerEndpoints.Length} nodes.");
        }

        /// <summary>
        /// Parse a comma-separated list of host:port pairs into IPEndpoints.
        /// Example: "192.168.1.10:9500,192.168.1.11:9500"
        /// </summary>
        public static IPEndPoint[] ParsePeers(string peersString)
        {
            if (string.IsNullOrWhiteSpace(peersString))
                throw new ArgumentException("Peers string is empty.", nameof(peersString));

            var parts = peersString.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var endpoints = new IPEndPoint[parts.Length];

            for (int i = 0; i < parts.Length; i++)
            {
                var segments = parts[i].Split(':');
                if (segments.Length != 2)
                    throw new FormatException(
                        $"Invalid peer endpoint '{parts[i]}'. Expected format: host:port");

                if (!int.TryParse(segments[1], out int port) || port < 1 || port > 65535)
                    throw new FormatException(
                        $"Invalid port in peer endpoint '{parts[i]}'.");

                if (IPAddress.TryParse(segments[0], out var ip))
                {
                    endpoints[i] = new IPEndPoint(ip, port);
                }
                else
                {
                    // Resolve hostname.
                    var addresses = Dns.GetHostAddresses(segments[0]);
                    if (addresses.Length == 0)
                        throw new InvalidOperationException($"Could not resolve hostname '{segments[0]}'.");
                    endpoints[i] = new IPEndPoint(addresses[0], port);
                }
            }

            return endpoints;
        }

        /// <summary>
        /// Try to build a <see cref="DistributedTpConfig"/> from environment variables.
        /// Returns null if the required variables are not set.
        ///
        /// Environment variables:
        ///   TENSORSHARP_TP_NODE_ID  — this node's ID (required for distributed mode)
        ///   TENSORSHARP_TP_PEERS    — comma-separated host:port list (required for distributed mode)
        /// </summary>
        public static DistributedTpConfig TryFromEnvironment(int localDegree)
        {
            string nodeIdStr = Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID");
            string peersStr = Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS");

            if (string.IsNullOrEmpty(nodeIdStr) || string.IsNullOrEmpty(peersStr))
                return null;

            if (!int.TryParse(nodeIdStr, out int nodeId))
                throw new FormatException($"TENSORSHARP_TP_NODE_ID must be an integer, got '{nodeIdStr}'.");

            var peers = ParsePeers(peersStr);
            return new DistributedTpConfig(nodeId, localDegree, peers);
        }
    }
}
