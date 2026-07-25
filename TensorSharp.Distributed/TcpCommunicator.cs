using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace TensorSharp.Distributed
{
    /// <summary>
    /// Collective operations over a TCP mesh network. Each node knows its own
    /// rank and the endpoints of every other node. Connections are established
    /// once at construction and reused for all subsequent operations.
    ///
    /// Wire protocol (per message):
    ///   [4 bytes] payload length (little-endian int32, excludes this header)
    ///   [1 byte ] message type
    ///   [N bytes] payload
    ///
    /// AllReduce algorithm (reduce-to-zero + broadcast):
    ///   1. Every non-zero node sends its partial to node 0
    ///   2. Node 0 accumulates: sum += partial
    ///   3. Node 0 broadcasts the reduced result to every other node
    ///   4. Barrier so all nodes observe the result before proceeding
    ///
    /// This mirrors the local <c>CudaP2PCommunicator</c> algorithm and
    /// preserves the bitwise-determinism invariant: because node 0 sums in
    /// a fixed rank order, every node receives an identical result.
    /// </summary>
    public sealed class TcpCommunicator : IDisposable
    {
        private const byte MsgAllReduceContribute = 0x01;
        private const byte MsgAllReduceResult     = 0x02;
        private const byte MsgBarrierAck          = 0x03;
        private const byte MsgControl             = 0x04;

        private readonly int _rank;
        private readonly int _worldSize;
        private readonly TcpListener _listener;
        private readonly TcpClient[] _peers;   // indexed by remote rank
        private readonly NetworkStream[] _streams;
        private readonly object _sendLock = new();
        private bool _disposed;

        // Reusable receive buffer to avoid per-call allocation.
        private byte[] _recvBuffer = Array.Empty<byte>();

        /// <param name="rank">This node's rank (0..worldSize-1).</param>
        /// <param name="endpoints">
        /// Endpoints for every node in the group, indexed by rank.
        /// The entry at <paramref name="rank"/> is the local listen address.
        /// </param>
        public TcpCommunicator(int rank, IPEndPoint[] endpoints)
        {
            _rank = rank;
            _worldSize = endpoints.Length;

            if (_worldSize < 2)
                throw new ArgumentException("TCP communicator requires at least 2 nodes.", nameof(endpoints));
            if ((uint)rank >= (uint)_worldSize)
                throw new ArgumentOutOfRangeException(nameof(rank));

            _peers = new TcpClient[_worldSize];
            _streams = new NetworkStream[_worldSize];

            // Start listening.
            _listener = new TcpListener(endpoints[rank]);
            _listener.Start();

            try
            {
                // Nodes are typically launched by hand, seconds or minutes apart,
                // so peers may not be listening yet when we start connecting. Retry
                // outbound connects (and tolerate the OS not having bound the peer's
                // listener yet) until a deadline instead of failing on the first
                // ECONNREFUSED. Override the window with TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS.
                TimeSpan connectTimeout = ResolveConnectTimeout();
                DateTime deadline = DateTime.UtcNow + connectTimeout;

                // Lower-ranked node initiates the connection to higher-ranked nodes.
                // Higher-ranked nodes accept. This gives exactly one connection per pair.

                int recvTimeoutMs = ResolveReceiveTimeoutMs();

                // Initiate outbound connections to higher-ranked peers.
                for (int r = rank + 1; r < _worldSize; r++)
                {
                    var client = ConnectWithRetry(endpoints[r], r, deadline);
                    _peers[r] = client;
                    _streams[r] = client.GetStream();
                    _streams[r].ReadTimeout = recvTimeoutMs;

                    // Handshake: tell the remote peer our rank so it can
                    // map this connection to the correct slot.
                    byte[] rankBytes = new byte[4];
                    BinaryPrimitives.WriteInt32LittleEndian(rankBytes, rank);
                    _streams[r].Write(rankBytes, 0, 4);
                    _streams[r].Flush();
                }

                // Accept inbound connections from lower-ranked peers.
                // AcceptTcpClient() does not guarantee rank order, so each
                // inbound connection advertises its rank in a 4-byte header.
                for (int r = 0; r < rank; r++)
                {
                    var client = _listener.AcceptTcpClient();
                    client.NoDelay = true;
                    var stream = client.GetStream();
                    stream.ReadTimeout = recvTimeoutMs;

                    byte[] rankBytes = ReadExact(stream, 4);
                    int remoteRank = BinaryPrimitives.ReadInt32LittleEndian(rankBytes);
                    if (remoteRank < 0 || remoteRank >= _worldSize || remoteRank == rank)
                        throw new InvalidOperationException(
                            $"Received invalid rank {remoteRank} during handshake (this node is rank {rank}).");
                    if (_peers[remoteRank] != null)
                        throw new InvalidOperationException(
                            $"Duplicate inbound connection from rank {remoteRank}.");

                    _peers[remoteRank] = client;
                    _streams[remoteRank] = stream;
                }
            }
            catch
            {
                Dispose();
                throw;
            }

            Console.WriteLine($"[TcpCommunicator] Rank {rank}/{_worldSize} connected to all peers.");
        }

        public int Rank => _rank;
        public int WorldSize => _worldSize;

        private static TimeSpan ResolveConnectTimeout()
        {
            string s = Environment.GetEnvironmentVariable("TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS");
            if (!string.IsNullOrWhiteSpace(s) && int.TryParse(s, out int secs) && secs > 0)
                return TimeSpan.FromSeconds(secs);
            return TimeSpan.FromSeconds(120);
        }

        /// <summary>
        /// Per-receive timeout for blocking reads. If a peer does not deliver a
        /// message within this window the receive throws instead of hanging
        /// indefinitely (the OS TCP keepalive default is often 2+ hours).
        /// Override with TENSORSHARP_TP_RECV_TIMEOUT_SECONDS.
        /// </summary>
        private static int ResolveReceiveTimeoutMs()
        {
            string s = Environment.GetEnvironmentVariable("TENSORSHARP_TP_RECV_TIMEOUT_SECONDS");
            if (!string.IsNullOrWhiteSpace(s) && int.TryParse(s, out int secs) && secs > 0)
                return secs * 1000;
            return 120_000;
        }

        /// <summary>
        /// Connect to a peer, retrying until <paramref name="deadline"/>. Tolerates
        /// the peer's listener not being up yet (connection refused / unreachable),
        /// which is the normal case when nodes are started a few seconds apart.
        /// </summary>
        private TcpClient ConnectWithRetry(IPEndPoint endpoint, int remoteRank, DateTime deadline)
        {
            const int retryDelayMs = 250;
            bool announced = false;
            int attempts = 0;

            while (true)
            {
                var client = new TcpClient { NoDelay = true };
                try
                {
                    client.Connect(endpoint);
                    if (announced)
                        Console.WriteLine($"[TcpCommunicator] Rank {_rank}: connected to rank {remoteRank} at {endpoint}.");
                    return client;
                }
                catch (SocketException) when (DateTime.UtcNow < deadline)
                {
                    client.Dispose();
                    attempts++;
                    if (!announced)
                    {
                        Console.WriteLine($"[TcpCommunicator] Rank {_rank}: waiting for rank {remoteRank} at {endpoint} " +
                            $"(retrying up to {(deadline - DateTime.UtcNow).TotalSeconds:F0}s)...");
                        announced = true;
                    }
                    Thread.Sleep(retryDelayMs);
                }
                catch (SocketException ex)
                {
                    client.Dispose();
                    throw new InvalidOperationException(
                        $"Rank {_rank} could not connect to rank {remoteRank} at {endpoint} after {attempts} attempts. " +
                        "Check the peer is running, the host:port is reachable, and firewall rules allow it. " +
                        "Increase the window with TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS.", ex);
                }
            }
        }

        /// <summary>
        /// In-place AllReduce (sum) across all nodes. The buffer is modified
        /// to contain the element-wise sum of all nodes' inputs.
        /// All nodes must call this concurrently with the same buffer length.
        /// </summary>
        public unsafe void AllReduce(Span<float> buffer)
        {
            int byteCount = buffer.Length * sizeof(float);

            if (_rank == 0)
            {
                // Node 0: receive partials from all other nodes and accumulate.
                for (int r = 1; r < _worldSize; r++)
                {
                    var (msgType, payload) = ReceiveMessage(r);
                    if (msgType != MsgAllReduceContribute)
                        throw new InvalidOperationException(
                            $"Expected AllReduceContribute from rank {r}, got message type 0x{msgType:X2}.");

                    fixed (float* bufPtr = buffer)
                    fixed (byte* payPtr = payload)
                    {
                        float* src = (float*)payPtr;
                        for (int i = 0; i < buffer.Length; i++)
                            bufPtr[i] += src[i];
                    }
                }

                // Broadcast the reduced result to all other nodes.
                byte[] resultBytes = new byte[byteCount];
                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)bufPtr, resultBytes, 0, byteCount);

                for (int r = 1; r < _worldSize; r++)
                    SendMessage(r, MsgAllReduceResult, resultBytes);
            }
            else
            {
                // Non-zero node: send partial to node 0, then receive the result.
                byte[] partialBytes = new byte[byteCount];
                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)bufPtr, partialBytes, 0, byteCount);

                SendMessage(0, MsgAllReduceContribute, partialBytes);

                var (msgType, payload) = ReceiveMessage(0);
                if (msgType != MsgAllReduceResult)
                    throw new InvalidOperationException(
                        $"Expected AllReduceResult from rank 0, got message type 0x{msgType:X2}.");

                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy(payload, 0, (IntPtr)bufPtr, byteCount);
            }

            // Barrier so all nodes have the result before any proceeds.
            Barrier();
        }

        /// <summary>
        /// Driver→worker control broadcast. Called on rank 0 (the driver node)
        /// to send an op code plus an int payload to every other node. Workers
        /// call <see cref="ReceiveControl"/> to read it. Unlike AllReduce this is
        /// one-way (no barrier): it precedes the forward pass whose AllReduces
        /// then provide the synchronisation.
        /// </summary>
        public void BroadcastControl(int op, int[] payload)
        {
            int n = payload?.Length ?? 0;
            byte[] buf = new byte[8 + n * 4];
            BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan(0, 4), op);
            BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan(4, 4), n);
            for (int i = 0; i < n; i++)
                BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan(8 + i * 4, 4), payload[i]);

            for (int r = 0; r < _worldSize; r++)
            {
                if (r == _rank) continue;
                SendMessage(r, MsgControl, buf);
            }
        }

        /// <summary>
        /// Worker-side receive for <see cref="BroadcastControl"/>. Blocks until
        /// rank 0 sends the next control message, then returns (op, payload).
        /// </summary>
        public (int op, int[] payload) ReceiveControl()
        {
            var (msgType, data) = ReceiveMessage(0);
            if (msgType != MsgControl)
                throw new InvalidOperationException(
                    $"Expected Control from rank 0, got message type 0x{msgType:X2}.");

            int op = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(0, 4));
            int n = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(4, 4));
            var payload = new int[n];
            for (int i = 0; i < n; i++)
                payload[i] = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(8 + i * 4, 4));
            return (op, payload);
        }

        /// <summary>
        /// Broadcast a buffer from <paramref name="rootRank"/> to all nodes.
        /// </summary>
        public unsafe void Broadcast(Span<float> buffer, int rootRank)
        {
            int byteCount = buffer.Length * sizeof(float);

            if (_rank == rootRank)
            {
                byte[] data = new byte[byteCount];
                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)bufPtr, data, 0, byteCount);

                for (int r = 0; r < _worldSize; r++)
                {
                    if (r == rootRank) continue;
                    SendMessage(r, MsgAllReduceResult, data);
                }
            }
            else
            {
                var (msgType, payload) = ReceiveMessage(rootRank);
                if (msgType != MsgAllReduceResult)
                    throw new InvalidOperationException(
                        $"Expected Broadcast from rank {rootRank}, got message type 0x{msgType:X2}.");

                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy(payload, 0, (IntPtr)bufPtr, byteCount);
            }

            Barrier();
        }

        /// <summary>
        /// Block until all nodes have reached this point. Uses a simple
        /// gather-then-release protocol through node 0.
        /// </summary>
        public void Barrier()
        {
            if (_rank == 0)
            {
                // Receive ack from every other node.
                for (int r = 1; r < _worldSize; r++)
                {
                    var (msgType, _) = ReceiveMessage(r);
                    if (msgType != MsgBarrierAck)
                        throw new InvalidOperationException(
                            $"Expected BarrierAck from rank {r}, got 0x{msgType:X2}.");
                }

                // Release all nodes.
                for (int r = 1; r < _worldSize; r++)
                    SendMessage(r, MsgBarrierAck, Array.Empty<byte>());
            }
            else
            {
                SendMessage(0, MsgBarrierAck, Array.Empty<byte>());
                var (msgType, _) = ReceiveMessage(0);
                if (msgType != MsgBarrierAck)
                    throw new InvalidOperationException(
                        $"Expected BarrierAck from rank 0, got 0x{msgType:X2}.");
            }
        }

        private void SendMessage(int destRank, byte msgType, byte[] payload)
        {
            var stream = _streams[destRank];
            int totalLen = 1 + payload.Length;

            byte[] header = new byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(header, totalLen);

            lock (_sendLock)
            {
                stream.Write(header, 0, 4);
                stream.WriteByte(msgType);
                if (payload.Length > 0)
                    stream.Write(payload, 0, payload.Length);
                stream.Flush();
            }
        }

        private (byte msgType, byte[] payload) ReceiveMessage(int srcRank)
        {
            var stream = _streams[srcRank];

            // Read 4-byte length header.
            byte[] header = ReadExact(stream, 4);
            int payloadLen = BinaryPrimitives.ReadInt32LittleEndian(header);

            if (payloadLen < 1)
                throw new InvalidOperationException("Received message with empty payload (missing type byte).");

            // Read type byte + payload.
            byte[] full = ReadExact(stream, payloadLen);
            byte msgType = full[0];
            byte[] payload = new byte[payloadLen - 1];
            if (payload.Length > 0)
                Array.Copy(full, 1, payload, 0, payload.Length);

            return (msgType, payload);
        }

        private static byte[] ReadExact(NetworkStream stream, int count)
        {
            byte[] buf = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int n;
                try
                {
                    n = stream.Read(buf, offset, count - offset);
                }
                catch (IOException ex) when (ex.InnerException is SocketException se
                    && (se.SocketErrorCode == SocketError.TimedOut || se.SocketErrorCode == SocketError.WouldBlock))
                {
                    throw new TimeoutException(
                        $"Timed out waiting for a peer message ({count} bytes expected, {offset} received). " +
                        "The remote node may be stuck, still loading, or unreachable. " +
                        "Adjust with TENSORSHARP_TP_RECV_TIMEOUT_SECONDS.", ex);
                }
                if (n == 0)
                    throw new EndOfStreamException("Peer closed the connection.");
                offset += n;
            }
            return buf;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            for (int r = 0; r < _worldSize; r++)
            {
                _streams[r]?.Dispose();
                _peers[r]?.Dispose();
            }

            try { _listener?.Stop(); } catch { }
        }
    }
}
