#!/usr/bin/env python3
"""Measure the actual TCP path used by a two-node validation run.

Start --listen on node 1, then point node 0 at that listener (or its SSH forward).
This is a round-trip echo measurement, not an unidirectional fabric benchmark.
"""
import argparse
import json
from pathlib import Path
import socket
import statistics
import struct
import time


def receive(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = sock.recv(count - len(chunks))
        if not chunk:
            raise EOFError("Connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listen", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9501)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.listen:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((a.host, a.port))
            server.listen(1)
            conn, _ = server.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    size = struct.unpack("<I", receive(conn, 4))[0]
                    if size == 0:
                        return
                    conn.sendall(receive(conn, size))
    else:
        result = {"endpoint": [a.host, a.port], "measurements": []}
        with socket.create_connection((a.host, a.port), timeout=30) as client:
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for size, repeats in ((4, 100), (4096, 100), (1048576, 10)):
                payload = bytes((i % 251 for i in range(size)))
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    client.sendall(struct.pack("<I", size) + payload)
                    actual = receive(client, size)
                    times.append(time.perf_counter() - start)
                    if actual != payload:
                        raise ValueError("Echo payload mismatch")
                median = statistics.median(times)
                result["measurements"].append({"bytes_each_direction": size, "repeats": repeats,
                    "median_round_trip_ms": median * 1000, "p95_round_trip_ms": sorted(times)[int(repeats * .95) - 1] * 1000,
                    "bidirectional_payload_mib_per_second": 2 * size / median / 1048576})
            client.sendall(struct.pack("<I", 0))
        encoded = json.dumps(result, indent=2) + "\n"
        if a.output:
            a.output.write_text(encoded)
        print(encoded)


if __name__ == "__main__":
    main()
