# -*- coding: utf-8 -*-
"""Small UDP probe for radar simulator packets.

Run this before starting the full MHT pipeline. It only binds the radar receive
address and prints packet length plus the first bytes, so it is useful for
checking IP/port/protocol mismatches.
"""

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.app_config import (
    FRAME_HEAD_END,
    FRAME_HEAD_POINT,
    FRAME_HEAD_STATUS,
    FRAME_HEAD_TRACK,
    HOST_IP,
    HOST_PORT,
)


FRAME_NAMES = {
    FRAME_HEAD_STATUS: "STATUS",
    FRAME_HEAD_POINT: "POINT",
    FRAME_HEAD_TRACK: "TRACK",
    FRAME_HEAD_END: "END",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST_IP)
    parser.add_argument("--port", type=int, default=HOST_PORT)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((args.host, args.port))

    print(f"[probe] listening on {args.host}:{args.port} for {args.seconds:.0f}s")
    print("[probe] expected frame heads:")
    for head, name in FRAME_NAMES.items():
        print(f"  {name:<6} 0x{head:08X}")

    deadline = time.time() + args.seconds
    count = 0
    type_counts = {name: 0 for name in FRAME_NAMES.values()}
    type_counts["UNKNOWN"] = 0

    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(65536)
        except socket.timeout:
            continue

        count += 1
        head = struct.unpack("<I", data[:4])[0] if len(data) >= 4 else None
        frame_len = struct.unpack("<I", data[4:8])[0] if len(data) >= 8 else None
        name = FRAME_NAMES.get(head, "UNKNOWN")
        type_counts[name] += 1
        hex_prefix = data[:24].hex(" ")
        print(
            f"[probe] #{count} from {addr[0]}:{addr[1]} bytes={len(data)} "
            f"head={f'0x{head:08X}' if head is not None else 'n/a'} "
            f"type={name} frame_len={frame_len} prefix={hex_prefix}",
            flush=True,
        )

    summary = ", ".join(f"{name}={value}" for name, value in type_counts.items())
    print(f"[probe] done. packets={count}, {summary}")
    return 0 if count else 2


if __name__ == "__main__":
    raise SystemExit(main())
