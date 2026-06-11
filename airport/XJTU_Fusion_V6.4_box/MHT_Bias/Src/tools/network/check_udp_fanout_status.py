"""Print current UDP fanout runtime status and relevant UDP listeners."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
STATUS_FILE = SRC_ROOT / "data" / "udp_fanout_status.json"
WATCH_SECONDS = 5.0


def load_status() -> dict | None:
    if not STATUS_FILE.exists():
        return None
    with STATUS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_ports() -> None:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-NetUDPEndpoint | "
            "Where-Object { $_.LocalPort -in 9000,9966,20000,19966,29000,29966 } | "
            "Select-Object LocalAddress,LocalPort,OwningProcess,"
            "@{Name='ProcessName';Expression={(Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName}} | "
            "Sort-Object LocalPort,LocalAddress | "
            "Format-Table -AutoSize"
        ),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    print("[fanout-check] UDP listeners:")
    print(completed.stdout.strip() or "(none)")


def summarize_status(status: dict, label: str) -> dict[str, int]:
    print(f"[fanout-check] {label}: {status.get('updated_at_text')}")
    counts: dict[str, int] = {}
    for stream in status.get("streams", []):
        name = stream.get("name")
        packet_count = int(stream.get("packet_count") or 0)
        counts[name] = packet_count
        print(
            f"  {name}: packets={packet_count}, bytes={stream.get('byte_count')}, "
            f"last_src={stream.get('last_src')}, age={stream.get('last_packet_age_sec')}s"
        )
        for target in stream.get("targets", []):
            print(
                f"    -> {target.get('name')}@{target.get('host')}:{target.get('port')} "
                f"send={target.get('send_count')} err={target.get('error_count')} "
                f"last_error={target.get('last_error')}"
            )
    return counts


def main() -> int:
    print_ports()
    first = load_status()
    if first is None:
        print(f"[fanout-check] missing status file: {STATUS_FILE}")
        print("[fanout-check] restart main2.py so the updated fanout can create it.")
        return 2

    first_counts = summarize_status(first, "initial")
    print(f"[fanout-check] watching {WATCH_SECONDS:.0f}s for packet count changes...")
    time.sleep(WATCH_SECONDS)

    second = load_status()
    if second is None:
        print("[fanout-check] status file disappeared.")
        return 2
    second_counts = summarize_status(second, "after-watch")

    for name in sorted(set(first_counts) | set(second_counts)):
        delta = second_counts.get(name, 0) - first_counts.get(name, 0)
        print(f"[fanout-check] delta {name}: {delta} packets/{WATCH_SECONDS:.0f}s")

    radar_delta = second_counts.get("radar", 0) - first_counts.get("radar", 0)
    if radar_delta <= 0:
        print(
            "[fanout-check][RESULT] radar packets are NOT entering fanout. "
            "Check radar software forwarding destination IP/port and fanout bind address. "
            "Current default is radar software -> 127.0.0.1:9000."
        )
    else:
        print(
            "[fanout-check][RESULT] radar packets entered fanout. "
            "If eosmsv4 still shows nothing, the issue is downstream eosmsv4 parsing/listener behavior."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
