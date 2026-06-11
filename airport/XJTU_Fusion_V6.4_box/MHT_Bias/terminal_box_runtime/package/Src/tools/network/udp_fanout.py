"""UDP fan-out helper for running eosmsv4 and main2.py at the same time.

The device can only send each UDP stream to one local IP:port. This helper owns
that device-facing port and copies every datagram to multiple local consumers.

Default layout:
  Radar relay   -> 127.0.0.1:9000    -> eosmsv4:20000, main2.py:29000
  Optical device -> 192.168.0.9:9966 -> eosmsv4:19966, main2.py:29966

For this to work:
  1. Keep radar software forwarding output at 127.0.0.1:9000.
  2. Configure eosmsv4 radar device/plugin port to 20000.
  3. Configure eosmsv4 optical port eoptclport to 19966.
  4. Start main2.py with AIRR_RADAR_LISTEN_PORT=29000 and
     AIRR_OPTICAL_LOCAL_PORT=29966.

Run order:
  1. Stop eosmsv4.exe and main2.py.
  2. Run configure_eosmsv4_for_fanout.py once, then restart eosmsv4.
  3. Start this script.
  4. Start main2.py through run_main2_with_fanout_ports.ps1.
"""

from __future__ import annotations

import socket
import threading
import time
import os
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOCAL_IP = "192.168.0.9"
RADAR_INGRESS_IP = os.environ.get("AIRR_RADAR_FANOUT_INGRESS_IP", "127.0.0.1")
OPTICAL_INGRESS_IP = os.environ.get("AIRR_OPTICAL_FANOUT_INGRESS_IP", LOCAL_IP)

# Device-facing ports. These must match what the radar/optical device sends to.
RADAR_SOURCE_PORT = int(os.environ.get("AIRR_RADAR_FANOUT_INGRESS_PORT", "9000"))
OPTICAL_SOURCE_PORT = 9966

# Downstream consumers. Keep each consumer on a unique port.
RADAR_TARGETS = [
    # eosmsv4's lz plugin may bind as IPv4 or IPv6 depending on startup state.
    # Send both forms; the duplicate only goes to eosmsv4, not to main2.
    ("127.0.0.1", 20000, "eosmsv4-radar-v4"),
    ("::1", 20000, "eosmsv4-radar-v6"),
    ("127.0.0.1", 29000, "main2-radar"),
]
OPTICAL_TARGETS = [
    ("127.0.0.1", 19966, "eosmsv4-optical"),
    (LOCAL_IP, 29966, "main2-optical"),
]

PRINT_INTERVAL_SEC = 5.0
PRINT_STATS = os.environ.get("AIRR_FANOUT_PRINT_STATS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CHECK_DOWNSTREAM_PORTS = os.environ.get("AIRR_FANOUT_CHECK_TARGETS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
STATUS_FILE = Path(
    os.environ.get(
        "AIRR_FANOUT_STATUS_FILE",
        str(Path(__file__).resolve().parents[2] / "data" / "udp_fanout_status.json"),
    )
)
STATUS_WRITE_INTERVAL_SEC = 1.0
STATUS_LOCK = threading.Lock()
ACTIVE_FANOUTS: list["UdpFanout"] = []


@dataclass(frozen=True)
class Target:
    host: str
    port: int
    name: str

    @property
    def addr(self):
        if ":" in self.host:
            return self.host, self.port, 0, 0
        return self.host, self.port


def _targets(items: Iterable[tuple]) -> list[Target]:
    return [Target(host, port, name) for host, port, name in items]


def _udp_listeners_by_port() -> dict[int, list[tuple[str, int]]]:
    if os.name != "nt":
        return {}

    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-NetUDPEndpoint | "
                "Select-Object LocalAddress,LocalPort,OwningProcess | "
                "ConvertTo-Json -Compress"
            ),
        ]
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}

        import json

        data = json.loads(completed.stdout)
        if isinstance(data, dict):
            data = [data]
        listeners: dict[int, list[tuple[str, int]]] = {}
        for item in data:
            try:
                port = int(item.get("LocalPort"))
                pid = int(item.get("OwningProcess"))
                addr = str(item.get("LocalAddress"))
            except (TypeError, ValueError):
                continue
            listeners.setdefault(port, []).append((addr, pid))
        return listeners
    except Exception:
        return {}


def warn_if_targets_not_listening(targets: list[Target]) -> None:
    if not CHECK_DOWNSTREAM_PORTS:
        return
    listeners = _udp_listeners_by_port()
    if not listeners:
        return

    for target in targets:
        if target.port in (RADAR_SOURCE_PORT, OPTICAL_SOURCE_PORT):
            continue
        if target.port not in listeners:
            print(
                f"[fanout] downstream target {target.name}@{target.host}:{target.port} "
                "is not listening yet."
            )


class UdpFanout(threading.Thread):
    def __init__(self, name: str, bind_host: str, bind_port: int, targets: list[Target]):
        super().__init__(daemon=True)
        self.name = name
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.targets = targets
        self.stop_event = threading.Event()
        self.packet_count = 0
        self.byte_count = 0
        self.start_error = None
        self.last_src = None
        self.last_packet_time = None
        self.target_send_counts = {target.name: 0 for target in targets}
        self.target_error_counts = {target.name: 0 for target in targets}
        self.target_last_errors = {target.name: None for target in targets}

    def snapshot(self) -> dict:
        now = time.time()
        return {
            "name": self.name,
            "bind_host": self.bind_host,
            "bind_port": self.bind_port,
            "alive": self.is_alive(),
            "start_error": str(self.start_error) if self.start_error else None,
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "last_src": self.last_src,
            "last_packet_time": self.last_packet_time,
            "last_packet_age_sec": (
                round(now - self.last_packet_time, 3)
                if self.last_packet_time is not None
                else None
            ),
            "targets": [
                {
                    "name": target.name,
                    "host": target.host,
                    "port": target.port,
                    "send_count": self.target_send_counts.get(target.name, 0),
                    "error_count": self.target_error_counts.get(target.name, 0),
                    "last_error": self.target_last_errors.get(target.name),
                }
                for target in self.targets
            ],
        }

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.bind_host, self.bind_port))
        except OSError as exc:
            self.start_error = exc
            print(
                f"[{self.name}] bind failed {self.bind_host}:{self.bind_port}: {exc}. "
                "Close eosmsv4/main2 instances that already own this port, then restart."
            )
            sock.close()
            return
        sock.settimeout(0.5)

        out_sock4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        out_sock6 = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        last_print = time.time()
        print(
            f"[{self.name}] listening {self.bind_host}:{self.bind_port} -> "
            + ", ".join(f"{t.name}@{t.host}:{t.port}" for t in self.targets)
        )
        write_status_snapshot()

        last_status_write = time.time()
        while not self.stop_event.is_set():
            try:
                data, src = sock.recvfrom(65535)
            except socket.timeout:
                now = time.time()
                if now - last_status_write >= STATUS_WRITE_INTERVAL_SEC:
                    write_status_snapshot()
                    last_status_write = now
                continue
            except OSError as exc:
                print(f"[{self.name}] socket error: {exc}")
                break

            self.packet_count += 1
            self.byte_count += len(data)
            self.last_src = f"{src[0]}:{src[1]}"
            self.last_packet_time = time.time()
            for target in self.targets:
                try:
                    out_sock = out_sock6 if ":" in target.host else out_sock4
                    out_sock.sendto(data, target.addr)
                except OSError as exc:
                    self.target_error_counts[target.name] += 1
                    self.target_last_errors[target.name] = str(exc)
                    print(f"[{self.name}] forward to {target.name} failed: {exc}")
                else:
                    self.target_send_counts[target.name] += 1

            now = time.time()
            if now - last_status_write >= STATUS_WRITE_INTERVAL_SEC:
                write_status_snapshot()
                last_status_write = now
            if PRINT_STATS and now - last_print >= PRINT_INTERVAL_SEC:
                print(
                    f"[{self.name}] packets={self.packet_count} bytes={self.byte_count} "
                    f"last_src={src[0]}:{src[1]}"
                )
                last_print = now

        sock.close()
        out_sock4.close()
        out_sock6.close()

    def stop(self) -> None:
        self.stop_event.set()


def main() -> int:
    fanouts = create_default_fanouts()
    set_active_fanouts(fanouts)
    warn_if_targets_not_listening(_targets(RADAR_TARGETS + OPTICAL_TARGETS))

    for fanout in fanouts:
        fanout.start()

    time.sleep(0.8)
    failed = [fanout for fanout in fanouts if not fanout.is_alive()]
    if failed:
        print(
            "[fanout] startup failed: "
            + ", ".join(
                f"{f.name}@{f.bind_host}:{f.bind_port}"
                + (f" ({f.start_error})" if f.start_error else "")
                for f in failed
            )
            + f" is not listening. Check whether another process owns {RADAR_SOURCE_PORT}/9966."
        )
        for fanout in fanouts:
            fanout.stop()
        for fanout in fanouts:
            fanout.join(timeout=2.0)
        return 1

    print("[fanout] running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[fanout] stopping...")
        for fanout in fanouts:
            fanout.stop()
        for fanout in fanouts:
            fanout.join(timeout=2.0)
    return 0


def create_default_fanouts() -> list[UdpFanout]:
    return [
        UdpFanout("radar", RADAR_INGRESS_IP, RADAR_SOURCE_PORT, _targets(RADAR_TARGETS)),
        UdpFanout("optical", OPTICAL_INGRESS_IP, OPTICAL_SOURCE_PORT, _targets(OPTICAL_TARGETS)),
    ]


def set_active_fanouts(fanouts: list[UdpFanout]) -> None:
    ACTIVE_FANOUTS.clear()
    ACTIVE_FANOUTS.extend(fanouts)
    write_status_snapshot()


def write_status_snapshot() -> None:
    with STATUS_LOCK:
        try:
            STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": time.time(),
                "updated_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status_file": str(STATUS_FILE),
                "streams": [fanout.snapshot() for fanout in ACTIVE_FANOUTS],
            }
            with STATUS_FILE.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            # Status output is diagnostic only; never let it break packet forwarding.
            pass


def start_default_fanouts() -> list[UdpFanout]:
    fanouts = create_default_fanouts()
    set_active_fanouts(fanouts)
    for fanout in fanouts:
        fanout.start()
    return fanouts


def stop_fanouts(fanouts: list[UdpFanout]) -> None:
    for fanout in fanouts:
        fanout.stop()
    for fanout in fanouts:
        fanout.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
