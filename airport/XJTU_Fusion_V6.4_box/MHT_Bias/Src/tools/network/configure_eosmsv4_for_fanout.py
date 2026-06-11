"""Configure eosmsv4 to receive fan-out downstream UDP streams.

This script changes only the local eosmsv4 configuration files, with backups.
It does not start or stop eosmsv4.

Fan-out topology:
  radar software -> 127.0.0.1:9000 -> eosmsv4:20000, main2.py:29000
  device optical -> 192.168.0.9:9966 -> eosmsv4:19966, main2.py:29966
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


EOSMSV4_ROOT = Path(r"D:\desk\tra\eosmsv4yibiaodin")
RUNTIME_PROPS = EOSMSV4_ROOT / "files" / "props" / "runtimeprops.json"
RADAR_PROPS = [
    EOSMSV4_ROOT / "plugins" / "radar" / "yzf" / "props.ini",
    EOSMSV4_ROOT / "plugins" / "radar" / "yf" / "props.ini",
]

EOSMSV4_RADAR_PORT = 20000
EOSMSV4_OPTICAL_PORT = 19966


def backup(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.fanout_backup_{stamp}")
    shutil.copy2(path, backup_path)
    print(f"[fanout-config] backup: {backup_path}")


def read_text_with_encoding(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "gbk", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def configure_runtime_props() -> None:
    backup(RUNTIME_PROPS)
    with RUNTIME_PROPS.open("r", encoding="utf-8") as f:
        data = json.load(f)
    old_port = data.get("eoptclport")
    data["eoptclport"] = EOSMSV4_OPTICAL_PORT
    with RUNTIME_PROPS.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(
        f"[fanout-config] eosmsv4 optical port: {old_port} -> {EOSMSV4_OPTICAL_PORT}"
    )


def configure_radar_props(path: Path) -> None:
    backup(path)
    text, encoding = read_text_with_encoding(path)
    old_port = None
    lines = []
    saw_ip = False
    saw_port = False
    for line in text.splitlines():
        if line.startswith("RadarTerminalIP="):
            saw_ip = True
            lines.append("RadarTerminalIP=127.0.0.1")
        elif line.startswith("RadarTerminalPort="):
            saw_port = True
            old_port = line.split("=", 1)[1].strip()
            lines.append(f"RadarTerminalPort={EOSMSV4_RADAR_PORT}")
        else:
            lines.append(line)

    if not saw_ip:
        lines.append("RadarTerminalIP=127.0.0.1")
    if not saw_port:
        lines.append(f"RadarTerminalPort={EOSMSV4_RADAR_PORT}")

    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    print(
        f"[fanout-config] {path.name} radar port: {old_port} -> {EOSMSV4_RADAR_PORT}"
    )


def main() -> int:
    configure_runtime_props()
    for radar_prop in RADAR_PROPS:
        configure_radar_props(radar_prop)
    print("[fanout-config] done. Restart eosmsv4 after running this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
