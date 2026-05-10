# -*- coding: utf-8 -*-
"""Run main2.py and drive the interactive calibration commands automatically."""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SRC_DIR = Path(__file__).resolve().parents[1]
TRACK_LOG = SRC_DIR / "data" / "track_log.txt"
CALIB_PARAMS = SRC_DIR / "calibration_data" / "calibration_params.json"
POSITION_OFFSET = SRC_DIR / "calibration_data" / "position_offset.json"


def _reader(proc, log_path, lines):
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        for raw in iter(proc.stdout.readline, ""):
            if raw == "":
                break
            line = raw.rstrip("\n")
            lines.append(line)
            log_file.write(line + "\n")
            log_file.flush()
            print(line, flush=True)


def _latest_tracks():
    if not TRACK_LOG.exists():
        return []

    lines = TRACK_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    frame_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "|" in lines[i] and ("Radar-" in "\n".join(lines[i + 1:i + 8]) or "航迹" in lines[i]):
            frame_idx = i
            break

    search_lines = lines[frame_idx + 1:] if frame_idx >= 0 else lines[-80:]
    tracks = []
    for line in search_lines:
        if "Radar-" not in line:
            continue
        id_match = re.search(r"(Radar-\d+)", line)
        if not id_match:
            continue
        number_matches = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]
        if not number_matches:
            continue
        # The first number usually belongs to the Radar-N id; distance is the next
        # metric value on current log lines.
        range_value = number_matches[1] if len(number_matches) > 1 else number_matches[0]
        tracks.append({"track_id": id_match.group(1), "range": range_value, "line": line})

    deduped = {}
    for track in tracks:
        deduped[track["track_id"]] = track
    return sorted(deduped.values(), key=lambda item: item["range"])


def _send(proc, command):
    print(f"\n[AUTO] > {command}", flush=True)
    proc.stdin.write(command + "\n")
    proc.stdin.flush()


def _wait_for_track(timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        tracks = _latest_tracks()
        if tracks:
            return tracks[0]
        time.sleep(1.0)
    return None


def _load_result():
    result = {}
    if CALIB_PARAMS.exists():
        result["calibration_params"] = json.loads(CALIB_PARAMS.read_text(encoding="utf-8"))
    if POSITION_OFFSET.exists():
        result["position_offset"] = json.loads(POSITION_OFFSET.read_text(encoding="utf-8"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Target id, for example Radar-3 or 3.")
    parser.add_argument("--startup-timeout", type=int, default=90)
    parser.add_argument("--sample-seconds", type=int, default=25)
    parser.add_argument("--log", default=str(SRC_DIR / "data" / "auto_calibrate_run.log"))
    args = parser.parse_args()

    command = [sys.executable, "-u", str(SRC_DIR / "main2.py")]
    proc = subprocess.Popen(
        command,
        cwd=str(SRC_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines = []
    reader = threading.Thread(target=_reader, args=(proc, Path(args.log), lines), daemon=True)
    reader.start()

    try:
        time.sleep(5)
        if proc.poll() is not None:
            raise RuntimeError(f"main2.py exited early with code {proc.returncode}")

        target = args.target
        if target and target.isdigit():
            target = f"Radar-{target}"
        if target is None:
            print("[AUTO] waiting for radar tracks...", flush=True)
            chosen = _wait_for_track(args.startup_timeout)
            if chosen is None:
                raise RuntimeError("Timed out waiting for a Radar-* target in track_log.txt")
            target = chosen["track_id"]
            print(f"[AUTO] selected nearest target: {target} ({chosen['range']:.1f}m)", flush=True)

        _send(proc, "list")
        time.sleep(1)
        _send(proc, f"cal {target}")
        time.sleep(args.sample_seconds)
        _send(proc, "cstat")
        time.sleep(1)
        _send(proc, "done")
        time.sleep(3)
        _send(proc, "cres")
        time.sleep(2)

        result = _load_result()
        print("\n[AUTO] calibration result files:", flush=True)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"\n[AUTO][ERROR] {exc}", flush=True)
        return 1
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        reader.join(timeout=2)
        print(f"[AUTO] log saved to {args.log}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
