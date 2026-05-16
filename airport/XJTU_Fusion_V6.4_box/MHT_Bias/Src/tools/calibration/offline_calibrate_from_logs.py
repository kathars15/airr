# -*- coding: utf-8 -*-
r"""Offline bearing-only radar/optical position calibration from saved logs.

Inputs:
  - data/track_log.txt: radar tracks with wall-clock frame timestamps
  - data/optical_measurements.csv: optical angle packets saved while tracking

Example:
  python .\tools\calibration\offline_calibrate_from_logs.py --target 5
  python .\tools\calibration\offline_calibrate_from_logs.py --target Radar-12 --window 0.5 --save
"""

import argparse
import bisect
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.app_config import CALIBRATION_MIN_RANGE, OPTICAL_MEASUREMENTS_FILE, TRACK_LOG_FILE  # noqa: E402
from core.calibration import DEFAULT_CALIBRATION_DATA_DIR  # noqa: E402
from tools.cal_offset import calculate_calibration_from_measurements  # noqa: E402


def normalize_track_id(value):
    text = str(value).strip()
    if text.isdigit():
        return f"Radar-{text}"
    return text


def parse_track_log(path, target_id=None):
    target_id = normalize_track_id(target_id) if target_id else None
    records = []
    current_ts = None

    frame_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    track_re = re.compile(r"^\s*(Radar-\d+)(?:\(raw=([^)]+)\))?:")
    number_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            frame_match = frame_re.match(line)
            if frame_match:
                dt = datetime.strptime(frame_match.group(1), "%Y-%m-%d %H:%M:%S")
                current_ts = time.mktime(dt.timetuple())
                continue

            if current_ts is None:
                continue

            track_match = track_re.match(line)
            if not track_match:
                continue

            track_id = track_match.group(1)
            if target_id and track_id != target_id:
                continue

            payload = line.split(":", 1)[1] if ":" in line else line
            nums = [float(x) for x in number_re.findall(payload)]
            if len(nums) < 4:
                continue

            records.append({
                "timestamp": current_ts,
                "track_id": track_id,
                "raw_display_id": track_match.group(2),
                "range": nums[0],
                "azimuth": nums[1],
                "pitch": nums[2],
                "speed": nums[3],
                "line": line.strip(),
            })

    return records


def parse_optical_csv(path):
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                status = int(float(row.get("status", "")))
                if status != 2:
                    continue
                records.append({
                    "timestamp": float(row["timestamp"]),
                    "azimuth": float(row["azimuth"]),
                    "pitch": float(row["pitch"]),
                    "status": status,
                    "range": float(row["range"]) if row.get("range") else 0.0,
                })
            except (KeyError, TypeError, ValueError):
                continue

    records.sort(key=lambda item: item["timestamp"])
    return records


def pair_by_time(radar_records, optical_records, window):
    optical_ts = [item["timestamp"] for item in optical_records]
    pairs = []

    for radar in radar_records:
        idx = bisect.bisect_left(optical_ts, radar["timestamp"])
        candidates = []
        if idx < len(optical_records):
            candidates.append(optical_records[idx])
        if idx > 0:
            candidates.append(optical_records[idx - 1])
        if not candidates:
            continue

        optical = min(candidates, key=lambda item: abs(item["timestamp"] - radar["timestamp"]))
        dt = abs(optical["timestamp"] - radar["timestamp"])
        if dt <= window:
            pairs.append((radar, optical, dt))

    return pairs


def filter_pairs_by_range(pairs, min_range):
    return [
        pair for pair in pairs
        if float(pair[0].get("range", 0.0) or 0.0) >= min_range
    ]


def radar_to_position(azimuth, pitch, range_m):
    az = math.radians(azimuth)
    el = math.radians(pitch)
    return np.array([
        range_m * math.cos(el) * math.sin(az),
        range_m * math.cos(el) * math.cos(az),
        range_m * math.sin(el),
    ], dtype=float)


def optical_to_direction(azimuth, pitch):
    az = math.radians(azimuth)
    el = math.radians(pitch)
    return np.array([
        math.cos(el) * math.sin(az),
        math.cos(el) * math.cos(az),
        math.sin(el),
    ], dtype=float)


def robust_mask(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return np.ones(len(values), dtype=bool)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-9:
        return np.ones(len(values), dtype=bool)
    robust_z = 0.6745 * (values - med) / mad
    return np.abs(robust_z) <= 3.5


def solve_offset(positions, dirs):
    a_rows = []
    b_rows = []
    for p, u in zip(positions, dirs):
        ux = np.array([
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0],
        ])
        a_rows.append(ux)
        b_rows.append(ux @ p)
    A = np.vstack(a_rows)
    b = np.concatenate(b_rows)
    offset, *_ = np.linalg.lstsq(A, b, rcond=None)
    return offset


def angular_errors(offset, positions, dirs):
    errors = []
    for p, u in zip(positions, dirs):
        rel = p - offset
        norm = np.linalg.norm(rel)
        if norm <= 1e-9:
            continue
        dot = np.clip(float(np.dot(rel / norm, u)), -1.0, 1.0)
        errors.append(math.degrees(math.acos(dot)))
    return np.asarray(errors, dtype=float)


def calculate_offset(pairs):
    measurements = pairs_to_cal_offset_measurements(pairs)
    result = calculate_calibration_from_measurements(measurements)
    if not result.get("success"):
        raise RuntimeError(result.get("reason", "cal_offset calculation failed"))

    validation = result.get("validation", {})
    az_errors = np.asarray(validation.get("azimuth_errors", []), dtype=float)
    pitch_errors = np.asarray(validation.get("pitch_errors", []), dtype=float)
    errors = np.sqrt(az_errors ** 2 + pitch_errors ** 2)

    return result["offset"], errors, pairs


def pairs_to_cal_offset_measurements(pairs):
    measurements = []
    for radar, optical, _ in pairs:
        measurements.append({
            "radar_az": radar["azimuth"],
            "radar_pitch": radar["pitch"],
            "radar_range": radar["range"],
            "optical_az": optical["azimuth"],
            "optical_pitch": optical["pitch"],
        })
    return measurements


def pairs_to_source_measurements(pairs, target_id=None, min_range=None, window=None):
    samples = []
    for radar, optical, dt in pairs:
        samples.append({
            "index": len(samples) + 1,
            "time_diff_sec": float(dt),
            "radar": {
                "timestamp": float(radar["timestamp"]),
                "track_id": radar.get("track_id"),
                "raw_display_id": radar.get("raw_display_id"),
                "azimuth": float(radar["azimuth"]),
                "pitch": float(radar["pitch"]),
                "range": float(radar["range"]),
                "speed": float(radar["speed"]) if radar.get("speed") is not None else None,
            },
            "optical": {
                "timestamp": float(optical["timestamp"]),
                "azimuth": float(optical["azimuth"]),
                "pitch": float(optical["pitch"]),
                "range": float(optical.get("range", 0.0) or 0.0),
                "status": optical.get("status"),
            },
        })

    return {
        "target_id": normalize_track_id(target_id) if target_id else None,
        "sample_count": len(samples),
        "min_radar_range_m": float(min_range) if min_range is not None else None,
        "pair_time_window_sec": float(window) if window is not None else None,
        "samples": samples,
    }


def summarize_raw_angle_diffs(pairs):
    az_diffs = []
    pitch_diffs = []
    for radar, optical, _ in pairs:
        az_diff = ((radar["azimuth"] - optical["azimuth"] + 180.0) % 360.0) - 180.0
        pitch_diff = radar["pitch"] - optical["pitch"]
        az_diffs.append(az_diff)
        pitch_diffs.append(pitch_diff)
    return np.asarray(az_diffs, dtype=float), np.asarray(pitch_diffs, dtype=float)


def save_position_offset(offset, errors, path, source_measurements=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "dx": float(offset[0]),
        "dy": float(offset[1]),
        "dz": float(offset[2]),
        "timestamp": time.time(),
        "sample_count": int(len(errors)),
        "use_position": True,
        "method": "offline_bearing_only",
    }
    if source_measurements is not None:
        data["source_measurements"] = source_measurements
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def main():
    parser = argparse.ArgumentParser(description="Offline radar/optical bearing-only calibration")
    parser.add_argument("--target", required=True, help="Target ID, e.g. 5 or Radar-5")
    parser.add_argument("--track-log", default=TRACK_LOG_FILE)
    parser.add_argument("--optical", default=OPTICAL_MEASUREMENTS_FILE)
    parser.add_argument("--window", type=float, default=0.5, help="Max radar/optical time difference in seconds")
    parser.add_argument(
        "--min-range",
        type=float,
        default=CALIBRATION_MIN_RANGE,
        help="Only use radar/optical pairs whose radar range is at least this many meters",
    )
    parser.add_argument("--save", action="store_true", help="Save result to calibration_data/position_offset.json")
    parser.add_argument("--force", action="store_true", help="Allow saving even when the estimated offset is suspicious")
    parser.add_argument("--max-offset", type=float, default=20.0, help="Reject --save if offset norm exceeds this many meters")
    parser.add_argument(
        "--output",
        default=os.path.join(DEFAULT_CALIBRATION_DATA_DIR, "position_offset.json"),
        help="Output JSON path when --save is used",
    )
    args = parser.parse_args()

    radar_records = parse_track_log(args.track_log, args.target)
    optical_records = parse_optical_csv(args.optical)
    all_pairs = pair_by_time(radar_records, optical_records, args.window)
    pairs = filter_pairs_by_range(all_pairs, args.min_range)

    print(f"[offline] target: {normalize_track_id(args.target)}")
    print(f"[offline] radar records: {len(radar_records)}")
    print(f"[offline] optical records(status=2): {len(optical_records)}")
    print(f"[offline] paired samples(window={args.window:.2f}s): {len(all_pairs)}")
    print(f"[offline] paired samples(range>={args.min_range:.0f}m): {len(pairs)}")

    if len(pairs) < 5:
        print("[offline] not enough pairs; need at least 5")
        return 2

    offset, errors, kept_pairs = calculate_offset(pairs)
    az_diffs, pitch_diffs = summarize_raw_angle_diffs(kept_pairs)
    positions = np.asarray([
        radar_to_position(r["azimuth"], r["pitch"], r["range"])
        for r, _, _ in kept_pairs
    ], dtype=float)
    dirs = np.asarray([
        optical_to_direction(o["azimuth"], o["pitch"])
        for _, o, _ in kept_pairs
    ], dtype=float)
    zero_errors = angular_errors(np.zeros(3), positions, dirs)
    offset_norm = float(np.linalg.norm(offset))

    print(f"[offline] kept samples after robust filter: {len(kept_pairs)}")
    print(
        "[offline] raw angle diff radar-optical: "
        f"az mean={np.mean(az_diffs):.2f}deg std={np.std(az_diffs):.2f}deg, "
        f"pitch mean={np.mean(pitch_diffs):.2f}deg std={np.std(pitch_diffs):.2f}deg"
    )
    print(f"[offline] zero-offset angular residual: mean={np.mean(zero_errors):.2f}deg, std={np.std(zero_errors):.2f}deg")
    print(f"[offline] optical offset relative to radar:")
    print(f"          dx={offset[0]:.2f}m, dy={offset[1]:.2f}m, dz={offset[2]:.2f}m")
    print(f"[offline] offset norm: {offset_norm:.2f}m")
    print(f"[offline] angular residual: mean={np.mean(errors):.2f}deg, std={np.std(errors):.2f}deg")
    if offset_norm > args.max_offset:
        print(
            f"[offline][warning] estimated offset exceeds {args.max_offset:.1f}m. "
            "This is usually angle bias, wrong target pairing, or weak geometry rather than real installation offset."
        )

    print("[offline] sample pairs:")
    for radar, optical, dt in kept_pairs[:8]:
        print(
            f"  dt={dt:.2f}s "
            f"radar({radar['azimuth']:.1f}deg,{radar['pitch']:.1f}deg,{radar['range']:.0f}m) -> "
            f"optical({optical['azimuth']:.1f}deg,{optical['pitch']:.1f}deg)"
        )

    if args.save:
        if offset_norm > args.max_offset and not args.force:
            print("[offline] not saved. Use --force only if you are sure this large offset is physically correct.")
            return 3
        source_measurements = pairs_to_source_measurements(
            kept_pairs,
            target_id=args.target,
            min_range=args.min_range,
            window=args.window,
        )
        data = save_position_offset(offset, errors, args.output, source_measurements)
        print(f"[offline] saved: {args.output}")
        print(f"[offline] saved samples: {data['sample_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
