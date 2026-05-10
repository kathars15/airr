# -*- coding: utf-8 -*-
"""Check whether radar TRACK range behaves like slant range or horizontal range.

The radar protocol parser exposes both range and height. This script compares:
  slant model:      height ~= range * sin(pitch)
  horizontal model: height ~= range * tan(pitch)
"""

import csv
import math
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.app_config import RAW_TRACKS_FILE  # noqa: E402


def _as_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(path):
    with open(path, newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = "timestamp" in sample.splitlines()[0].lower()
        if has_header:
            reader = csv.DictReader(f)
            return list(reader)

        rows = []
        for row in csv.reader(f):
            if len(row) >= 8:
                rows.append({
                    "timestamp": row[0],
                    "track_id": row[1],
                    "range": row[2],
                    "azimuth": row[3],
                    "pitch": row[4],
                    "speed": row[5],
                    "target_type": row[6],
                    "height": row[7],
                })
        return rows


def _median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else RAW_TRACKS_FILE
    if not os.path.exists(path):
        print(f"[range-mode] raw track file not found: {path}")
        return 1

    rows = _load_rows(path)
    usable = []
    for row in rows:
        rng = _as_float(row.get("range"))
        pitch = _as_float(row.get("pitch"))
        height = _as_float(row.get("height"))
        if rng is None or pitch is None or height is None:
            continue
        if rng <= 0 or abs(pitch) >= 89.0:
            continue
        pitch_rad = math.radians(pitch)
        slant_h = rng * math.sin(pitch_rad)
        horizontal_h = rng * math.tan(pitch_rad)
        usable.append({
            "track_id": row.get("track_id", ""),
            "range": rng,
            "pitch": pitch,
            "height": height,
            "slant_error": abs(height - slant_h),
            "horizontal_error": abs(height - horizontal_h),
        })

    if not usable:
        print("[range-mode] no usable rows with height. Run main2.py after this patch to collect new raw_tracks rows.")
        return 2

    slant_errors = [r["slant_error"] for r in usable]
    horizontal_errors = [r["horizontal_error"] for r in usable]
    slant_med = _median(slant_errors)
    horizontal_med = _median(horizontal_errors)
    slant_mean = sum(slant_errors) / len(slant_errors)
    horizontal_mean = sum(horizontal_errors) / len(horizontal_errors)

    print(f"[range-mode] file: {path}")
    print(f"[range-mode] usable rows: {len(usable)}")
    print(f"[range-mode] slant model      median error={slant_med:.2f}m mean error={slant_mean:.2f}m")
    print(f"[range-mode] horizontal model  median error={horizontal_med:.2f}m mean error={horizontal_mean:.2f}m")

    if slant_med < horizontal_med * 0.7:
        print("[range-mode] verdict: range is closer to 3D slant/euclidean distance")
    elif horizontal_med < slant_med * 0.7:
        print("[range-mode] verdict: range is closer to horizontal ground distance")
    else:
        print("[range-mode] verdict: inconclusive; check radar protocol or collect targets with larger pitch variation")

    print("[range-mode] sample rows:")
    for row in usable[-5:]:
        print(
            "  "
            f"id={row['track_id']} range={row['range']:.1f}m pitch={row['pitch']:.1f}deg "
            f"height={row['height']:.1f}m "
            f"slant_err={row['slant_error']:.1f}m horizontal_err={row['horizontal_error']:.1f}m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
