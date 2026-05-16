#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline analysis for point-track results vs raw radar tracks.

This script does not require shared timestamps. It focuses on:
1. range-pitch trend comparison
2. nearest-neighbor matching by range/azimuth
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys


SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.app_config import POINT_TRACK_RESULTS_FILE, RAW_TRACKS_FILE  # noqa: E402
from core.track_smoothing import angle_delta_deg, smooth_rows_by_track_id  # noqa: E402


def load_raw_tracks(path):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "track_id": str(row.get("track_id", "")).strip(),
                        "range": float(row["range"]),
                        "azimuth": float(row["azimuth"]),
                        "pitch": float(row["pitch"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def load_point_track_results(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.endswith(","):
        content = content[:-1].rstrip()
    if content.startswith("[") and not content.endswith("]"):
        content = content + "\n]"
    frames = json.loads(content)

    rows = []
    for frame in frames:
        for item in frame.get("targets", []):
            rows.append(
                {
                    "timestamp": float(frame.get("timestamp", 0.0)),
                    "track_id": item.get("track_id"),
                    "range": float(item.get("range", 0.0)),
                    "azimuth": float(item.get("azimuth", 0.0)),
                    "pitch": float(item.get("pitch", 0.0)),
                }
            )
    return rows


def summarize_trend(name, rows):
    if not rows:
        return {"name": name, "count": 0}
    ranges = [r["range"] for r in rows]
    pitches = [r["pitch"] for r in rows]
    return {
        "name": name,
        "count": len(rows),
        "range_min": min(ranges),
        "range_max": max(ranges),
        "pitch_mean": statistics.mean(pitches),
        "pitch_std": statistics.pstdev(pitches) if len(pitches) > 1 else 0.0,
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
    }


def nearest_match(point_rows, raw_rows, max_range_gap=80.0, max_az_gap=20.0):
    matches = []
    used_raw = set()
    candidates = []

    for pi, point in enumerate(point_rows):
        for ri, raw in enumerate(raw_rows):
            dr = abs(point["range"] - raw["range"])
            da = abs(angle_delta_deg(point["azimuth"], raw["azimuth"]))
            if dr > max_range_gap or da > max_az_gap:
                continue
            dp = abs(point["pitch"] - raw["pitch"])
            score = dr * 0.10 + da * 10.0 + dp * 12.0
            candidates.append((score, pi, ri, dr, da, dp))

    candidates.sort(key=lambda item: item[0])
    used_point = set()

    for score, pi, ri, dr, da, dp in candidates:
        if pi in used_point or ri in used_raw:
            continue
        used_point.add(pi)
        used_raw.add(ri)
        matches.append(
            {
                "point_track_id": point_rows[pi]["track_id"],
                "raw_track_id": raw_rows[ri]["track_id"],
                "point_range": point_rows[pi]["range"],
                "raw_range": raw_rows[ri]["range"],
                "range_diff": point_rows[pi]["range"] - raw_rows[ri]["range"],
                "point_azimuth": point_rows[pi]["azimuth"],
                "raw_azimuth": raw_rows[ri]["azimuth"],
                "azimuth_diff": angle_delta_deg(point_rows[pi]["azimuth"], raw_rows[ri]["azimuth"]),
                "point_pitch": point_rows[pi]["pitch"],
                "raw_pitch": raw_rows[ri]["pitch"],
                "pitch_diff": point_rows[pi]["pitch"] - raw_rows[ri]["pitch"],
                "point_raw_azimuth": point_rows[pi].get("raw_azimuth", point_rows[pi]["azimuth"]),
                "raw_raw_azimuth": raw_rows[ri].get("raw_azimuth", raw_rows[ri]["azimuth"]),
                "raw_azimuth_diff": angle_delta_deg(
                    point_rows[pi].get("raw_azimuth", point_rows[pi]["azimuth"]),
                    raw_rows[ri].get("raw_azimuth", raw_rows[ri]["azimuth"]),
                ),
                "point_raw_pitch": point_rows[pi].get("raw_pitch", point_rows[pi]["pitch"]),
                "raw_raw_pitch": raw_rows[ri].get("raw_pitch", raw_rows[ri]["pitch"]),
                "raw_pitch_diff": point_rows[pi].get("raw_pitch", point_rows[pi]["pitch"]) - raw_rows[ri].get("raw_pitch", raw_rows[ri]["pitch"]),
                "score": score,
            }
        )
    return matches


def print_trend(summary):
    if summary["count"] == 0:
        print(f"[offline] {summary['name']}: no rows")
        return
    print(
        f"[offline] {summary['name']}: count={summary['count']} "
        f"range={summary['range_min']:.1f}-{summary['range_max']:.1f}m "
        f"pitch_mean={summary['pitch_mean']:.2f}deg "
        f"pitch_std={summary['pitch_std']:.2f}deg "
        f"pitch_min={summary['pitch_min']:.2f}deg "
        f"pitch_max={summary['pitch_max']:.2f}deg"
    )


def print_match_summary(matches):
    if not matches:
        print("[offline] no nearest matches found")
        return
    range_abs = [abs(m["range_diff"]) for m in matches]
    az_abs = [abs(m["azimuth_diff"]) for m in matches]
    pitch_abs = [abs(m["pitch_diff"]) for m in matches]
    print(f"[offline] nearest matches: {len(matches)}")
    print(
        "[offline] matched mean abs error: "
        f"range={statistics.mean(range_abs):.2f}m, "
        f"az={statistics.mean(az_abs):.2f}deg, "
        f"pitch={statistics.mean(pitch_abs):.2f}deg"
    )
    print(
        "[offline] matched max abs error: "
        f"range={max(range_abs):.2f}m, "
        f"az={max(az_abs):.2f}deg, "
        f"pitch={max(pitch_abs):.2f}deg"
    )
    raw_az_abs = [abs(m["raw_azimuth_diff"]) for m in matches if "raw_azimuth_diff" in m]
    raw_pitch_abs = [abs(m["raw_pitch_diff"]) for m in matches if "raw_pitch_diff" in m]
    if raw_az_abs and raw_pitch_abs:
        print(
            "[offline] raw mean abs error before smoothing: "
            f"az={statistics.mean(raw_az_abs):.2f}deg, "
            f"pitch={statistics.mean(raw_pitch_abs):.2f}deg"
        )

    worst = sorted(matches, key=lambda m: abs(m["pitch_diff"]), reverse=True)[:10]
    print("[offline] top pitch-diff rows:")
    for item in worst:
        print(
            f"  point={item['point_track_id']:<8} raw={item['raw_track_id']:<8} "
            f"range={item['point_range']:.1f}/{item['raw_range']:.1f}m "
            f"pitch={item['point_pitch']:.2f}/{item['raw_pitch']:.2f}deg "
            f"dp={item['pitch_diff']:+.2f}deg"
        )


def write_match_csv(path, matches):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "point_track_id",
                "raw_track_id",
                "point_range",
                "raw_range",
                "range_diff",
                "point_azimuth",
                "raw_azimuth",
                "azimuth_diff",
                "point_pitch",
                "raw_pitch",
                "pitch_diff",
                "point_raw_azimuth",
                "raw_raw_azimuth",
                "raw_azimuth_diff",
                "point_raw_pitch",
                "raw_raw_pitch",
                "raw_pitch_diff",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(matches)


def main():
    parser = argparse.ArgumentParser(description="Offline analysis for point-track results vs raw tracks")
    parser.add_argument("--raw", default=RAW_TRACKS_FILE)
    parser.add_argument("--point-track", default=POINT_TRACK_RESULTS_FILE)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--no-smooth", action="store_true", help="Use raw one-frame azimuth/pitch values")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.raw):
        print(f"[offline] raw file missing: {args.raw}")
        return 2
    if not os.path.exists(args.point_track):
        print(f"[offline] point-track file missing: {args.point_track}")
        return 3

    raw_rows = load_raw_tracks(args.raw)
    point_rows = load_point_track_results(args.point_track)
    if not args.no_smooth:
        raw_rows = smooth_rows_by_track_id(raw_rows, window_size=args.smooth_window, id_field="track_id")
        point_rows = smooth_rows_by_track_id(point_rows, window_size=args.smooth_window, id_field="track_id")

    print_trend(summarize_trend("raw_tracks", raw_rows))
    print_trend(summarize_trend("point_track_results", point_rows))

    matches = nearest_match(point_rows, raw_rows)
    print_match_summary(matches)

    if args.output_csv:
        write_match_csv(args.output_csv, matches)
        print(f"[offline] wrote csv: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
