#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare debug point-track results against raw radar TRACK records.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.app_config import POINT_TRACK_RESULTS_FILE, POINT_VS_RAW_COMPARE_FILE, RAW_TRACKS_FILE  # noqa: E402
from core.track_smoothing import angle_delta_deg, smooth_rows_by_track_id  # noqa: E402


MATCHED = "matched"
UNMATCHED_POINT = "unmatched_point_track"
UNMATCHED_RAW = "unmatched_raw_track"


def load_raw_tracks(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "raw_track_id": str(row.get("track_id", "")).strip(),
                        "range": float(row["range"]),
                        "azimuth": float(row["azimuth"]),
                        "pitch": float(row["pitch"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def load_point_track_results(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        return rows
    if content.endswith(","):
        content = content[:-1].rstrip()
    if content.startswith("[") and not content.endswith("]"):
        content = content + "\n]"
    data = json.loads(content)
    for frame in data:
        timestamp = float(frame.get("timestamp", 0.0))
        for item in frame.get("targets", []):
            rows.append(
                {
                    "timestamp": timestamp,
                    "point_track_id": item.get("track_id"),
                    "range": float(item.get("range", 0.0)),
                    "azimuth": float(item.get("azimuth", 0.0)),
                    "pitch": float(item.get("pitch", 0.0)),
                }
            )
    return rows


def prepare_rows(rows, smooth, window_size):
    if not smooth:
        return [dict(row) for row in rows]
    return smooth_rows_by_track_id(rows, window_size=window_size, id_field="point_track_id" if rows and "point_track_id" in rows[0] else "raw_track_id")


def build_cost(point_track, raw_track):
    dt = abs(point_track["timestamp"] - raw_track["timestamp"])
    dr = abs(point_track["range"] - raw_track["range"])
    daz = abs(angle_delta_deg(point_track["azimuth"], raw_track["azimuth"]))
    dpitch = abs(point_track["pitch"] - raw_track["pitch"])
    score = dt * 40.0 + dr * 0.10 + daz * 12.0 + dpitch * 14.0
    return dt, dr, daz, dpitch, score


def greedy_one_to_one(point_tracks, raw_tracks, time_window):
    candidates = []
    for pi, point_track in enumerate(point_tracks):
        for ri, raw_track in enumerate(raw_tracks):
            dt, dr, daz, dpitch, score = build_cost(point_track, raw_track)
            if dt <= time_window:
                candidates.append((score, pi, ri, dt, dr, daz, dpitch))

    candidates.sort(key=lambda item: item[0])
    used_point = set()
    used_raw = set()
    matches = []

    for score, pi, ri, dt, dr, daz, dpitch in candidates:
        if pi in used_point or ri in used_raw:
            continue
        used_point.add(pi)
        used_raw.add(ri)
        matches.append(
            {
                "match_status": MATCHED,
                "point_track_id": point_tracks[pi]["point_track_id"],
                "raw_track_id": raw_tracks[ri]["raw_track_id"],
                "timestamp_point": point_tracks[pi]["timestamp"],
                "timestamp_raw": raw_tracks[ri]["timestamp"],
                "time_diff": dt,
                "point_range": point_tracks[pi]["range"],
                "raw_range": raw_tracks[ri]["range"],
                "range_diff": point_tracks[pi]["range"] - raw_tracks[ri]["range"],
                "point_azimuth": point_tracks[pi]["azimuth"],
                "raw_azimuth": raw_tracks[ri]["azimuth"],
                "azimuth_diff": angle_delta_deg(point_tracks[pi]["azimuth"], raw_tracks[ri]["azimuth"]),
                "point_pitch": point_tracks[pi]["pitch"],
                "raw_pitch": raw_tracks[ri]["pitch"],
                "pitch_diff": point_tracks[pi]["pitch"] - raw_tracks[ri]["pitch"],
                "point_raw_azimuth": point_tracks[pi].get("raw_azimuth", point_tracks[pi]["azimuth"]),
                "raw_raw_azimuth": raw_tracks[ri].get("raw_azimuth", raw_tracks[ri]["azimuth"]),
                "raw_azimuth_diff": angle_delta_deg(
                    point_tracks[pi].get("raw_azimuth", point_tracks[pi]["azimuth"]),
                    raw_tracks[ri].get("raw_azimuth", raw_tracks[ri]["azimuth"]),
                ),
                "point_raw_pitch": point_tracks[pi].get("raw_pitch", point_tracks[pi]["pitch"]),
                "raw_raw_pitch": raw_tracks[ri].get("raw_pitch", raw_tracks[ri]["pitch"]),
                "raw_pitch_diff": point_tracks[pi].get("raw_pitch", point_tracks[pi]["pitch"]) - raw_tracks[ri].get("raw_pitch", raw_tracks[ri]["pitch"]),
                "score": score,
            }
        )

    for pi, point_track in enumerate(point_tracks):
        if pi in used_point:
            continue
        matches.append(
            {
                "match_status": UNMATCHED_POINT,
                "point_track_id": point_track["point_track_id"],
                "raw_track_id": "",
                "timestamp_point": point_track["timestamp"],
                "timestamp_raw": "",
                "time_diff": "",
                "point_range": point_track["range"],
                "raw_range": "",
                "range_diff": "",
                "point_azimuth": point_track["azimuth"],
                "raw_azimuth": "",
                "azimuth_diff": "",
                "point_pitch": point_track["pitch"],
                "raw_pitch": "",
                "pitch_diff": "",
                "score": "",
            }
        )

    for ri, raw_track in enumerate(raw_tracks):
        if ri in used_raw:
            continue
        matches.append(
            {
                "match_status": UNMATCHED_RAW,
                "point_track_id": "",
                "raw_track_id": raw_track["raw_track_id"],
                "timestamp_point": "",
                "timestamp_raw": raw_track["timestamp"],
                "time_diff": "",
                "point_range": "",
                "raw_range": raw_track["range"],
                "range_diff": "",
                "point_azimuth": "",
                "raw_azimuth": raw_track["azimuth"],
                "azimuth_diff": "",
                "point_pitch": "",
                "raw_pitch": raw_track["pitch"],
                "pitch_diff": "",
                "score": "",
            }
        )

    return matches


def summarize(matches):
    matched = [m for m in matches if m["match_status"] == MATCHED]
    unmatched_point = sum(1 for m in matches if m["match_status"] == UNMATCHED_POINT)
    unmatched_raw = sum(1 for m in matches if m["match_status"] == UNMATCHED_RAW)

    summary = {
        "matched_pairs": len(matched),
        "unmatched_point_tracks": unmatched_point,
        "unmatched_raw_tracks": unmatched_raw,
    }

    if matched:
        range_abs = [abs(m["range_diff"]) for m in matched]
        az_abs = [abs(m["azimuth_diff"]) for m in matched]
        pitch_abs = [abs(m["pitch_diff"]) for m in matched]
        summary.update(
            {
                "mean_abs_range_diff_m": statistics.mean(range_abs),
                "max_abs_range_diff_m": max(range_abs),
                "mean_abs_azimuth_diff_deg": statistics.mean(az_abs),
                "max_abs_azimuth_diff_deg": max(az_abs),
                "mean_abs_pitch_diff_deg": statistics.mean(pitch_abs),
                "max_abs_pitch_diff_deg": max(pitch_abs),
            }
        )
        raw_az_abs = [abs(m["raw_azimuth_diff"]) for m in matched if "raw_azimuth_diff" in m]
        raw_pitch_abs = [abs(m["raw_pitch_diff"]) for m in matched if "raw_pitch_diff" in m]
        if raw_az_abs and raw_pitch_abs:
            summary.update(
                {
                    "mean_abs_raw_azimuth_diff_deg": statistics.mean(raw_az_abs),
                    "mean_abs_raw_pitch_diff_deg": statistics.mean(raw_pitch_abs),
                }
            )
    return summary


def print_summary(summary):
    print("[point-debug] matched_pairs:", summary.get("matched_pairs", 0))
    print("[point-debug] unmatched_point_tracks:", summary.get("unmatched_point_tracks", 0))
    print("[point-debug] unmatched_raw_tracks:", summary.get("unmatched_raw_tracks", 0))
    if "mean_abs_range_diff_m" in summary:
        print(
            "[point-debug] mean abs error: "
            f"range={summary['mean_abs_range_diff_m']:.2f}m, "
            f"az={summary['mean_abs_azimuth_diff_deg']:.2f}deg, "
            f"pitch={summary['mean_abs_pitch_diff_deg']:.2f}deg"
        )
        print(
            "[point-debug] max abs error: "
            f"range={summary['max_abs_range_diff_m']:.2f}m, "
            f"az={summary['max_abs_azimuth_diff_deg']:.2f}deg, "
            f"pitch={summary['max_abs_pitch_diff_deg']:.2f}deg"
        )
        if "mean_abs_raw_azimuth_diff_deg" in summary:
            print(
                "[point-debug] raw mean abs error before smoothing: "
                f"az={summary['mean_abs_raw_azimuth_diff_deg']:.2f}deg, "
                f"pitch={summary['mean_abs_raw_pitch_diff_deg']:.2f}deg"
            )


def write_compare_csv(path, matches):
    fieldnames = [
        "match_status",
        "point_track_id",
        "raw_track_id",
        "timestamp_point",
        "timestamp_raw",
        "time_diff",
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
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def run_compare(raw_path, point_track_path, output_path, time_window=0.5, smooth=True, smooth_window=5):
    raw_rows = load_raw_tracks(raw_path)
    point_rows = load_point_track_results(point_track_path)
    raw_rows = prepare_rows(raw_rows, smooth=smooth, window_size=smooth_window)
    point_rows = prepare_rows(point_rows, smooth=smooth, window_size=smooth_window)
    matches = greedy_one_to_one(point_rows, raw_rows, time_window=time_window)
    summary = summarize(matches)
    write_compare_csv(output_path, matches)
    return summary, matches


def main():
    parser = argparse.ArgumentParser(description="Compare debug point-track results against raw track packets")
    parser.add_argument("--raw", default=RAW_TRACKS_FILE)
    parser.add_argument("--point-track", default=POINT_TRACK_RESULTS_FILE)
    parser.add_argument("--output", default=POINT_VS_RAW_COMPARE_FILE)
    parser.add_argument("--time-window", type=float, default=0.5)
    parser.add_argument("--no-smooth", action="store_true", help="Use raw one-frame azimuth/pitch values")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.raw):
        print(f"[point-debug] raw file missing: {args.raw}")
        return 2
    if not os.path.exists(args.point_track):
        print(f"[point-debug] point-track file missing: {args.point_track}")
        return 3

    summary, _ = run_compare(
        args.raw,
        args.point_track,
        args.output,
        time_window=args.time_window,
        smooth=not args.no_smooth,
        smooth_window=args.smooth_window,
    )
    print_summary(summary)
    print(f"[point-debug] compare csv: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
