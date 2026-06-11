# -*- coding: utf-8 -*-
"""Offline radar-optical time sync diagnosis for saved calibration sessions."""

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.calibration import DEFAULT_CALIBRATION_DATA_DIR  # noqa: E402
from tools.calibration.replay_calibration_session import (  # noqa: E402
    filter_source_measurements,
    list_session_files,
    load_session,
    parse_range_arg,
)


def angle_diff_deg(target_angle, source_angle):
    return (float(target_angle) - float(source_angle) + 180.0) % 360.0 - 180.0


def compute_stats(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def prepare_samples(source_measurements, range_filter=None):
    samples = []
    for item in source_measurements.get("samples", []):
        radar = item.get("radar", {})
        optical = item.get("optical", {})
        radar_range = float(radar.get("range", 0) or 0)
        if range_filter is not None:
            range_min, range_max = range_filter
            if not (range_min <= radar_range <= range_max):
                continue
        samples.append({
            "index": int(item.get("index", len(samples) + 1)),
            "radar_range": radar_range,
            "radar_ts": float(radar.get("timestamp", 0) or 0),
            "optical_ts": float(optical.get("timestamp", 0) or 0),
            "radar_az": float(radar.get("azimuth", 0) or 0),
            "radar_pitch": float(radar.get("pitch", 0) or 0),
            "optical_az": float(optical.get("azimuth", 0) or 0),
            "optical_pitch": float(optical.get("pitch", 0) or 0),
            "pair_time_diff_sec": float(item.get("time_diff_sec", 0) or 0),
        })
    samples.sort(key=lambda item: item["optical_ts"])
    return samples


def interpolate_errors(samples, delta_t):
    if len(samples) < 2:
        return None

    shifted_times = np.array([item["radar_ts"] + delta_t for item in samples], dtype=float)
    optical_times = np.array([item["optical_ts"] for item in samples], dtype=float)

    overlap_start = max(float(np.min(shifted_times)), float(np.min(optical_times)))
    overlap_end = min(float(np.max(shifted_times)), float(np.max(optical_times)))
    if overlap_end <= overlap_start:
        return None

    mask = (optical_times >= overlap_start) & (optical_times <= overlap_end)
    if int(np.count_nonzero(mask)) < 2:
        return None

    radar_az = np.unwrap(np.deg2rad([item["radar_az"] for item in samples]))
    radar_pitch = np.array([item["radar_pitch"] for item in samples], dtype=float)
    optical_az = np.array([item["optical_az"] for item in samples], dtype=float)[mask]
    optical_pitch = np.array([item["optical_pitch"] for item in samples], dtype=float)[mask]
    optical_times = optical_times[mask]

    interp_az = np.rad2deg(np.interp(optical_times, shifted_times, radar_az))
    interp_pitch = np.interp(optical_times, shifted_times, radar_pitch)
    az_errors = np.array([angle_diff_deg(optical_az[i], interp_az[i]) for i in range(len(optical_times))], dtype=float)
    pitch_errors = optical_pitch - interp_pitch
    az_rmse = float(np.sqrt(np.mean(az_errors ** 2)))
    pitch_rmse = float(np.sqrt(np.mean(pitch_errors ** 2)))
    total_rmse = float(math.sqrt(az_rmse ** 2 + pitch_rmse ** 2))

    return {
        "az_errors": az_errors,
        "pitch_errors": pitch_errors,
        "az_rmse": az_rmse,
        "pitch_rmse": pitch_rmse,
        "total_rmse": total_rmse,
        "overlap_count": int(optical_times.size),
        "overlap_mask": mask,
    }


def scan_time_offsets(samples, delta_min, delta_max, delta_step):
    rows = []
    delta = delta_min
    while delta <= delta_max + 1e-9:
        result = interpolate_errors(samples, delta)
        if result is not None:
            rows.append({
                "delta_t": float(delta),
                "az_rmse": result["az_rmse"],
                "pitch_rmse": result["pitch_rmse"],
                "total_rmse": result["total_rmse"],
            })
        delta += delta_step
    return rows


def suspicious_samples(samples, az_errors, pitch_errors, limit=5):
    rows = []
    for item, az_err, pitch_err in zip(samples, az_errors, pitch_errors):
        rows.append({
            "index": item["index"],
            "radar_range": item["radar_range"],
            "pair_time_diff_sec": item["pair_time_diff_sec"],
            "az_error_deg": float(az_err),
            "pitch_error_deg": float(pitch_err),
            "combined_error_deg": float(math.sqrt(float(az_err) ** 2 + float(pitch_err) ** 2)),
        })
    rows.sort(key=lambda item: item["combined_error_deg"], reverse=True)
    return rows[:limit]


def save_plots(output_dir, prefix, rows, before_result, after_result):
    os.makedirs(output_dir, exist_ok=True)

    rmse_path = os.path.join(output_dir, f"{prefix}_delta_rmse.png")
    plt.figure(figsize=(10, 6))
    plt.plot([row["delta_t"] for row in rows], [row["az_rmse"] for row in rows], label="azimuth RMSE")
    plt.plot([row["delta_t"] for row in rows], [row["pitch_rmse"] for row in rows], label="pitch RMSE")
    plt.plot([row["delta_t"] for row in rows], [row["total_rmse"] for row in rows], label="total RMSE")
    plt.xlabel("delta_t (s)")
    plt.ylabel("RMSE (deg)")
    plt.title("Radar-Optical Time Offset Scan")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rmse_path, dpi=150)
    plt.close()

    compare_path = os.path.join(output_dir, f"{prefix}_before_after_errors.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(before_result["az_errors"], label="before", color="tab:red")
    axes[0].plot(after_result["az_errors"], label="after", color="tab:green")
    axes[0].set_ylabel("az error (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(before_result["pitch_errors"], label="before", color="tab:red")
    axes[1].plot(after_result["pitch_errors"], label="after", color="tab:green")
    axes[1].set_ylabel("pitch error (deg)")
    axes[1].set_xlabel("sample index")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Radar-Optical Residuals Before/After Best Time Shift")
    fig.tight_layout()
    fig.savefig(compare_path, dpi=150)
    plt.close(fig)

    return rmse_path, compare_path


def main():
    parser = argparse.ArgumentParser(description="Diagnose radar-optical time synchronization from one calibration session")
    parser.add_argument("--input", help="Path to one saved calibration session JSON file")
    parser.add_argument("--drop-index", type=int, action="append", default=[], help="Drop one paired sample by index")
    parser.add_argument("--drop-range", type=parse_range_arg, action="append", default=[], help="Drop paired samples within one radar range interval min:max")
    parser.add_argument("--range", dest="range_filter", type=parse_range_arg, help="Only analyze one radar range interval min:max")
    parser.add_argument("--delta-min", type=float, default=-1.0, help="Minimum delta_t to scan (seconds)")
    parser.add_argument("--delta-max", type=float, default=1.0, help="Maximum delta_t to scan (seconds)")
    parser.add_argument("--delta-step", type=float, default=0.01, help="delta_t scan step (seconds)")
    args = parser.parse_args()

    session_files = list_session_files()
    session_path = args.input or (session_files[0] if session_files else None)
    if not session_path:
        print("[sync] no calibration sessions found")
        return 1

    session = load_session(session_path)
    source_measurements = filter_source_measurements(
        session.get("source_measurements", {}),
        set(args.drop_index),
        list(args.drop_range),
    )
    samples = prepare_samples(source_measurements, range_filter=args.range_filter)
    if len(samples) < 2:
        print("[sync] not enough samples to diagnose time sync")
        return 1

    pair_stats = compute_stats([item["pair_time_diff_sec"] for item in samples])
    rows = scan_time_offsets(samples, args.delta_min, args.delta_max, args.delta_step)
    if not rows:
        print("[sync] time-offset scan produced no valid interpolation range")
        return 1

    best_row = min(rows, key=lambda item: item["total_rmse"])
    before_result = interpolate_errors(samples, 0.0)
    after_result = interpolate_errors(samples, best_row["delta_t"])
    if before_result is None or after_result is None:
        print("[sync] failed to compute before/after interpolation results")
        return 1

    output_dir = os.path.join(DEFAULT_CALIBRATION_DATA_DIR, "time_sync_reports")
    session_stem = os.path.splitext(os.path.basename(session_path))[0]
    rmse_path, compare_path = save_plots(output_dir, session_stem, rows, before_result, after_result)
    overlap_samples = [item for item, keep in zip(samples, after_result["overlap_mask"]) if keep]
    suspicious = suspicious_samples(overlap_samples, after_result["az_errors"], after_result["pitch_errors"])

    result = {
        "session_path": session_path,
        "sample_count": len(samples),
        "pair_time_diff_stats": pair_stats,
        "best_time_offset_sec": float(best_row["delta_t"]),
        "rmse_before": {
            "azimuth_deg": before_result["az_rmse"],
            "pitch_deg": before_result["pitch_rmse"],
            "total_deg": before_result["total_rmse"],
        },
        "rmse_after": {
            "azimuth_deg": after_result["az_rmse"],
            "pitch_deg": after_result["pitch_rmse"],
            "total_deg": after_result["total_rmse"],
        },
        "segment_summaries": [],
        "suspicious_samples": suspicious,
        "plots": {
            "delta_rmse": rmse_path,
            "before_after": compare_path,
        },
    }
    if args.range_filter is not None:
        result["segment_summaries"].append({
            "range_min_m": args.range_filter[0],
            "range_max_m": args.range_filter[1],
            "sample_count": len(overlap_samples),
            "best_time_offset_sec": float(best_row["delta_t"]),
            "rmse_after": result["rmse_after"],
        })

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{session_stem}_time_sync.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[sync] session: {session_path}")
    print(f"[sync] sample_count: {len(samples)}")
    print(f"[sync] overlap_count: {len(overlap_samples)}")
    print(f"[sync] pair_time_diff mean={pair_stats['mean']:.4f}s median={pair_stats['median']:.4f}s std={pair_stats['std']:.4f}s")
    print(f"[sync] best_time_offset_sec: {best_row['delta_t']:.4f}s")
    print(
        f"[sync] RMSE before: az={result['rmse_before']['azimuth_deg']:.4f}°, "
        f"pitch={result['rmse_before']['pitch_deg']:.4f}°, total={result['rmse_before']['total_deg']:.4f}°"
    )
    print(
        f"[sync] RMSE after:  az={result['rmse_after']['azimuth_deg']:.4f}°, "
        f"pitch={result['rmse_after']['pitch_deg']:.4f}°, total={result['rmse_after']['total_deg']:.4f}°"
    )
    print(f"[sync] JSON report: {json_path}")
    print(f"[sync] plots: {rmse_path}, {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
