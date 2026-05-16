# -*- coding: utf-8 -*-
r"""Fit simple calibration curves from one saved calibration session and draw plots.

Examples:
  python .\tools\calibration\fit_calibration_curves.py
  python .\tools\calibration\fit_calibration_curves.py --input ..\calibration_data\cal_sessions\cal_session_xxx.json
"""

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

min_range = 900


SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.calibration import CALIBRATION_SESSION_DIR, DEFAULT_CALIBRATION_DATA_DIR  # noqa: E402


def session_dir():
    return os.path.join(DEFAULT_CALIBRATION_DATA_DIR, CALIBRATION_SESSION_DIR)


def newest_session():
    base = session_dir()
    if not os.path.isdir(base):
        return None
    items = []
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isfile(path) and name.lower().endswith(".json"):
            items.append(path)
    if not items:
        return None
    items.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return items[0]


def load_pairs(path, min_range=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("source_measurements", {}).get("samples", [])
    if not samples:
        raise ValueError("session file has no source_measurements.samples")
    rows = []
    for item in samples:
        radar = item.get("radar", {})
        optical = item.get("optical", {})
        radar_az = float(radar["azimuth"])
        radar_pitch = float(radar["pitch"])
        radar_range = float(radar["range"])
        optical_az = float(optical["azimuth"])
        optical_pitch = float(optical["pitch"])
        if min_range is not None and radar_range < float(min_range):
            continue
        delta_az = (optical_az - radar_az + 180.0) % 360.0 - 180.0
        delta_pitch = optical_pitch - radar_pitch
        rows.append(
            {
                "range": radar_range,
                "radar_az": radar_az,
                "radar_pitch": radar_pitch,
                "optical_az": optical_az,
                "optical_pitch": optical_pitch,
                "delta_az": delta_az,
                "delta_pitch": delta_pitch,
            }
        )
    rows.sort(key=lambda item: item["range"])
    return rows


def fit_poly(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    pred = np.polyval(coeffs, x)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return coeffs, pred, rmse


def split_segments(rows, pitch_jump_deg=1.0, range_gap_m=150.0):
    if not rows:
        return []
    segments = [[rows[0]]]
    for row in rows[1:]:
        prev = segments[-1][-1]
        if abs(row["radar_pitch"] - prev["radar_pitch"]) > pitch_jump_deg or row["range"] - prev["range"] > range_gap_m:
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments


def plot_curve(ax, x, y, degree, color, label_prefix):
    coeffs, pred, rmse = fit_poly(x, y, degree)
    order = np.argsort(x)
    ax.plot(x[order], pred[order], color=color, linewidth=2, label=f"{label_prefix} deg{degree} RMSE={rmse:.3f}")
    return coeffs, rmse

def main():
    parser = argparse.ArgumentParser(description="Fit and plot calibration curves from one saved session")
    parser.add_argument("--input", help="Path to one saved calibration session JSON file")
    parser.add_argument("--degree", type=int, default=3, help="Polynomial degree for global fitting")
    parser.add_argument("--min-range", type=float, default=min_range, help="Only use samples whose radar range is at least this value")
    args = parser.parse_args()

    input_path = args.input or newest_session()
    if not input_path:
        print("[fit] no calibration session found")
        return 1

    rows = load_pairs(input_path, min_range=args.min_range)
    if len(rows) < 3:
        print(f"[fit] not enough samples after filtering, current={len(rows)}")
        return 1
    x = np.array([row["range"] for row in rows], dtype=float)
    y_az = np.array([row["delta_az"] for row in rows], dtype=float)
    y_pitch = np.array([row["delta_pitch"] for row in rows], dtype=float)
    radar_pitch = np.array([row["radar_pitch"] for row in rows], dtype=float)
    optical_pitch = np.array([row["optical_pitch"] for row in rows], dtype=float)

    coeffs_az, _pred_az, rmse_az = fit_poly(x, y_az, args.degree)
    coeffs_pitch, _pred_pitch, rmse_pitch = fit_poly(x, y_pitch, args.degree)

    segments = split_segments(rows)
    segment_pitch_rmses = []
    for segment in segments:
        if len(segment) < 3:
            continue
        sx = np.array([row["range"] for row in segment], dtype=float)
        sy = np.array([row["delta_pitch"] for row in segment], dtype=float)
        degree = min(2, len(segment) - 1)
        _coeffs, _pred, seg_rmse = fit_poly(sx, sy, degree)
        segment_pitch_rmses.append((segment[0]["range"], segment[-1]["range"], seg_rmse, len(segment)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    axes[0].scatter(x, y_az, color="#1f77b4", s=26, label="samples")
    plot_curve(axes[0], x, y_az, args.degree, "#d62728", "global")
    axes[0].set_title("Azimuth offset vs radar range")
    axes[0].set_xlabel("Radar range (m)")
    axes[0].set_ylabel("Optical az - Radar az (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    radar_coeffs, radar_pred, radar_rmse = fit_poly(x, radar_pitch, args.degree)
    optical_coeffs, optical_pred, optical_rmse = fit_poly(x, optical_pitch, args.degree)
    order = np.argsort(x)
    axes[1].scatter(x, radar_pitch, color="#1f77b4", s=26, label="radar pitch samples")
    axes[1].scatter(x, optical_pitch, color="#d62728", s=26, label="optical pitch samples")
    axes[1].plot(x[order], radar_pred[order], color="#1f77b4", linewidth=2, label=f"radar fit deg{args.degree} RMSE={radar_rmse:.3f}")
    axes[1].plot(x[order], optical_pred[order], color="#d62728", linewidth=2, label=f"optical fit deg{args.degree} RMSE={optical_rmse:.3f}")
    axes[1].set_title("Radar / Optical pitch vs radar range")
    axes[1].set_xlabel("Radar range (m)")
    axes[1].set_ylabel("Pitch (deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_dir = os.path.join(os.getcwd(), "calibration_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fit_calibration_curves.png")
    fig.savefig(output_path, dpi=180)

    print(f"[fit] session: {input_path}")
    print(f"[fit] sample_count: {len(rows)}")
    if args.min_range is not None:
        print(f"[fit] min_range filter: {args.min_range:.1f} m")
    print(f"[fit] global azimuth deg{args.degree} RMSE: {rmse_az:.3f} deg")
    print(f"[fit] global pitch offset deg{args.degree} RMSE: {rmse_pitch:.3f} deg")
    print(f"[fit] radar pitch deg{args.degree} RMSE: {radar_rmse:.3f} deg")
    print(f"[fit] optical pitch deg{args.degree} RMSE: {optical_rmse:.3f} deg")
    if segment_pitch_rmses:
        print("[fit] segmented pitch fits:")
        for start_r, end_r, seg_rmse, count in segment_pitch_rmses:
            print(f"  {start_r:.0f}-{end_r:.0f}m n={count} RMSE={seg_rmse:.3f} deg")
    print(f"[fit] plot saved: {output_path}")
    print(f"[fit] azimuth coeffs: {coeffs_az.tolist()}")
    print(f"[fit] pitch coeffs: {coeffs_pitch.tolist()}")
    print(f"[fit] radar pitch coeffs: {radar_coeffs.tolist()}")
    print(f"[fit] optical pitch coeffs: {optical_coeffs.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
