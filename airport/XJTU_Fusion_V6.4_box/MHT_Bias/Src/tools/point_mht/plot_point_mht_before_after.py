# -*- coding: utf-8 -*-
"""Compare one filtered point-record run before vs after MHT.

Tune parameters directly in the config block below.
"""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.point_mht.replay_points_mht_compare import (
    angle_delta_deg,
    iter_point_frames,
    load_point_records,
    run_point_mht,
)


# ==================== Config ====================
POINT_FILE = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_121233\point_records.csv"
)
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
OUTPUT_PNG = OUTPUT_DIR / "point_mht_before_after.png"
OUTPUT_CSV = OUTPUT_DIR / "point_mht_before_after.csv"

# Point filtering
AZIMUTH_MIN_DEG = 120.0
AZIMUTH_MAX_DEG = 135.0
TRUE_ONLY = True
MAX_FRAME_GAP = 0.08

# MHT tuning
Q_SCALE = 0.01
MAX_VEL = 10.0
N_SCAN = 1
P_DEATH = 2e-2
CLUSTER_DISTANCE = 10.0
RESOLVED_TIME_WINDOW = 2.0
RESOLVED_MIN_DETECT = 1
MAX_DETECT_TIME = 20.0

# Matching raw representative <-> MHT output
MATCH_RANGE_WEIGHT = 0.10
MATCH_AZ_WEIGHT = 12.0
MATCH_PITCH_WEIGHT = 14.0
# ===============================================


def in_azimuth_window(value):
    value = float(value) % 360.0
    if AZIMUTH_MIN_DEG <= AZIMUTH_MAX_DEG:
        return AZIMUTH_MIN_DEG <= value <= AZIMUTH_MAX_DEG
    return value >= AZIMUTH_MIN_DEG or value <= AZIMUTH_MAX_DEG


def build_args():
    return SimpleNamespace(
        cluster_distance=CLUSTER_DISTANCE,
        q_scale=Q_SCALE,
        p_death=P_DEATH,
        max_vel=MAX_VEL,
        n_scan=N_SCAN,
        resolved_time_window=RESOLVED_TIME_WINDOW,
        resolved_min_detect=RESOLVED_MIN_DETECT,
        max_detect_time=MAX_DETECT_TIME,
        max_frame_gap=MAX_FRAME_GAP,
        true_only=TRUE_ONLY,
    )


def filter_rows(rows):
    out = []
    for row in rows:
        az = row.get("azimuth")
        if az is None or not in_azimuth_window(az):
            continue
        if TRUE_ONLY and str(row.get("is_true_point", "")).strip() != "1":
            continue
        pitch_plot = row.get("pitch_enu")
        if pitch_plot is None:
            pitch_raw = row.get("pitch")
            pitch_plot = -float(pitch_raw) if pitch_raw is not None else None
        else:
            pitch_plot = float(pitch_plot)
        out.append(row)
        out[-1]["pitch_plot"] = pitch_plot
    return out


def representative_raw_frames(rows):
    reps = []
    for frame_no, frame_rows in iter_point_frames(rows, max_frame_gap=MAX_FRAME_GAP):
        if not frame_rows:
            continue
        timestamp = float(frame_rows[0]["timestamp"])
        range_mean = float(np.mean([r["range"] for r in frame_rows]))
        az_mean = float(np.mean([r["azimuth"] for r in frame_rows]))
        pitch_plot_mean = float(np.mean([r["pitch_plot"] for r in frame_rows]))
        reps.append(
            {
                "frame": frame_no,
                "timestamp": timestamp,
                "raw_count": len(frame_rows),
                "raw_range": range_mean,
                "raw_azimuth": az_mean,
                "raw_pitch": pitch_plot_mean,
            }
        )
    return reps


def flatten_mht_frames(frames):
    rows = []
    for frame in frames:
        for target in frame.get("targets", []):
            rows.append(
                {
                    "frame": int(frame["frame"]),
                    "timestamp": float(frame["timestamp"]),
                    "track_id": target["track_id"],
                    "range": float(target["range"]),
                    "azimuth": float(target["azimuth"]),
                    "pitch": float(target["pitch"]),
                }
            )
    return rows


def pick_best_target(raw_rep, candidates):
    def score(item):
        return (
            abs(item["range"] - raw_rep["raw_range"]) * MATCH_RANGE_WEIGHT
            + abs(angle_delta_deg(item["azimuth"], raw_rep["raw_azimuth"])) * MATCH_AZ_WEIGHT
            + abs(item["pitch"] - raw_rep["raw_pitch"]) * MATCH_PITCH_WEIGHT
        )

    return min(candidates, key=score)


def join_before_after(raw_reps, frames):
    by_frame = {}
    for item in flatten_mht_frames(frames):
        by_frame.setdefault(item["frame"], []).append(item)

    rows = []
    for raw_rep in raw_reps:
        candidates = by_frame.get(raw_rep["frame"], [])
        best = pick_best_target(raw_rep, candidates) if candidates else None
        rows.append(
            {
                "frame": raw_rep["frame"],
                "timestamp": raw_rep["timestamp"],
                "raw_count": raw_rep["raw_count"],
                "raw_range": raw_rep["raw_range"],
                "raw_azimuth": raw_rep["raw_azimuth"],
                "raw_pitch": raw_rep["raw_pitch"],
                "filtered_range": best["range"] if best else "",
                "filtered_azimuth": best["azimuth"] if best else "",
                "filtered_pitch": best["pitch"] if best else "",
                "track_id": best["track_id"] if best else "",
            }
        )
    return rows


def write_csv(rows):
    import csv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "timestamp",
                "raw_count",
                "raw_range",
                "raw_azimuth",
                "raw_pitch",
                "filtered_range",
                "filtered_azimuth",
                "filtered_pitch",
                "track_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x = [row["frame"] for row in rows]
    raw_range = [row["raw_range"] for row in rows]
    raw_az = [row["raw_azimuth"] for row in rows]
    raw_pitch = [row["raw_pitch"] for row in rows]
    filtered_range = [float(row["filtered_range"]) if row["filtered_range"] != "" else np.nan for row in rows]
    filtered_az = [float(row["filtered_azimuth"]) if row["filtered_azimuth"] != "" else np.nan for row in rows]
    filtered_pitch = [float(row["filtered_pitch"]) if row["filtered_pitch"] != "" else np.nan for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    series = [
        ("Range", raw_range, filtered_range, "Range (m)"),
        ("Azimuth", raw_az, filtered_az, "Azimuth (deg)"),
        ("Pitch", raw_pitch, filtered_pitch, "Pitch (deg)"),
    ]
    for ax, (title, before, after, ylabel) in zip(axes, series):
        ax.plot(x, before, marker="o", linewidth=1.6, color="#1f77b4", label=f"before MHT {title.lower()}")
        ax.plot(x, after, marker="o", linewidth=1.6, color="#d62728", label=f"after MHT {title.lower()}")
        ax.set_title(f"{title}: before vs after MHT")
        ax.set_xlabel("Frame index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Point records before/after MHT | az={AZIMUTH_MIN_DEG:.0f}-{AZIMUTH_MAX_DEG:.0f} deg | "
        f"Q={Q_SCALE}, cluster={CLUSTER_DISTANCE}, max_vel={MAX_VEL}",
        fontsize=12,
    )
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def main():
    rows = load_point_records(str(POINT_FILE))
    rows = filter_rows(rows)
    raw_reps = representative_raw_frames(rows)
    args = build_args()
    frames, _log_lines = run_point_mht(rows, args)
    compare_rows = join_before_after(raw_reps, frames)
    write_csv(compare_rows)
    plot_rows(compare_rows)
    print(f"[plot] input: {POINT_FILE}")
    print(f"[plot] filtered point rows: {len(rows)}")
    print(f"[plot] raw representative frames: {len(raw_reps)}")
    print(f"[plot] replay frames: {len(frames)}")
    print(f"[plot] output png: {OUTPUT_PNG}")
    print(f"[plot] output csv: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
