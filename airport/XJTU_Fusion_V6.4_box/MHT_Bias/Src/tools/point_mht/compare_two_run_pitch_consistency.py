# -*- coding: utf-8 -*-
"""Compare pitch-vs-range consistency across two flight runs.

This overlays algorithm-confirmed tracks from track_results.json for two runs
in one figure, with a binned mean curve overlaid for each run.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================== Config ====================
RUN_A = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_120434"
)
RUN_B = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_121233"
)
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
OUTPUT_PNG = OUTPUT_DIR / "compare_two_run_pitch_consistency.png"

BIN_SIZE_M = 70.0
AZIMUTH_MIN_DEG = 120.0
AZIMUTH_MAX_DEG = 135.0
TRACK_COLOR = "#d62728"
RUN_COLORS = ["#1f77b4", "#ff7f0e"]
MIN_TRACK_POINTS = 4
# ===============================================


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def in_azimuth_window(value):
    if value is None:
        return False
    value = value % 360.0
    if AZIMUTH_MIN_DEG <= AZIMUTH_MAX_DEG:
        return AZIMUTH_MIN_DEG <= value <= AZIMUTH_MAX_DEG
    return value >= AZIMUTH_MIN_DEG or value <= AZIMUTH_MAX_DEG


def load_track_rows(run_dir: Path):
    path = run_dir / "track_results.json"
    rows = []
    if not path.exists():
        return rows
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for frame in data:
        for target in frame.get("targets", []):
            rng = to_float(target.get("range"))
            pitch = to_float(target.get("pitch"))
            az = to_float(target.get("azimuth"))
            if None in (rng, pitch, az):
                continue
            if not in_azimuth_window(az):
                continue
            rows.append(
                {
                    "range": rng,
                    "pitch": pitch,
                    "azimuth": az,
                    "timestamp": to_float(frame.get("timestamp"), 0.0),
                    "track_id": str(target.get("track_id", "")).strip(),
                }
            )
    rows.sort(key=lambda item: (item["range"], item["timestamp"]))
    return rows


def load_main_track_rows(run_dir: Path):
    path = run_dir / "track_results.json"
    rows = load_track_rows(run_dir)
    if not rows:
        return rows

    tracks = {}
    for row in rows:
        tracks.setdefault(row["track_id"], []).append(row)

    filtered = []
    for track_id, track_rows in tracks.items():
        if len(track_rows) < MIN_TRACK_POINTS:
            continue
        filtered.extend(track_rows)

    filtered.sort(key=lambda item: (item["range"], item["timestamp"]))
    return filtered


def bin_means(rows, bin_size_m):
    bins = {}
    for row in rows:
        bin_id = int(row["range"] // bin_size_m) * bin_size_m
        bins.setdefault(bin_id, []).append(row["pitch"])
    xs = []
    ys = []
    counts = []
    for bin_id in sorted(bins):
        values = bins[bin_id]
        if len(values) < 2:
            continue
        xs.append(bin_id + bin_size_m / 2.0)
        ys.append(statistics.mean(values))
        counts.append(len(values))
    return xs, ys, counts


def summarize(rows):
    if not rows:
        return "n=0"
    pitches = [row["pitch"] for row in rows]
    ranges = [row["range"] for row in rows]
    return f"n={len(rows)} range={min(ranges):.1f}-{max(ranges):.1f}m pitch_std={statistics.pstdev(pitches) if len(pitches) > 1 else 0.0:.2f}deg"


def plot_series(ax, rows, title, scatter_color, mean_color, mean_label):
    if not rows:
        ax.set_title(f"{title} | no data")
        ax.grid(True, alpha=0.3)
        return

    x = [row["range"] for row in rows]
    y = [row["pitch"] for row in rows]
    ax.scatter(x, y, s=14, alpha=0.35, color=scatter_color, label="confirmed track samples")

    bx, by, counts = bin_means(rows, BIN_SIZE_M)
    if bx:
        ax.plot(bx, by, color=mean_color, linewidth=2.2, marker="o", label=mean_label)

    return bx, by


def main():
    run_a_tracks = load_main_track_rows(RUN_A)
    run_b_tracks = load_main_track_rows(RUN_B)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)
    plot_series(
        ax,
        run_a_tracks,
        f"{RUN_A.name} vs {RUN_B.name} | confirmed tracks",
        RUN_COLORS[0],
        "#1f77b4",
        f"{RUN_A.name} bin mean ({BIN_SIZE_M:.0f}m)",
    )
    plot_series(
        ax,
        run_b_tracks,
        f"{RUN_A.name} vs {RUN_B.name} | confirmed tracks",
        RUN_COLORS[1],
        "#ff7f0e",
        f"{RUN_B.name} bin mean ({BIN_SIZE_M:.0f}m)",
    )

    ax.set_title(
        f"Cross-run consistency: confirmed tracks pitch vs range | az {AZIMUTH_MIN_DEG:.1f}-{AZIMUTH_MAX_DEG:.1f} deg"
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Pitch (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    print(f"[plot] run A tracks: {len(run_a_tracks)}")
    print(f"[plot] run B tracks: {len(run_b_tracks)}")
    print(f"[plot] output: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
