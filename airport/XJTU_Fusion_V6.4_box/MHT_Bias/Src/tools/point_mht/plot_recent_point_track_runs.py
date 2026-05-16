# -*- coding: utf-8 -*-
"""Plot raw point-record range-vs-time scatter for the latest two runs.

Same style as point_records_time_range_az124_130_latest.png.
Edit the config block directly. No CLI arguments.
"""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==================== Config ====================
FLIGHT_RUNS_DIR = Path(r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs")
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
AZIMUTH_MIN_DEG = 124.0
AZIMUTH_MAX_DEG = 130.0
OUTPUT_PREFIX = "recent_point_time_range"
# ===============================================


def load_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def in_azimuth_window(value):
    if value is None:
        return False
    value = value % 360.0
    if AZIMUTH_MIN_DEG <= AZIMUTH_MAX_DEG:
        return AZIMUTH_MIN_DEG <= value <= AZIMUTH_MAX_DEG
    return value >= AZIMUTH_MIN_DEG or value <= AZIMUTH_MAX_DEG


def load_run(run_dir):
    point_file = run_dir / "point_records.csv"
    point_rows = load_csv_rows(point_file) if point_file.exists() else []

    rows = []
    for row in point_rows:
        ts = to_float(row.get("timestamp"))
        rng = to_float(row.get("range"))
        az = to_float(row.get("azimuth"))
        if None in (ts, rng, az):
            continue
        if in_azimuth_window(az):
            rows.append(
                {
                    "timestamp": ts,
                    "range": rng,
                    "azimuth": az,
                    "is_true_point": str(row.get("is_true_point", "")).strip(),
                }
            )

    rows.sort(key=lambda item: item["timestamp"])
    return {
        "run_dir": run_dir,
        "point_file": point_file,
        "rows": rows,
    }


def relative_times(rows):
    if not rows:
        return []
    t0 = rows[0]["timestamp"]
    return [row["timestamp"] - t0 for row in rows]


def plot_single_run(run_data, output_path):
    rows = run_data["rows"]
    x = relative_times(rows)
    y = [row["range"] for row in rows]
    c = ["#d62728" if row["is_true_point"] == "1" else "#1f77b4" for row in rows]

    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, c=c, s=18, alpha=0.75)
    plt.title(f"{run_data['run_dir'].name} | point records: range vs time (azimuth {AZIMUTH_MIN_DEG:.0f}-{AZIMUTH_MAX_DEG:.0f} deg)")
    plt.xlabel("Time since first sample (s)")
    plt.ylabel("Range (m)")
    plt.grid(True, alpha=0.3)
    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="is_true_point=0/other", markerfacecolor="#1f77b4", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="is_true_point=1", markerfacecolor="#d62728", markersize=8),
    ]
    plt.legend(handles=legend_items)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_combined(runs, output_path):
    plt.figure(figsize=(12, 7))
    colors = ["#1f77b4", "#d62728"]
    for idx, run_data in enumerate(runs[:2]):
        rows = run_data["rows"]
        x = relative_times(rows)
        y = [row["range"] for row in rows]
        plt.scatter(x, y, c=colors[idx % len(colors)], s=16, alpha=0.55, label=run_data["run_dir"].name)

    plt.title(f"Latest two runs | point records: range vs time (azimuth {AZIMUTH_MIN_DEG:.0f}-{AZIMUTH_MAX_DEG:.0f} deg)")
    plt.xlabel("Time since first sample (s)")
    plt.ylabel("Range (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        [path for path in FLIGHT_RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:2]
    run_dirs = list(reversed(run_dirs))
    if len(run_dirs) < 2:
        raise SystemExit("Need at least two flight_run directories")

    runs = [load_run(run_dir) for run_dir in run_dirs]

    for run_data in runs:
        out = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{run_data['run_dir'].name}.png"
        plot_single_run(run_data, out)
        print(f"[plot] single: {out}")
        print(f"[plot]   rows in az window: {len(run_data['rows'])}")

    combined_out = OUTPUT_DIR / f"{OUTPUT_PREFIX}_latest_two_runs.png"
    plot_combined(runs, combined_out)
    print(f"[plot] combined: {combined_out}")


if __name__ == "__main__":
    main()
