# -*- coding: utf-8 -*-
"""Plot the main extracted MHT tracks from the latest two runs.

Edit the config block below directly. No CLI arguments.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================== Config ====================
RUN_A = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_120434\point_track_results_az126_129.json"
)
RUN_B = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_121233\point_track_results_az126_129.json"
)
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
OUTPUT_PNG = OUTPUT_DIR / "latest_mht_main_tracks_compare.png"
# ===============================================


def load_main_track(path):
    with open(path, encoding="utf-8") as f:
        frames = json.load(f)

    tracks = {}
    for frame in frames:
        ts = float(frame["timestamp"])
        for target in frame.get("targets", []):
            tracks.setdefault(target["track_id"], []).append(
                {
                    "timestamp": ts,
                    "range": float(target["range"]),
                    "azimuth": float(target["azimuth"]),
                    "pitch": float(target["pitch"]),
                }
            )

    if not tracks:
        raise RuntimeError(f"no targets found: {path}")

    main_id, main_rows = max(tracks.items(), key=lambda item: len(item[1]))
    main_rows.sort(key=lambda item: item["timestamp"])
    t0 = main_rows[0]["timestamp"]
    for row in main_rows:
        row["time_rel"] = row["timestamp"] - t0
    return path, main_id, main_rows


def plot_series(ax, rows, color, label, key, ylabel):
    ax.plot(
        [row["time_rel"] for row in rows],
        [row[key] for row in rows],
        marker="o",
        linewidth=1.8,
        color=color,
        label=label,
    )
    ax.set_xlabel("Time since first track point (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path_a, id_a, rows_a = load_main_track(RUN_A)
    path_b, id_b, rows_b = load_main_track(RUN_B)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    plot_series(axes[0], rows_a, "#1f77b4", f"{path_a.parent.name} | {id_a}", "range", "Range (m)")
    plot_series(axes[0], rows_b, "#d62728", f"{path_b.parent.name} | {id_b}", "range", "Range (m)")
    axes[0].set_title("Main MHT track comparison: Range vs Time")

    plot_series(axes[1], rows_a, "#1f77b4", f"{path_a.parent.name} | {id_a}", "azimuth", "Azimuth (deg)")
    plot_series(axes[1], rows_b, "#d62728", f"{path_b.parent.name} | {id_b}", "azimuth", "Azimuth (deg)")
    axes[1].set_title("Main MHT track comparison: Azimuth vs Time")

    plot_series(axes[2], rows_a, "#1f77b4", f"{path_a.parent.name} | {id_a}", "pitch", "Pitch (deg)")
    plot_series(axes[2], rows_b, "#d62728", f"{path_b.parent.name} | {id_b}", "pitch", "Pitch (deg)")
    axes[2].set_title("Main MHT track comparison: Pitch vs Time")

    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    print(f"[plot] run_a: {path_a}")
    print(f"[plot] main track a: {id_a} ({len(rows_a)} points)")
    print(f"[plot] run_b: {path_b}")
    print(f"[plot] main track b: {id_b} ({len(rows_b)} points)")
    print(f"[plot] output: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
