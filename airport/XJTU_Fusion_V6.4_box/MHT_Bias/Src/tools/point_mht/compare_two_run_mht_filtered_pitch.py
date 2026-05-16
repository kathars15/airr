# -*- coding: utf-8 -*-
"""Replay two point-record runs through MHT and compare filtered main tracks.

Tune the config block directly, then run:
    python tools/point_mht/compare_two_run_mht_filtered_pitch.py
"""

from pathlib import Path
from types import SimpleNamespace
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.point_mht.replay_points_mht_compare import load_point_records, run_point_mht


# ==================== Config ====================
RUN_A_POINT_FILE = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_120434\point_records_az126_129.csv"
)
RUN_B_POINT_FILE = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_121233\point_records_az126_129.csv"
)
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
OUTPUT_PNG = OUTPUT_DIR / "compare_two_run_mht_filtered_pitch.png"
OUTPUT_CSV = OUTPUT_DIR / "compare_two_run_mht_filtered_pitch.csv"

# Point filtering
AZIMUTH_MIN_DEG = 120.0
AZIMUTH_MAX_DEG = 135.0
TRUE_ONLY = True
MAX_FRAME_GAP = 0.08

# Main-track extraction
MIN_TRACK_POINTS = 4
MIN_TRACK_RANGE_M = 300.0
BIN_SIZE_M = 100.0
DIFF_PASS_THRESHOLD_DEG = 0.5

# MHT tuning
Q_SCALE = 0.1
MAX_VEL = 15.0
N_SCAN = 1
P_DEATH = 1e-2
CLUSTER_DISTANCE = 50.0
RESOLVED_TIME_WINDOW = 2.0
RESOLVED_MIN_DETECT = 1
MAX_DETECT_TIME = 20.0
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


def filter_point_rows(rows):
    filtered = []
    for row in rows:
        az = row.get("azimuth")
        if az is None or not in_azimuth_window(az):
            continue
        if TRUE_ONLY and str(row.get("is_true_point", "")).strip() != "1":
            continue
        filtered.append(row)
    return filtered


def extract_tracks(frames):
    tracks = {}
    for frame in frames:
        timestamp = float(frame["timestamp"])
        for target in frame.get("targets", []):
            track_id = str(target.get("track_id", "")).strip()
            if not track_id:
                continue
            tracks.setdefault(track_id, []).append(
                {
                    "timestamp": timestamp,
                    "range": float(target["range"]),
                    "azimuth": float(target["azimuth"]),
                    "pitch": float(target["pitch"]),
                    "track_id": track_id,
                }
            )
    for rows in tracks.values():
        rows.sort(key=lambda item: item["timestamp"])
    return tracks


def choose_main_track(tracks):
    candidates = []
    for track_id, rows in tracks.items():
        if len(rows) < MIN_TRACK_POINTS:
            continue
        max_range = max(row["range"] for row in rows)
        if max_range < MIN_TRACK_RANGE_M:
            continue
        candidates.append((track_id, rows))
    if not candidates:
        raise RuntimeError(
            f"no track reaches MIN_TRACK_POINTS={MIN_TRACK_POINTS} "
            f"and MIN_TRACK_RANGE_M={MIN_TRACK_RANGE_M}"
        )
    track_id, rows = max(
        candidates,
        key=lambda item: (
            item[1][-1]["timestamp"] - item[1][0]["timestamp"],
            max(row["range"] for row in item[1]) - min(row["range"] for row in item[1]),
            len(item[1]),
        ),
    )
    first_ts = rows[0]["timestamp"]
    enriched = []
    for row in rows:
        item = dict(row)
        item["time_rel"] = row["timestamp"] - first_ts
        enriched.append(item)
    return track_id, enriched


def bin_pitch_means(rows):
    bins = {}
    for row in rows:
        bin_id = int(row["range"] // BIN_SIZE_M) * BIN_SIZE_M
        bins.setdefault(bin_id, []).append(row["pitch"])
    out = []
    for bin_id in sorted(bins):
        values = bins[bin_id]
        if len(values) < 2:
            continue
        out.append(
            {
                "range_center": bin_id + BIN_SIZE_M / 2.0,
                "pitch_mean": statistics.mean(values),
                "count": len(values),
            }
        )
    return out


def bin_azimuth_means(rows):
    bins = {}
    for row in rows:
        bin_id = int(row["range"] // BIN_SIZE_M) * BIN_SIZE_M
        bins.setdefault(bin_id, []).append(row["azimuth"])
    out = []
    for bin_id in sorted(bins):
        values = bins[bin_id]
        if len(values) < 2:
            continue
        out.append(
            {
                "range_center": bin_id + BIN_SIZE_M / 2.0,
                "azimuth_mean": statistics.mean(values),
                "count": len(values),
            }
        )
    return out


def compare_bin_means(a_means, b_means):
    by_a = {row["range_center"]: row for row in a_means}
    by_b = {row["range_center"]: row for row in b_means}
    common = sorted(set(by_a) & set(by_b))
    rows = []
    for key in common:
        diff = by_a[key]["pitch_mean"] - by_b[key]["pitch_mean"]
        rows.append(
            {
                "range_center": key,
                "run_a_pitch_mean": by_a[key]["pitch_mean"],
                "run_b_pitch_mean": by_b[key]["pitch_mean"],
                "pitch_diff": diff,
                "abs_pitch_diff": abs(diff),
                "run_a_count": by_a[key]["count"],
                "run_b_count": by_b[key]["count"],
            }
        )
    return rows


def compare_bin_means_generic(a_means, b_means, key_name, diff_name, abs_diff_name):
    by_a = {row["range_center"]: row for row in a_means}
    by_b = {row["range_center"]: row for row in b_means}
    common = sorted(set(by_a) & set(by_b))
    rows = []
    for key in common:
        diff = by_a[key][key_name] - by_b[key][key_name]
        rows.append(
            {
                "range_center": key,
                f"run_a_{key_name}": by_a[key][key_name],
                f"run_b_{key_name}": by_b[key][key_name],
                diff_name: diff,
                abs_diff_name: abs(diff),
                "run_a_count": by_a[key]["count"],
                "run_b_count": by_b[key]["count"],
            }
        )
    return rows


def write_csv(main_a_rows, main_b_rows, pitch_diff_rows, azimuth_diff_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "section",
            "run_label",
            "track_id",
            "time_rel",
            "timestamp",
            "range",
            "azimuth",
            "pitch",
            "range_center",
            "run_a_pitch_mean",
            "run_b_pitch_mean",
            "pitch_diff",
            "abs_pitch_diff",
            "run_a_count",
            "run_b_count",
            "run_a_azimuth_mean",
            "run_b_azimuth_mean",
            "azimuth_diff",
            "abs_azimuth_diff",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in main_a_rows:
            writer.writerow(
                {
                    "section": "main_track",
                    "run_label": "run_a",
                    "track_id": row["track_id"],
                    "time_rel": row["time_rel"],
                    "timestamp": row["timestamp"],
                    "range": row["range"],
                    "azimuth": row["azimuth"],
                    "pitch": row["pitch"],
                }
            )
        for row in main_b_rows:
            writer.writerow(
                {
                    "section": "main_track",
                    "run_label": "run_b",
                    "track_id": row["track_id"],
                    "time_rel": row["time_rel"],
                    "timestamp": row["timestamp"],
                    "range": row["range"],
                    "azimuth": row["azimuth"],
                    "pitch": row["pitch"],
                }
            )
        for row in pitch_diff_rows:
            writer.writerow(
                {
                    "section": "pitch_bin_diff",
                    "range_center": row["range_center"],
                    "run_a_pitch_mean": row["run_a_pitch_mean"],
                    "run_b_pitch_mean": row["run_b_pitch_mean"],
                    "pitch_diff": row["pitch_diff"],
                    "abs_pitch_diff": row["abs_pitch_diff"],
                    "run_a_count": row["run_a_count"],
                    "run_b_count": row["run_b_count"],
                }
            )
        for row in azimuth_diff_rows:
            writer.writerow(
                {
                    "section": "azimuth_bin_diff",
                    "range_center": row["range_center"],
                    "run_a_azimuth_mean": row["run_a_azimuth_mean"],
                    "run_b_azimuth_mean": row["run_b_azimuth_mean"],
                    "azimuth_diff": row["azimuth_diff"],
                    "abs_azimuth_diff": row["abs_azimuth_diff"],
                    "run_a_count": row["run_a_count"],
                    "run_b_count": row["run_b_count"],
                }
            )


def plot_overlay(run_a_name, run_b_name, main_a_rows, main_b_rows, pitch_diff_rows, azimuth_diff_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 18), constrained_layout=True)

    ax = axes[0]
    ax.scatter(
        [row["range"] for row in main_a_rows],
        [row["pitch"] for row in main_a_rows],
        s=18,
        alpha=0.35,
        color="#1f77b4",
        label=f"{run_a_name} filtered samples",
    )
    ax.scatter(
        [row["range"] for row in main_b_rows],
        [row["pitch"] for row in main_b_rows],
        s=18,
        alpha=0.35,
        color="#d62728",
        label=f"{run_b_name} filtered samples",
    )

    a_means = bin_pitch_means(main_a_rows)
    b_means = bin_pitch_means(main_b_rows)
    ax.plot(
        [row["range_center"] for row in a_means],
        [row["pitch_mean"] for row in a_means],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label=f"{run_a_name} bin mean ({BIN_SIZE_M:.0f}m)",
    )
    ax.plot(
        [row["range_center"] for row in b_means],
        [row["pitch_mean"] for row in b_means],
        marker="o",
        linewidth=2.2,
        color="#d62728",
        label=f"{run_b_name} bin mean ({BIN_SIZE_M:.0f}m)",
    )
    ax.set_title(
        f"MHT filtered main-track pitch vs range | az {AZIMUTH_MIN_DEG:.1f}-{AZIMUTH_MAX_DEG:.1f} deg | "
        f"Q={Q_SCALE}, max_vel={MAX_VEL}, n_scan={N_SCAN}, min_track_range={MIN_TRACK_RANGE_M:.0f}m"
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Pitch (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    if pitch_diff_rows:
        ax2.plot(
            [row["range_center"] for row in pitch_diff_rows],
            [row["abs_pitch_diff"] for row in pitch_diff_rows],
            marker="o",
            linewidth=2.0,
            color="#2ca02c",
            label="abs pitch diff between run means",
        )
    ax2.axhline(DIFF_PASS_THRESHOLD_DEG, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"{DIFF_PASS_THRESHOLD_DEG:.1f} deg target")
    ax2.set_title("Cross-run pitch difference after MHT filtering")
    ax2.set_xlabel("Range bin center (m)")
    ax2.set_ylabel("Absolute pitch diff (deg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[2]
    ax3.scatter(
        [row["range"] for row in main_a_rows],
        [row["azimuth"] for row in main_a_rows],
        s=18,
        alpha=0.35,
        color="#1f77b4",
        label=f"{run_a_name} filtered samples",
    )
    ax3.scatter(
        [row["range"] for row in main_b_rows],
        [row["azimuth"] for row in main_b_rows],
        s=18,
        alpha=0.35,
        color="#d62728",
        label=f"{run_b_name} filtered samples",
    )
    a_az_means = bin_azimuth_means(main_a_rows)
    b_az_means = bin_azimuth_means(main_b_rows)
    ax3.plot(
        [row["range_center"] for row in a_az_means],
        [row["azimuth_mean"] for row in a_az_means],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label=f"{run_a_name} az bin mean ({BIN_SIZE_M:.0f}m)",
    )
    ax3.plot(
        [row["range_center"] for row in b_az_means],
        [row["azimuth_mean"] for row in b_az_means],
        marker="o",
        linewidth=2.2,
        color="#d62728",
        label=f"{run_b_name} az bin mean ({BIN_SIZE_M:.0f}m)",
    )
    ax3.set_title("MHT filtered main-track azimuth vs range")
    ax3.set_xlabel("Range (m)")
    ax3.set_ylabel("Azimuth (deg)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = axes[3]
    if azimuth_diff_rows:
        ax4.plot(
            [row["range_center"] for row in azimuth_diff_rows],
            [row["abs_azimuth_diff"] for row in azimuth_diff_rows],
            marker="o",
            linewidth=2.0,
            color="#9467bd",
            label="abs azimuth diff between run means",
        )
    ax4.axhline(DIFF_PASS_THRESHOLD_DEG, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"{DIFF_PASS_THRESHOLD_DEG:.1f} deg target")
    ax4.set_title("Cross-run azimuth difference after MHT filtering")
    ax4.set_xlabel("Range bin center (m)")
    ax4.set_ylabel("Absolute azimuth diff (deg)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def summarize_diff(diff_rows):
    if not diff_rows:
        return "no common bins after filtering"
    abs_diffs = [row["abs_pitch_diff"] for row in diff_rows]
    within = sum(1 for value in abs_diffs if value <= DIFF_PASS_THRESHOLD_DEG)
    return (
        f"common_bins={len(diff_rows)} "
        f"mean_abs_diff={statistics.mean(abs_diffs):.3f}deg "
        f"median_abs_diff={statistics.median(abs_diffs):.3f}deg "
        f"max_abs_diff={max(abs_diffs):.3f}deg "
        f"within_{DIFF_PASS_THRESHOLD_DEG:.1f}deg={within}/{len(diff_rows)}"
    )


def replay_one_run(point_file):
    rows = load_point_records(str(point_file))
    rows = filter_point_rows(rows)
    frames, _log_lines = run_point_mht(rows, build_args())
    tracks = extract_tracks(frames)
    track_id, main_rows = choose_main_track(tracks)
    return rows, frames, track_id, main_rows


def main():
    run_a_name = RUN_A_POINT_FILE.parent.name
    run_b_name = RUN_B_POINT_FILE.parent.name

    point_a_rows, frames_a, track_id_a, main_a_rows = replay_one_run(RUN_A_POINT_FILE)
    point_b_rows, frames_b, track_id_b, main_b_rows = replay_one_run(RUN_B_POINT_FILE)
    pitch_diff_rows = compare_bin_means(bin_pitch_means(main_a_rows), bin_pitch_means(main_b_rows))
    azimuth_diff_rows = compare_bin_means_generic(
        bin_azimuth_means(main_a_rows),
        bin_azimuth_means(main_b_rows),
        "azimuth_mean",
        "azimuth_diff",
        "abs_azimuth_diff",
    )

    write_csv(main_a_rows, main_b_rows, pitch_diff_rows, azimuth_diff_rows)
    plot_overlay(run_a_name, run_b_name, main_a_rows, main_b_rows, pitch_diff_rows, azimuth_diff_rows)

    print(f"[compare] run_a point rows after filter: {len(point_a_rows)}")
    print(f"[compare] run_a replay frames: {len(frames_a)}")
    print(
        f"[compare] run_a main track: {track_id_a} ({len(main_a_rows)} points, "
        f"duration={main_a_rows[-1]['time_rel']:.2f}s)"
    )
    print(f"[compare] run_b point rows after filter: {len(point_b_rows)}")
    print(f"[compare] run_b replay frames: {len(frames_b)}")
    print(
        f"[compare] run_b main track: {track_id_b} ({len(main_b_rows)} points, "
        f"duration={main_b_rows[-1]['time_rel']:.2f}s)"
    )
    print(f"[compare] pitch diff summary: {summarize_diff(pitch_diff_rows)}")
    print(f"[compare] azimuth diff summary: {summarize_diff(azimuth_diff_rows)}")
    print(f"[compare] output png: {OUTPUT_PNG}")
    print(f"[compare] output csv: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
