# -*- coding: utf-8 -*-
"""Compare radar pitch against range across two raw pair record runs, ignoring track_id splits."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "..", "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "calibration_data")
RAW_PAIR_DIR = os.path.join(SRC_DIR, "calibration_data", "raw_pair_records")


def load_run(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data.get("samples", []):
        radar = item.get("radar", {})
        rows.append(
            {
                "range": float(radar.get("range", 0.0)),
                "pitch": float(radar.get("pitch", 0.0)),
            }
        )
    rows.sort(key=lambda item: item["range"])
    return data.get("target_id", os.path.basename(path)), rows


def latest_two_inputs():
    if not os.path.isdir(RAW_PAIR_DIR):
        return None, None
    files = [
        os.path.join(RAW_PAIR_DIR, name)
        for name in os.listdir(RAW_PAIR_DIR)
        if name.lower().endswith(".json") and name.startswith("raw_pairs_")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    if len(files) < 2:
        return None, None
    return files[1], files[0]


def latest_n_inputs(count):
    if not os.path.isdir(RAW_PAIR_DIR):
        return []
    files = [
        os.path.join(RAW_PAIR_DIR, name)
        for name in os.listdir(RAW_PAIR_DIR)
        if name.lower().endswith(".json") and name.startswith("raw_pairs_")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return list(reversed(files[: max(0, int(count))]))


def main():
    parser = argparse.ArgumentParser(description="Compare radar pitch between two calibration runs")
    parser.add_argument("--input-a", help="First raw_pairs_*.json")
    parser.add_argument("--input-b", help="Second raw_pairs_*.json")
    parser.add_argument("--latest", type=int, help="Use latest N raw_pairs_*.json files")
    parser.add_argument("--output", help="Optional output PNG path")
    args = parser.parse_args()

    selected_inputs = []
    if args.latest:
        selected_inputs = latest_n_inputs(args.latest)
        if len(selected_inputs) < 2:
            print("[compare] need at least two raw_pairs_*.json files for --latest")
            return 1

    input_a = args.input_a
    input_b = args.input_b
    if not selected_inputs and (not input_a or not input_b):
        auto_a, auto_b = latest_two_inputs()
        if not auto_a or not auto_b:
            print("[compare] need --input-a and --input-b, or at least two raw_pairs_*.json files in raw_pair_records")
            return 1
        input_a = input_a or auto_a
        input_b = input_b or auto_b
        selected_inputs = [input_a, input_b]
    elif not selected_inputs:
        selected_inputs = [input_a, input_b]

    runs = []
    for path in selected_inputs:
        label, rows = load_run(path)
        if not rows:
            print(f"[compare] input has no samples: {path}")
            return 1
        runs.append((path, label, rows))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(
        OUTPUT_DIR,
        "compare_radar_pitch_latest.png",
    )

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]
    for index, (path, label, rows) in enumerate(runs):
        ax.plot(
            [item["range"] for item in rows],
            [item["pitch"] for item in rows],
            marker="o",
            linewidth=1.8,
            color=colors[index % len(colors)],
            label=f"{label} | {os.path.basename(path)}",
        )

    ax.set_title("Radar pitch vs range across runs (ignore track_id)")
    ax.set_xlabel("Radar range (m)")
    ax.set_ylabel("Radar pitch (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(output_path, dpi=180)

    for path, label, rows in runs:
        print(f"[compare] input: {path}")
        print(f"[compare] label: {label}")
        print(f"[compare] samples: {len(rows)}")
    print(f"[compare] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
