# -*- coding: utf-8 -*-
"""Plot radar/optical pitch against radar range for each track in one raw pair record file."""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "..", "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "calibration_data")


def default_input():
    base = os.path.join(SRC_DIR, "calibration_data", "raw_pair_records")
    if not os.path.isdir(base):
        return None
    files = [
        os.path.join(base, name)
        for name in os.listdir(base)
        if name.lower().endswith(".json")
    ]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_groups(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups = defaultdict(list)
    for item in data.get("samples", []):
        radar = item.get("radar", {})
        optical = item.get("optical", {})
        track_id = radar.get("track_id", "unknown")
        groups[track_id].append(
            {
                "range": float(radar.get("range", 0.0)),
                "radar_pitch": float(radar.get("pitch", 0.0)),
                "optical_pitch": float(optical.get("pitch", 0.0)),
            }
        )
    for values in groups.values():
        values.sort(key=lambda item: item["range"])
    return data, groups


def main():
    parser = argparse.ArgumentParser(description="Plot radar/optical pitch by track")
    parser.add_argument("--input", help="Path to raw_pairs_*.json")
    args = parser.parse_args()

    input_path = args.input or default_input()
    if not input_path:
        print("[plot] no raw pair record found")
        return 1

    data, groups = load_groups(input_path)
    if not groups:
        print("[plot] no samples found")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_id = data.get("target_id", "unknown")
    output_path = os.path.join(OUTPUT_DIR, f"pitch_by_track_{os.path.splitext(os.path.basename(input_path))[0]}.png")

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 4.2 * len(groups)), constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, (track_id, values) in zip(axes, sorted(groups.items())):
        x = [item["range"] for item in values]
        radar_pitch = [item["radar_pitch"] for item in values]
        optical_pitch = [item["optical_pitch"] for item in values]

        ax.plot(x, radar_pitch, marker="o", linewidth=1.8, color="#1f77b4", label="radar pitch")
        ax.plot(x, optical_pitch, marker="o", linewidth=1.8, color="#d62728", label="optical pitch")
        ax.set_title(f"{track_id} | target label = {target_id}")
        ax.set_xlabel("Radar range (m)")
        ax.set_ylabel("Pitch (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.savefig(output_path, dpi=180)
    print(f"[plot] input: {input_path}")
    print(f"[plot] groups: {', '.join(sorted(groups.keys()))}")
    print(f"[plot] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
