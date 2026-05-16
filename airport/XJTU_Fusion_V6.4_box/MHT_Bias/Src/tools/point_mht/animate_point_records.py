# -*- coding: utf-8 -*-
"""Animate point records and overlay algorithm-confirmed tracks.

Default behavior:
- Plot raw point detections in gray.
- Plot algorithm-confirmed tracks from replay_points_mht_compare.py in red.

Edit the config block below directly. No CLI arguments.
"""

from pathlib import Path
import csv
import io
import math
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image


# ==================== Config ====================
POINT_FILE = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_120434\point_records.csv"
)
TRACK_RESULTS_FILE = Path(
    r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\flight_data_runs\flight_run_20260515_120434\track_results.json"
)
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
OUTPUT_GIF = OUTPUT_DIR / "point_records_playback.gif"

AZIMUTH_MIN_DEG = 120
AZIMUTH_MAX_DEG = 140.0
TRUE_ONLY = True
MAX_FRAME_GAP = 0.08
TRACK_TIMESTAMP_OFFSET = None  # None = auto-estimate from the first point/track timestamps

SHOW_CONFIRMED_TRACKS = True
SHOW_RAW_POINTS = True
CONFIRMED_ONLY = False
TAIL_FRAMES = 3

FIG_SIZE = (10, 10)
DPI = 130
GIF_FRAME_DURATION = 0.18
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


def polar_to_xy(range_m, azimuth_deg):
    azimuth_rad = math.radians(azimuth_deg)
    x = range_m * math.sin(azimuth_rad)
    y = range_m * math.cos(azimuth_rad)
    return x, y


def load_point_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ts = to_float(row.get("timestamp"))
            rng = to_float(row.get("range"))
            az = to_float(row.get("azimuth"))
            pitch = to_float(row.get("pitch"))
            is_true = str(row.get("is_true_point", "")).strip()
            frame_cnt = str(row.get("frame_cnt", "")).strip()
            if None in (ts, rng, az, pitch):
                continue
            if not in_azimuth_window(az):
                continue
            if TRUE_ONLY and is_true != "1":
                continue
            frame = frame_cnt or f"t:{ts:.6f}"
            x, y = polar_to_xy(rng, az)
            rows.append(
                {
                    "timestamp": ts,
                    "frame": frame,
                    "frame_no": int(frame_cnt) if frame_cnt.isdigit() else None,
                    "range": rng,
                    "azimuth": az,
                    "pitch": pitch,
                    "x": x,
                    "y": y,
                }
            )
    rows.sort(key=lambda item: (item["timestamp"], item["frame"]))
    return rows


def group_frames(rows):
    frames = []
    current = []
    current_key = None
    for row in rows:
        key = row["frame_no"] if row.get("frame_no") is not None else (row["timestamp"], row["frame"])
        if current and key != current_key:
            frames.append(current)
            current = []
        current.append(row)
        current_key = key
    if current:
        frames.append(current)
    return frames


def load_confirmed_tracks(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    frames = {}
    for item in data:
        frame_no = int(item.get("frame", len(frames) + 1))
        timestamp = to_float(item.get("timestamp"), float(frame_no))
        targets = []
        for target in item.get("targets", []):
            if CONFIRMED_ONLY and not str(target.get("track_id", "")).strip():
                continue
            rng = to_float(target.get("range"))
            az = to_float(target.get("azimuth"))
            pitch = to_float(target.get("pitch"))
            if None in (rng, az, pitch):
                continue
            x, y = polar_to_xy(rng, az)
            targets.append(
                {
                    "track_id": str(target.get("track_id", "")),
                    "range": rng,
                    "azimuth": az,
                    "pitch": pitch,
                    "x": x,
                    "y": y,
                    "raw_display_id": str(target.get("raw_display_id", "")),
                    "raw_match_score": to_float(target.get("raw_match_score"), None),
                }
            )
        frames[frame_no] = {
            "frame": frame_no,
            "timestamp": timestamp,
            "target_count": len(targets),
            "targets": targets,
        }
    return frames


def estimate_track_timestamp_offset(point_frames, track_frames):
    if TRACK_TIMESTAMP_OFFSET is not None:
        return TRACK_TIMESTAMP_OFFSET
    if not point_frames or not track_frames:
        return 0.0
    point_ts = point_frames[0][0]["timestamp"]
    track_ts = min(frame["timestamp"] for frame in track_frames.values())
    return point_ts - track_ts


def align_track_frames(track_frames, timestamp_offset):
    aligned = []
    for frame in track_frames.values():
        aligned.append(
            {
                "frame": frame["frame"],
                "timestamp": frame["timestamp"],
                "aligned_timestamp": frame["timestamp"] + timestamp_offset,
                "target_count": frame["target_count"],
                "targets": frame["targets"],
            }
        )
    aligned.sort(key=lambda item: item["aligned_timestamp"])
    return aligned


def match_track_frame(point_timestamp, aligned_track_frames):
    if not aligned_track_frames:
        return None
    best = min(aligned_track_frames, key=lambda item: abs(item["aligned_timestamp"] - point_timestamp))
    if abs(best["aligned_timestamp"] - point_timestamp) > 2.0:
        return None
    return best


def compute_limit(point_frames, track_frames):
    values = [0.0]
    for frame in point_frames:
        for row in frame:
            values.extend([row["x"], row["y"]])
    for frame in track_frames.values():
        for row in frame.get("targets", []):
            values.extend([row["x"], row["y"]])
    vmax = max(abs(v) for v in values) if values else 10.0
    return max(vmax, 50.0) * 1.05


def render_frame(point_frames, track_frames, aligned_track_frames, idx, limit):
    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
    ax.scatter([0], [0], s=70, marker="x", color="#d62728", label="radar center")

    start = max(0, idx - TAIL_FRAMES + 1)
    trail_frames = point_frames[start : idx + 1]
    total = len(trail_frames)
    for j, frame_rows in enumerate(trail_frames):
        alpha = 0.12 + 0.50 * ((j + 1) / max(total, 1))
        size = 12 + 3 * ((j + 1) / max(total, 1))
        xs = [row["x"] for row in frame_rows]
        ys = [row["y"] for row in frame_rows]
        if SHOW_RAW_POINTS:
            ax.scatter(xs, ys, s=size, color="#7f7f7f", alpha=alpha, marker="o", linewidths=0)

    frame_key = point_frames[idx][0].get("frame_no")
    if frame_key is None:
        frame_key = idx + 1
    point_ts = point_frames[idx][0]["timestamp"]
    current_tracks = match_track_frame(point_ts, aligned_track_frames) or {}
    targets = current_tracks.get("targets", [])
    if SHOW_CONFIRMED_TRACKS and targets:
        xs = [t["x"] for t in targets]
        ys = [t["y"] for t in targets]
        labels = [t["track_id"] or "confirmed" for t in targets]
        ax.scatter(xs, ys, s=90, color="#d62728", alpha=0.95, marker="^", label="confirmed tracks")
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#8c1d18")

    ts = point_frames[idx][0]["timestamp"]
    confirmed_n = len(targets)
    raw_n = len(point_frames[idx])
    ax.set_title(
        f"Point playback | raw={raw_n} | confirmed={confirmed_n} | "
        f"frame {frame_key}/{len(point_frames)} | t={ts:.2f}"
    )
    ax.set_xlabel("East X (m)")
    ax.set_ylabel("North Y (m)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def main():
    point_rows = load_point_rows(POINT_FILE)
    point_frames = group_frames(point_rows)
    if not point_frames:
        raise SystemExit("no point frames after filtering")

    track_frames = load_confirmed_tracks(TRACK_RESULTS_FILE)
    if not track_frames:
        print(f"[gif] warning: no confirmed track frames found: {TRACK_RESULTS_FILE}")
        track_frames = {}
    point_frame_keys = {row.get("frame_no") for frame in point_frames for row in frame if row.get("frame_no") is not None}
    timestamp_offset = estimate_track_timestamp_offset(point_frames, track_frames)
    aligned_track_frames = align_track_frames(track_frames, timestamp_offset)
    if track_frames:
        print(f"[gif] track timestamp offset: {timestamp_offset:.3f}s")
        print(
            "[gif] point ts range: "
            f"{point_frames[0][0]['timestamp']:.3f} -> {point_frames[-1][0]['timestamp']:.3f}"
        )
        print(
            "[gif] track ts range: "
            f"{aligned_track_frames[0]['aligned_timestamp']:.3f} -> {aligned_track_frames[-1]['aligned_timestamp']:.3f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    limit = compute_limit(point_frames, track_frames)
    images = [render_frame(point_frames, track_frames, aligned_track_frames, idx, limit) for idx in range(len(point_frames))]
    imageio.mimsave(OUTPUT_GIF, images, duration=GIF_FRAME_DURATION, loop=0)

    confirmed_total = sum(frame.get("target_count", 0) for frame in track_frames.values())
    print(f"[gif] input points: {POINT_FILE}")
    print(f"[gif] confirmed tracks: {TRACK_RESULTS_FILE}")
    print(f"[gif] filtered point rows: {len(point_rows)}")
    print(f"[gif] point frames: {len(point_frames)}")
    print(f"[gif] confirmed targets: {confirmed_total}")
    print(f"[gif] output: {OUTPUT_GIF}")


if __name__ == "__main__":
    main()
