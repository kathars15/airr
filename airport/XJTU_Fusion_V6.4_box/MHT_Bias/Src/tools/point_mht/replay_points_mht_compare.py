#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay saved POINT records through MHT, then compare with saved raw TRACK rows.

Use this after changing the point/MHT algorithm:
    python tools/point_mht/replay_points_mht_compare.py

The script does not require radar hardware. It reads data/point_records.csv
(or the old data/radar_calibration_data.csv), runs the current MHT code again,
and writes fresh offline results for comparison.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from copy import deepcopy
from itertools import count

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MHT_BIAS_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(MHT_BIAS_DIR)
for path in (SRC_DIR, MHT_BIAS_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.app_config import DATA_DIR, RAW_TRACKS_FILE  # noqa: E402
from main2 import enu_to_radar_polar  # noqa: E402
from MHT.POMHT import POMHT_Bias  # noqa: E402
from common.clusters import Clustering_Obs  # noqa: E402
from Sensor_Config.sensor_config import Sensor_Config  # noqa: E402


DEFAULT_POINT_RECORDS_FILE = os.path.join(DATA_DIR, "point_records.csv")
LEGACY_POINT_RECORDS_FILE = os.path.join(DATA_DIR, "radar_calibration_data.csv")
DEFAULT_RESULTS_FILE = os.path.join(DATA_DIR, "offline_point_track_results.json")
DEFAULT_LOG_FILE = os.path.join(DATA_DIR, "offline_point_track_log.txt")
DEFAULT_COMPARE_FILE = os.path.join(DATA_DIR, "offline_point_vs_raw_compare.csv")
DEFAULT_PAIRS_COMPARE_FILE = os.path.join(DATA_DIR, "mht_pairs_compare_latest.csv")
DEFAULT_PAIRS_PLOT_FILE = os.path.join(os.path.abspath(os.path.join(SRC_DIR, "..", "..", "..", "..")), "calibration_data", "mht_pairs_compare_latest.png")
PAIRS_JSON_DIR = os.path.join(SRC_DIR, "calibration_data", "raw_pair_records")
PAIRS_TEXT_DIR = os.path.join(SRC_DIR, "calibration_data", "readable_pair_logs")

DEFAULT_Q_SCALE = 0.01
DEFAULT_MAX_VEL = 15.0
DEFAULT_N_SCAN = 1
DEFAULT_P_DEATH = 1e-2
DEFAULT_RESOLVED_TIME_WINDOW = 2.0
DEFAULT_RESOLVED_MIN_DETECT = 1
DEFAULT_MAX_DETECT_TIME = 20.0


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def angle_delta_deg(a, b):
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def polar_to_enu(range_m, azimuth_deg, pitch_deg):
    azimuth_rad = math.radians(azimuth_deg)
    pitch_rad = math.radians(pitch_deg)
    x = range_m * math.cos(pitch_rad) * math.sin(azimuth_rad)
    y = range_m * math.cos(pitch_rad) * math.cos(azimuth_rad)
    z = range_m * math.sin(pitch_rad)
    return np.array([[x], [y], [z]], dtype=float)


def choose_point_file(path):
    if path:
        return path
    if os.path.exists(DEFAULT_POINT_RECORDS_FILE):
        return DEFAULT_POINT_RECORDS_FILE
    return LEGACY_POINT_RECORDS_FILE


def choose_latest_pairs_json():
    if not os.path.isdir(PAIRS_JSON_DIR):
        return ""
    files = [
        os.path.join(PAIRS_JSON_DIR, name)
        for name in os.listdir(PAIRS_JSON_DIR)
        if name.lower().endswith(".json") and name.startswith("raw_pairs_")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0] if files else ""


def infer_pairs_json_from_text(path):
    name = os.path.basename(path)
    if name.startswith("readable_pairs_"):
        raw_name = "raw_pairs_" + name[len("readable_pairs_") :]
        raw_name = os.path.splitext(raw_name)[0] + ".json"
        candidate = os.path.join(PAIRS_JSON_DIR, raw_name)
        if os.path.exists(candidate):
            return candidate
    return ""


def parse_readable_pairs_text(path):
    pattern = re.compile(
        r"雷达\(([-+]?\d+(?:\.\d+)?)°,\s*([-+]?\d+(?:\.\d+)?)°,\s*([-+]?\d+(?:\.\d+)?)m\)\s*->\s*"
        r"光电\(([-+]?\d+(?:\.\d+)?)°,\s*([-+]?\d+(?:\.\d+)?)°,\s*([-+]?\d+(?:\.\d+)?)m\),\s*时间差=([-+]?\d+(?:\.\d+)?)s"
    )
    samples = []
    current_time = 0.0
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            radar_az, radar_pitch, radar_range, optical_az, optical_pitch, optical_range, time_diff = [float(v) for v in match.groups()]
            samples.append(
                {
                    "index": len(samples) + 1,
                    "time_diff_sec": time_diff,
                    "radar": {
                        "timestamp": current_time,
                        "track_id": "pairs-text",
                        "azimuth": radar_az,
                        "pitch": radar_pitch,
                        "range": radar_range,
                    },
                    "optical": {
                        "timestamp": current_time + time_diff,
                        "azimuth": optical_az,
                        "pitch": optical_pitch,
                        "range": optical_range,
                        "status": 2,
                    },
                }
            )
            current_time += 1.0
    return {
        "target_id": os.path.splitext(os.path.basename(path))[0],
        "sample_count": len(samples),
        "samples": samples,
    }


def load_pairs_source(args):
    if args.pairs_json:
        with open(args.pairs_json, "r", encoding="utf-8") as f:
            return json.load(f), args.pairs_json
    if args.pairs_text:
        inferred = infer_pairs_json_from_text(args.pairs_text)
        if inferred:
            with open(inferred, "r", encoding="utf-8") as f:
                return json.load(f), inferred
        return parse_readable_pairs_text(args.pairs_text), args.pairs_text
    if args.latest_pairs:
        latest = choose_latest_pairs_json()
        if latest:
            with open(latest, "r", encoding="utf-8") as f:
                return json.load(f), latest
    return None, ""


def load_pair_rows(source_data):
    rows = []
    for item in source_data.get("samples", []):
        radar = item.get("radar", {})
        timestamp = to_float(radar.get("timestamp"), None)
        if timestamp is None:
            timestamp = float(len(rows))
        rows.append(
            {
                "index": len(rows) + 1,
                "timestamp": float(timestamp),
                "target_id": radar.get("track_id") or source_data.get("target_id", ""),
                "range": float(radar.get("range", 0.0)),
                "azimuth": float(radar.get("azimuth", 0.0)),
                "pitch": float(radar.get("pitch", 0.0)),
                "azimuth_relative": float(radar.get("azimuth", 0.0)),
                "pitch_enu": -float(radar.get("pitch", 0.0)),
                "speed": 0.0,
                "doppler": 0.0,
                "target_type": "",
                "is_true_point": "1",
                "radar_heading": 0.0,
                "frame_cnt": str(len(rows) + 1),
            }
        )
    return rows


def load_point_records(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows = []
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        has_header = "timestamp" in first_line.lower()
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                item = {
                    "timestamp": to_float(row.get("timestamp")),
                    "target_id": row.get("target_id", ""),
                    "range": to_float(row.get("range")),
                    "azimuth": to_float(row.get("azimuth")),
                    "pitch": to_float(row.get("pitch")),
                    "azimuth_relative": to_float(row.get("azimuth_relative")),
                    "pitch_enu": to_float(row.get("pitch_enu")),
                    "speed": to_float(row.get("speed"), 0.0),
                    "doppler": to_float(row.get("doppler"), 0.0),
                    "target_type": row.get("target_type", ""),
                    "is_true_point": row.get("is_true_point", ""),
                    "radar_heading": to_float(row.get("radar_heading"), 0.0),
                    "frame_cnt": row.get("frame_cnt", ""),
                }
                if item["timestamp"] is not None and item["range"] is not None:
                    rows.append(item)
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                item = {
                    "timestamp": to_float(row[0]),
                    "azimuth": to_float(row[1]),
                    "pitch": to_float(row[2]),
                    "azimuth_relative": None,
                    "pitch_enu": None,
                    "range": to_float(row[3]),
                    "target_id": row[4] if len(row) > 4 else "",
                    "speed": to_float(row[5], 0.0) if len(row) > 5 else 0.0,
                    "doppler": 0.0,
                    "target_type": "",
                    "is_true_point": "",
                    "radar_heading": 0.0,
                    "frame_cnt": "",
                }
                if item["timestamp"] is not None and item["range"] is not None:
                    rows.append(item)

    filtered = []
    for r in rows:
        if r["range"] is None or r["range"] <= 0 or r["azimuth"] is None or r["pitch"] is None:
            continue
        if r.get("azimuth_relative") is None:
            r["azimuth_relative"] = r["azimuth"]
            r["azimuth"] = (r["azimuth"] + float(r.get("radar_heading") or 0.0)) % 360.0
        if r.get("pitch_enu") is None:
            r["pitch_enu"] = -r["pitch"]
        filtered.append(r)
    rows = [
        r for r in filtered
        if r["range"] is not None and r["range"] > 0
        and r["azimuth"] is not None and r["pitch"] is not None
    ]
    rows.sort(key=lambda r: (r["timestamp"], str(r.get("frame_cnt", "")), str(r.get("target_id", ""))))
    return rows


def load_raw_tracks(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows = []
    seen = set()
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "timestamp": to_float(row.get("timestamp")),
                "raw_track_id": str(row.get("track_id", "")).strip(),
                "range": to_float(row.get("range")),
                "azimuth": to_float(row.get("azimuth")),
                "pitch": to_float(row.get("pitch")),
            }
            if None in (item["timestamp"], item["range"], item["azimuth"], item["pitch"]):
                continue
            key = (
                item["raw_track_id"],
                round(item["range"], 1),
                round(item["azimuth"], 2),
                round(item["pitch"], 2),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def normalize_raw_timebase(raw_rows, point_rows):
    if not raw_rows or not point_rows:
        return raw_rows
    point_mid = statistics.median([r["timestamp"] for r in point_rows])
    same_scale = [r for r in raw_rows if abs(r["timestamp"] - point_mid) < 86400.0]
    if same_scale:
        return same_scale
    return raw_rows


def iter_point_frames(rows, max_frame_gap=0.08):
    frame_id = count(1)
    current = []
    current_key = None

    for row in rows:
        frame_cnt = str(row.get("frame_cnt", "")).strip()
        key = frame_cnt if frame_cnt and frame_cnt.lower() != "nan" and frame_cnt != "0" else None
        if key is None:
            timestamp = row["timestamp"]
            if current and abs(timestamp - current[-1]["timestamp"]) <= max_frame_gap:
                key = current_key
            else:
                key = f"t:{timestamp:.6f}"

        if current and key != current_key:
            yield next(frame_id), current
            current = []
        current.append(row)
        current_key = key

    if current:
        yield next(frame_id), current


def build_mht_params(args):
    dim_d = 3
    return {
        "Lambda_NT": 1,
        "Q_k": np.identity(dim_d) * args.q_scale,
        "Max_Vel": args.max_vel,
        "N_Scan": args.n_scan,
        "Pg": 0.999,
        "P_death": args.p_death,
        "dim_d": dim_d,
        "Debug_Params": {"Debug": False, "Begin_Frame": 30},
        "Resolved_Time_Window": args.resolved_time_window,
        "Resolved_Min_Detect": args.resolved_min_detect,
        "max_detect_time": args.max_detect_time,
    }


def print_param_snapshot(args):
    print("[pairs-replay] params:")
    print(f"  q_scale={args.q_scale}")
    print(f"  max_vel={args.max_vel}")
    print(f"  n_scan={args.n_scan}")
    print(f"  p_death={args.p_death}")
    print(f"  resolved_time_window={args.resolved_time_window}")
    print(f"  resolved_min_detect={args.resolved_min_detect}")
    print(f"  max_detect_time={args.max_detect_time}")


def run_point_mht(point_rows, args):
    mht_params = build_mht_params(args)
    cluster_sigma = np.diag([10.0, 10.0, 10.0])
    sensor_config = deepcopy(Sensor_Config["Radar"])
    sensor_config["Biased_Ignore"] = True

    tracker = None
    label_id_map = {}
    next_label = 1
    frames_out = []
    log_lines = []
    last_timestamp = None

    for frame_no, frame_rows in iter_point_frames(point_rows, max_frame_gap=args.max_frame_gap):
        if args.true_only:
            frame_rows = [r for r in frame_rows if str(r.get("is_true_point", "")).strip() == "1"]
        if not frame_rows:
            continue

        timestamp = float(frame_rows[0]["timestamp"])
        if last_timestamp is not None and timestamp <= last_timestamp:
            timestamp = last_timestamp + 1e-3
        last_timestamp = timestamp

        obs_input = [polar_to_enu(r["range"], r["azimuth"], r["pitch_enu"]) for r in frame_rows]
        infos_input = [
            {
                "target_id": r.get("target_id"),
                "range": r["range"],
                "azimuth": r["azimuth"],
                "pitch": r["pitch"],
                "azimuth_relative": r.get("azimuth_relative"),
                "pitch_enu": r.get("pitch_enu"),
                "speed": r.get("speed", 0.0),
                "doppler": r.get("doppler", 0.0),
                "target_type": r.get("target_type", ""),
                "is_true_point": r.get("is_true_point", ""),
                "radar_heading": r.get("radar_heading", 0.0),
            }
            for r in frame_rows
        ]

        obs_clusters, obs_indices = Clustering_Obs(
            obs_k=obs_input,
            Clustering_Type="DBSCAN",
            eps=args.cluster_distance,
            min_samples=1,
            Sigma=cluster_sigma,
        )
        if not obs_clusters:
            continue

        obs_k = []
        cluster_infos = []
        for cluster, indices in zip(obs_clusters, obs_indices):
            obs_k.append(np.mean(np.concatenate(cluster, axis=1), axis=1).reshape(-1, 1))
            first_index = indices[0] if indices else 0
            info = dict(infos_input[first_index])
            info["cluster_size"] = len(indices)
            info["cluster_point_ids"] = ",".join(str(infos_input[i].get("target_id", "")) for i in indices)
            cluster_infos.append(info)

        if tracker is None:
            tracker = POMHT_Bias(
                Lambda_NT=mht_params["Lambda_NT"],
                obs_k=obs_k,
                timestamp=timestamp,
                sensor_config=sensor_config,
                Q_k=mht_params["Q_k"],
                Max_Vel=mht_params["Max_Vel"],
                N_Scan=mht_params["N_Scan"],
                Pg=mht_params["Pg"],
                P_death=mht_params["P_death"],
                dim_d=mht_params["dim_d"],
                Debug_Params=mht_params["Debug_Params"],
                extra_infos=cluster_infos,
                Resolved_Time_Window=mht_params["Resolved_Time_Window"],
                Resolved_Min_Detect=mht_params["Resolved_Min_Detect"],
                max_detect_time=mht_params["max_detect_time"],
            )
        else:
            tracker.forward(
                timestamp=timestamp,
                obs_k=obs_k,
                sensor_config=sensor_config,
                extra_infos=cluster_infos,
            )

        targets = []
        if hasattr(tracker, "Output_Nodes") and tracker.Output_Nodes:
            for node in deepcopy(tracker.Output_Nodes[-1]).values():
                if node.label not in label_id_map:
                    label_id_map[node.label] = next_label
                    next_label += 1
                track_id = f"Replay-{label_id_map[node.label]}"
                pos_enu = node.x_k_k[:3, :]
                vel_enu = node.x_k_k[3:6, :]
                polar = enu_to_radar_polar(pos_enu, radar_heading_deg=None)
                speed = float(np.linalg.norm(vel_enu))
                source_info = {}
                if getattr(node, "obs_id", None):
                    obs_idx = node.obs_id[-1]
                    if isinstance(obs_idx, (int, np.integer)) and 0 <= obs_idx < len(cluster_infos):
                        source_info = cluster_infos[obs_idx]
                targets.append(
                    {
                        "track_id": track_id,
                        "range": float(polar["range"]),
                        "azimuth": float(polar["azimuth"]),
                        "azimuth_relative": float(polar["azimuth_relative"]),
                        "pitch": float(polar["pitch"]),
                        "speed": speed,
                        "vel_x": float(vel_enu[0, 0]),
                        "vel_y": float(vel_enu[1, 0]),
                        "vel_z": float(vel_enu[2, 0]),
                        "source_point_id": source_info.get("target_id", ""),
                        "source_cluster_size": source_info.get("cluster_size", ""),
                        "source_cluster_point_ids": source_info.get("cluster_point_ids", ""),
                    }
                )

        frames_out.append(
            {
                "frame": frame_no,
                "timestamp": timestamp,
                "point_count": len(frame_rows),
                "cluster_count": len(obs_k),
                "target_count": len(targets),
                "targets": targets,
            }
        )
        for target in targets:
            log_lines.append(
                f"frame={frame_no} {target['track_id']}: "
                f"range={target['range']:.1f}m az={target['azimuth']:.2f}deg "
                f"pitch={target['pitch']:.2f}deg speed={target['speed']:.1f}m/s"
            )

    return frames_out, log_lines


def flatten_replay_results(frames):
    rows = []
    for frame in frames:
        for target in frame.get("targets", []):
            rows.append(
                {
                    "timestamp": float(frame["timestamp"]),
                    "point_track_id": target["track_id"],
                    "range": float(target["range"]),
                    "azimuth": float(target["azimuth"]),
                    "pitch": float(target["pitch"]),
                    "speed": float(target.get("speed", 0.0)),
                }
            )
    return rows


def flatten_pairs_results(frames, pair_rows, flip_filtered_pitch=False):
    rows = []
    for index, pair_row in enumerate(pair_rows):
        frame = frames[index] if index < len(frames) else {}
        targets = frame.get("targets", [])
        target = targets[0] if targets else {}
        filtered_range = to_float(target.get("range"), None)
        filtered_az = to_float(target.get("azimuth"), None)
        filtered_pitch = to_float(target.get("pitch"), None)
        if filtered_pitch is not None and flip_filtered_pitch:
            filtered_pitch = -filtered_pitch
        rows.append(
            {
                "index": index + 1,
                "timestamp": float(pair_row["timestamp"]),
                "raw_range": float(pair_row["range"]),
                "raw_azimuth": float(pair_row["azimuth"]),
                "raw_pitch": float(pair_row["pitch"]),
                "filtered_range": filtered_range,
                "filtered_azimuth": filtered_az,
                "filtered_pitch": filtered_pitch,
                "delta_range": (filtered_range - pair_row["range"]) if filtered_range is not None else "",
                "delta_azimuth": angle_delta_deg(filtered_az, pair_row["azimuth"]) if filtered_az is not None else "",
                "delta_pitch": (filtered_pitch - pair_row["pitch"]) if filtered_pitch is not None else "",
            }
        )
    return rows


def build_match_candidates(point_tracks, raw_tracks, args):
    candidates = []
    for pi, point in enumerate(point_tracks):
        for ri, raw in enumerate(raw_tracks):
            dt = abs(point["timestamp"] - raw["timestamp"])
            dr = abs(point["range"] - raw["range"])
            da = abs(angle_delta_deg(point["azimuth"], raw["azimuth"]))
            dp = abs(point["pitch"] - raw["pitch"])
            if args.match_mode == "time" and dt > args.time_window:
                continue
            if args.match_mode == "spatial" and (dr > args.max_range_gap or da > args.max_az_gap):
                continue
            if args.match_mode == "auto":
                if dt <= args.time_window:
                    pass
                elif dr <= args.max_range_gap and da <= args.max_az_gap:
                    pass
                else:
                    continue
            score = dt * 40.0 + dr * 0.10 + da * 12.0 + dp * 14.0
            if args.match_mode == "spatial":
                score = dr * 0.10 + da * 12.0 + dp * 14.0
            candidates.append((score, pi, ri, dt, dr, da, dp))
    candidates.sort(key=lambda item: item[0])
    return candidates


def match_replay_to_raw(point_tracks, raw_tracks, args):
    candidates = build_match_candidates(point_tracks, raw_tracks, args)
    used_point = set()
    used_raw = set()
    matches = []

    for score, pi, ri, dt, dr, da, dp in candidates:
        if pi in used_point or ri in used_raw:
            continue
        used_point.add(pi)
        used_raw.add(ri)
        point = point_tracks[pi]
        raw = raw_tracks[ri]
        matches.append(
            {
                "match_status": "matched",
                "point_track_id": point["point_track_id"],
                "raw_track_id": raw["raw_track_id"],
                "timestamp_point": point["timestamp"],
                "timestamp_raw": raw["timestamp"],
                "time_diff": dt,
                "point_range": point["range"],
                "raw_range": raw["range"],
                "range_diff": point["range"] - raw["range"],
                "point_azimuth": point["azimuth"],
                "raw_azimuth": raw["azimuth"],
                "azimuth_diff": angle_delta_deg(point["azimuth"], raw["azimuth"]),
                "point_pitch": point["pitch"],
                "raw_pitch": raw["pitch"],
                "pitch_diff": point["pitch"] - raw["pitch"],
                "score": score,
            }
        )

    for pi, point in enumerate(point_tracks):
        if pi not in used_point:
            matches.append(
                {
                    "match_status": "unmatched_point_track",
                    "point_track_id": point["point_track_id"],
                    "raw_track_id": "",
                    "timestamp_point": point["timestamp"],
                    "timestamp_raw": "",
                    "time_diff": "",
                    "point_range": point["range"],
                    "raw_range": "",
                    "range_diff": "",
                    "point_azimuth": point["azimuth"],
                    "raw_azimuth": "",
                    "azimuth_diff": "",
                    "point_pitch": point["pitch"],
                    "raw_pitch": "",
                    "pitch_diff": "",
                    "score": "",
                }
            )

    for ri, raw in enumerate(raw_tracks):
        if ri not in used_raw:
            matches.append(
                {
                    "match_status": "unmatched_raw_track",
                    "point_track_id": "",
                    "raw_track_id": raw["raw_track_id"],
                    "timestamp_point": "",
                    "timestamp_raw": raw["timestamp"],
                    "time_diff": "",
                    "point_range": "",
                    "raw_range": raw["range"],
                    "range_diff": "",
                    "point_azimuth": "",
                    "raw_azimuth": raw["azimuth"],
                    "azimuth_diff": "",
                    "point_pitch": "",
                    "raw_pitch": raw["pitch"],
                    "pitch_diff": "",
                    "score": "",
                }
            )
    return matches


def write_json(path, frames):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False, indent=2)


def write_log(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_compare_csv(path, matches):
    fieldnames = [
        "match_status",
        "point_track_id",
        "raw_track_id",
        "timestamp_point",
        "timestamp_raw",
        "time_diff",
        "point_range",
        "raw_range",
        "range_diff",
        "point_azimuth",
        "raw_azimuth",
        "azimuth_diff",
        "point_pitch",
        "raw_pitch",
        "pitch_diff",
        "score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def write_pairs_compare_csv(path, rows):
    fieldnames = [
        "index",
        "timestamp",
        "raw_range",
        "raw_azimuth",
        "raw_pitch",
        "filtered_range",
        "filtered_azimuth",
        "filtered_pitch",
        "delta_range",
        "delta_azimuth",
        "delta_pitch",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairs_plot(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = [row["index"] for row in rows]
    raw_range = [row["raw_range"] for row in rows]
    raw_az = [row["raw_azimuth"] for row in rows]
    raw_pitch = [row["raw_pitch"] for row in rows]
    filtered_range = [row["filtered_range"] for row in rows]
    filtered_az = [row["filtered_azimuth"] for row in rows]
    filtered_pitch = [row["filtered_pitch"] for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    series = [
        ("Range", raw_range, filtered_range, "Range (m)"),
        ("Azimuth", raw_az, filtered_az, "Azimuth (deg)"),
        ("Pitch", raw_pitch, filtered_pitch, "Pitch (deg)"),
    ]
    for ax, (title, raw_values, filtered_values, ylabel) in zip(axes, series):
        ax.plot(x, raw_values, marker="o", linewidth=1.6, color="#1f77b4", label=f"raw {title.lower()}")
        ax.plot(x, filtered_values, marker="o", linewidth=1.6, color="#d62728", label=f"filtered {title.lower()}")
        ax.set_title(f"Raw vs filtered {title}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(point_rows, raw_rows, frames, replay_rows, matches, args):
    matched = [m for m in matches if m["match_status"] == "matched"]
    unmatched_point = sum(1 for m in matches if m["match_status"] == "unmatched_point_track")
    unmatched_raw = sum(1 for m in matches if m["match_status"] == "unmatched_raw_track")
    print("[replay] point records:", len(point_rows))
    print("[replay] raw track rows:", len(raw_rows))
    print("[replay] replay frames:", len(frames))
    print("[replay] replay target rows:", len(replay_rows))
    print(f"[replay] match mode: {args.match_mode}")
    print("[replay] matched pairs:", len(matched))
    print("[replay] unmatched replay tracks:", unmatched_point)
    print("[replay] unmatched raw tracks:", unmatched_raw)

    if matched:
        range_abs = [abs(m["range_diff"]) for m in matched]
        az_abs = [abs(m["azimuth_diff"]) for m in matched]
        pitch_abs = [abs(m["pitch_diff"]) for m in matched]
        print(
            "[replay] mean abs error: "
            f"range={statistics.mean(range_abs):.2f}m, "
            f"az={statistics.mean(az_abs):.2f}deg, "
            f"pitch={statistics.mean(pitch_abs):.2f}deg"
        )
        print(
            "[replay] max abs error: "
            f"range={max(range_abs):.2f}m, "
            f"az={max(az_abs):.2f}deg, "
            f"pitch={max(pitch_abs):.2f}deg"
        )
        worst = sorted(matched, key=lambda m: abs(m["pitch_diff"]), reverse=True)[:5]
        print("[replay] worst pitch rows:")
        for row in worst:
            print(
                f"  {row['point_track_id']} vs raw={row['raw_track_id']} "
                f"range={row['point_range']:.1f}/{row['raw_range']:.1f}m "
                f"pitch={row['point_pitch']:.2f}/{row['raw_pitch']:.2f}deg "
                f"diff={row['pitch_diff']:+.2f}deg"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay saved POINT records through current MHT and compare against raw TRACK records."
    )
    parser.add_argument("--points", default="", help="point_records.csv path; defaults to data/point_records.csv")
    parser.add_argument("--pairs-json", default="", help="raw_pairs_*.json path; preferred pairs replay input")
    parser.add_argument("--pairs-text", default="", help="readable_pairs_*.txt path; will infer same-name raw_pairs_*.json if possible")
    parser.add_argument("--latest-pairs", type=int, default=0, help="Use latest N pairs files; v1 uses the latest one when N>=1")
    parser.add_argument("--raw", default=RAW_TRACKS_FILE, help="raw_tracks.csv path")
    parser.add_argument("--results", default=DEFAULT_RESULTS_FILE, help="output replay JSON")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help="output replay text log")
    parser.add_argument("--compare", default=DEFAULT_COMPARE_FILE, help="output comparison CSV")
    parser.add_argument("--pairs-compare", default=DEFAULT_PAIRS_COMPARE_FILE, help="pairs replay comparison CSV")
    parser.add_argument("--pairs-plot", default=DEFAULT_PAIRS_PLOT_FILE, help="pairs replay plot PNG")
    parser.add_argument("--cluster-distance", type=float, default=50.0)
    parser.add_argument("--q-scale", type=float, default=DEFAULT_Q_SCALE, help="Primary tuning knob: Q_k = I * q_scale")
    parser.add_argument("--p-death", type=float, default=DEFAULT_P_DEATH)
    parser.add_argument("--max-vel", type=float, default=DEFAULT_MAX_VEL)
    parser.add_argument("--n-scan", type=int, default=DEFAULT_N_SCAN)
    parser.add_argument("--resolved-time-window", type=float, default=DEFAULT_RESOLVED_TIME_WINDOW)
    parser.add_argument("--resolved-min-detect", type=int, default=DEFAULT_RESOLVED_MIN_DETECT)
    parser.add_argument("--max-detect-time", type=float, default=DEFAULT_MAX_DETECT_TIME)
    parser.add_argument("--max-frame-gap", type=float, default=0.08)
    parser.add_argument("--true-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-mode", choices=("auto", "time", "spatial"), default="auto")
    parser.add_argument("--time-window", type=float, default=0.5)
    parser.add_argument("--max-range-gap", type=float, default=100.0)
    parser.add_argument("--max-az-gap", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs_source, pairs_source_path = load_pairs_source(args)
    if pairs_source is not None:
        print_param_snapshot(args)
        pair_rows = load_pair_rows(pairs_source)
        if not pair_rows:
            print(f"[pairs-replay] no usable samples: {pairs_source_path}")
            return 4
        frames, log_lines = run_point_mht(pair_rows, args)
        replay_rows = flatten_pairs_results(frames, pair_rows, flip_filtered_pitch=True)
        write_json(args.results, frames)
        write_log(args.log, log_lines)
        write_pairs_compare_csv(args.pairs_compare, replay_rows)
        write_pairs_plot(args.pairs_plot, replay_rows)
        print(f"[pairs-replay] input: {pairs_source_path}")
        print(f"[pairs-replay] samples: {len(pair_rows)}")
        print(f"[pairs-replay] result json: {args.results}")
        print(f"[pairs-replay] result log: {args.log}")
        print(f"[pairs-replay] compare csv: {args.pairs_compare}")
        print(f"[pairs-replay] compare plot: {args.pairs_plot}")
        return 0

    point_path = choose_point_file(args.points)
    point_rows = load_point_records(point_path)
    raw_rows = normalize_raw_timebase(load_raw_tracks(args.raw), point_rows)
    if not point_rows:
        print(f"[replay] no point records: {point_path}")
        return 2
    if not raw_rows:
        print(f"[replay] no raw tracks: {args.raw}")
        return 3

    frames, log_lines = run_point_mht(point_rows, args)
    replay_rows = flatten_replay_results(frames)
    matches = match_replay_to_raw(replay_rows, raw_rows, args)

    write_json(args.results, frames)
    write_log(args.log, log_lines)
    write_compare_csv(args.compare, matches)

    print_summary(point_rows, raw_rows, frames, replay_rows, matches, args)
    print(f"[replay] point file: {point_path}")
    print(f"[replay] result json: {args.results}")
    print(f"[replay] result log: {args.log}")
    print(f"[replay] compare csv: {args.compare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
