# -*- coding: utf-8 -*-
"""Offline 2D-segment joint position-bias + angle-bias estimator."""

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.app_config import (  # noqa: E402
    CALIBRATION_MAX_OFFSET_X_M,
    CALIBRATION_MAX_OFFSET_Y_M,
    CALIBRATION_MAX_OFFSET_Z_M,
)

# ======================================================================
# Edit here
# ======================================================================
INPUT_SESSION_FILES = [
        r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\calibration_data\cal_sessions\cal_session_20260519_113114_Radar-24.json",
        r"D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src\calibration_data\cal_sessions\cal_session_20260518_140007_Radar-10.json",
]

DROP_INDEXES = []
DROP_BY_SESSION = {
    "cal_session_20260518_140007_Radar-10.json": [37,38],
    # "cal_session_20260519_113114_Radar-24.json": [17, 18],
}
DROP_RANGE_SEGMENTS = []
ONLY_RANGE_SEGMENTS = []

MAX_AZ_BIAS_DEG = 100.0
MAX_PITCH_BIAS_DEG = 100.0
MIN_SAMPLES = 5
OUTPUT_DIR = Path(r"D:\desk\airr\calibration_data")
OUTPUT_BASENAME = "offline_joint_bias_estimator"

RANGE_BIN_SIZE_M = 100.0
RANGE_BIN_MIN_SAMPLES = 3
RANGE_USABLE_MEAN_ERROR_DEG = 5.0
RANGE_USABLE_MAX_ERROR_DEG = 12.0

AZ_BIN_SIZE_DEG = 10.0
AZ_BIN_MIN_SAMPLES = 3
AZ_USABLE_MEAN_ERROR_DEG = 5.0
AZ_USABLE_MAX_ERROR_DEG = 12.0
# ======================================================================


def angle_diff_deg(target_angle, source_angle):
    return (float(target_angle) - float(source_angle) + 180.0) % 360.0 - 180.0


def polar_to_vector(az_deg, pitch_deg, range_m):
    az = math.radians(float(az_deg))
    pitch = math.radians(float(pitch_deg))
    cp = math.cos(pitch)
    return np.array([
        range_m * cp * math.sin(az),
        range_m * cp * math.cos(az),
        range_m * math.sin(pitch),
    ], dtype=float)


def should_keep_sample(sample, session_name):
    idx = int(sample.get("index", -1))
    if idx in DROP_INDEXES:
        return False
    if idx in set(DROP_BY_SESSION.get(session_name, [])):
        return False

    radar = sample.get("radar", {})
    radar_range = float(radar.get("range", 0) or 0)
    for range_min, range_max in DROP_RANGE_SEGMENTS:
        if range_min <= radar_range <= range_max:
            return False

    if ONLY_RANGE_SEGMENTS:
        keep = False
        for range_min, range_max in ONLY_RANGE_SEGMENTS:
            if range_min <= radar_range <= range_max:
                keep = True
                break
        if not keep:
            return False

    return True


def load_measurements():
    measurements = []
    used_sessions = []
    for session_path in INPUT_SESSION_FILES:
        path = Path(session_path)
        if not path.exists():
            print(f"[joint] skip missing session: {path}")
            continue
        session_name = path.name
        data = json.loads(path.read_text(encoding="utf-8"))
        used_sessions.append(str(path))
        source = data.get("source_measurements", {})
        for sample in source.get("samples", []):
            if not should_keep_sample(sample, session_name):
                continue
            radar = sample.get("radar", {})
            optical = sample.get("optical", {})
            try:
                measurements.append({
                    "session_path": str(path),
                    "index": int(sample.get("index", len(measurements) + 1)),
                    "radar_range": float(radar["range"]),
                    "radar_az": float(radar["azimuth"]),
                    "radar_pitch": float(radar["pitch"]),
                    "optical_az": float(optical["azimuth"]),
                    "optical_pitch": float(optical["pitch"]),
                    "time_diff_sec": float(sample.get("time_diff_sec", 0) or 0),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return measurements, used_sessions


def summarize_angle_offsets(measurements):
    az_diffs = [angle_diff_deg(m["optical_az"], m["radar_az"]) for m in measurements]
    pitch_diffs = [m["optical_pitch"] - m["radar_pitch"] for m in measurements]
    return {
        "azimuth_offset_median": float(np.median(az_diffs)) if az_diffs else 0.0,
        "pitch_offset_median": float(np.median(pitch_diffs)) if pitch_diffs else 0.0,
        "azimuth_offset_mean": float(np.mean(az_diffs)) if az_diffs else 0.0,
        "pitch_offset_mean": float(np.mean(pitch_diffs)) if pitch_diffs else 0.0,
        "azimuth_offset_std": float(np.std(az_diffs)) if az_diffs else 0.0,
        "pitch_offset_std": float(np.std(pitch_diffs)) if pitch_diffs else 0.0,
    }


def params_from_dict(params_dict):
    return [
        float(params_dict["dx_m"]),
        float(params_dict["dy_m"]),
        float(params_dict["dz_m"]),
        float(params_dict["delta_az_deg"]),
        float(params_dict["delta_pitch_deg"]),
    ]


def predict_angles(measurement, params):
    dx, dy, dz, delta_az_deg, delta_pitch_deg = params
    radar_pos = polar_to_vector(measurement["radar_az"], measurement["radar_pitch"], measurement["radar_range"])
    rel = radar_pos - np.array([dx, dy, dz], dtype=float)
    rel_range = float(np.linalg.norm(rel))
    pred_az = math.degrees(math.atan2(rel[0], rel[1])) % 360.0
    pred_pitch = math.degrees(math.asin(rel[2] / rel_range)) if rel_range > 1e-9 else 0.0
    pred_az = (pred_az + delta_az_deg) % 360.0
    pred_pitch = pred_pitch + delta_pitch_deg
    return pred_az, pred_pitch


def residuals(params, measurements):
    values = []
    for m in measurements:
        pred_az, pred_pitch = predict_angles(m, params)
        values.append(angle_diff_deg(pred_az, m["optical_az"]))
        values.append(pred_pitch - m["optical_pitch"])
    return np.array(values, dtype=float)


def compute_component_errors(params, measurements):
    az_errors = []
    pitch_errors = []
    total_errors = []
    for m in measurements:
        pred_az, pred_pitch = predict_angles(m, params)
        az_err = angle_diff_deg(pred_az, m["optical_az"])
        pitch_err = pred_pitch - m["optical_pitch"]
        az_errors.append(abs(az_err))
        pitch_errors.append(abs(pitch_err))
        total_errors.append(math.sqrt(az_err ** 2 + pitch_err ** 2))
    return {
        "mean_az_error_deg": float(np.mean(az_errors)) if az_errors else 0.0,
        "max_az_error_deg": float(np.max(az_errors)) if az_errors else 0.0,
        "mean_pitch_error_deg": float(np.mean(pitch_errors)) if pitch_errors else 0.0,
        "max_pitch_error_deg": float(np.max(pitch_errors)) if pitch_errors else 0.0,
        "mean_error_deg": float(np.mean(total_errors)) if total_errors else 0.0,
        "std_error_deg": float(np.std(total_errors)) if total_errors else 0.0,
        "max_error_deg": float(np.max(total_errors)) if total_errors else 0.0,
    }


def calculate_geometry_condition(measurements):
    if not measurements:
        return float("inf")
    eye = np.eye(3)
    a_blocks = []
    for m in measurements:
        az = math.radians(m["optical_az"])
        pitch = math.radians(m["optical_pitch"])
        u = np.array([
            math.cos(pitch) * math.sin(az),
            math.cos(pitch) * math.cos(az),
            math.sin(pitch),
        ], dtype=float)
        u = u / max(np.linalg.norm(u), 1e-9)
        a_blocks.append(eye - np.outer(u, u))
    A = np.vstack(a_blocks)
    ata = A.T @ A
    singular_values = np.linalg.svd(ata, compute_uv=False)
    if singular_values[-1] <= 1e-12:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def fit_joint_params(measurements):
    angle_stats = summarize_angle_offsets(measurements)
    x0 = np.array([0.0, 0.0, 0.0, angle_stats["azimuth_offset_median"], angle_stats["pitch_offset_median"]], dtype=float)
    lower = np.array([
        -CALIBRATION_MAX_OFFSET_X_M,
        -CALIBRATION_MAX_OFFSET_Y_M,
        -CALIBRATION_MAX_OFFSET_Z_M,
        -MAX_AZ_BIAS_DEG,
        -MAX_PITCH_BIAS_DEG,
    ], dtype=float)
    upper = np.array([
        CALIBRATION_MAX_OFFSET_X_M,
        CALIBRATION_MAX_OFFSET_Y_M,
        CALIBRATION_MAX_OFFSET_Z_M,
        MAX_AZ_BIAS_DEG,
        MAX_PITCH_BIAS_DEG,
    ], dtype=float)
    x0 = np.clip(x0, lower, upper)

    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        args=(measurements,),
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        verbose=0,
    )
    if not result.success:
        return None, f"optimization failed: {result.message}"

    params = [float(v) for v in result.x]
    return {
        "parameters": {
            "dx_m": params[0],
            "dy_m": params[1],
            "dz_m": params[2],
            "delta_az_deg": params[3],
            "delta_pitch_deg": params[4],
        },
        "angle_stats": angle_stats,
        "errors": compute_component_errors(params, measurements),
        "geometry_condition": calculate_geometry_condition(measurements),
    }, None


def fit_joint_params_per_2d_bin(measurements):
    grouped = {}
    for m in measurements:
        range_bin = int(float(m["radar_range"]) // RANGE_BIN_SIZE_M) * int(RANGE_BIN_SIZE_M)
        az = float(m["radar_az"]) % 360.0
        az_bin = int(az // AZ_BIN_SIZE_DEG) * int(AZ_BIN_SIZE_DEG)
        grouped.setdefault((range_bin, az_bin), []).append(m)

    segments = []
    for (range_bin, az_bin), rows in sorted(grouped.items()):
        if len(rows) < max(RANGE_BIN_MIN_SAMPLES, AZ_BIN_MIN_SAMPLES):
            continue
        fitted, err = fit_joint_params(rows)
        if err is not None:
            continue
        params = fitted["parameters"]
        errors = fitted["errors"]
        ranges = [float(item["radar_range"]) for item in rows]
        az_values = [float(item["radar_az"]) % 360.0 for item in rows]
        usable = (
            errors["mean_error_deg"] <= max(RANGE_USABLE_MEAN_ERROR_DEG, AZ_USABLE_MEAN_ERROR_DEG)
            and errors["max_error_deg"] <= max(RANGE_USABLE_MAX_ERROR_DEG, AZ_USABLE_MAX_ERROR_DEG)
            and fitted["geometry_condition"] <= 1.0e6
            and abs(params["dx_m"]) <= CALIBRATION_MAX_OFFSET_X_M
            and abs(params["dy_m"]) <= CALIBRATION_MAX_OFFSET_Y_M
            and abs(params["dz_m"]) <= CALIBRATION_MAX_OFFSET_Z_M
        )
        segments.append({
            "range_min_m": float(min(ranges)),
            "range_max_m": float(max(ranges)),
            "range_bin_start_m": float(range_bin),
            "range_bin_end_m": float(range_bin + RANGE_BIN_SIZE_M),
            "az_min_deg": float(min(az_values)),
            "az_max_deg": float(max(az_values)),
            "az_bin_start_deg": float(az_bin),
            "az_bin_end_deg": float(az_bin + AZ_BIN_SIZE_DEG),
            "sample_count": len(rows),
            "geometry_condition": float(fitted["geometry_condition"]),
            "usable": usable,
            "usable_reason": "2d_segment_fit" if usable else "2d_segment_not_usable",
            "segment_type": "joint_position_angle_bias_2d",
            "segment_solver": "offline_joint_position_angle_bias",
            "parameters": params,
            "angle_stats": fitted["angle_stats"],
            "errors": errors,
        })
    return segments


def build_range_bin_summaries(params, measurements):
    grouped = {}
    for m in measurements:
        range_bin = int(float(m["radar_range"]) // RANGE_BIN_SIZE_M) * int(RANGE_BIN_SIZE_M)
        grouped.setdefault(range_bin, []).append(m)
    summaries = []
    for range_bin, rows in sorted(grouped.items()):
        stats = compute_component_errors(params, rows)
        ranges = [float(item["radar_range"]) for item in rows]
        usable = len(rows) >= RANGE_BIN_MIN_SAMPLES and stats["mean_error_deg"] <= RANGE_USABLE_MEAN_ERROR_DEG and stats["max_error_deg"] <= RANGE_USABLE_MAX_ERROR_DEG
        summaries.append({
            "range_min_m": float(min(ranges)),
            "range_max_m": float(max(ranges)),
            "sample_count": len(rows),
            "mean_error_deg": stats["mean_error_deg"],
            "max_error_deg": stats["max_error_deg"],
            "usable": usable,
        })
    return summaries


def build_az_bin_summaries(params, measurements):
    grouped = {}
    for m in measurements:
        az = float(m["radar_az"]) % 360.0
        az_bin = int(az // AZ_BIN_SIZE_DEG) * int(AZ_BIN_SIZE_DEG)
        grouped.setdefault(az_bin, []).append(m)
    summaries = []
    for az_bin, rows in sorted(grouped.items()):
        stats = compute_component_errors(params, rows)
        az_values = [float(item["radar_az"]) % 360.0 for item in rows]
        usable = len(rows) >= AZ_BIN_MIN_SAMPLES and stats["mean_error_deg"] <= AZ_USABLE_MEAN_ERROR_DEG and stats["max_error_deg"] <= AZ_USABLE_MAX_ERROR_DEG
        summaries.append({
            "az_min_deg": float(min(az_values)),
            "az_max_deg": float(max(az_values)),
            "sample_count": len(rows),
            "mean_error_deg": stats["mean_error_deg"],
            "max_error_deg": stats["max_error_deg"],
            "usable": usable,
        })
    return summaries


def build_2d_bin_summaries(params, measurements):
    grouped = {}
    for m in measurements:
        range_bin = int(float(m["radar_range"]) // RANGE_BIN_SIZE_M) * int(RANGE_BIN_SIZE_M)
        az = float(m["radar_az"]) % 360.0
        az_bin = int(az // AZ_BIN_SIZE_DEG) * int(AZ_BIN_SIZE_DEG)
        grouped.setdefault((range_bin, az_bin), []).append(m)
    summaries = []
    for (_range_bin, _az_bin), rows in sorted(grouped.items()):
        stats = compute_component_errors(params, rows)
        ranges = [float(item["radar_range"]) for item in rows]
        az_values = [float(item["radar_az"]) % 360.0 for item in rows]
        usable = len(rows) >= max(RANGE_BIN_MIN_SAMPLES, AZ_BIN_MIN_SAMPLES) and stats["mean_error_deg"] <= max(RANGE_USABLE_MEAN_ERROR_DEG, AZ_USABLE_MEAN_ERROR_DEG) and stats["max_error_deg"] <= max(RANGE_USABLE_MAX_ERROR_DEG, AZ_USABLE_MAX_ERROR_DEG)
        summaries.append({
            "range_min_m": float(min(ranges)),
            "range_max_m": float(max(ranges)),
            "az_min_deg": float(min(az_values)),
            "az_max_deg": float(max(az_values)),
            "sample_count": len(rows),
            "mean_error_deg": stats["mean_error_deg"],
            "max_error_deg": stats["max_error_deg"],
            "usable": usable,
        })
    return summaries


def merge_usable_ranges(bin_summaries):
    return [item for item in bin_summaries if item["usable"]]


def merge_usable_az_ranges(bin_summaries):
    return [item for item in bin_summaries if item["usable"]]


def merge_usable_range_az_ranges(segment_fits):
    return [
        {
            "range_min_m": item["range_min_m"],
            "range_max_m": item["range_max_m"],
            "az_min_deg": item["az_min_deg"],
            "az_max_deg": item["az_max_deg"],
            "sample_count": item["sample_count"],
            "mean_error_deg": item["errors"]["mean_error_deg"],
            "max_error_deg": item["errors"]["max_error_deg"],
            "parameters": item["parameters"],
            "errors": item["errors"],
            "angle_stats": item["angle_stats"],
        }
        for item in segment_fits
        if item["usable"]
    ]


def save_range_error_plot(bin_summaries):
    if not bin_summaries:
        return None
    xs = [0.5 * (item["range_min_m"] + item["range_max_m"]) for item in bin_summaries]
    ys = [item["mean_error_deg"] for item in bin_summaries]
    path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_range_errors.png"
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Radar range (m)")
    plt.ylabel("Mean error (deg)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def save_az_error_plot(bin_summaries):
    if not bin_summaries:
        return None
    xs = [0.5 * (item["az_min_deg"] + item["az_max_deg"]) for item in bin_summaries]
    ys = [item["mean_error_deg"] for item in bin_summaries]
    path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_az_errors.png"
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Radar azimuth (deg)")
    plt.ylabel("Mean error (deg)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def save_range_az_error_plot(segment_fits):
    if not segment_fits:
        return None
    xs = [0.5 * (item["range_min_m"] + item["range_max_m"]) for item in segment_fits]
    ys = [0.5 * (item["az_min_deg"] + item["az_max_deg"]) for item in segment_fits]
    cs = [item["errors"]["mean_error_deg"] for item in segment_fits]
    sizes = [max(20, item["sample_count"] * 20) for item in segment_fits]
    path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_range_az_errors.png"
    plt.figure(figsize=(11, 7))
    sc = plt.scatter(xs, ys, c=cs, s=sizes, cmap="viridis", alpha=0.85, edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, label="mean error (deg)")
    plt.xlabel("Radar range (m)")
    plt.ylabel("Radar azimuth (deg)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def main():
    measurements, used_sessions = load_measurements()
    if len(measurements) < MIN_SAMPLES:
        print(f"[joint] not enough samples, current={len(measurements)}, need>={MIN_SAMPLES}")
        return 1

    global_fit, err = fit_joint_params(measurements)
    if err is not None:
        print(f"[joint] {err}")
        return 1

    global_params = global_fit["parameters"]
    global_errors = global_fit["errors"]
    global_geometry_condition = global_fit["geometry_condition"]
    global_angle_stats = global_fit["angle_stats"]

    segment_fits = fit_joint_params_per_2d_bin(measurements)
    range_bin_summaries = build_range_bin_summaries(params_from_dict(global_params), measurements)
    az_bin_summaries = build_az_bin_summaries(params_from_dict(global_params), measurements)
    usable_ranges = merge_usable_ranges(range_bin_summaries)
    usable_az_ranges = merge_usable_az_ranges(az_bin_summaries)
    usable_range_az_ranges = merge_usable_range_az_ranges(segment_fits)

    range_plot_path = save_range_error_plot(range_bin_summaries)
    az_plot_path = save_az_error_plot(az_bin_summaries)
    range_az_plot_path = save_range_az_error_plot(segment_fits)

    summary = {
        "used_sessions": used_sessions,
        "sample_count": len(measurements),
        "drop_indexes": list(DROP_INDEXES),
        "drop_by_session": {key: list(value) for key, value in DROP_BY_SESSION.items()},
        "drop_range_segments": list(DROP_RANGE_SEGMENTS),
        "only_range_segments": list(ONLY_RANGE_SEGMENTS),
        "method": "offline_joint_position_angle_bias_2d_segments",
        "global_parameters": global_params,
        "global_errors": global_errors,
        "global_geometry_condition": global_geometry_condition,
        "global_angle_stats": global_angle_stats,
        "range_bin_summaries": range_bin_summaries,
        "az_bin_summaries": az_bin_summaries,
        "usable_ranges": usable_ranges,
        "usable_az_ranges": usable_az_ranges,
        "usable_range_az_ranges": usable_range_az_ranges,
        "segment_fits": segment_fits,
        "range_error_plot": range_plot_path,
        "az_error_plot": az_plot_path,
        "range_az_error_plot": range_az_plot_path,
        "saved_at": float(time.time()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[joint] used sessions:")
    for path in used_sessions:
        print(f"  {path}")
    print(f"[joint] sample_count: {len(measurements)}")
    print(
        f"[joint] global ref: dx={global_params['dx_m']:.3f}m, dy={global_params['dy_m']:.3f}m, dz={global_params['dz_m']:.3f}m, "
        f"delta_az={global_params['delta_az_deg']:.3f}°, delta_pitch={global_params['delta_pitch_deg']:.3f}°"
    )
    print(
        f"[joint] global errors: mean={global_errors['mean_error_deg']:.3f}°, std={global_errors['std_error_deg']:.3f}°, "
        f"max={global_errors['max_error_deg']:.3f}°"
    )
    print(f"[joint] global geometry_condition: {global_geometry_condition:.3f}")

    print("[joint] fitted 2D segments:")
    if segment_fits:
        for item in segment_fits:
            p = item["parameters"]
            e = item["errors"]
            print(
                f"  range {item['range_min_m']:.0f}-{item['range_max_m']:.0f}m, "
                f"az {item['az_min_deg']:.0f}-{item['az_max_deg']:.0f}deg -> "
                f"dx={p['dx_m']:.3f}m, dy={p['dy_m']:.3f}m, dz={p['dz_m']:.3f}m, "
                f"daz={p['delta_az_deg']:.3f}°, dp={p['delta_pitch_deg']:.3f}°, "
                f"mean={e['mean_error_deg']:.3f}°, usable={item['usable']}"
            )
    else:
        print("  none")

    if usable_ranges:
        print("[joint] usable ranges:")
        for item in usable_ranges:
            print(f"  {item['range_min_m']:.0f}-{item['range_max_m']:.0f}m (n={item['sample_count']}, mean={item['mean_error_deg']:.3f}°)")
    else:
        print("[joint] usable ranges: none")

    if usable_az_ranges:
        print("[joint] usable azimuth ranges:")
        for item in usable_az_ranges:
            print(f"  {item['az_min_deg']:.0f}-{item['az_max_deg']:.0f}deg (n={item['sample_count']}, mean={item['mean_error_deg']:.3f}°)")
    else:
        print("[joint] usable azimuth ranges: none")

    if usable_range_az_ranges:
        print("[joint] usable 2D cells:")
        for item in usable_range_az_ranges:
            print(
                f"  range {item['range_min_m']:.0f}-{item['range_max_m']:.0f}m, "
                f"az {item['az_min_deg']:.0f}-{item['az_max_deg']:.0f}deg "
                f"(n={item['sample_count']}, mean={item['mean_error_deg']:.3f}°, max={item['max_error_deg']:.3f}°)"
            )
    else:
        print("[joint] usable 2D cells: none")

    if range_plot_path:
        print(f"[joint] range error plot: {range_plot_path}")
    if az_plot_path:
        print(f"[joint] azimuth error plot: {az_plot_path}")
    if range_az_plot_path:
        print(f"[joint] range-azimuth error plot: {range_az_plot_path}")
    print(f"[joint] summary saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
