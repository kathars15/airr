# -*- coding: utf-8 -*-
"""Offline radar-range + optical-bearing fusion and augmented-bias diagnostic.

This script is intentionally standalone. It uses raw_pairs_*.json records to
test whether the current data are suitable for an augmented-bias filter:

1. radar range + optical az/pitch -> pseudo ENU points
2. pseudo ENU points -> constant-velocity Kalman smoothing
3. radar/optical measurements -> augmented EKF with radar bias states

The first two steps are the practical "optical direction + radar scale" path.
The third step is a diagnostic filter that estimates how radar angle bias
changes during the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SRC_DIR = Path(__file__).resolve().parents[2]
DESK_ROOT = SRC_DIR.parents[3]
RAW_PAIR_DIR = SRC_DIR / "calibration_data" / "raw_pair_records"
OUTPUT_DIR = DESK_ROOT / "calibration_data"


# Default inputs. Edit this list when testing another run.
INPUT_RAW_PAIR_FILES = [
    RAW_PAIR_DIR / "raw_pairs_20260519_113114_Radar-24.json",
]

# Optional manual drop list. Keys can be a full file name or a stem.
DROP_BY_FILE: Dict[str, List[int]] = {
    # "raw_pairs_20260519_113114_Radar-24.json": [17, 18],
}


# Measurement trust. The radar pitch sigma is deliberately large because the
# field data show distance-dependent plateaus and jumps.
RADAR_RANGE_SIGMA_M = 20.0
RADAR_AZ_SIGMA_DEG = 1.2
RADAR_PITCH_SIGMA_DEG = 5.0
OPTICAL_AZ_SIGMA_DEG = 0.15
OPTICAL_PITCH_SIGMA_DEG = 0.15


# Filter dynamics.
CV_ACCEL_SIGMA_MPS2 = 4.0
EKF_ACCEL_SIGMA_MPS2 = 5.0
EKF_B_RANGE_RW_M_PER_SQRT_S = 0.5
EKF_B_AZ_RW_DEG_PER_SQRT_S = 0.02
EKF_B_PITCH0_RW_DEG_PER_SQRT_S = 0.02
EKF_B_PITCH_SLOPE_RW_DEG_PER_KM_SQRT_S = 0.02


# Suitability checks are intentionally conservative for "augmented bias stable".
# A run may still be useful for pseudo ENU validation even if the bias estimate
# itself is unstable.
MAX_STABLE_B_RANGE_M = 350.0
MAX_STABLE_B_AZ_DEG = 30.0
MAX_STABLE_B_PITCH0_DEG = 12.0
MAX_STABLE_B_PITCH_SLOPE_DEG_PER_KM = 12.0
MAX_STABLE_EKF_OPTICAL_RMSE_DEG = 1.0
MAX_STABLE_EKF_RADAR_PITCH_MODEL_RMSE_DEG = 2.0


def wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def wrap_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def rmse(values: Sequence[float]) -> float:
    return math.sqrt(mean([float(v) ** 2 for v in values])) if values else float("nan")


def std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    ma = mean(a)
    mb = mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 1e-12 or vb <= 1e-12:
        return float("nan")
    return float(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb))


def linfit(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    if len(x) < 2:
        return 0.0, y[0] if y else 0.0, 0.0
    mx = mean(x)
    my = mean(y)
    den = sum((xi - mx) ** 2 for xi in x)
    if den <= 1e-12:
        return 0.0, my, rmse([yi - my for yi in y])
    slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / den
    intercept = my - slope * mx
    residual = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
    return float(slope), float(intercept), rmse(residual)


def stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "rmse": float("nan")}
    return {
        "mean": mean(values),
        "std": std(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "rmse": rmse(values),
    }


def polar_to_enu(range_m: float, az_deg: float, pitch_deg: float) -> np.ndarray:
    az = math.radians(float(az_deg))
    pitch = math.radians(float(pitch_deg))
    r = float(range_m)
    return np.array(
        [
            r * math.cos(pitch) * math.sin(az),
            r * math.cos(pitch) * math.cos(az),
            r * math.sin(pitch),
        ],
        dtype=float,
    )


def enu_to_polar(pos: Sequence[float]) -> Dict[str, float]:
    e, n, u = [float(v) for v in pos[:3]]
    r = math.sqrt(e * e + n * n + u * u)
    if r <= 1e-9:
        return {"range": 0.0, "azimuth": 0.0, "pitch": 0.0}
    az = math.degrees(math.atan2(e, n)) % 360.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, u / r))))
    return {"range": r, "azimuth": az, "pitch": pitch}


def load_raw_pairs(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples") or data.get("source_measurements", {}).get("samples") or []
    drop = set(DROP_BY_FILE.get(path.name, []) + DROP_BY_FILE.get(path.stem, []))
    rows = []
    for item in samples:
        index = int(item.get("index", len(rows) + 1))
        if index in drop:
            continue
        radar = item.get("radar", {})
        optical = item.get("optical", {})
        try:
            rows.append(
                {
                    "index": index,
                    "timestamp": float(radar.get("timestamp", item.get("timestamp", index))),
                    "time_diff_sec": float(item.get("time_diff_sec", 0.0) or 0.0),
                    "track_id": str(radar.get("track_id", "")),
                    "radar_range": float(radar["range"]),
                    "radar_azimuth": float(radar["azimuth"]) % 360.0,
                    "radar_pitch": float(radar["pitch"]),
                    "optical_azimuth": float(optical["azimuth"]) % 360.0,
                    "optical_pitch": float(optical["pitch"]),
                    "optical_status": int(optical.get("status", 0) or 0),
                    "radar_device_timestamp": radar.get("device_timestamp"),
                    "optical_device_timestamp": optical.get("device_timestamp"),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort(key=lambda row: (row["timestamp"], row["index"]))
    return rows


def latest_inputs(count: int) -> List[Path]:
    files = sorted(RAW_PAIR_DIR.glob("raw_pairs_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return list(reversed(files[: max(1, int(count))]))


def cv_kalman_filter(times: Sequence[float], positions: Sequence[np.ndarray]) -> List[np.ndarray]:
    if not positions:
        return []
    n = len(positions)
    x = np.zeros((6, 1), dtype=float)
    x[:3, 0] = positions[0]
    if n > 1:
        dt0 = max(1e-3, times[1] - times[0])
        x[3:, 0] = (positions[1] - positions[0]) / dt0
    p = np.diag([80.0**2, 80.0**2, 40.0**2, 50.0**2, 50.0**2, 20.0**2])
    h = np.zeros((3, 6), dtype=float)
    h[:, :3] = np.eye(3)
    r = np.diag([25.0**2, 25.0**2, 15.0**2])
    out = []
    last_t = times[0]
    for t, z_pos in zip(times, positions):
        dt = max(1e-3, min(10.0, float(t - last_t)))
        last_t = t
        f = np.eye(6)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        q = np.zeros((6, 6), dtype=float)
        accel_var = CV_ACCEL_SIGMA_MPS2**2
        block = np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=float) * accel_var
        for axis in range(3):
            q[axis, axis] = block[0, 0]
            q[axis, axis + 3] = block[0, 1]
            q[axis + 3, axis] = block[1, 0]
            q[axis + 3, axis + 3] = block[1, 1]
        x = f @ x
        p = f @ p @ f.T + q
        z = np.asarray(z_pos, dtype=float).reshape(3, 1)
        y = z - h @ x
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ y
        p = (np.eye(6) - k @ h) @ p
        out.append(x.copy())
    return out


def pitch_jacobian(e: float, n: float, u: float) -> Tuple[np.ndarray, float, float]:
    horiz = math.sqrt(max(1e-12, e * e + n * n))
    rho2 = max(1e-12, e * e + n * n + u * u)
    rho = math.sqrt(rho2)
    jac = np.array(
        [
            -u * e / (rho2 * horiz),
            -u * n / (rho2 * horiz),
            horiz / rho2,
        ],
        dtype=float,
    )
    return jac, rho, horiz


def augmented_ekf(
    rows: Sequence[Dict[str, float]],
    pseudo_positions: Sequence[np.ndarray],
) -> List[Dict[str, object]]:
    if not rows:
        return []
    times = [row["timestamp"] for row in rows]
    ranges = [row["radar_range"] for row in rows]
    ref_range = mean(ranges)
    pitch_res_deg = [wrap_deg(row["radar_pitch"] - row["optical_pitch"]) for row in rows]
    az_res_deg = [wrap_deg(row["radar_azimuth"] - row["optical_azimuth"]) for row in rows]
    slope_deg_per_m, intercept_deg, _ = linfit(ranges, pitch_res_deg)
    init_pitch0_deg = intercept_deg + slope_deg_per_m * ref_range
    init_pitch_slope_deg_per_km = slope_deg_per_m * 1000.0

    x = np.zeros((10, 1), dtype=float)
    x[:3, 0] = pseudo_positions[0]
    if len(rows) > 1:
        dt0 = max(1e-3, times[1] - times[0])
        x[3:6, 0] = (pseudo_positions[1] - pseudo_positions[0]) / dt0
    x[6, 0] = 0.0
    x[7, 0] = math.radians(statistics.median(az_res_deg))
    x[8, 0] = math.radians(init_pitch0_deg)
    x[9, 0] = math.radians(init_pitch_slope_deg_per_km)

    p = np.diag(
        [
            100.0**2,
            100.0**2,
            60.0**2,
            60.0**2,
            60.0**2,
            30.0**2,
            80.0**2,
            math.radians(8.0) ** 2,
            math.radians(8.0) ** 2,
            math.radians(8.0) ** 2,
        ]
    )

    r_diag = np.array(
        [
            RADAR_RANGE_SIGMA_M**2,
            math.radians(RADAR_AZ_SIGMA_DEG) ** 2,
            math.radians(RADAR_PITCH_SIGMA_DEG) ** 2,
            math.radians(OPTICAL_AZ_SIGMA_DEG) ** 2,
            math.radians(OPTICAL_PITCH_SIGMA_DEG) ** 2,
        ],
        dtype=float,
    )
    r_mat = np.diag(r_diag)

    out = []
    last_t = times[0]
    eye = np.eye(10)
    for row in rows:
        dt = max(1e-3, min(10.0, row["timestamp"] - last_t))
        last_t = row["timestamp"]

        f = np.eye(10)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        q = np.zeros((10, 10), dtype=float)
        accel_var = EKF_ACCEL_SIGMA_MPS2**2
        block = np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=float) * accel_var
        for axis in range(3):
            q[axis, axis] = block[0, 0]
            q[axis, axis + 3] = block[0, 1]
            q[axis + 3, axis] = block[1, 0]
            q[axis + 3, axis + 3] = block[1, 1]
        q[6, 6] = (EKF_B_RANGE_RW_M_PER_SQRT_S**2) * dt
        q[7, 7] = (math.radians(EKF_B_AZ_RW_DEG_PER_SQRT_S) ** 2) * dt
        q[8, 8] = (math.radians(EKF_B_PITCH0_RW_DEG_PER_SQRT_S) ** 2) * dt
        q[9, 9] = (math.radians(EKF_B_PITCH_SLOPE_RW_DEG_PER_KM_SQRT_S) ** 2) * dt

        x = f @ x
        p = f @ p @ f.T + q

        e, n, u = x[0, 0], x[1, 0], x[2, 0]
        rho = max(1e-9, math.sqrt(e * e + n * n + u * u))
        horiz2 = max(1e-12, e * e + n * n)
        az = math.atan2(e, n)
        pitch = math.atan2(u, math.sqrt(horiz2))
        drho_dp = np.array([e / rho, n / rho, u / rho], dtype=float)
        daz_dp = np.array([n / horiz2, -e / horiz2, 0.0], dtype=float)
        dpitch_dp, _, _ = pitch_jacobian(e, n, u)
        range_term = (rho - ref_range) / 1000.0
        pitch_bias = x[8, 0] + x[9, 0] * range_term

        h_val = np.array(
            [
                rho + x[6, 0],
                az + x[7, 0],
                pitch + pitch_bias,
                az,
                pitch,
            ],
            dtype=float,
        ).reshape(5, 1)
        z = np.array(
            [
                row["radar_range"],
                math.radians(row["radar_azimuth"]),
                math.radians(row["radar_pitch"]),
                math.radians(row["optical_azimuth"]),
                math.radians(row["optical_pitch"]),
            ],
            dtype=float,
        ).reshape(5, 1)

        h_jac = np.zeros((5, 10), dtype=float)
        h_jac[0, :3] = drho_dp
        h_jac[0, 6] = 1.0
        h_jac[1, :3] = daz_dp
        h_jac[1, 7] = 1.0
        h_jac[2, :3] = dpitch_dp + (x[9, 0] / 1000.0) * drho_dp
        h_jac[2, 8] = 1.0
        h_jac[2, 9] = range_term
        h_jac[3, :3] = daz_dp
        h_jac[4, :3] = dpitch_dp

        innovation = z - h_val
        for angle_idx in (1, 2, 3, 4):
            innovation[angle_idx, 0] = wrap_rad(innovation[angle_idx, 0])

        s_mat = h_jac @ p @ h_jac.T + r_mat
        k = p @ h_jac.T @ np.linalg.inv(s_mat)
        x = x + k @ innovation
        p = (eye - k @ h_jac) @ p

        polar = enu_to_polar(x[:3, 0])
        rho_after = polar["range"]
        range_term_after = (rho_after - ref_range) / 1000.0
        b_pitch_after = x[8, 0] + x[9, 0] * range_term_after
        out.append(
            {
                "state": x.copy(),
                "covariance_diag": np.diag(p).copy(),
                "polar": polar,
                "b_range_m": float(x[6, 0]),
                "b_az_deg": math.degrees(float(x[7, 0])),
                "b_pitch0_deg": math.degrees(float(x[8, 0])),
                "b_pitch_slope_deg_per_km": math.degrees(float(x[9, 0])),
                "b_pitch_at_range_deg": math.degrees(float(b_pitch_after)),
                "ref_range_m": ref_range,
                "innovation": innovation[:, 0].copy(),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_run(output_png: Path, label: str, csv_rows: Sequence[Dict[str, object]]) -> None:
    ranges = [float(r["radar_range_m"]) for r in csv_rows]
    radar_az = [float(r["radar_azimuth_deg"]) for r in csv_rows]
    optical_az = [float(r["optical_azimuth_deg"]) for r in csv_rows]
    ekf_az = [float(r["ekf_azimuth_deg"]) for r in csv_rows]
    radar_pitch = [float(r["radar_pitch_deg"]) for r in csv_rows]
    optical_pitch = [float(r["optical_pitch_deg"]) for r in csv_rows]
    cv_pitch = [float(r["cv_pitch_deg"]) for r in csv_rows]
    ekf_pitch = [float(r["ekf_pitch_deg"]) for r in csv_rows]
    daz = [float(r["radar_minus_optical_az_deg"]) for r in csv_rows]
    dpitch = [float(r["radar_minus_optical_pitch_deg"]) for r in csv_rows]
    b_az = [float(r["ekf_b_az_deg"]) for r in csv_rows]
    b_pitch = [float(r["ekf_b_pitch_at_range_deg"]) for r in csv_rows]
    e_cv = [float(r["cv_e_m"]) for r in csv_rows]
    n_cv = [float(r["cv_n_m"]) for r in csv_rows]
    e_ekf = [float(r["ekf_e_m"]) for r in csv_rows]
    n_ekf = [float(r["ekf_n_m"]) for r in csv_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    fig.suptitle(f"Augmented bearing fusion diagnostic: {label}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(ranges, radar_az, "o-", color="#4c78a8", label="radar az", alpha=0.65)
    ax.plot(ranges, ekf_az, "s-", color="#54a24b", label="aug EKF az", alpha=0.75)
    ax.plot(
        ranges,
        optical_az,
        "o--",
        color="#f58518",
        linewidth=2.2,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="optical az",
        zorder=5,
    )
    ax.set_title("Azimuth")
    ax.set_xlabel("Radar range (m)")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ranges, radar_pitch, "o-", color="#4c78a8", label="radar pitch", alpha=0.65)
    ax.plot(ranges, cv_pitch, "^-", color="#54a24b", label="CV pseudo pitch", alpha=0.45)
    ax.plot(ranges, ekf_pitch, "s-", color="#e45756", label="aug EKF pitch", alpha=0.65)
    ax.plot(
        ranges,
        optical_pitch,
        "o--",
        color="#f58518",
        linewidth=2.4,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="optical pitch",
        zorder=6,
    )
    ax.set_title("Pitch")
    ax.set_xlabel("Radar range (m)")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(ranges, daz, "o-", label="radar-optical az residual")
    ax.plot(ranges, b_az, "s-", label="EKF estimated b_az")
    ax.plot(ranges, dpitch, "o-", label="radar-optical pitch residual")
    ax.plot(ranges, b_pitch, "s-", label="EKF estimated b_pitch(range)")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Estimated radar biases")
    ax.set_xlabel("Radar range (m)")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(e_cv, n_cv, "o-", label="CV pseudo EN")
    ax.plot(e_ekf, n_ekf, "s-", label="aug EKF EN")
    ax.set_title("ENU ground track")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def plot_raw_angles(output_png: Path, label: str, csv_rows: Sequence[Dict[str, object]]) -> None:
    ranges = [float(r["radar_range_m"]) for r in csv_rows]
    radar_az = [float(r["radar_azimuth_deg"]) for r in csv_rows]
    optical_az = [float(r["optical_azimuth_deg"]) for r in csv_rows]
    radar_pitch = [float(r["radar_pitch_deg"]) for r in csv_rows]
    optical_pitch = [float(r["optical_pitch_deg"]) for r in csv_rows]
    daz = [float(r["radar_minus_optical_az_deg"]) for r in csv_rows]
    dpitch = [float(r["radar_minus_optical_pitch_deg"]) for r in csv_rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=160, sharex=True)
    fig.suptitle(f"Raw radar vs optical angles: {label}", fontsize=14)

    ax = axes[0]
    ax.plot(ranges, radar_az, "o-", color="#4c78a8", label="radar az")
    ax.plot(
        ranges,
        optical_az,
        "o--",
        color="#f58518",
        linewidth=2.2,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="optical az",
    )
    ax.set_ylabel("Azimuth (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ranges, radar_pitch, "o-", color="#4c78a8", label="radar pitch")
    ax.plot(
        ranges,
        optical_pitch,
        "o--",
        color="#f58518",
        linewidth=2.2,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="optical pitch",
    )
    ax.set_ylabel("Pitch (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(ranges, daz, "o-", color="#4c78a8", label="radar - optical az")
    ax.plot(ranges, dpitch, "s-", color="#e45756", label="radar - optical pitch")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Radar range (m)")
    ax.set_ylabel("Residual (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def process_run(path: Path, output_dir: Path) -> Dict[str, object]:
    rows = load_raw_pairs(path)
    if len(rows) < 3:
        raise ValueError(f"not enough valid samples in {path}")

    t0 = rows[0]["timestamp"]
    times = [row["timestamp"] for row in rows]
    rel_times = [t - t0 for t in times]
    pseudo_positions = [
        polar_to_enu(row["radar_range"], row["optical_azimuth"], row["optical_pitch"])
        for row in rows
    ]
    radar_positions = [
        polar_to_enu(row["radar_range"], row["radar_azimuth"], row["radar_pitch"])
        for row in rows
    ]
    cv_states = cv_kalman_filter(times, pseudo_positions)
    ekf_rows = augmented_ekf(rows, pseudo_positions)

    csv_rows = []
    radar_opt_az_res = []
    radar_opt_pitch_res = []
    ekf_opt_az_res = []
    ekf_opt_pitch_res = []
    ekf_radar_pitch_model_res = []
    ranges = []
    for i, row in enumerate(rows):
        pseudo_polar = enu_to_polar(pseudo_positions[i])
        radar_polar = enu_to_polar(radar_positions[i])
        cv_state = cv_states[i]
        cv_polar = enu_to_polar(cv_state[:3, 0])
        ekf = ekf_rows[i]
        ekf_state = ekf["state"]
        ekf_polar = ekf["polar"]
        radar_minus_opt_az = wrap_deg(row["radar_azimuth"] - row["optical_azimuth"])
        radar_minus_opt_pitch = wrap_deg(row["radar_pitch"] - row["optical_pitch"])
        ekf_minus_opt_az = wrap_deg(ekf_polar["azimuth"] - row["optical_azimuth"])
        ekf_minus_opt_pitch = wrap_deg(ekf_polar["pitch"] - row["optical_pitch"])
        radar_pitch_model_res = wrap_deg(row["radar_pitch"] - (ekf_polar["pitch"] + ekf["b_pitch_at_range_deg"]))
        radar_opt_az_res.append(radar_minus_opt_az)
        radar_opt_pitch_res.append(radar_minus_opt_pitch)
        ekf_opt_az_res.append(ekf_minus_opt_az)
        ekf_opt_pitch_res.append(ekf_minus_opt_pitch)
        ekf_radar_pitch_model_res.append(radar_pitch_model_res)
        ranges.append(row["radar_range"])
        csv_rows.append(
            {
                "index": row["index"],
                "time_sec": rel_times[i],
                "timestamp": row["timestamp"],
                "time_diff_sec": row["time_diff_sec"],
                "track_id": row["track_id"],
                "optical_status": row["optical_status"],
                "radar_range_m": row["radar_range"],
                "radar_azimuth_deg": row["radar_azimuth"],
                "radar_pitch_deg": row["radar_pitch"],
                "optical_azimuth_deg": row["optical_azimuth"],
                "optical_pitch_deg": row["optical_pitch"],
                "radar_minus_optical_az_deg": radar_minus_opt_az,
                "radar_minus_optical_pitch_deg": radar_minus_opt_pitch,
                "pseudo_e_m": pseudo_positions[i][0],
                "pseudo_n_m": pseudo_positions[i][1],
                "pseudo_u_m": pseudo_positions[i][2],
                "pseudo_range_m": pseudo_polar["range"],
                "pseudo_azimuth_deg": pseudo_polar["azimuth"],
                "pseudo_pitch_deg": pseudo_polar["pitch"],
                "radar_e_m": radar_positions[i][0],
                "radar_n_m": radar_positions[i][1],
                "radar_u_m": radar_positions[i][2],
                "radar_recomputed_azimuth_deg": radar_polar["azimuth"],
                "radar_recomputed_pitch_deg": radar_polar["pitch"],
                "cv_e_m": cv_state[0, 0],
                "cv_n_m": cv_state[1, 0],
                "cv_u_m": cv_state[2, 0],
                "cv_ve_mps": cv_state[3, 0],
                "cv_vn_mps": cv_state[4, 0],
                "cv_vu_mps": cv_state[5, 0],
                "cv_range_m": cv_polar["range"],
                "cv_azimuth_deg": cv_polar["azimuth"],
                "cv_pitch_deg": cv_polar["pitch"],
                "ekf_e_m": ekf_state[0, 0],
                "ekf_n_m": ekf_state[1, 0],
                "ekf_u_m": ekf_state[2, 0],
                "ekf_ve_mps": ekf_state[3, 0],
                "ekf_vn_mps": ekf_state[4, 0],
                "ekf_vu_mps": ekf_state[5, 0],
                "ekf_range_m": ekf_polar["range"],
                "ekf_azimuth_deg": ekf_polar["azimuth"],
                "ekf_pitch_deg": ekf_polar["pitch"],
                "ekf_minus_optical_az_deg": ekf_minus_opt_az,
                "ekf_minus_optical_pitch_deg": ekf_minus_opt_pitch,
                "ekf_b_range_m": ekf["b_range_m"],
                "ekf_b_az_deg": ekf["b_az_deg"],
                "ekf_b_pitch0_deg": ekf["b_pitch0_deg"],
                "ekf_b_pitch_slope_deg_per_km": ekf["b_pitch_slope_deg_per_km"],
                "ekf_b_pitch_at_range_deg": ekf["b_pitch_at_range_deg"],
                "ekf_radar_pitch_model_residual_deg": radar_pitch_model_res,
            }
        )

    range_span = max(ranges) - min(ranges)
    optical_status_counts: Dict[str, int] = {}
    track_counts: Dict[str, int] = {}
    for row in rows:
        optical_status_counts[str(row["optical_status"])] = optical_status_counts.get(str(row["optical_status"]), 0) + 1
        track_counts[row["track_id"]] = track_counts.get(row["track_id"], 0) + 1

    pitch_slope_deg_per_m, pitch_intercept_deg, pitch_fit_rmse = linfit(ranges, radar_opt_pitch_res)
    az_slope_deg_per_m, az_intercept_deg, az_fit_rmse = linfit(ranges, radar_opt_az_res)
    pseudo_enu_reasons = []
    if range_span >= 300.0:
        pseudo_enu_reasons.append("range span is enough for radar-range + optical-bearing validation")
    if abs(corr(ranges, radar_opt_pitch_res)) >= 0.6:
        pseudo_enu_reasons.append("pitch residual is strongly range-correlated")
    if optical_status_counts.get("2", 0) >= max(3, int(0.8 * len(rows))):
        pseudo_enu_reasons.append("optical tracking status is mostly valid")
    pseudo_enu_feasible = len(pseudo_enu_reasons) >= 2

    run_name = path.stem
    csv_path = output_dir / f"{run_name}_augmented_bearing_samples.csv"
    json_path = output_dir / f"{run_name}_augmented_bearing_summary.json"
    png_path = output_dir / f"{run_name}_augmented_bearing_diagnostics.png"
    raw_angles_png_path = output_dir / f"{run_name}_raw_radar_vs_optical_angles.png"
    write_csv(csv_path, csv_rows)
    plot_run(png_path, run_name, csv_rows)
    plot_raw_angles(raw_angles_png_path, run_name, csv_rows)

    final_bias = ekf_rows[-1]
    ekf_optical_total_rmse = math.sqrt(
        rmse(ekf_opt_az_res) ** 2 + rmse(ekf_opt_pitch_res) ** 2
    )
    stable_checks = {
        "pseudo_enu_feasible": pseudo_enu_feasible,
        "b_range_within_limit": abs(final_bias["b_range_m"]) <= MAX_STABLE_B_RANGE_M,
        "b_az_within_limit": abs(final_bias["b_az_deg"]) <= MAX_STABLE_B_AZ_DEG,
        "b_pitch0_within_limit": abs(final_bias["b_pitch0_deg"]) <= MAX_STABLE_B_PITCH0_DEG,
        "b_pitch_slope_within_limit": abs(final_bias["b_pitch_slope_deg_per_km"]) <= MAX_STABLE_B_PITCH_SLOPE_DEG_PER_KM,
        "ekf_optical_rmse_within_limit": ekf_optical_total_rmse <= MAX_STABLE_EKF_OPTICAL_RMSE_DEG,
        "radar_pitch_model_rmse_within_limit": rmse(ekf_radar_pitch_model_res) <= MAX_STABLE_EKF_RADAR_PITCH_MODEL_RMSE_DEG,
    }
    augmented_bias_stable = all(stable_checks.values())
    failed_stable_checks = [name for name, ok in stable_checks.items() if not ok]

    summary = {
        "input": str(path),
        "sample_count": len(rows),
        "track_counts": track_counts,
        "optical_status_counts": optical_status_counts,
        "range_m": {"min": min(ranges), "max": max(ranges), "span": range_span},
        "time_diff_sec": stats([row["time_diff_sec"] for row in rows]),
        "radar_minus_optical_az_deg": stats(radar_opt_az_res),
        "radar_minus_optical_pitch_deg": stats(radar_opt_pitch_res),
        "ekf_minus_optical_az_deg": stats(ekf_opt_az_res),
        "ekf_minus_optical_pitch_deg": stats(ekf_opt_pitch_res),
        "ekf_radar_pitch_model_residual_deg": stats(ekf_radar_pitch_model_res),
        "range_correlation": {
            "az_residual": corr(ranges, radar_opt_az_res),
            "pitch_residual": corr(ranges, radar_opt_pitch_res),
        },
        "linear_bias_fit": {
            "az_slope_deg_per_m": az_slope_deg_per_m,
            "az_intercept_deg": az_intercept_deg,
            "az_rmse_deg": az_fit_rmse,
            "pitch_slope_deg_per_m": pitch_slope_deg_per_m,
            "pitch_slope_deg_per_km": pitch_slope_deg_per_m * 1000.0,
            "pitch_intercept_deg": pitch_intercept_deg,
            "pitch_rmse_deg": pitch_fit_rmse,
        },
        "final_augmented_bias": {
            "b_range_m": final_bias["b_range_m"],
            "b_az_deg": final_bias["b_az_deg"],
            "b_pitch0_deg": final_bias["b_pitch0_deg"],
            "b_pitch_slope_deg_per_km": final_bias["b_pitch_slope_deg_per_km"],
            "b_pitch_at_final_range_deg": final_bias["b_pitch_at_range_deg"],
            "ref_range_m": final_bias["ref_range_m"],
        },
        "filter_config": {
            "radar_range_sigma_m": RADAR_RANGE_SIGMA_M,
            "radar_az_sigma_deg": RADAR_AZ_SIGMA_DEG,
            "radar_pitch_sigma_deg": RADAR_PITCH_SIGMA_DEG,
            "optical_az_sigma_deg": OPTICAL_AZ_SIGMA_DEG,
            "optical_pitch_sigma_deg": OPTICAL_PITCH_SIGMA_DEG,
        },
        "pseudo_enu_feasible": pseudo_enu_feasible,
        "pseudo_enu_reasons": pseudo_enu_reasons,
        "augmented_bias_stable": augmented_bias_stable,
        "augmented_bias_stability_checks": stable_checks,
        "augmented_bias_failed_checks": failed_stable_checks,
        "suitable_for_augmented_bias_filter": augmented_bias_stable,
        "outputs": {
            "csv": str(csv_path),
            "summary_json": str(json_path),
            "diagnostic_png": str(png_path),
            "raw_angles_png": str(raw_angles_png_path),
        },
        "notes": [
            "Pseudo ENU assumes the optical and radar origins are close enough for this offline test.",
            "The augmented EKF estimates effective radar biases, not physical installation parameters.",
            "A strong range correlation means a constant pitch bias is not enough.",
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def resolve_inputs(args: argparse.Namespace) -> List[Path]:
    if args.input:
        return [Path(item) for item in args.input]
    if args.latest:
        return latest_inputs(args.latest)
    return [Path(item) for item in INPUT_RAW_PAIR_FILES]


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline augmented radar/optical bearing fusion diagnostic")
    parser.add_argument("--input", action="append", help="raw_pairs_*.json path. Repeat to process multiple runs.")
    parser.add_argument("--latest", type=int, help="Process latest N raw_pairs_*.json files.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    inputs = resolve_inputs(args)
    if not inputs:
        print("[aug] no input files")
        return 1

    summaries = []
    for path in inputs:
        if not path.exists():
            print(f"[aug] missing input: {path}")
            return 1
        print(f"[aug] processing: {path}")
        summary = process_run(path, output_dir)
        summaries.append(summary)
        final_bias = summary["final_augmented_bias"]
        print(f"[aug] samples: {summary['sample_count']}")
        print(
            "[aug] radar-opt residual RMSE: "
            f"az={summary['radar_minus_optical_az_deg']['rmse']:.3f}deg, "
            f"pitch={summary['radar_minus_optical_pitch_deg']['rmse']:.3f}deg"
        )
        print(
            "[aug] EKF-opt residual RMSE: "
            f"az={summary['ekf_minus_optical_az_deg']['rmse']:.3f}deg, "
            f"pitch={summary['ekf_minus_optical_pitch_deg']['rmse']:.3f}deg"
        )
        print(
            "[aug] final bias: "
            f"b_range={final_bias['b_range_m']:.2f}m, "
            f"b_az={final_bias['b_az_deg']:.2f}deg, "
            f"b_pitch0={final_bias['b_pitch0_deg']:.2f}deg, "
            f"b_pitch_slope={final_bias['b_pitch_slope_deg_per_km']:.2f}deg/km"
        )
        print(f"[aug] suitable_for_augmented_bias_filter: {summary['suitable_for_augmented_bias_filter']}")
        print(f"[aug] csv: {summary['outputs']['csv']}")
        print(f"[aug] png: {summary['outputs']['diagnostic_png']}")

    if len(summaries) > 1:
        combined = {
            "inputs": [item["input"] for item in summaries],
            "sample_count": sum(item["sample_count"] for item in summaries),
            "runs": summaries,
        }
        combined_path = output_dir / "offline_augmented_bearing_filter_summary.json"
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"[aug] combined summary: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
