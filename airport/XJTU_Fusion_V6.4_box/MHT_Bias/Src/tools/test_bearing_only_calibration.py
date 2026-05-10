# -*- coding: utf-8 -*-
"""Synthetic test for bearing-only radar/optical position calibration.

Radar supplies target range/azimuth/pitch. Optical supplies only azimuth/pitch.
The solver should recover the optical position offset relative to the radar.
"""

import math
import os
import sys
import tempfile

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.calibration import RadarOpticalCalibrator  # noqa: E402


def vector_to_polar(vec):
    x, y, z = vec
    rng = float(np.linalg.norm(vec))
    az = math.degrees(math.atan2(x, y)) % 360.0
    pitch = math.degrees(math.asin(z / rng)) if rng > 0 else 0.0
    return az, pitch, rng


def polar_to_vector(az, pitch, rng):
    az_rad = math.radians(az)
    pitch_rad = math.radians(pitch)
    return np.array([
        rng * math.cos(pitch_rad) * math.sin(az_rad),
        rng * math.cos(pitch_rad) * math.cos(az_rad),
        rng * math.sin(pitch_rad),
    ])


def run_case(noise_deg=0.0, seed=7):
    rng = np.random.default_rng(seed)
    true_offset = np.array([18.0, -7.5, 3.2])
    calibrator = RadarOpticalCalibrator(data_dir=tempfile.mkdtemp(prefix="bearing_cal_"))

    target_specs = [
        (20, 2, 420),
        (55, 5, 650),
        (95, -1, 800),
        (140, 8, 520),
        (185, 4, 900),
        (230, -3, 760),
        (280, 6, 610),
        (330, 1, 700),
    ]

    for i, (radar_az, radar_pitch, radar_range) in enumerate(target_specs):
        target_pos = polar_to_vector(radar_az, radar_pitch, radar_range)
        opt_vec = target_pos - true_offset
        opt_az, opt_pitch, _ = vector_to_polar(opt_vec)
        opt_az = (opt_az + rng.normal(0.0, noise_deg)) % 360.0
        opt_pitch = opt_pitch + rng.normal(0.0, noise_deg)

        calibrator.radar_measurements.append({
            "timestamp": float(i),
            "azimuth": radar_az,
            "pitch": radar_pitch,
            "range": radar_range,
        })
        calibrator.optical_measurements.append({
            "timestamp": float(i),
            "azimuth": opt_az,
            "pitch": opt_pitch,
            "range": 0,
        })

    ok = calibrator.calculate_position_offset()
    estimated = np.array([
        calibrator.position_offset["dx"],
        calibrator.position_offset["dy"],
        calibrator.position_offset["dz"],
    ])
    err = estimated - true_offset

    print(f"noise_deg={noise_deg}")
    print(f"true_offset      dx={true_offset[0]:8.3f} dy={true_offset[1]:8.3f} dz={true_offset[2]:8.3f}")
    print(f"estimated_offset dx={estimated[0]:8.3f} dy={estimated[1]:8.3f} dz={estimated[2]:8.3f}")
    print(f"error            dx={err[0]:8.3f} dy={err[1]:8.3f} dz={err[2]:8.3f}")
    print(f"norm_error={np.linalg.norm(err):.3f}m")
    print(f"ok={ok}")
    return ok, float(np.linalg.norm(err))


def main():
    ok0, err0 = run_case(noise_deg=0.0)
    print("-" * 72)
    ok1, err1 = run_case(noise_deg=0.05)

    if not ok0 or err0 > 1e-6:
        raise SystemExit("perfect-data test failed")
    if not ok1 or err1 > 3.0:
        raise SystemExit("noisy-data test failed")
    print("bearing-only calibration synthetic test passed")


if __name__ == "__main__":
    main()
