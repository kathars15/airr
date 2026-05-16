# -*- coding: utf-8 -*-
r"""Replay one saved calibration session through the current online calibration pipeline.

Examples:
  python .\tools\calibration\replay_calibration_session.py
  python .\tools\calibration\replay_calibration_session.py --list
  python .\tools\calibration\replay_calibration_session.py --input .\calibration_data\cal_sessions\cal_session_20260514_101530_Radar-3.json
  python .\tools\calibration\replay_calibration_session.py --input session.json --dry-run
"""

import argparse
import json
import os
import sys
from copy import deepcopy

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.calibration import CALIBRATION_SESSION_DIR, DEFAULT_CALIBRATION_DATA_DIR, calibrator  # noqa: E402


def get_session_dir():
    session_dir = os.path.join(DEFAULT_CALIBRATION_DATA_DIR, CALIBRATION_SESSION_DIR)
    return session_dir


def list_session_files():
    session_dir = get_session_dir()
    if not os.path.isdir(session_dir):
        return []
    items = []
    for name in os.listdir(session_dir):
        path = os.path.join(session_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".json"):
            items.append(path)
    items.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return items


def load_session(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("session file is not a JSON object")
    source = data.get("source_measurements", {})
    if not isinstance(source, dict) or not isinstance(source.get("samples", []), list):
        raise ValueError("session file missing source_measurements.samples")
    return data


def normalize_for_compare(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"timestamp", "saved_at"}:
                continue
            result[key] = normalize_for_compare(item)
        return result
    if isinstance(value, list):
        return [normalize_for_compare(item) for item in value]
    return value


def first_diff(lhs, rhs, prefix="root"):
    if type(lhs) is not type(rhs):
        return f"{prefix}: type {type(lhs).__name__} != {type(rhs).__name__}"
    if isinstance(lhs, dict):
        lhs_keys = set(lhs.keys())
        rhs_keys = set(rhs.keys())
        if lhs_keys != rhs_keys:
            only_l = sorted(lhs_keys - rhs_keys)
            only_r = sorted(rhs_keys - lhs_keys)
            return f"{prefix}: key mismatch only_left={only_l} only_right={only_r}"
        for key in sorted(lhs.keys()):
            diff = first_diff(lhs[key], rhs[key], f"{prefix}.{key}")
            if diff:
                return diff
        return None
    if isinstance(lhs, list):
        if len(lhs) != len(rhs):
            return f"{prefix}: len {len(lhs)} != {len(rhs)}"
        for index, (left_item, right_item) in enumerate(zip(lhs, rhs)):
            diff = first_diff(left_item, right_item, f"{prefix}[{index}]")
            if diff:
                return diff
        return None
    if lhs != rhs:
        return f"{prefix}: {lhs!r} != {rhs!r}"
    return None


def build_current_summary():
    return {
        "calibration_result": deepcopy(calibrator._with_source_measurements(calibrator.calibration_result)),
        "position_offset": deepcopy(calibrator._with_source_measurements(calibrator.position_offset)),
        "segments": deepcopy(calibrator.segmented_6dof_params),
        "stability_report": deepcopy(calibrator.radar_stability_report),
    }


def main():
    parser = argparse.ArgumentParser(description="Replay one saved calibration session")
    parser.add_argument("--input", help="Path to one saved session JSON file")
    parser.add_argument("--list", action="store_true", help="List available session files and exit")
    parser.add_argument("--dry-run", action="store_true", help="Recompute but do not overwrite standard output files")
    args = parser.parse_args()

    session_files = list_session_files()

    if args.list:
        if not session_files:
            print("[replay] no calibration sessions found")
            return 0
        print("[replay] available calibration sessions:")
        for path in session_files:
            print(f"  {path}")
        return 0

    session_path = args.input
    if not session_path:
        if not session_files:
            print("[replay] no calibration sessions found")
            return 1
        session_path = session_files[0]

    session = load_session(session_path)
    source_measurements = session.get("source_measurements", {})
    stability_samples = session.get("radar_stability_samples", [])
    target_history = session.get("target_history", [])

    sample_count = int(source_measurements.get("sample_count", 0))
    print(f"[replay] session: {session_path}")
    print(f"[replay] target: {session.get('target_id')}")
    print(f"[replay] sample_count: {sample_count}")
    print(f"[replay] mode: {'dry-run' if args.dry_run else 'overwrite standard outputs'}")

    calibrator.load_source_measurements(
        source_measurements,
        target_history=target_history,
        stability_samples=stability_samples,
    )
    ok = calibrator.finalize_calibration_from_loaded_samples(save_outputs=not args.dry_run)
    if not ok:
        print("[replay] recompute failed")
        return 1

    print(calibrator.format_radar_stability_summary())

    expected_summary = session.get("result_summary")
    if isinstance(expected_summary, dict):
        current_summary = build_current_summary()
        expected_norm = normalize_for_compare(expected_summary)
        current_norm = normalize_for_compare(current_summary)
        diff = first_diff(expected_norm, current_norm)
        if diff is None:
            print("[replay] comparison: MATCH (ignoring timestamps/runtime-only metadata)")
        else:
            print(f"[replay] comparison: DIFF -> {diff}")

    if not args.dry_run:
        print(f"[replay] outputs written to: {DEFAULT_CALIBRATION_DATA_DIR}")
        print(f"[replay] calibration_params.json")
        print(f"[replay] position_offset.json (if position solver succeeds)")
        print(f"[replay] 6dof_params.json")
        print(f"[replay] radar_stability_report.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
