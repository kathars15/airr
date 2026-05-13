# -*- coding: utf-8 -*-

import time

from core.calibration import calibrator
from core.console_utils import safe_print


def start_calibration(track_id):
    """Start calibration."""
    if calibrator.start_calibration(track_id):
        safe_print(f"[calibration] start target {track_id}")
        safe_print("[calibration] keep target stable, radar scan period is 4s, recommend 10-20 samples")
        safe_print("[calibration] input 'done' to stop and calculate parameters")
        return True
    return False


def stop_calibration():
    """Stop calibration."""
    return calibrator.stop_calibration()


def get_calibration_status():
    """Get calibration status."""
    status = calibrator.get_status()
    if status['mode']:
        safe_print(
            f"[calibration] active: target={status['target_id']}, paired={status['radar_samples']}, "
            f"stability_samples={status.get('radar_stability_samples', 0)}"
        )
        history = status.get('target_history', [])
        if history:
            safe_print(f"[calibration] target history: {', '.join(history)}")
    else:
        safe_print(f"[calibration] idle, has_params: {'yes' if status['has_calibration'] else 'no'}")

    stable_ranges = status.get('radar_stable_ranges', [])
    if stable_ranges:
        text = ", ".join(
            f"{item['range_min_m']:.0f}-{item['range_max_m']:.0f}m"
            f"(n={item['sample_count']}, std<={item['pitch_std_max_deg']:.2f}deg)"
            for item in stable_ranges
        )
        safe_print(f"[stability] pitch-stable ranges: {text}")
    else:
        safe_print("[stability] no pitch-stable range detected yet")
    return status


def show_calibration_result():
    """Show calibration result."""
    result = calibrator.calibration_result
    position = calibrator.position_offset
    segmented = getattr(calibrator, 'segmented_6dof_params', [])

    if result['sample_count'] > 0:
        safe_print("\n" + "=" * 50)
        safe_print("Current calibration params:")
        safe_print(f"  azimuth offset: {result['azimuth_offset']:.2f}deg")
        safe_print(f"  pitch offset: {result['pitch_offset']:.2f}deg")
        safe_print(f"  sample count: {result['sample_count']}")
        if result.get('method'):
            safe_print(f"  method: {result['method']}")
        safe_print(f"  time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))}")

        if position.get('sample_count', 0) > 0:
            safe_print("")
            safe_print("Position offset params:")
            safe_print(f"  dx: {position['dx']:.2f}m")
            safe_print(f"  dy: {position['dy']:.2f}m")
            safe_print(f"  dz: {position['dz']:.2f}m")
            safe_print(f"  enabled: {'yes' if position.get('use_position', False) else 'no'}")
            safe_print(f"  method: {position.get('method', 'unknown')}")
            if 'mean_error_deg' in position:
                safe_print(
                    f"  reprojection error: mean={position['mean_error_deg']:.3f}deg, "
                    f"max={position.get('max_error_deg', 0.0):.3f}deg"
                )

        if segmented:
            safe_print("")
            safe_print("Segmented 6DoF params:")
            for item in segmented:
                safe_print(
                    f"  {item.get('range_label', 'unknown')}: "
                    f"dx={item.get('dx', 0.0):.2f}m, dy={item.get('dy', 0.0):.2f}m, "
                    f"dz={item.get('dz', 0.0):.2f}m, n={item.get('sample_count', 0)}, "
                    f"mean={item.get('mean_error_deg', 0.0):.3f}deg"
                )
        safe_print("=" * 50)
    else:
        safe_print("No calibration params yet")

    safe_print(calibrator.format_radar_stability_summary())


def clear_calibration():
    """Clear calibration params."""
    calibrator.clear_calibration()
