# -*- coding: utf-8 -*-
"""Estimate target ENU position from radar/MHT range and optical bearing."""

import math
import time


def polar_to_enu(range_m, azimuth_deg, pitch_deg):
    az = math.radians(float(azimuth_deg))
    pitch = math.radians(float(pitch_deg))
    r = float(range_m)
    return (
        r * math.cos(pitch) * math.sin(az),
        r * math.cos(pitch) * math.cos(az),
        r * math.sin(pitch),
    )


class TruePositionEstimator:
    """Small online estimator for the current optically tracked target.

    The output is an estimated ENU position, not an externally verified truth.
    It assumes the optical direction and radar/MHT range can be combined in the
    same ENU convention used elsewhere in the project.
    """

    def __init__(
        self,
        confirm_frames=3,
        max_optical_age_sec=0.5,
        print_interval_sec=0.5,
    ):
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_optical_age_sec = float(max_optical_age_sec)
        self.print_interval_sec = float(print_interval_sec)
        self._tracking_count = 0
        self._tracking_track_id = None
        self._last_state_by_track = {}
        self._last_print_time = 0.0

    def reset(self):
        self._tracking_count = 0
        self._tracking_track_id = None
        self._last_state_by_track.clear()
        self._last_print_time = 0.0

    def estimate(
        self,
        track_id,
        raw_display_id,
        range_m,
        optical_state,
        fusion_time=None,
        now=None,
    ):
        now = float(now if now is not None else time.time())
        fusion_time = float(fusion_time if fusion_time is not None else now)
        track_id = str(track_id)
        optical_state = dict(optical_state or {})

        status = optical_state.get("current_status")
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None

        azimuth = optical_state.get("latest_azimuth")
        pitch = optical_state.get("latest_pitch")
        angle_time = optical_state.get("latest_angle_host_time")

        if status != 2:
            self._tracking_count = 0
            self._tracking_track_id = None
            return self._skip("optical_not_tracking")

        if azimuth is None or pitch is None:
            self._tracking_count = 0
            self._tracking_track_id = None
            return self._skip("missing_optical_angle")

        try:
            azimuth = float(azimuth)
            pitch = float(pitch)
            range_m = float(range_m)
        except (TypeError, ValueError):
            return self._skip("invalid_input")

        if range_m <= 0.0:
            return self._skip("invalid_range")

        if angle_time is None:
            return self._skip("missing_optical_angle_time")

        try:
            angle_time = float(angle_time)
        except (TypeError, ValueError):
            return self._skip("invalid_optical_angle_time")

        age = now - angle_time
        if age < 0.0:
            age = 0.0
        if age > self.max_optical_age_sec:
            self._tracking_count = 0
            self._tracking_track_id = None
            return self._skip("optical_angle_stale", optical_angle_age_sec=age)

        if self._tracking_track_id != track_id:
            self._tracking_track_id = track_id
            self._tracking_count = 0
        self._tracking_count += 1
        if self._tracking_count < self.confirm_frames:
            return self._skip(
                "tracking_not_confirmed",
                optical_angle_age_sec=age,
                optical_tracking_frames=self._tracking_count,
            )

        enu = polar_to_enu(range_m, azimuth, pitch)
        previous = self._last_state_by_track.get(track_id)
        velocity = None
        speed = None
        if previous is not None:
            dt = fusion_time - previous["fusion_time"]
            if dt > 1e-3:
                velocity = tuple((enu[i] - previous["enu"][i]) / dt for i in range(3))
                speed = math.sqrt(sum(v * v for v in velocity))

        self._last_state_by_track[track_id] = {
            "enu": enu,
            "fusion_time": fusion_time,
        }

        should_print = now - self._last_print_time >= self.print_interval_sec
        if should_print:
            self._last_print_time = now

        return {
            "used": True,
            "skip_reason": None,
            "track_id": track_id,
            "raw_display_id": raw_display_id,
            "estimated_true_enu": {
                "east_m": enu[0],
                "north_m": enu[1],
                "up_m": enu[2],
            },
            "estimated_true_velocity_enu": (
                {
                    "east_mps": velocity[0],
                    "north_mps": velocity[1],
                    "up_mps": velocity[2],
                }
                if velocity is not None
                else None
            ),
            "estimated_true_speed_mps": speed,
            "range_m": range_m,
            "optical_azimuth_deg": azimuth,
            "optical_pitch_deg": pitch,
            "optical_angle_age_sec": age,
            "optical_tracking_frames": self._tracking_count,
            "true_position_source": "mht_range_plus_optical_bearing",
            "should_print": should_print,
        }

    @staticmethod
    def _skip(reason, **extra):
        result = {
            "used": False,
            "skip_reason": reason,
            "true_position_source": None,
            "should_print": False,
        }
        result.update(extra)
        return result
