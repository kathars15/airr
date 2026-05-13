# -*- coding: utf-8 -*-
"""Small helpers for suppressing radar TRACK angle jitter."""

import math
import statistics
from collections import deque


def angle_delta_deg(a, b):
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def circular_mean_deg(values):
    if not values:
        return 0.0
    sin_sum = sum(math.sin(math.radians(v)) for v in values)
    cos_sum = sum(math.cos(math.radians(v)) for v in values)
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return float(values[-1]) % 360.0
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


class TrackAngleSmoother:
    """Per-track trailing-window smoother for range/azimuth/pitch measurements."""

    def __init__(self, window_size=5, max_age_sec=15.0):
        self.window_size = max(1, int(window_size))
        self.max_age_sec = float(max_age_sec)
        self._history = {}

    def reset(self):
        self._history.clear()

    def smooth(self, track_id, timestamp, range_m, azimuth_deg, pitch_deg):
        try:
            timestamp = float(timestamp)
            range_m = float(range_m)
            azimuth_deg = float(azimuth_deg) % 360.0
            pitch_deg = float(pitch_deg)
        except (TypeError, ValueError):
            return range_m, azimuth_deg, pitch_deg

        key = str(track_id)
        history = self._history.get(key)
        if history is None:
            history = deque(maxlen=self.window_size)
            self._history[key] = history
        elif history and timestamp - history[-1]["timestamp"] > self.max_age_sec:
            history.clear()

        history.append({
            "timestamp": timestamp,
            "range": range_m,
            "azimuth": azimuth_deg,
            "pitch": pitch_deg,
        })

        if len(history) == 1:
            return range_m, azimuth_deg, pitch_deg

        ranges = [item["range"] for item in history]
        azimuths = [item["azimuth"] for item in history]
        pitches = [item["pitch"] for item in history]

        return (
            float(statistics.median(ranges)),
            circular_mean_deg(azimuths),
            float(statistics.median(pitches)),
        )

    def smooth_track(self, track, timestamp):
        smoothed = dict(track)
        range_m, azimuth, pitch = self.smooth(
            track.get("display_id", track.get("track_id")),
            timestamp,
            track.get("range"),
            track.get("azimuth"),
            track.get("pitch"),
        )
        smoothed["raw_range"] = track.get("range")
        smoothed["raw_azimuth"] = track.get("azimuth")
        smoothed["raw_pitch"] = track.get("pitch")
        smoothed["range"] = range_m
        smoothed["azimuth"] = azimuth
        smoothed["pitch"] = pitch
        smoothed["angle_smoothed"] = True
        return smoothed


def smooth_rows_by_track_id(rows, window_size=5, max_age_sec=15.0, id_field="track_id"):
    smoother = TrackAngleSmoother(window_size=window_size, max_age_sec=max_age_sec)
    smoothed_rows = []
    for row in sorted(rows, key=lambda item: item.get("timestamp", 0.0)):
        item = dict(row)
        range_m, azimuth, pitch = smoother.smooth(
            item.get(id_field, ""),
            item.get("timestamp", 0.0),
            item.get("range"),
            item.get("azimuth"),
            item.get("pitch"),
        )
        item["raw_range"] = item.get("range")
        item["raw_azimuth"] = item.get("azimuth")
        item["raw_pitch"] = item.get("pitch")
        item["range"] = range_m
        item["azimuth"] = azimuth
        item["pitch"] = pitch
        item["angle_smoothed"] = True
        smoothed_rows.append(item)
    return smoothed_rows


class OpticalAngleSmoother:
    """Per-device trailing-window smoother for optical angle/range measurements.

    Only used during calibration; the tracking path receives raw optical data.
    """

    def __init__(self, window_size=3, max_age_sec=5.0):
        self.window_size = max(1, int(window_size))
        self.max_age_sec = float(max_age_sec)
        self._history = deque(maxlen=self.window_size)

    def reset(self):
        self._history.clear()

    def smooth(self, timestamp, azimuth_deg, pitch_deg, range_m=None):
        try:
            timestamp = float(timestamp)
            azimuth_deg = float(azimuth_deg) % 360.0
            pitch_deg = float(pitch_deg)
            range_m = float(range_m) if range_m is not None else None
        except (TypeError, ValueError):
            return azimuth_deg, pitch_deg, range_m

        entry = {
            "timestamp": timestamp,
            "azimuth": azimuth_deg,
            "pitch": pitch_deg,
        }
        if range_m is not None:
            entry["range"] = range_m

        if self._history and timestamp - self._history[-1]["timestamp"] > self.max_age_sec:
            self._history.clear()

        self._history.append(entry)

        if len(self._history) == 1:
            return azimuth_deg, pitch_deg, range_m

        azimuths = [item["azimuth"] for item in self._history]
        pitches = [item["pitch"] for item in self._history]
        ranges = [item["range"] for item in self._history if "range" in item]

        smoothed_az = circular_mean_deg(azimuths)
        smoothed_pitch = float(statistics.median(pitches))
        smoothed_range = float(statistics.median(ranges)) if ranges else range_m

        return smoothed_az, smoothed_pitch, smoothed_range
