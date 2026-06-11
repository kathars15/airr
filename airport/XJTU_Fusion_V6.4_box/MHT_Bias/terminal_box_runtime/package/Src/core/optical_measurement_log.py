# -*- coding: utf-8 -*-

import csv
import os
import threading

from core.app_config import OPTICAL_MEASUREMENTS_FILE

_log_lock = threading.Lock()


def append_optical_measurement(timestamp, azimuth, pitch, status, range_m):
    """Persist optical angle/status reports for offline radar/optical matching."""
    try:
        with _log_lock:
            write_header = not os.path.exists(OPTICAL_MEASUREMENTS_FILE)
            with open(OPTICAL_MEASUREMENTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['timestamp', 'azimuth', 'pitch', 'status', 'range'])
                writer.writerow([timestamp, azimuth, pitch, status, range_m])
    except Exception:
        # Keep UDP receive path non-blocking and quiet if logging fails once.
        pass
