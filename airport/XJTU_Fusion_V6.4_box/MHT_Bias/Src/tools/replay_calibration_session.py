#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry for calibration session replay.

New location:
    tools/calibration/replay_calibration_session.py
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tools.calibration.replay_calibration_session import main


if __name__ == "__main__":
    raise SystemExit(main())
