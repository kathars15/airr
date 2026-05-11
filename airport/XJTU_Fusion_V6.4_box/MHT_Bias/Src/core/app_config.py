# -*- coding: utf-8 -*-

import os

MAX_RANGE = 2500
FAKE_DIS = 70

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MHT_BIAS_PATH = os.path.join(PROJECT_ROOT, 'MHT_Bias')

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

RAW_TRACKS_FILE = os.path.join(DATA_DIR, 'raw_tracks.csv')
OPTICAL_MEASUREMENTS_FILE = os.path.join(DATA_DIR, 'optical_measurements.csv')
TRACK_RESULTS_FILE = os.path.join(DATA_DIR, 'track_results.json')
TRACK_LOG_FILE = os.path.join(DATA_DIR, 'track_log.txt')
CALIBRATION_FILE = os.path.join(DATA_DIR, 'radar_calibration_data.csv')

RADAR_IP = "192.168.0.99"
HOST_IP = "127.0.0.1"  # 推荐，监听所有本机地址


RADAR_PORT = 8080
HOST_PORT = 9000

OPTICAL_IP = "192.168.0.98"
OPTICAL_LOCAL_IP = "192.168.0.9"

OPTICAL_PORT = 9966
OPTICAL_AI_TEMPLATE = 1

# Optical angle/target reports are configured at 100 ms. Keep calibration pairing
# tight enough to avoid matching the wrong radar scan moment.
CALIBRATION_PAIR_TIME_WINDOW = 0.15

# Radar angles are unreliable at close range in the current field setup.
# Online and offline calibration only use paired samples at or beyond this range.
CALIBRATION_MIN_RANGE = 800.0

FRAME_HEAD_STATUS = 0xA0A0A7A7
FRAME_TAIL_STATUS = 0x7A7A0A0A
FRAME_HEAD_POINT = 0xA1A1A8A8
FRAME_TAIL_POINT = 0x8A8A1A1A
FRAME_HEAD_TRACK = 0xA3A3AAAA
FRAME_TAIL_TRACK = 0xAAAA3A3A
FRAME_HEAD_END = 0xA8A8AFAF
FRAME_TAIL_END = 0xFAFA8A8A
FRAME_HEAD_CONTROL = 0xAA55
FRAME_TAIL_CONTROL = 0x55AA
