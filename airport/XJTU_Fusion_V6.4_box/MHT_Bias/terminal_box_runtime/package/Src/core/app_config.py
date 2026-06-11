# -*- coding: utf-8 -*-

import os


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in ('0', 'false', 'no', 'off')


MAX_RANGE = 2500
FAKE_DIS = 200

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MHT_BIAS_PATH = os.path.join(PROJECT_ROOT, 'MHT_Bias')

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
FLIGHT_RUNS_DIR = os.path.join(SCRIPT_DIR, 'flight_data_runs')
os.makedirs(FLIGHT_RUNS_DIR, exist_ok=True)
CALIBRATION_DATA_DIR = os.path.join(SCRIPT_DIR, 'calibration_data')
os.makedirs(CALIBRATION_DATA_DIR, exist_ok=True)
PARSE_LOG_DIR = os.path.join(CALIBRATION_DATA_DIR, 'parse_logs')
os.makedirs(PARSE_LOG_DIR, exist_ok=True)

DEBUG_POINT_MHT = True

RAW_TRACKS_FILE = os.path.join(DATA_DIR, 'raw_tracks.csv')
CALIBRATION_FILE = os.path.join(DATA_DIR, 'radar_calibration_data.csv')
OPTICAL_MEASUREMENTS_FILE = os.path.join(DATA_DIR, 'optical_measurements.csv')
OPTICAL_STATUS_FILE = os.path.join(DATA_DIR, 'optical_status.json')
CV_DETECTION_RESULTS_FILE = os.path.join(DATA_DIR, 'cv_detection_result.json')
TRACK_RESULTS_FILE = os.path.join(DATA_DIR, 'track_results.json')
TRACK_LOG_FILE = os.path.join(DATA_DIR, 'track_log.txt')
TRUE_POSITION_LOG_DIR = os.path.join(FLIGHT_RUNS_DIR, 'true_position_logs')
os.makedirs(TRUE_POSITION_LOG_DIR, exist_ok=True)
TERMINAL_RESULT_LOG_DIR = os.path.join(FLIGHT_RUNS_DIR, 'terminal_box_logs')
os.makedirs(TERMINAL_RESULT_LOG_DIR, exist_ok=True)
POINT_RECORDS_FILE = os.path.join(DATA_DIR, 'point_records.csv')
POINT_TRACK_RESULTS_FILE = os.path.join(DATA_DIR, 'point_track_results.json')
POINT_TRACK_LOG_FILE = os.path.join(DATA_DIR, 'point_track_log.txt')
POINT_VS_RAW_COMPARE_FILE = os.path.join(DATA_DIR, 'point_vs_raw_track_compare.csv')
RADAR_PARSE_LOG_FILE = os.path.join(PARSE_LOG_DIR, 'radar_parse_debug.log')

RADAR_IP = "192.168.0.99"
HOST_IP = "127.0.0.1"  # 推荐，监听所有本机地址


RADAR_PORT = 8080
ENABLE_MANAGED_UDP_FANOUT = _env_bool('AIRR_ENABLE_MANAGED_UDP_FANOUT', True)
RADAR_FANOUT_INGRESS_PORT = int(os.environ.get('AIRR_RADAR_FANOUT_INGRESS_PORT', '9000'))
RADAR_FANOUT_MAIN_PORT = int(os.environ.get('AIRR_RADAR_FANOUT_MAIN_PORT', '29000'))
OPTICAL_FANOUT_INGRESS_PORT = int(os.environ.get('AIRR_OPTICAL_FANOUT_INGRESS_PORT', '9966'))
OPTICAL_FANOUT_MAIN_PORT = int(os.environ.get('AIRR_OPTICAL_FANOUT_MAIN_PORT', '29966'))
HOST_PORT = int(os.environ.get(
    'AIRR_RADAR_LISTEN_PORT',
    str(RADAR_FANOUT_MAIN_PORT if ENABLE_MANAGED_UDP_FANOUT else 9000),
))

OPTICAL_IP = "192.168.0.98"
OPTICAL_LOCAL_IP = "192.168.0.9"

OPTICAL_DEVICE_PORT = int(os.environ.get('AIRR_OPTICAL_DEVICE_PORT', '9966'))
OPTICAL_PORT = int(os.environ.get(
    'AIRR_OPTICAL_LOCAL_PORT',
    str(OPTICAL_FANOUT_MAIN_PORT if ENABLE_MANAGED_UDP_FANOUT else OPTICAL_DEVICE_PORT),
))
OPTICAL_REPORT_IP = os.environ.get('AIRR_OPTICAL_REPORT_IP', OPTICAL_LOCAL_IP)
OPTICAL_REPORT_PORT = int(os.environ.get(
    'AIRR_OPTICAL_REPORT_PORT',
    str(OPTICAL_FANOUT_INGRESS_PORT if ENABLE_MANAGED_UDP_FANOUT else OPTICAL_PORT),
))
OPTICAL_AI_TEMPLATE = 1

ENABLE_MANAGED_CV_DETECTION = _env_bool('AIRR_ENABLE_MANAGED_CV_DETECTION', True)
CV_DETECTION_SCRIPT = os.environ.get(
    'AIRR_CV_DETECTION_SCRIPT',
    os.path.join(MHT_BIAS_PATH, 'CV', 'code_image', 'rtsp_detect_show.py'),
)
CV_MODEL_DIR = os.environ.get('AIRR_CV_MODEL_DIR', 'train10')
CV_MODEL_WEIGHTS_FILE = os.environ.get(
    'AIRR_CV_MODEL_WEIGHTS_FILE',
    os.path.join(MHT_BIAS_PATH, 'CV', CV_MODEL_DIR, 'weights', 'best.pt'),
)

# Optical angle/target reports are configured at 100 ms. Keep calibration pairing
# tight enough to avoid matching the wrong radar scan moment.
CALIBRATION_PAIR_TIME_WINDOW = 0.15

# Radar angles are unreliable at close range in the current field setup.
# Online and offline calibration only use paired samples at or beyond this range.
CALIBRATION_MIN_RANGE = 200.0

# Task-driven position-bias calibration assumes the two sensors are already
# mechanically leveled, so only a small relative translation is physically
# plausible.
CALIBRATION_MAX_OFFSET_X_M = 20000.0
CALIBRATION_MAX_OFFSET_Y_M = 20000.0
CALIBRATION_MAX_OFFSET_Z_M = 20000.0

# Position-bias solutions must stay within these quality limits to be marked
# usable for online pointing.
CALIBRATION_MAX_MEAN_ERROR_DEG = 30
CALIBRATION_MAX_MAX_ERROR_DEG = 80
CALIBRATION_MAX_MEAN_AZ_ERROR_DEG = 25
CALIBRATION_MAX_MEAN_PITCH_ERROR_DEG = 25
CALIBRATION_MAX_GEOMETRY_COND = 1.0e5

# Online estimated target ENU from MHT range + optical bearing.
ENABLE_TRUE_POSITION_OUTPUT = True
TRUE_POSITION_CONFIRM_FRAMES = 3
TRUE_POSITION_MAX_OPTICAL_AGE_SEC = 0.5
TRUE_POSITION_PRINT_INTERVAL_SEC = 0.5
TRUE_POSITION_USE_MHT_RANGE = True
TRUE_POSITION_REQUIRE_CURRENT_GUIDED_TARGET = True
TRUE_POSITION_MAX_RADAR_AGE_SEC = 2.0

# Terminal-box mode: no optical guidance, no interactive console.
ENABLE_TERMINAL_BOX_MODE = _env_bool('AIRR_ENABLE_TERMINAL_BOX_MODE', False)
ENABLE_OPTICAL_GUIDANCE = _env_bool('AIRR_ENABLE_OPTICAL_GUIDANCE', False)
ENABLE_INTERACTIVE_CONSOLE = _env_bool('AIRR_ENABLE_INTERACTIVE_CONSOLE', False)
ENABLE_LOCAL_CV_WHEN_GPU_AVAILABLE = _env_bool('AIRR_ENABLE_LOCAL_CV_WHEN_GPU_AVAILABLE', True)
ENABLE_LOCAL_CV_WHEN_CPU_ONLY = _env_bool('AIRR_ENABLE_LOCAL_CV_WHEN_CPU_ONLY', False)
TERMINAL_PRINT_INTERVAL_SEC = float(os.environ.get('AIRR_TERMINAL_PRINT_INTERVAL_SEC', '0.5'))
TERMINAL_TARGET_MATCH_MAX_AZ_DIFF_DEG = float(os.environ.get('AIRR_TERMINAL_TARGET_MATCH_MAX_AZ_DIFF_DEG', '5.0'))
TERMINAL_TARGET_MATCH_MAX_RADAR_AGE_SEC = float(os.environ.get('AIRR_TERMINAL_TARGET_MATCH_MAX_RADAR_AGE_SEC', '2.0'))

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
