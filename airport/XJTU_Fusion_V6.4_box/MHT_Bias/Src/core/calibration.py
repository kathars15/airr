"""
雷达与光电校准模块 - 支持6DoF（位置+旋转）版本（已修复）
"""
from core.app_config import FAKE_DIS
import time
import json
import os
from collections import deque
import numpy as np

try:
    from core.app_config import CALIBRATION_MIN_RANGE, CALIBRATION_PAIR_TIME_WINDOW, SCRIPT_DIR
except ImportError:
    CALIBRATION_MIN_RANGE = 700.0
    CALIBRATION_PAIR_TIME_WINDOW = 0.15
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CALIBRATION_DATA_DIR = os.path.join(SCRIPT_DIR, "calibration_data")


class RadarOpticalCalibrator:
    def __init__(self, data_dir=DEFAULT_CALIBRATION_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.radar_measurements = []
        self.optical_measurements = []
        self.current_target_id = None
        self.calibration_mode = False

        self.radar_buffer = deque(maxlen=10)
        self.optical_buffer = deque(maxlen=10)

        # 角度偏移（备用）
        self.calibration_result = {
            'azimuth_offset': 0.0,
            'pitch_offset': 0.0,
            'timestamp': 0,
            'sample_count': 0
        }

        # ==================== 6DoF 参数 ====================
        self.RANGE_THRESHOLD = 1200.0

        self.far_params = {
            'dx': -385.7, 'dy': -74.2, 'dz': 19.8,
            'az_rotation': -15.19,
            'pitch_rotation': -1.87,
            'mean_error_deg': 0.87,
            'timestamp': 0,
            'sample_count': 23
        }

        self.near_params = {
            'dx': -451.3, 'dy': -98.6, 'dz': 35.4,
            'az_rotation': -15.27,
            'pitch_rotation': -2.84,
            'mean_error_deg': 1.68,
            'timestamp': 0,
            'sample_count': 6
        }
        # =================================================

        self.load_calibration()
        self.load_6dof_params()

    # ==================== 6DoF 核心 ====================
    def apply_6dof_calibration(self, radar_azimuth, radar_pitch, radar_range):
        if radar_range is None or radar_range <= 0:
            return self.apply_angle_offset(radar_azimuth, radar_pitch)

        params = self.near_params if radar_range < self.RANGE_THRESHOLD else self.far_params

        if params.get('mean_error_deg', 0) == 0:
            return self.apply_angle_offset(radar_azimuth, radar_pitch)

        # 雷达极坐标转3D
        az_rad = np.radians(radar_azimuth)
        pitch_rad = np.radians(radar_pitch)
        x = radar_range * np.cos(pitch_rad) * np.sin(az_rad)
        y = radar_range * np.cos(pitch_rad) * np.cos(az_rad)
        z = radar_range * np.sin(pitch_rad)

        # 应用旋转
        az_rot = np.radians(params['az_rotation'])
        pitch_rot = np.radians(params['pitch_rotation'])

        x1 = x * np.cos(az_rot) - y * np.sin(az_rot)
        y1 = x * np.sin(az_rot) + y * np.cos(az_rot)
        z1 = z

        x2 = x1
        y2 = y1 * np.cos(pitch_rot) - z1 * np.sin(pitch_rot)
        z2 = y1 * np.sin(pitch_rot) + z1 * np.cos(pitch_rot)

        # 应用位置偏移
        rel_x = x2 - params['dx']
        rel_y = y2 - params['dy']
        rel_z = z2 - params['dz']

        # 转回光电角度
        opt_range = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        opt_az = np.degrees(np.arctan2(rel_x, rel_y))
        if opt_az < 0:
            opt_az += 360
        opt_pitch = np.degrees(np.arcsin(rel_z / opt_range)) if opt_range > 0 else 0
        opt_range = radar_range
        return opt_az, opt_pitch, opt_range - FAKE_DIS

    def apply_calibration(self, radar_azimuth, radar_pitch, radar_range=None):
        return self.apply_6dof_calibration(radar_azimuth, radar_pitch, radar_range)

    def apply_angle_offset(self, radar_azimuth, radar_pitch):
        calibrated_az = (radar_azimuth + self.calibration_result['azimuth_offset']) % 360
        calibrated_pitch = radar_pitch + self.calibration_result['pitch_offset']
        return calibrated_az, calibrated_pitch, None

    # ==================== 参数保存/加载 ====================
    def load_calibration(self):
        path = os.path.join(self.data_dir, 'calibration_params.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.calibration_result.update(json.load(f))

    def save_calibration(self):
        path = os.path.join(self.data_dir, 'calibration_params.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.calibration_result, f, indent=2)

    def load_6dof_params(self):
        path = os.path.join(self.data_dir, '6dof_params.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.far_params.update(data.get('far', {}))
                self.near_params.update(data.get('near', {}))
            print("[校准] 已加载6DoF参数")

    def save_6dof_params(self):
        path = os.path.join(self.data_dir, '6dof_params.json')
        data = {'far': self.far_params, 'near': self.near_params, 'timestamp': time.time()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("[校准] 6DoF参数已保存")

    # ==================== 其他必要方法 ====================
    def start_calibration(self, target_id):
        self.calibration_mode = True
        self.current_target_id = target_id
        self.radar_measurements = []
        self.optical_measurements = []
        self.radar_buffer.clear()
        self.optical_buffer.clear()
        print(f"[校准] 开始校准，目标: {target_id}")
        return True

    def stop_calibration(self):
        self.calibration_mode = False
        if len(self.radar_measurements) < 5:
            print(f"[校准] 数据不足，当前{len(self.radar_measurements)}组")
            return False
        print("[校准] 校准完成")
        self.save_6dof_params()
        return True

    def get_status(self):
        return {
            'mode': self.calibration_mode,
            'target_id': self.current_target_id,
            'has_6dof': self.far_params.get('mean_error_deg', 0) > 0
        }


# 全局实例
calibrator = RadarOpticalCalibrator()