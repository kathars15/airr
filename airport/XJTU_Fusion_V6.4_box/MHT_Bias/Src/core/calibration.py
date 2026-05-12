# calibration.py
"""
雷达与光电校准模块。

在线校准仍保留原来的采集、配对、保存源数据能力；发送光电时优先使用
6DoF（位置+旋转）参数，缺少 6DoF 参数时回退到位置偏移/角度偏移。
"""
from core.app_config import FAKE_DIS
import json
import os
import time
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

        self.calibration_result = {
            'azimuth_offset': 0.0,
            'pitch_offset': 0.0,
            'azimuth_scale': 1.0,
            'pitch_scale': 1.0,
            'timestamp': 0,
            'sample_count': 0,
        }

        self.position_offset = {
            'dx': 0.0,
            'dy': 0.0,
            'dz': 0.0,
            'timestamp': 0,
            'sample_count': 0,
            'use_position': False,
        }

        self.RANGE_THRESHOLD = 1200.0
        self.far_params = {
            'dx': -385.7, 'dy': -74.2, 'dz': 19.8,
            'az_rotation': -15.19,
            'pitch_rotation': -1.87,
            'mean_error_deg': 0.87,
            'timestamp': 0,
            'sample_count': 23,
        }
        self.near_params = {
            'dx': -451.3, 'dy': -98.6, 'dz': 35.4,
            'az_rotation': -15.27,
            'pitch_rotation': -2.84,
            'mean_error_deg': 1.68,
            'timestamp': 0,
            'sample_count': 6,
        }

        self.load_calibration()
        self.load_position_offset()
        self.load_6dof_params()

    def start_calibration(self, target_id):
        """开始校准。"""
        self.calibration_mode = True
        self.current_target_id = target_id
        self.radar_measurements = []
        self.optical_measurements = []
        self.radar_buffer.clear()
        self.optical_buffer.clear()
        print(f"[校准] 开始校准，目标: {target_id}")
        print(f"[校准] 雷达扫描周期: 4秒，建议采集10-20组数据")
        print(f"[校准] 距离过滤: 仅采集雷达距离 >= {CALIBRATION_MIN_RANGE:.0f}m 的配对数据")
        return True

    def stop_calibration(self):
        """停止校准并计算/保存参数。"""
        self.calibration_mode = False

        if len(self.radar_measurements) < 5:
            print(f"[校准] 数据不足，需要至少5组数据，当前有{len(self.radar_measurements)}组")
            return False

        print("[校准] 使用 cal_offset 位置偏移算法计算参数...")
        if self.calculate_position_offset():
            self.save_calibration()
            self.save_position_offset()
            self.save_6dof_params()
            print("[校准] cal_offset 位置偏移校准完成！")
            print(
                f"[校准] 光电相对于雷达: 东偏移 {self.position_offset['dx']:.2f}m, "
                f"北偏移 {self.position_offset['dy']:.2f}m, 高偏移 {self.position_offset['dz']:.2f}m"
            )
            print(f"[校准] 样本数量: {self.position_offset['sample_count']}")
            mean_error = self.position_offset.get('mean_error_deg')
            max_error = self.position_offset.get('max_error_deg')
            if mean_error is not None and max_error is not None:
                print(f"[校准] 重投影角度误差: 均值 {mean_error:.3f}°, 最大 {max_error:.3f}°")
            return True

        print("[校准] cal_offset 位置偏移校准失败，回退到角度偏移校准...")
        self._calculate_calibration()
        self.save_calibration()
        self.save_6dof_params()
        print("[校准] 角度偏移校准完成！")
        print(f"[校准] 方位角偏移: {self.calibration_result['azimuth_offset']:.2f}°")
        print(f"[校准] 俯仰角偏移: {self.calibration_result['pitch_offset']:.2f}°")
        print(f"[校准] 样本数量: {self.calibration_result['sample_count']}")
        return True

    def add_radar_measurement(self, track_id, azimuth, pitch, range_m, timestamp=None):
        """添加雷达测量数据。"""
        if not self.calibration_mode or track_id != self.current_target_id:
            return

        if timestamp is None:
            timestamp = time.time()

        self.radar_buffer.append({
            'timestamp': timestamp,
            'track_id': track_id,
            'azimuth': azimuth,
            'pitch': pitch,
            'range': range_m,
        })
        self._try_pair_measurement()

    def add_optical_measurement(self, azimuth, pitch, timestamp=None, optical_status=None, opt_range=None):
        """添加光电实际角度和距离。"""
        if not self.calibration_mode:
            return

        if optical_status != 2:
            return

        if timestamp is None:
            timestamp = time.time()

        self.optical_buffer.append({
            'timestamp': timestamp,
            'azimuth': azimuth,
            'pitch': pitch,
            'range': opt_range if opt_range else 0,
            'status': optical_status,
        })
        self._try_pair_measurement()

    def _try_pair_measurement(self):
        """尝试配对雷达和光电测量数据。"""
        if len(self.radar_buffer) == 0 or len(self.optical_buffer) == 0:
            return

        best_pair = None
        best_diff = float('inf')

        for r in list(self.radar_buffer):
            for o in list(self.optical_buffer):
                time_diff = abs(r['timestamp'] - o['timestamp'])
                radar_range = r.get('range', 0) or 0
                if radar_range < CALIBRATION_MIN_RANGE:
                    continue
                if time_diff <= CALIBRATION_PAIR_TIME_WINDOW and time_diff < best_diff:
                    best_pair = (r, o)
                    best_diff = time_diff

        if best_pair:
            r, o = best_pair
            self.radar_measurements.append(r)
            self.optical_measurements.append(o)
            print(
                f"[校准] 配对成功: 雷达({r['azimuth']:.1f}°, {r['pitch']:.1f}°, {r['range']:.0f}m) -> "
                f"光电({o['azimuth']:.1f}°, {o['pitch']:.1f}°, {o['range']:.0f}m), 时间差={best_diff:.2f}s"
            )
            self.radar_buffer.clear()
            self.optical_buffer.clear()

    def _calculate_calibration(self):
        """计算角度偏移校准参数。"""
        try:
            from tools.cal_offset import summarize_angle_offsets
        except ImportError:
            summarize_angle_offsets = None

        if summarize_angle_offsets:
            measurements = self.to_cal_offset_measurements()
            stats = summarize_angle_offsets(measurements)
            self.calibration_result = {
                'azimuth_offset': float(stats['azimuth_offset_median']),
                'pitch_offset': float(stats['pitch_offset_median']),
                'azimuth_scale': 1.0,
                'pitch_scale': 1.0,
                'timestamp': time.time(),
                'sample_count': len(measurements),
                'method': 'cal_offset_angle_median',
            }
            return

        n = len(self.radar_measurements)
        radar_az = np.array([m['azimuth'] for m in self.radar_measurements], dtype=float)
        radar_pitch = np.array([m['pitch'] for m in self.radar_measurements], dtype=float)
        opt_az = np.array([m['azimuth'] for m in self.optical_measurements], dtype=float)
        opt_pitch = np.array([m['pitch'] for m in self.optical_measurements], dtype=float)
        az_diff = (opt_az - radar_az + 180.0) % 360.0 - 180.0
        pitch_diff = opt_pitch - radar_pitch

        self.calibration_result = {
            'azimuth_offset': float(np.median(az_diff)),
            'pitch_offset': float(np.median(pitch_diff)),
            'azimuth_scale': 1.0,
            'pitch_scale': 1.0,
            'timestamp': time.time(),
            'sample_count': n,
            'method': 'angle_median',
        }

    def calculate_position_offset(self):
        """使用 tools/cal_offset.py 中的离线同款算法计算位置偏移。"""
        measurements = self.to_cal_offset_measurements()
        if len(measurements) < 5:
            print(f"[校准] cal_offset 位置偏移需要至少5组有效配对数据，当前{len(measurements)}组")
            return False

        try:
            from tools.cal_offset import calculate_calibration_from_measurements
        except ImportError as exc:
            print(f"[校准] 无法导入 tools.cal_offset: {exc}")
            return False

        result = calculate_calibration_from_measurements(measurements)
        angle_stats = result.get('angle_stats', {})
        self.calibration_result = {
            'azimuth_offset': float(angle_stats.get('azimuth_offset_median', 0.0)),
            'pitch_offset': float(angle_stats.get('pitch_offset_median', 0.0)),
            'azimuth_scale': 1.0,
            'pitch_scale': 1.0,
            'timestamp': time.time(),
            'sample_count': int(result.get('sample_count', len(measurements))),
            'method': 'cal_offset_angle_median',
        }

        if not result.get('success'):
            print(f"[校准] {result.get('reason', 'cal_offset 计算失败')}")
            return False

        offset = result['offset']
        validation = result.get('validation', {})
        dx, dy, dz = offset

        print("[校准] cal_offset 位置偏移计算:")
        print(f"       光电相对于雷达: 东偏移 {dx:.2f}m, 北偏移 {dy:.2f}m, 高偏移 {dz:.2f}m")
        print(
            "       重投影角度误差: "
            f"均值 {validation.get('mean_error_deg', 0.0):.3f}°, "
            f"标准差 {validation.get('std_error_deg', 0.0):.3f}°, "
            f"最大 {validation.get('max_error_deg', 0.0):.3f}°"
        )

        self.position_offset = {
            'dx': float(dx),
            'dy': float(dy),
            'dz': float(dz),
            'timestamp': time.time(),
            'sample_count': int(result.get('sample_count', len(measurements))),
            'use_position': True,
            'method': result.get('method', 'cal_offset_least_squares'),
            'mean_error_deg': float(validation.get('mean_error_deg', 0.0)),
            'std_error_deg': float(validation.get('std_error_deg', 0.0)),
            'max_error_deg': float(validation.get('max_error_deg', 0.0)),
            'azimuth_offset_median': self.calibration_result['azimuth_offset'],
            'pitch_offset_median': self.calibration_result['pitch_offset'],
        }
        return True

    def to_cal_offset_measurements(self):
        """转换在线配对数据为 tools/cal_offset.py 使用的测量格式。"""
        measurements = []
        for r, o in zip(self.radar_measurements, self.optical_measurements):
            try:
                radar_range = float(r.get('range', 0))
                if radar_range < CALIBRATION_MIN_RANGE:
                    continue
                measurements.append({
                    'radar_az': float(r['azimuth']),
                    'radar_pitch': float(r['pitch']),
                    'radar_range': radar_range,
                    'optical_az': float(o['azimuth']),
                    'optical_pitch': float(o['pitch']),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return measurements

    def get_source_measurements(self):
        """生成本次校准使用的原始配对数据快照。"""
        samples = []
        for r, o in zip(self.radar_measurements, self.optical_measurements):
            try:
                radar_range = float(r.get('range', 0))
                if radar_range < CALIBRATION_MIN_RANGE:
                    continue

                radar_ts = float(r['timestamp']) if r.get('timestamp') is not None else None
                optical_ts = float(o['timestamp']) if o.get('timestamp') is not None else None
                time_diff = None
                if radar_ts is not None and optical_ts is not None:
                    time_diff = abs(radar_ts - optical_ts)

                samples.append({
                    'index': len(samples) + 1,
                    'time_diff_sec': time_diff,
                    'radar': {
                        'timestamp': radar_ts,
                        'track_id': r.get('track_id', self.current_target_id),
                        'azimuth': float(r['azimuth']),
                        'pitch': float(r['pitch']),
                        'range': radar_range,
                    },
                    'optical': {
                        'timestamp': optical_ts,
                        'azimuth': float(o['azimuth']),
                        'pitch': float(o['pitch']),
                        'range': float(o.get('range', 0) or 0),
                        'status': o.get('status'),
                    },
                })
            except (KeyError, TypeError, ValueError):
                continue

        return {
            'target_id': self.current_target_id,
            'sample_count': len(samples),
            'min_radar_range_m': float(CALIBRATION_MIN_RANGE),
            'pair_time_window_sec': float(CALIBRATION_PAIR_TIME_WINDOW),
            'samples': samples,
        }

    def _with_source_measurements(self, result):
        data = dict(result)
        source = self.get_source_measurements()
        if source['sample_count'] == 0 and isinstance(result.get('source_measurements'), dict):
            source = result['source_measurements']
        data['source_measurements'] = source
        return data

    def has_6dof_params(self):
        return (
            self.far_params.get('sample_count', 0) > 0
            and self.near_params.get('sample_count', 0) > 0
            and (
                self.far_params.get('mean_error_deg', 0) > 0
                or self.near_params.get('mean_error_deg', 0) > 0
            )
        )

    def apply_calibration(self, radar_azimuth, radar_pitch, radar_range=None):
        """应用校准参数，将雷达角度转换为光电应该转到的角度。"""
        if self.has_6dof_params() and radar_range is not None and radar_range > 0:
            return self.apply_6dof_calibration(radar_azimuth, radar_pitch, radar_range)

        if self.position_offset.get('use_position', False) and radar_range is not None and radar_range > 0:
            return self.apply_position_offset(radar_azimuth, radar_pitch, radar_range)

        return self.apply_angle_offset(radar_azimuth, radar_pitch)

    def apply_angle_offset(self, radar_azimuth, radar_pitch):
        calibrated_az = (radar_azimuth + self.calibration_result['azimuth_offset']) % 360
        calibrated_pitch = radar_pitch + self.calibration_result['pitch_offset']
        return calibrated_az, calibrated_pitch, None

    def apply_position_offset(self, radar_azimuth, radar_pitch, radar_range):
        if self.position_offset['sample_count'] == 0:
            return self.apply_angle_offset(radar_azimuth, radar_pitch)

        az_rad = np.radians(radar_azimuth)
        pitch_rad = np.radians(radar_pitch)
        target_x = radar_range * np.cos(pitch_rad) * np.sin(az_rad)
        target_y = radar_range * np.cos(pitch_rad) * np.cos(az_rad)
        target_z = radar_range * np.sin(pitch_rad)

        rel_x = target_x - self.position_offset['dx']
        rel_y = target_y - self.position_offset['dy']
        rel_z = target_z - self.position_offset['dz']

        opt_range = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        opt_azimuth = np.degrees(np.arctan2(rel_x, rel_y))
        if opt_azimuth < 0:
            opt_azimuth += 360
        opt_pitch = np.degrees(np.arcsin(rel_z / opt_range)) if opt_range > 0 else 0

        return opt_azimuth, opt_pitch, max(0.0, radar_range - FAKE_DIS)

    def apply_6dof_calibration(self, radar_azimuth, radar_pitch, radar_range):
        if radar_range is None or radar_range <= 0:
            return self.apply_angle_offset(radar_azimuth, radar_pitch)

        params = self.near_params if radar_range < self.RANGE_THRESHOLD else self.far_params
        if params.get('sample_count', 0) <= 0:
            return self.apply_position_offset(radar_azimuth, radar_pitch, radar_range)

        az_rad = np.radians(radar_azimuth)
        pitch_rad = np.radians(radar_pitch)
        x = radar_range * np.cos(pitch_rad) * np.sin(az_rad)
        y = radar_range * np.cos(pitch_rad) * np.cos(az_rad)
        z = radar_range * np.sin(pitch_rad)

        az_rot = np.radians(params.get('az_rotation', 0.0))
        pitch_rot = np.radians(params.get('pitch_rotation', 0.0))

        x1 = x * np.cos(az_rot) - y * np.sin(az_rot)
        y1 = x * np.sin(az_rot) + y * np.cos(az_rot)
        z1 = z

        x2 = x1
        y2 = y1 * np.cos(pitch_rot) - z1 * np.sin(pitch_rot)
        z2 = y1 * np.sin(pitch_rot) + z1 * np.cos(pitch_rot)

        rel_x = x2 - params.get('dx', 0.0)
        rel_y = y2 - params.get('dy', 0.0)
        rel_z = z2 - params.get('dz', 0.0)

        opt_range = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        opt_az = np.degrees(np.arctan2(rel_x, rel_y))
        if opt_az < 0:
            opt_az += 360
        opt_pitch = np.degrees(np.arcsin(rel_z / opt_range)) if opt_range > 0 else 0

        return opt_az, opt_pitch, max(0.0, radar_range - FAKE_DIS)

    def load_calibration(self):
        file_path = os.path.join(self.data_dir, 'calibration_params.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.calibration_result.update(loaded)
            print(
                f"[校准] 已加载角度偏移: 方位={self.calibration_result['azimuth_offset']:.2f}°, "
                f"俯仰={self.calibration_result['pitch_offset']:.2f}°"
            )
            return True
        return False

    def save_calibration(self):
        file_path = os.path.join(self.data_dir, 'calibration_params.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._with_source_measurements(self.calibration_result), f, indent=2, ensure_ascii=False)
        print(f"[校准] 角度偏移参数已保存: {file_path}")

    def load_position_offset(self):
        file_path = os.path.join(self.data_dir, 'position_offset.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.position_offset.update(loaded)
            print(
                f"[校准] 已加载位置偏移: DX={self.position_offset['dx']:.2f}m, "
                f"DY={self.position_offset['dy']:.2f}m, DZ={self.position_offset['dz']:.2f}m"
            )
            return True
        return False

    def save_position_offset(self):
        file_path = os.path.join(self.data_dir, 'position_offset.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._with_source_measurements(self.position_offset), f, indent=2, ensure_ascii=False)
        print(f"[校准] 位置偏移已保存: {file_path}")

    def load_6dof_params(self):
        file_path = os.path.join(self.data_dir, '6dof_params.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.far_params.update(data.get('far', {}))
                self.near_params.update(data.get('near', {}))
            print("[校准] 已加载6DoF参数")
            return True
        return False

    def save_6dof_params(self):
        file_path = os.path.join(self.data_dir, '6dof_params.json')
        data = {
            'far': self.far_params,
            'near': self.near_params,
            'range_threshold': self.RANGE_THRESHOLD,
            'timestamp': time.time(),
            'source_measurements': self.get_source_measurements(),
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[校准] 6DoF参数已保存: {file_path}")

    def get_status(self):
        return {
            'mode': self.calibration_mode,
            'target_id': self.current_target_id,
            'radar_samples': len(self.radar_measurements),
            'optical_samples': len(self.optical_measurements),
            'has_calibration': self.calibration_result['sample_count'] > 0,
            'has_position': self.position_offset['sample_count'] > 0,
            'use_position': self.position_offset.get('use_position', False),
            'has_6dof': self.has_6dof_params(),
        }

    def clear_calibration(self):
        """清除校准参数和本次采集样本。"""
        self.radar_measurements = []
        self.optical_measurements = []
        self.radar_buffer.clear()
        self.optical_buffer.clear()
        self.calibration_result = {
            'azimuth_offset': 0.0,
            'pitch_offset': 0.0,
            'azimuth_scale': 1.0,
            'pitch_scale': 1.0,
            'timestamp': 0,
            'sample_count': 0,
        }
        self.position_offset = {
            'dx': 0.0,
            'dy': 0.0,
            'dz': 0.0,
            'timestamp': 0,
            'sample_count': 0,
            'use_position': False,
        }
        self.far_params.update({'timestamp': 0, 'sample_count': 0, 'mean_error_deg': 0.0})
        self.near_params.update({'timestamp': 0, 'sample_count': 0, 'mean_error_deg': 0.0})
        self.save_calibration()
        self.save_position_offset()
        self.save_6dof_params()
        print("[校准] 已清除所有校准参数")


calibrator = RadarOpticalCalibrator()
