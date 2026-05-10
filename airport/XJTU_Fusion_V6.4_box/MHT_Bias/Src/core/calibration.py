# calibration.py
"""
雷达与光电校准模块
功能：通过同一目标的雷达测量值和光电实际角度，计算校准参数
支持两种校准方式：
1. 角度偏移校准（适用于安装位置很近）
2. 位置偏移校准（适用于安装位置分开，需要光电测距）
"""
from core.app_config import FAKE_DIS
import time
import json
import os
from collections import deque
import numpy as np

try:
    from core.app_config import CALIBRATION_PAIR_TIME_WINDOW, SCRIPT_DIR
except ImportError:
    CALIBRATION_PAIR_TIME_WINDOW = 0.15
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CALIBRATION_DATA_DIR = os.path.join(SCRIPT_DIR, "calibration_data")

class RadarOpticalCalibrator:
    def __init__(self, data_dir=DEFAULT_CALIBRATION_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 存储校准数据
        self.radar_measurements = []   # 雷达测量值
        self.optical_measurements = [] # 光电实际值（包含距离）
        
        # 当前跟踪的目标
        self.current_target_id = None
        self.calibration_mode = False
        
        # 数据缓冲
        self.radar_buffer = deque(maxlen=10)    # 存储最近10次雷达数据
        self.optical_buffer = deque(maxlen=10)  # 存储最近10次光电数据
        
        # 角度偏移校准结果
        self.calibration_result = {
            'azimuth_offset': 0.0,      # 方位角偏移
            'pitch_offset': 0.0,        # 俯仰角偏移
            'azimuth_scale': 1.0,       # 方位角缩放
            'pitch_scale': 1.0,         # 俯仰角缩放
            'timestamp': 0,
            'sample_count': 0
        }
        
        # 位置偏移校准结果（需要光电测距）
        self.position_offset = {
            'dx': 0.0,      # 光电相对于雷达的东向偏移（米）
            'dy': 0.0,      # 光电相对于雷达的北向偏移（米）
            'dz': 0.0,      # 光电相对于雷达的高度偏移（米）
            'timestamp': 0,
            'sample_count': 0,
            'use_position': False   # 是否使用位置偏移校准
        }
        
        # 加载已有校准参数
        self.load_calibration()
        self.load_position_offset()
    
    def start_calibration(self, target_id):
        """开始校准"""
        self.calibration_mode = True
        self.current_target_id = target_id
        self.radar_measurements = []
        self.optical_measurements = []
        self.radar_buffer.clear()
        self.optical_buffer.clear()
        print(f"[校准] 开始校准，目标: {target_id}")
        print(f"[校准] 雷达扫描周期: 4秒，建议采集10-20组数据")
        return True
    
    def stop_calibration(self):
        """停止校准并计算参数"""
        self.calibration_mode = False
        
        if len(self.radar_measurements) < 5:
            print(f"[校准] 数据不足，需要至少5组数据，当前有{len(self.radar_measurements)}组")
            return False
        
        # # 检查是否有光电距离数据
        # has_range = len(self.optical_measurements) >= 5
        
        # if has_range and len(self.radar_measurements) >= 5:
        #     # 有距离数据，使用位置偏移校准（更精确）
        #     print("[校准] 检测到光电距离数据，使用位置偏移校准...")
        #     success = self.calculate_position_offset()
        #     if success:
        #         self.position_offset['use_position'] = True
        #         self.save_position_offset()
        #         print(f"[校准] 位置偏移校准完成！")
        #         print(f"[校准] 光电相对于雷达: 东偏移 {self.position_offset['dx']:.2f}m, "
        #               f"北偏移 {self.position_offset['dy']:.2f}m, 高偏移 {self.position_offset['dz']:.2f}m")
        #         print(f"[校准] 样本数量: {self.position_offset['sample_count']}")
        #         return True
        #     else:
        #         print("[校准] 位置偏移校准失败，回退到角度偏移校准...")
        
        # 没有距离数据或位置偏移校准失败，使用角度偏移校准
        self._calculate_calibration()
        self.save_calibration()
        
        print(f"[校准] 角度偏移校准完成！")
        print(f"[校准] 方位角偏移: {self.calibration_result['azimuth_offset']:.2f}°")
        print(f"[校准] 俯仰角偏移: {self.calibration_result['pitch_offset']:.2f}°")
        print(f"[校准] 样本数量: {self.calibration_result['sample_count']}")
        
        return True

    def add_radar_measurement(self, track_id, azimuth, pitch, range_m, timestamp=None):
        """添加雷达测量数据"""
        if not self.calibration_mode or track_id != self.current_target_id:
            return
        
        if timestamp is None:
            timestamp = time.time()

            
        self.radar_buffer.append({
            'timestamp': timestamp,
            'azimuth': azimuth,
            'pitch': pitch,
            'range': range_m
        })
        
        self._try_pair_measurement()
    
    # OPTICAL_SEND_INTERVAL秒调用一次
    def add_optical_measurement(self, azimuth, pitch, timestamp=None, optical_status=None, opt_range=None):
        """添加光电实际角度和距离"""
        if not self.calibration_mode:
            return
        
        if optical_status != 2:  # 只在跟踪状态记录
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        
        self.optical_buffer.append({
            'timestamp': timestamp,
            'azimuth': azimuth,
            'pitch': pitch,
            'range': opt_range if opt_range else 0
        })
        
        self._try_pair_measurement()
        
    def _try_pair_measurement(self):
        """尝试配对雷达和光电测量数据"""
        if len(self.radar_buffer) == 0 or len(self.optical_buffer) == 0:
            return
        
        radar_data = list(self.radar_buffer)
        optical_data = list(self.optical_buffer)
        
        # 找到时间最接近的一对
        best_pair = None
        best_diff = float('inf')
        
        for r in radar_data:
            for o in optical_data:
                time_diff = abs(r['timestamp'] - o['timestamp'])
                radar_range = r.get('range', 0) or 0
                optical_range = 0
                if False and radar_range > 0 and optical_range > 0:
                    max_range_diff = max(80.0, radar_range * 0.20)
                    if abs(radar_range - optical_range) > max_range_diff:
                        continue
                if time_diff <= CALIBRATION_PAIR_TIME_WINDOW and time_diff < best_diff:
                    best_pair = (r, o)
                    best_diff = time_diff
        
        if best_pair:
            r, o = best_pair
            self.radar_measurements.append(r)
            self.optical_measurements.append(o)
            print(f"[校准] 配对成功: 雷达({r['azimuth']:.1f}°, {r['pitch']:.1f}°, {r['range']:.0f}m) -> "
                  f"光电({o['azimuth']:.1f}°, {o['pitch']:.1f}°, {o['range']:.0f}m), 时间差={best_diff:.2f}s")
            
            # 清空缓冲区，避免重复配对
            self.radar_buffer.clear()
            self.optical_buffer.clear()
    
    def _calculate_calibration(self):
        """计算角度偏移校准参数"""
        n = len(self.radar_measurements)
        
        radar_az = np.array([m['azimuth'] for m in self.radar_measurements])
        radar_pitch = np.array([m['pitch'] for m in self.radar_measurements])
        opt_az = np.array([m['azimuth'] for m in self.optical_measurements])
        opt_pitch = np.array([m['pitch'] for m in self.optical_measurements])
        
        # 处理角度环绕（360° -> 0°）
        for i in range(n):
            diff = opt_az[i] - radar_az[i]
            if diff > 180:
                radar_az[i] += 360
            elif diff < -180:
                radar_az[i] -= 360
        
        # 计算偏移
        az_diff = opt_az - radar_az
        pitch_diff = opt_pitch - radar_pitch
        
        # 去除异常值（3倍标准差）
        az_std = np.std(az_diff)
        pitch_std = np.std(pitch_diff)
        az_mean = np.mean(az_diff)
        pitch_mean = np.mean(pitch_diff)
        
        valid_mask = (np.abs(az_diff - az_mean) < 3 * az_std) & \
                     (np.abs(pitch_diff - pitch_mean) < 3 * pitch_std)
        
        if np.sum(valid_mask) < 3:
            print(f"[校准] 有效数据不足，使用全部数据")
            valid_mask = np.ones(n, dtype=bool)
        
        self.calibration_result = {
            'azimuth_offset': float(np.mean(az_diff[valid_mask])),
            'pitch_offset': float(np.mean(pitch_diff[valid_mask])),
            'azimuth_scale': 1.0,
            'pitch_scale': 1.0,
            'timestamp': time.time(),
            'sample_count': int(np.sum(valid_mask))
        }
    
    def calculate_position_offset(self):
        """
        利用光电测量的距离，计算光电相对于雷达的位置偏移（三维）
        
        原理：
        雷达测量的目标位置 = 光电位置 + 光电测量的目标位置
        因此：光电位置 = 雷达测量的目标位置 - 光电测量的目标位置
        """
        if len(self.radar_measurements) < 3:
            print(f"[校准] 位置偏移需要至少3组数据，当前有{len(self.radar_measurements)}组")
            return False
        
        radar_positions = []
        optical_positions = []
        
        for i in range(len(self.radar_measurements)):
            r = self.radar_measurements[i]
            o = self.optical_measurements[i]
            
            # 检查是否有光电距离
            opt_range = o.get('range', 0)
            if opt_range <= 0:
                print(f"[校准] 第{i+1}组数据缺少光电距离，跳过")
                continue
            
            # 雷达测量的目标 ENU 位置（雷达为原点）
            radar_az_rad = np.radians(r['azimuth'])
            radar_pitch_rad = np.radians(r['pitch'])
            radar_range = r['range']
            
            radar_x = radar_range * np.cos(radar_pitch_rad) * np.sin(radar_az_rad)
            radar_y = radar_range * np.cos(radar_pitch_rad) * np.cos(radar_az_rad)
            radar_z = radar_range * np.sin(radar_pitch_rad)
            
            # 光电测量的目标 ENU 位置（光电为原点）
            opt_az_rad = np.radians(o['azimuth'])
            opt_pitch_rad = np.radians(o['pitch'])
            
            opt_x = opt_range * np.cos(opt_pitch_rad) * np.sin(opt_az_rad)
            opt_y = opt_range * np.cos(opt_pitch_rad) * np.cos(opt_az_rad)
            opt_z = opt_range * np.sin(opt_pitch_rad)
            
            radar_positions.append([radar_x, radar_y, radar_z])
            optical_positions.append([opt_x, opt_y, opt_z])
        
        if len(radar_positions) < 3:
            print(f"[校准] 有效数据不足，需要至少3组，当前有{len(radar_positions)}组")
            return False
        
        # 转换为 numpy 数组
        radar_positions = np.array(radar_positions)
        optical_positions = np.array(optical_positions)
        
        # 计算平均位置
        radar_mean = np.mean(radar_positions, axis=0)
        optical_mean = np.mean(optical_positions, axis=0)
        
        # 光电相对于雷达的位置偏移
        dx = radar_mean[0] - optical_mean[0]
        dy = radar_mean[1] - optical_mean[1]
        dz = radar_mean[2] - optical_mean[2]
        
        # 计算残差，评估校准质量
        errors = []
        for i in range(len(radar_positions)):
            error = np.linalg.norm(radar_positions[i] - optical_positions[i] - np.array([dx, dy, dz]))
            errors.append(error)
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f"[校准] 位置偏移计算:")
        print(f"       光电相对于雷达: 东偏移 {dx:.2f}m, 北偏移 {dy:.2f}m, 高偏移 {dz:.2f}m")
        print(f"       平均残差: {mean_error:.2f}m, 标准差: {std_error:.2f}m")
        
        self.position_offset = {
            'dx': float(dx),
            'dy': float(dy),
            'dz': float(dz),
            'timestamp': time.time(),
            'sample_count': len(radar_positions),
            'use_position': True
        }
        
        return True
    
    def apply_calibration(self, radar_azimuth, radar_pitch, radar_range=None):
        """
        应用校准参数，将雷达角度转换为光电应该转到的角度
        
        优先使用位置偏移校准（如果可用且有距离），否则使用角度偏移校准
        """
        # 优先使用位置偏移校准
        if self.position_offset.get('use_position', False) and radar_range is not None and radar_range > 0:
            return self.apply_position_offset(radar_azimuth, radar_pitch, radar_range)
        else:
            return self.apply_angle_offset(radar_azimuth, radar_pitch)
    
    def apply_angle_offset(self, radar_azimuth, radar_pitch):
        """应用角度偏移校准"""
        calibrated_az = radar_azimuth + self.calibration_result['azimuth_offset']
        calibrated_pitch = radar_pitch + self.calibration_result['pitch_offset']
        
        # 归一化角度
        calibrated_az = calibrated_az % 360
        
        return calibrated_az, calibrated_pitch, None

    def apply_position_offset(self, radar_azimuth, radar_pitch, radar_range):
        """
        应用位置偏移校准，将雷达测量的目标转换为光电应该指向的角度
        """
        if self.position_offset['sample_count'] == 0:
            return self.apply_angle_offset(radar_azimuth, radar_pitch)
        
        # 1. 雷达测量的目标 ENU 位置
        az_rad = np.radians(radar_azimuth)
        pitch_rad = np.radians(radar_pitch)
        
        target_x = radar_range * np.cos(pitch_rad) * np.sin(az_rad)
        target_y = radar_range * np.cos(pitch_rad) * np.cos(az_rad)
        target_z = radar_range * np.sin(pitch_rad)
        
        # 2. 光电的位置（相对于雷达）
        dx = self.position_offset['dx']
        dy = self.position_offset['dy']
        dz = self.position_offset['dz']
        
        # 3. 目标相对于光电的位置
        rel_x = target_x - dx
        rel_y = target_y - dy
        rel_z = target_z - dz
        
        # 4. 计算光电应该转到的角度
        # opt_range = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        opt_range = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        opt_azimuth = np.degrees(np.arctan2(rel_x, rel_y))
        if opt_azimuth < 0:
            opt_azimuth += 360
        opt_pitch = np.degrees(np.arcsin(rel_z / opt_range)) if opt_range > 0 else 0
        
        opt_range = radar_range-FAKE_DIS
        return opt_azimuth, opt_pitch, opt_range
    
    def save_calibration(self):
        """保存角度偏移校准参数"""
        file_path = os.path.join(self.data_dir, 'calibration_params.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.calibration_result, f, indent=2)
        print(f"[校准] 角度偏移参数已保存: {file_path}")
    
    def load_calibration(self):
        """加载角度偏移校准参数"""
        file_path = os.path.join(self.data_dir, 'calibration_params.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.calibration_result.update(loaded)
            print(f"[校准] 已加载角度偏移: 方位={self.calibration_result['azimuth_offset']:.2f}°, "
                  f"俯仰={self.calibration_result['pitch_offset']:.2f}°")
            return True
        return False
    
    def save_position_offset(self):
        """保存位置偏移校准参数"""
        file_path = os.path.join(self.data_dir, 'position_offset.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.position_offset, f, indent=2)
        print(f"[校准] 位置偏移已保存: {file_path}")
    
    def load_position_offset(self):
        """加载位置偏移校准参数"""
        file_path = os.path.join(self.data_dir, 'position_offset.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.position_offset.update(loaded)
            print(f"[校准] 已加载位置偏移: DX={self.position_offset['dx']:.2f}m, "
                  f"DY={self.position_offset['dy']:.2f}m, DZ={self.position_offset['dz']:.2f}m")
            return True
        return False
    
    def get_status(self):
        """获取校准状态"""
        return {
            'mode': self.calibration_mode,
            'target_id': self.current_target_id,
            'radar_samples': len(self.radar_measurements),
            'optical_samples': len(self.optical_measurements),
            'has_calibration': self.calibration_result['sample_count'] > 0,
            'has_position': self.position_offset['sample_count'] > 0,
            'use_position': self.position_offset.get('use_position', False)
        }
    
    def clear_calibration(self):
        """清除所有校准参数"""
        self.calibration_result = {
            'azimuth_offset': 0.0,
            'pitch_offset': 0.0,
            'azimuth_scale': 1.0,
            'pitch_scale': 1.0,
            'timestamp': 0,
            'sample_count': 0
        }
        self.position_offset = {
            'dx': 0.0, 'dy': 0.0, 'dz': 0.0,
            'timestamp': 0, 'sample_count': 0, 'use_position': False
        }
        self.save_calibration()
        self.save_position_offset()
        print("[校准] 已清除所有校准参数")


# 全局校准器实例
def _angle_diff_deg(target_angle, source_angle):
    """Return signed shortest difference target - source in degrees."""
    return (target_angle - source_angle + 180.0) % 360.0 - 180.0


def _robust_mask(values, sigma=3.5):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=bool)

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-9:
        std = np.std(values)
        if std < 1e-9:
            return np.ones(len(values), dtype=bool)
        return np.abs(values - np.mean(values)) <= sigma * std

    robust_z = 0.6745 * (values - median) / mad
    return np.abs(robust_z) <= sigma


def _calculate_calibration_robust(self):
    """Calculate angle offsets from paired radar/optical measurements."""
    n = len(self.radar_measurements)

    radar_az = np.array([m['azimuth'] for m in self.radar_measurements], dtype=float)
    radar_pitch = np.array([m['pitch'] for m in self.radar_measurements], dtype=float)
    opt_az = np.array([m['azimuth'] for m in self.optical_measurements], dtype=float)
    opt_pitch = np.array([m['pitch'] for m in self.optical_measurements], dtype=float)

    az_diff = _angle_diff_deg(opt_az, radar_az)
    pitch_diff = opt_pitch - radar_pitch

    valid_mask = _robust_mask(az_diff) & _robust_mask(pitch_diff)
    if np.sum(valid_mask) < 3:
        print("[校准] 有效数据不足，使用全部数据")
        valid_mask = np.ones(n, dtype=bool)

    self.calibration_result = {
        'azimuth_offset': float(np.median(az_diff[valid_mask])),
        'pitch_offset': float(np.median(pitch_diff[valid_mask])),
        'azimuth_scale': 1.0,
        'pitch_scale': 1.0,
        'timestamp': time.time(),
        'sample_count': int(np.sum(valid_mask))
    }


def _calculate_position_offset_robust(self):
    """Estimate optical position offset relative to radar from paired measurements."""
    if len(self.radar_measurements) < 3:
        print(f"[校准] 位置偏移至少需要3组数据，当前{len(self.radar_measurements)}组")
        return False

    radar_positions = []
    optical_positions = []

    for i, (r, o) in enumerate(zip(self.radar_measurements, self.optical_measurements)):
        opt_range = o.get('range', 0)
        if opt_range <= 0:
            print(f"[校准] 第{i + 1}组数据缺少光电距离，跳过")
            continue

        radar_az_rad = np.radians(r['azimuth'])
        radar_pitch_rad = np.radians(r['pitch'])
        radar_range = r['range']
        radar_positions.append([
            radar_range * np.cos(radar_pitch_rad) * np.sin(radar_az_rad),
            radar_range * np.cos(radar_pitch_rad) * np.cos(radar_az_rad),
            radar_range * np.sin(radar_pitch_rad),
        ])

        opt_az_rad = np.radians(o['azimuth'])
        opt_pitch_rad = np.radians(o['pitch'])
        optical_positions.append([
            opt_range * np.cos(opt_pitch_rad) * np.sin(opt_az_rad),
            opt_range * np.cos(opt_pitch_rad) * np.cos(opt_az_rad),
            opt_range * np.sin(opt_pitch_rad),
        ])

    if len(radar_positions) < 3:
        print(f"[校准] 有效位置偏移数据不足，至少3组，当前{len(radar_positions)}组")
        return False

    radar_positions = np.array(radar_positions, dtype=float)
    optical_positions = np.array(optical_positions, dtype=float)
    offsets = radar_positions - optical_positions
    offset = np.median(offsets, axis=0)

    errors = [np.linalg.norm(row - offset) for row in offsets]
    error_mask = _robust_mask(errors)
    if np.sum(error_mask) >= 3:
        offset = np.median(offsets[error_mask], axis=0)
        errors = [np.linalg.norm(row - offset) for row in offsets[error_mask]]

    dx, dy, dz = offset
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print("[校准] 位置偏移计算:")
    print(f"       光电相对于雷达: 东偏移={dx:.2f}m, 北偏移={dy:.2f}m, 高偏移={dz:.2f}m")
    print(f"       平均残差: {mean_error:.2f}m, 标准差: {std_error:.2f}m")

    self.position_offset = {
        'dx': float(dx),
        'dy': float(dy),
        'dz': float(dz),
        'timestamp': time.time(),
        'sample_count': int(len(errors)),
        'use_position': True
    }

    return True


def _calculate_position_offset_bearing_only(self):
    """Estimate optical translation using radar range/angles and optical angles."""
    pairs = list(zip(self.radar_measurements, self.optical_measurements))
    if len(pairs) < 5:
        print(f"[calibration] need at least 5 paired samples, got {len(pairs)}")
        return False

    radar_positions = []
    optical_dirs = []
    for r, o in pairs:
        radar_az_rad = np.radians(r['azimuth'])
        radar_pitch_rad = np.radians(r['pitch'])
        radar_range = float(r['range'])
        radar_positions.append([
            radar_range * np.cos(radar_pitch_rad) * np.sin(radar_az_rad),
            radar_range * np.cos(radar_pitch_rad) * np.cos(radar_az_rad),
            radar_range * np.sin(radar_pitch_rad),
        ])

        opt_az_rad = np.radians(o['azimuth'])
        opt_pitch_rad = np.radians(o['pitch'])
        optical_dirs.append([
            np.cos(opt_pitch_rad) * np.sin(opt_az_rad),
            np.cos(opt_pitch_rad) * np.cos(opt_az_rad),
            np.sin(opt_pitch_rad),
        ])

    radar_positions = np.asarray(radar_positions, dtype=float)
    optical_dirs = np.asarray(optical_dirs, dtype=float)

    def solve_offset(positions, dirs):
        a_rows = []
        b_rows = []
        for p, u in zip(positions, dirs):
            ux = np.array([
                [0.0, -u[2], u[1]],
                [u[2], 0.0, -u[0]],
                [-u[1], u[0], 0.0],
            ])
            a_rows.append(ux)
            b_rows.append(ux @ p)
        A = np.vstack(a_rows)
        b = np.concatenate(b_rows)
        offset, *_ = np.linalg.lstsq(A, b, rcond=None)
        return offset

    def angular_errors_for(offset, positions, dirs):
        errors = []
        for p, u in zip(positions, dirs):
            rel = p - offset
            norm = np.linalg.norm(rel)
            if norm <= 1e-9:
                continue
            dot = np.clip(float(np.dot(rel / norm, u)), -1.0, 1.0)
            errors.append(np.degrees(np.arccos(dot)))
        return np.asarray(errors, dtype=float)

    offset = solve_offset(radar_positions, optical_dirs)
    errors = angular_errors_for(offset, radar_positions, optical_dirs)
    mask = _robust_mask(errors)
    if np.sum(mask) >= 5 and np.sum(mask) < len(radar_positions):
        radar_positions = radar_positions[mask]
        optical_dirs = optical_dirs[mask]
        offset = solve_offset(radar_positions, optical_dirs)
        errors = angular_errors_for(offset, radar_positions, optical_dirs)

    dx, dy, dz = offset
    mean_error = float(np.mean(errors)) if len(errors) else 0.0
    std_error = float(np.std(errors)) if len(errors) else 0.0

    print("[calibration] bearing-only position calibration complete")
    print(f"       optical offset relative to radar: dx={dx:.2f}m, dy={dy:.2f}m, dz={dz:.2f}m")
    print(f"       angular residual: mean={mean_error:.2f}deg, std={std_error:.2f}deg, samples={len(errors)}")

    self.position_offset = {
        'dx': float(dx),
        'dy': float(dy),
        'dz': float(dz),
        'timestamp': time.time(),
        'sample_count': int(len(errors)),
        'use_position': True,
        'method': 'bearing_only'
    }
    return True


RadarOpticalCalibrator._calculate_calibration = _calculate_calibration_robust
RadarOpticalCalibrator.calculate_position_offset = _calculate_position_offset_bearing_only


calibrator = RadarOpticalCalibrator()
