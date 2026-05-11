import numpy as np
from scipy.optimize import least_squares
from typing import List, Dict, Tuple


def angle_diff_deg(target_angle: float, source_angle: float) -> float:
    """计算 target - source 的最短有符号角度差。"""
    return (target_angle - source_angle + 180.0) % 360.0 - 180.0


def summarize_angle_offsets(measurements: List[Dict]) -> Dict:
    """统计零位置偏移假设下的角度差，用于诊断标定数据质量。"""
    az_diffs = []
    pitch_diffs = []
    for m in measurements:
        az_diffs.append(angle_diff_deg(m['optical_az'], m['radar_az']))
        pitch_diffs.append(m['optical_pitch'] - m['radar_pitch'])

    if not az_diffs:
        return {
            'azimuth_offset_median': 0.0,
            'pitch_offset_median': 0.0,
            'azimuth_offset_mean': 0.0,
            'pitch_offset_mean': 0.0,
            'azimuth_offset_std': 0.0,
            'pitch_offset_std': 0.0,
        }

    return {
        'azimuth_offset_median': float(np.median(az_diffs)),
        'pitch_offset_median': float(np.median(pitch_diffs)),
        'azimuth_offset_mean': float(np.mean(az_diffs)),
        'pitch_offset_mean': float(np.mean(pitch_diffs)),
        'azimuth_offset_std': float(np.std(az_diffs)),
        'pitch_offset_std': float(np.std(pitch_diffs)),
    }

def calculate_offset_from_measurements(measurements: List[Dict]) -> Tuple[np.ndarray, bool]:
    """
    根据雷达-光学配对测量值计算光学传感器相对于雷达的位置偏移
    """
    
    def polar_to_vector(az: float, pitch: float, rng: float = 1.0) -> np.ndarray:
        """极坐标转笛卡尔坐标 (北东天坐标系: y=北, x=东, z=天)"""
        az_rad = np.radians(az)
        pitch_rad = np.radians(pitch)
        cos_pitch = np.cos(pitch_rad)
        return np.array([
            rng * cos_pitch * np.sin(az_rad),  # x: 东
            rng * cos_pitch * np.cos(az_rad),  # y: 北
            rng * np.sin(pitch_rad)            # z: 天
        ])
    
    def residuals(params, meas):
        """残差函数"""
        dx, dy, dz = params[:3]
        lambdas = params[3:]  # 每个目标到光学传感器的距离
        
        residuals_vec = []
        for i, m in enumerate(meas):
            # 雷达测量的目标位置
            radar_pos = polar_to_vector(m['radar_az'], m['radar_pitch'], m['radar_range'])
            
            # 光学传感器位置（待求）
            optical_pos = np.array([dx, dy, dz])
            
            # 光学射线方向
            optical_dir = polar_to_vector(m['optical_az'], m['optical_pitch'], 1.0)
            
            # 预测的目标位置
            predicted_pos = optical_pos + lambdas[i] * optical_dir
            
            # 残差
            residuals_vec.extend(predicted_pos - radar_pos)
        
        return np.array(residuals_vec)
    
    # 参数初始化
    n_measurements = len(measurements)
    x0 = [0.0, 0.0, 0.0] + [m['radar_range'] * 0.8 for m in measurements]
    
    # 设置参数边界 - 限制偏移量在合理范围（-3000m 到 3000m）
    bounds_lower = [-3000.0, -3000.0, -3000.0] + [1.0] * n_measurements
    bounds_upper = [3000.0, 3000.0, 3000.0] + [np.inf] * n_measurements
    
    try:
        result = least_squares(
            residuals, x0, 
            bounds=(bounds_lower, bounds_upper),
            args=(measurements,),
            method='trf', 
            ftol=1e-12, 
            xtol=1e-12,
            verbose=0
        )
        
        if result.success:
            offset = result.x[:3]
            return offset, True
        else:
            print(f"优化未成功: {result.message}")
            return np.zeros(3), False
            
    except Exception as e:
        print(f"优化失败: {e}")
        return np.zeros(3), False


def validate_offset(offset: np.ndarray, measurements: List[Dict]) -> Dict:
    """
    验证偏移量的准确性（只对传入的测量集计算误差）
    """
    def polar_to_vector(az, pitch, rng=1.0):
        az_rad = np.radians(az)
        pitch_rad = np.radians(pitch)
        cp = np.cos(pitch_rad)
        return np.array([
            rng * cp * np.sin(az_rad),
            rng * cp * np.cos(az_rad),
            rng * np.sin(pitch_rad)
        ])
    
    if len(measurements) == 0:
        return {
            'mean_error_deg': float('inf'),
            'std_error_deg': float('inf'),
            'max_error_deg': float('inf'),
            'min_error_deg': float('inf'),
            'azimuth_errors': [],
            'pitch_errors': [],
        }
    
    errors = []
    azimuth_errors = []
    pitch_errors = []
    
    for m in measurements:
        # 雷达测量的目标位置
        radar_pos = polar_to_vector(m['radar_az'], m['radar_pitch'], m['radar_range'])
        
        # 根据偏移计算预测的光学测量
        optical_pos = offset
        optical_dir_from_radar = radar_pos - optical_pos
        optical_range = np.linalg.norm(optical_dir_from_radar)
        
        # 计算预测的光学角度
        pred_az = np.degrees(np.arctan2(optical_dir_from_radar[0], 
                                        optical_dir_from_radar[1])) % 360.0
        pred_pitch = np.degrees(np.arcsin(optical_dir_from_radar[2] / optical_range))
        
        # 角度误差
        az_error = abs(pred_az - m['optical_az'])
        az_error = min(az_error, 360 - az_error)
        pitch_error = abs(pred_pitch - m['optical_pitch'])
        
        azimuth_errors.append(az_error)
        pitch_errors.append(pitch_error)
        errors.append(np.sqrt(az_error**2 + pitch_error**2))
    
    return {
        'mean_error_deg': float(np.mean(errors)),
        'std_error_deg': float(np.std(errors)),
        'max_error_deg': float(np.max(errors)),
        'min_error_deg': float(np.min(errors)),
        'azimuth_errors': [float(x) for x in azimuth_errors],
        'pitch_errors': [float(x) for x in pitch_errors],
    }


def calculate_calibration_from_measurements(measurements: List[Dict], min_samples: int = 5) -> Dict:
    """
    统一的雷达-光电在线/离线位置偏移计算入口。

    measurements 中每项需要包含：
      radar_az, radar_pitch, radar_range, optical_az, optical_pitch
    """
    if len(measurements) < min_samples:
        return {
            'success': False,
            'reason': f'有效样本不足，需要至少{min_samples}组，当前{len(measurements)}组',
            'sample_count': len(measurements),
            'angle_stats': summarize_angle_offsets(measurements),
        }

    offset, ok = calculate_offset_from_measurements(measurements)
    angle_stats = summarize_angle_offsets(measurements)
    if not ok:
        return {
            'success': False,
            'reason': '位置偏移优化失败',
            'sample_count': len(measurements),
            'angle_stats': angle_stats,
        }

    validation = validate_offset(offset, measurements)
    return {
        'success': True,
        'offset': offset,
        'sample_count': len(measurements),
        'validation': validation,
        'angle_stats': angle_stats,
        'method': 'cal_offset_least_squares',
    }


def main():
    # ==================== 远距离测量数据（距离 > 700m）====================
    # 从你的日志中提取的远距离数据
    far_measurements = [
        # 第一组远距离数据
        {'radar_az': 148.1, 'radar_pitch': 4.1, 'radar_range': 845, 'optical_az': 147.8, 'optical_pitch': 4.0},
        {'radar_az': 148.1, 'radar_pitch': 3.9, 'radar_range': 873, 'optical_az': 147.8, 'optical_pitch': 3.9},
        {'radar_az': 148.1, 'radar_pitch': 3.9, 'radar_range': 901, 'optical_az': 147.7, 'optical_pitch': 3.7},
        {'radar_az': 148.1, 'radar_pitch': 3.9, 'radar_range': 929, 'optical_az': 147.8, 'optical_pitch': 3.5},
        {'radar_az': 148.8, 'radar_pitch': 3.1, 'radar_range': 992, 'optical_az': 147.8, 'optical_pitch': 3.4},
        {'radar_az': 149.1, 'radar_pitch': 3.8, 'radar_range': 1035, 'optical_az': 147.8, 'optical_pitch': 3.2},
        {'radar_az': 149.0, 'radar_pitch': 3.8, 'radar_range': 1070, 'optical_az': 147.8, 'optical_pitch': 3.1},
        {'radar_az': 148.7, 'radar_pitch': 3.1, 'radar_range': 1107, 'optical_az': 147.9, 'optical_pitch': 3.0},
        {'radar_az': 148.7, 'radar_pitch': 3.2, 'radar_range': 1146, 'optical_az': 147.9, 'optical_pitch': 2.9},
        {'radar_az': 148.7, 'radar_pitch': 2.9, 'radar_range': 1183, 'optical_az': 147.9, 'optical_pitch': 2.7},
        {'radar_az': 148.7, 'radar_pitch': 2.5, 'radar_range': 1220, 'optical_az': 147.9, 'optical_pitch': 2.6},
        {'radar_az': 148.7, 'radar_pitch': 2.1, 'radar_range': 1258, 'optical_az': 147.9, 'optical_pitch': 2.5},
        {'radar_az': 148.7, 'radar_pitch': 1.6, 'radar_range': 1293, 'optical_az': 148.0, 'optical_pitch': 2.4},
        
        # 第二组远距离数据
        {'radar_az': 149.2, 'radar_pitch': 3.2, 'radar_range': 1033, 'optical_az': 148.3, 'optical_pitch': 3.5},
        {'radar_az': 149.3, 'radar_pitch': 3.3, 'radar_range': 981, 'optical_az': 148.3, 'optical_pitch': 3.5},
        {'radar_az': 149.3, 'radar_pitch': 3.1, 'radar_range': 937, 'optical_az': 148.3, 'optical_pitch': 3.7},
        {'radar_az': 149.3, 'radar_pitch': 2.7, 'radar_range': 890, 'optical_az': 148.3, 'optical_pitch': 4.0},
        {'radar_az': 149.3, 'radar_pitch': 2.4, 'radar_range': 844, 'optical_az': 148.3, 'optical_pitch': 4.3},
        {'radar_az': 149.3, 'radar_pitch': 2.0, 'radar_range': 797, 'optical_az': 148.3, 'optical_pitch': 4.6},
        {'radar_az': 149.1, 'radar_pitch': 3.9, 'radar_range': 695, 'optical_az': 148.2, 'optical_pitch': 5.3},
    ]
    
    # ==================== 中距离数据（500-700m，用于对比）====================
    middle_measurements = [
        {'radar_az': 149.0, 'radar_pitch': 8.7, 'radar_range': 647, 'optical_az': 148.2, 'optical_pitch': 5.8},
        {'radar_az': 149.1, 'radar_pitch': 9.5, 'radar_range': 603, 'optical_az': 148.2, 'optical_pitch': 6.3},
        {'radar_az': 149.1, 'radar_pitch': 8.9, 'radar_range': 553, 'optical_az': 148.2, 'optical_pitch': 7.0},
        {'radar_az': 149.1, 'radar_pitch': 8.6, 'radar_range': 504, 'optical_az': 148.2, 'optical_pitch': 7.7},
        {'radar_az': 149.2, 'radar_pitch': 8.6, 'radar_range': 456, 'optical_az': 148.2, 'optical_pitch': 8.6},
        {'radar_az': 149.3, 'radar_pitch': 8.8, 'radar_range': 408, 'optical_az': 148.1, 'optical_pitch': 9.7},
        {'radar_az': 149.2, 'radar_pitch': 8.8, 'radar_range': 362, 'optical_az': 148.1, 'optical_pitch': 10.9},
    ]
    
    # ==================== 近距离异常数据（用于对比，不参与标定）====================
    close_abnormal = [
        {'radar_az': 149.3, 'radar_pitch': 8.9, 'radar_range': 315, 'optical_az': 148.0, 'optical_pitch': 12.8},
        {'radar_az': 149.3, 'radar_pitch': 8.8, 'radar_range': 268, 'optical_az': 148.0, 'optical_pitch': 15.1},
        {'radar_az': 149.3, 'radar_pitch': 8.7, 'radar_range': 222, 'optical_az': 147.9, 'optical_pitch': 16.5},
        {'radar_az': 149.3, 'radar_pitch': 8.6, 'radar_range': 179, 'optical_az': 147.8, 'optical_pitch': 17.6},
        {'radar_az': 149.3, 'radar_pitch': 8.5, 'radar_range': 128, 'optical_az': 147.8, 'optical_pitch': 19.3},
    ]
    
    # 合并所有远距离数据
    all_far_measurements = far_measurements
    
    print("=" * 80)
    print("传感器标定 - 基于远距离数据（距离 > 700m）")
    print("=" * 80)
    
    print(f"\n使用 {len(all_far_measurements)} 组远距离数据")
    
    # ==================== 角度偏差分析（先于位置标定）====================
    print("\n" + "=" * 80)
    print("第一步：角度偏差分析（零位移假设）")
    print("=" * 80)
    
    print("\n远距离数据角度差值:")
    print("序号 | 距离   | 雷达方位 | 雷达俯仰 | 光电方位 | 光电俯仰 | 方位差 | 俯仰差")
    print("-" * 85)
    
    az_diffs = []
    pitch_diffs = []
    
    for i, m in enumerate(all_far_measurements):
        az_diff = m['optical_az'] - m['radar_az']
        if az_diff > 180:
            az_diff -= 360
        elif az_diff < -180:
            az_diff += 360
        pitch_diff = m['optical_pitch'] - m['radar_pitch']
        
        az_diffs.append(az_diff)
        pitch_diffs.append(pitch_diff)
        
        print(f"{i+1:4d} | {m['radar_range']:4.0f}m | {m['radar_az']:7.1f}°  | {m['radar_pitch']:7.1f}°   | {m['optical_az']:7.1f}°  | {m['optical_pitch']:7.1f}°  | {az_diff:+6.2f}° | {pitch_diff:+6.2f}°")
    
    # 使用中位数（更鲁棒）
    az_offset_median = np.median(az_diffs)
    pitch_offset_median = np.median(pitch_diffs)
    az_offset_mean = np.mean(az_diffs)
    pitch_offset_mean = np.mean(pitch_diffs)
    
    print(f"\n角度差值统计:")
    print(f"  方位差: 中位数={az_offset_median:+.3f}°, 均值={az_offset_mean:+.3f}°, 标准差={np.std(az_diffs):.3f}°")
    print(f"  俯仰差: 中位数={pitch_offset_median:+.3f}°, 均值={pitch_offset_mean:+.3f}°, 标准差={np.std(pitch_diffs):.3f}°")
    
    # ==================== 位置偏移校准 ====================
    print("\n" + "=" * 80)
    print("第二步：位置偏移校准（带边界约束）")
    print("=" * 80)
    
    # 使用所有远距离数据
    print("\n[方法A] 使用全部远距离数据")
    offset_all, ok_all = calculate_offset_from_measurements(all_far_measurements)
    if ok_all:
        print(f"计算结果: dx={offset_all[0]:.3f}m, dy={offset_all[1]:.3f}m, dz={offset_all[2]:.3f}m")
        dist = np.linalg.norm(offset_all)
        print(f"总偏移距离: {dist:.1f}m")
        validation_all = validate_offset(offset_all, all_far_measurements)
        print(f"重投影角度误差: 均值={validation_all['mean_error_deg']:.3f}°, 最大={validation_all['max_error_deg']:.3f}°")
    
    # ==================== 最终推荐参数 ====================
    print("\n" + "=" * 80)
    print("最终推荐参数")
    print("=" * 80)
    
    # 检查角度差是否随距离变化
    # 计算近距离和远距离的角度差差异
    far_az = np.mean(az_diffs[:len(az_diffs)//2]) if len(az_diffs) > 4 else az_offset_median
    near_az = np.mean(az_diffs[-len(az_diffs)//2:]) if len(az_diffs) > 4 else az_offset_median
    
    if abs(far_az - near_az) > 0.5:
        print("\n⚠️ 角度差随距离变化明显，可能需要位置校准")
        if ok_all and dist < 50:
            print(f"\n推荐使用位置偏移校准:")
            print(f"  dx = {offset_all[0]:.2f} m  (东向)")
            print(f"  dy = {offset_all[1]:.2f} m  (北向)")
            print(f"  dz = {offset_all[2]:.2f} m  (天向)")
        else:
            print(f"\n⚠️ 位置校准结果偏移 {dist:.1f}m 超出合理范围，使用角度校准")
            print(f"\n推荐使用角度偏移校准:")
            print(f"  方位角偏移 = {az_offset_median:+.2f}°")
            print(f"  俯仰角偏移 = {pitch_offset_median:+.2f}°")
    else:
        print("\n✅ 角度差稳定，推荐使用角度偏移校准:")
        print(f"  方位角偏移 = {az_offset_median:+.2f}°")
        print(f"  俯仰角偏移 = {pitch_offset_median:+.2f}°")
    
    # ==================== 中距离数据验证 ====================
    print("\n" + "=" * 80)
    print("中距离数据验证（500-700m，用于检验校准效果）")
    print("=" * 80)
    
    if ok_all and dist < 50:
        # 用位置偏移参数验证中距离数据
        mid_validation = validate_offset(offset_all, middle_measurements)
        print(f"\n用位置偏移参数预测中距离角度误差:")
        print(f"  均值={mid_validation['mean_error_deg']:.3f}°, 最大={mid_validation['max_error_deg']:.3f}°")
    else:
        # 用角度偏移参数验证
        print(f"\n使用角度偏移 ({az_offset_median:+.2f}°, {pitch_offset_median:+.2f}°) 预测中距离:")
        for m in middle_measurements:
            pred_az = m['radar_az'] + az_offset_median
            pred_pitch = m['radar_pitch'] + pitch_offset_median
            az_error = abs(pred_az - m['optical_az'])
            az_error = min(az_error, 360 - az_error)
            pitch_error = abs(pred_pitch - m['optical_pitch'])
            print(f"  {m['radar_range']}m: 方位误差={az_error:.2f}°, 俯仰误差={pitch_error:.2f}°")


if __name__ == "__main__":
    main()
