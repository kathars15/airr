import numpy as np
def radar_to_optical_prediction(radar_az, radar_pitch, radar_range, offset):
    """
    根据雷达测量和标定偏移，预测光学测量值
    
    Args:
        radar_az, radar_pitch, radar_range: 雷达测量
        offset: {'dx', 'dy', 'dz'} 光学相对于雷达的位置
    
    Returns:
        dict: 预测的光学测量 {'azimuth', 'pitch', 'range'}
    """
    # 光学传感器位置（雷达坐标系）
    optical_pos = np.array([offset['dx'], offset['dy'], offset['dz']])
    
    # 雷达测量转目标位置
    az_rad = np.radians(radar_az)
    pitch_rad = np.radians(radar_pitch)
    cos_pitch = np.cos(pitch_rad)
    
    target = np.array([
        radar_range * cos_pitch * np.sin(az_rad),
        radar_range * cos_pitch * np.cos(az_rad),
        radar_range * np.sin(pitch_rad)
    ])
    
    # 光学视线
    optical_vector = target - optical_pos
    opt_range = np.linalg.norm(optical_vector)
    
    # 转光学极坐标
    opt_az = np.degrees(np.arctan2(optical_vector[0], optical_vector[1])) % 360.0
    opt_pitch = np.degrees(np.arcsin(optical_vector[2] / opt_range)) if opt_range > 0 else 0
    
    return {
        'azimuth': opt_az,
        'pitch': opt_pitch,
        'range': opt_range
    }


# 使用示例
offset = {'dx': 173.33, 'dy': -116.90, 'dz': 17.08}

# 第2组数据验证
pred = radar_to_optical_prediction(142.84, 5.3, 800, offset)
print(f"预测: 方位={pred['azimuth']:.1f}°, 俯仰={pred['pitch']:.1f}°, 距离={pred['range']:.1f}m")
print(f"实际: 方位=122.7°, 俯仰=4.8°")
print(f"距离转换: 雷达659m → 光学{pred['range']:.1f}m")