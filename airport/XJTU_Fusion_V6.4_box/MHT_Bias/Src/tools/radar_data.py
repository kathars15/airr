import pandas as pd
import numpy as np

# 读取保存的点迹数据
df = pd.read_csv('radar_calibration_data.csv', 
                  names=['timestamp', 'azimuth', 'pitch', 'range', 'target_id', 'speed'])

# 假设参考物在方位45°、距离120米附近
# 先筛选出参考物的数据
reference = df[
    (df['azimuth'].between(40, 50)) &      # 方位范围
    (df['range'].between(100, 150))         # 距离范围
]

# 取均值
radar_az_mean = reference['azimuth'].mean()
radar_pitch_mean = reference['pitch'].mean()
radar_range_mean = reference['range'].mean()

print(f"雷达测量均值:")
print(f"  方位角: {radar_az_mean:.2f}°")
print(f"  俯仰角: {radar_pitch_mean:.2f}°")
print(f"  距离: {radar_range_mean:.1f}m")
print(f"  标准差: 方位={reference['azimuth'].std():.2f}°, 距离={reference['range'].std():.1f}m")