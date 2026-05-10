import math
import numpy as np

class CoordinateTransformer:
    """雷达到光电的坐标转换器"""
    
    def __init__(self, radar_pos, optical_pos):
        """
        初始化转换器
        
        :param radar_pos: 雷达位置 (x, y, z) 单位：米
        :param optical_pos: 光电位置 (x, y, z) 单位：米
        """
        self.radar_pos = np.array(radar_pos).reshape(3, 1)
        self.optical_pos = np.array(optical_pos).reshape(3, 1)
        self.offset = self.optical_pos - self.radar_pos  # 光电相对于雷达的偏移
        
    def radar_polar_to_enu(self, distance, azimuth, pitch):
        """
        雷达极坐标 → ENU坐标（以雷达为原点）
        
        :param distance: 距离（米）
        :param azimuth: 方位角（度，真北为0，顺时针）
        :param pitch: 俯仰角（度，水平为0，向上为正）
        :return: (x, y, z) ENU坐标
        """
        az_rad = math.radians(azimuth)
        pitch_rad = math.radians(pitch)
        
        x = distance * math.cos(pitch_rad) * math.sin(az_rad)  # 东向
        y = distance * math.cos(pitch_rad) * math.cos(az_rad)  # 北向
        z = distance * math.sin(pitch_rad)                      # 天向
        
        return np.array([x, y, z]).reshape(3, 1)
    
    def enu_to_optical_polar(self, target_enu):
        """
        目标ENU坐标 → 光电极坐标（以光电为原点）
        
        :param target_enu: 目标绝对ENU坐标 (x, y, z)
        :return: (azimuth, pitch, distance) 光电应转到的角度和距离
        """
        # 目标相对于光电的位置
        relative_pos = target_enu - self.optical_pos
        
        x = relative_pos[0, 0]
        y = relative_pos[1, 0]
        z = relative_pos[2, 0]
        
        # 计算距离
        distance = math.sqrt(x**2 + y**2 + z**2)
        
        # 计算方位角（真北为0，顺时针）
        azimuth = math.degrees(math.atan2(x, y))
        if azimuth < 0:
            azimuth += 360
            
        # 计算俯仰角
        horizontal_dist = math.sqrt(x**2 + y**2)
        pitch = math.degrees(math.atan2(z, horizontal_dist))
        
        return azimuth, pitch, distance
    
    def radar_to_optical(self, distance, azimuth, pitch):
        """
        完整转换：雷达极坐标 → 光电极坐标
        
        :param distance: 雷达测量的距离（米）
        :param azimuth: 雷达测量的方位角（度）
        :param pitch: 雷达测量的俯仰角（度）
        :return: (光电方位角, 光电俯仰角, 光电距离)
        """
        # 步骤1：雷达极坐标 → 目标ENU坐标
        target_enu = self.radar_polar_to_enu(distance, azimuth, pitch)
        target_enu = target_enu + self.radar_pos  # 转为绝对坐标
        
        # 步骤2：目标ENU坐标 → 光电极坐标
        opt_az, opt_pitch, opt_dist = self.enu_to_optical_polar(target_enu)
        
        return opt_az, opt_pitch, opt_dist


# ========== 使用示例 ==========

# 雷达位置（ENU坐标系原点，设为 (0,0,0)）
radar_position = (0, 0, 0)

# 光电位置（相对于雷达，单位：米）
# 例如：光电在雷达东侧100米，北侧50米，高2米
optical_position = (100, 50, 2)  # (东, 北, 天)

# 创建转换器
transformer = CoordinateTransformer(radar_position, optical_position)

# 雷达测量值
radar_distance = 500      # 500米
radar_azimuth = 45        # 45度
radar_pitch = 10          # 10度

# 转换到光电角度
opt_az, opt_pitch, opt_dist = transformer.radar_to_optical(
    radar_distance, radar_azimuth, radar_pitch
)
s
print(f"雷达测量: 距离={radar_distance}m, 方位={radar_azimuth}°, 俯仰={radar_pitch}°")
print(f"光电应转: 方位={opt_az:.1f}°, 俯仰={opt_pitch:.1f}°, 距离={opt_dist:.0f}m")