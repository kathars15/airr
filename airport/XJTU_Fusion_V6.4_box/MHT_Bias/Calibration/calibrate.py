"""
标定执行脚本
从采集的数据中匹配点并执行标定
"""

import pandas as pd
import numpy as np
import os
from bias import RadarOpticalTracker


def load_calibration_data(radar_file="calibration_data/radar_points.csv",
                          optical_file="calibration_data/optical_angles.csv",
                          time_window=0.2):
    """
    加载采集的数据并匹配
    
    :param radar_file: 雷达原始点迹文件
    :param optical_file: 光电角度文件
    :param time_window: 时间匹配窗口（秒）
    :return: 匹配的点列表
    """
    if not os.path.exists(radar_file):
        print(f"[错误] 雷达文件不存在: {radar_file}")
        return []
    
    if not os.path.exists(optical_file):
        print(f"[错误] 光电文件不存在: {optical_file}")
        return []
    
    # 读取数据
    radar_df = pd.read_csv(radar_file)
    optical_df = pd.read_csv(optical_file)
    
    print(f"[加载] 雷达数据: {len(radar_df)}条")
    print(f"[加载] 光电数据: {len(optical_df)}条")
    
    # 时间匹配
    matched_points = []
    
    for _, opt_row in optical_df.iterrows():
        opt_time = opt_row['timestamp']
        opt_az = opt_row['azimuth']
        opt_pitch = opt_row['pitch']
        
        # 找时间最接近的雷达点
        time_diff = np.abs(radar_df['timestamp'] - opt_time)
        closest_idx = time_diff.idxmin()
        closest_time_diff = time_diff.iloc[closest_idx]
        
        if closest_time_diff < time_window:
            radar_row = radar_df.iloc[closest_idx]
            
            matched_points.append((
                float(radar_row['azimuth']),   # 雷达方位角
                float(radar_row['pitch']),     # 雷达俯仰角
                float(radar_row['distance']),  # 雷达距离
                float(opt_az),                 # 光电方位角
                float(opt_pitch)               # 光电俯仰角
            ))
    
    print(f"[匹配] 匹配到 {len(matched_points)} 个点")
    
    return matched_points


def calibrate_from_files(radar_pos, optical_pos, 
                         radar_file="calibration_data/radar_points.csv",
                         optical_file="calibration_data/optical_angles.csv",
                         calib_file="radar_optical_calibration.json"):
    """
    从采集的文件进行标定
    """
    # 加载匹配点
    points = load_calibration_data(radar_file, optical_file)
    
    if len(points) < 3:
        print(f"[错误] 标定点不足，需要至少3个点，当前{len(points)}个")
        return None
    
    # 创建转换器并标定
    tracker = RadarOpticalTracker(
        radar_pos=radar_pos,
        optical_pos=optical_pos,
        calib_file=calib_file
    )
    
    # 执行多点标定
    result = tracker.calibrate_multiple(points)
    
    return result


def verify_calibration(calib_file="radar_optical_calibration.json"):
    """验证标定效果"""
    from bias import RadarOpticalCalibration
    
    cal = RadarOpticalCalibration(
        radar_pos=(0, 0, 0),
        optical_pos=(0, 0, 0),
        calib_file=calib_file
    )
    
    print(f"\n[验证] 当前标定参数:")
    print(f"  水平偏差: {cal.yaw_bias:.3f}°")
    print(f"  俯仰偏差: {cal.pitch_bias:.3f}°")
    
    # 测试转换
    print(f"\n  测试转换（雷达角度 -> 光电角度）:")
    test_points = [
        (45, 5, 100),
        (90, 3, 200),
        (135, 2, 150),
    ]
    
    for az, pitch, dist in test_points:
        opt_az, opt_pitch, opt_dist = cal.convert(az, pitch, dist)
        print(f"    雷达({az}°, {pitch}°) -> 光电({opt_az:.1f}°, {opt_pitch:.1f}°)")


def main():
    """主函数"""
    print("=" * 60)
    print("雷达-光电标定程序")
    print("=" * 60)
    
    # 配置参数（根据实际修改）
    RADAR_POS = (0, 0, 0)        # 雷达位置（米）
    OPTICAL_POS = (2, 1, 0)      # 光电位置（米），根据实际安装位置修改
    
    # 标定文件路径
    RADAR_FILE = "calibration_data/radar_points.csv"
    OPTICAL_FILE = "calibration_data/optical_angles.csv"
    CALIB_FILE = "radar_optical_calibration.json"
    
    # 检查是否有采集数据
    import os
    if not os.path.exists(RADAR_FILE) or not os.path.exists(OPTICAL_FILE):
        print("\n[提示] 未找到标定数据文件！")
        print("请先运行主程序并开启数据采集模式，让目标飞行几分钟。")
        print(f"需要以下文件:")
        print(f"  - {RADAR_FILE}")
        print(f"  - {OPTICAL_FILE}")
        return
    
    # 执行标定
    print("\n[标定] 开始标定...")
    result = calibrate_from_files(
        radar_pos=RADAR_POS,
        optical_pos=OPTICAL_POS,
        radar_file=RADAR_FILE,
        optical_file=OPTICAL_FILE,
        calib_file=CALIB_FILE
    )
    
    if result:
        yaw_bias, pitch_bias, rms = result
        print(f"\n[完成] 标定成功！")
        print(f"  结果已保存到: {CALIB_FILE}")
    else:
        print(f"\n[失败] 标定失败，请确保采集了足够的数据")
    
    # 验证
    verify_calibration(CALIB_FILE)


if __name__ == "__main__":
    main()