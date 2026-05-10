"""
标定数据采集器
用于采集雷达原始点迹和光电角度的匹配数据
"""

import time
import csv
import os
import threading
import numpy as np
from collections import deque
from datetime import datetime


class CalibrationDataCollector:
    """标定数据采集器 - 用于雷达-光电标定"""
    
    def __init__(self, radar_callback=None, optical_callback=None):
        """
        :param radar_callback: 获取雷达原始点迹的回调函数
        :param optical_callback: 获取光电角度的回调函数
        """
        self.radar_callback = radar_callback
        self.optical_callback = optical_callback
        
        # 数据缓冲区
        self.radar_buffer = deque(maxlen=1000)   # (timestamp, az, pitch, dist)
        self.optical_buffer = deque(maxlen=1000) # (timestamp, az, pitch)
        
        # 采集状态
        self.is_collecting = False
        self.collect_thread = None
        
        # 文件
        self.radar_file = "calib_radar_raw.csv"
        self.optical_file = "calib_optical_raw.csv"
    
    def add_radar_point(self, timestamp, azimuth, pitch, distance, target_id=None):
        """添加雷达原始点迹（从你的点迹解析函数调用）"""
        self.radar_buffer.append({
            'timestamp': timestamp,
            'azimuth': azimuth,
            'pitch': pitch,
            'distance': distance,
            'target_id': target_id
        })
        
        # 实时保存到CSV
        if self.is_collecting:
            with open(self.radar_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, azimuth, pitch, distance, target_id])
    
    def add_optical_angle(self, timestamp, azimuth, pitch):
        """添加光电角度（从0x02上报获取）"""
        self.optical_buffer.append({
            'timestamp': timestamp,
            'azimuth': azimuth,
            'pitch': pitch
        })
        
        # 实时保存到CSV
        if self.is_collecting:
            with open(self.optical_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, azimuth, pitch])
    
    def start_collection(self, duration=None):
        """
        开始采集标定数据
        
        :param duration: 采集持续时间（秒），None表示手动停止
        """
        # 初始化文件
        with open(self.radar_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'azimuth', 'pitch', 'distance', 'target_id'])
        
        with open(self.optical_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'azimuth', 'pitch'])
        
        self.is_collecting = True
        self.radar_buffer.clear()
        self.optical_buffer.clear()
        
        print(f"\n[采集] 开始采集标定数据...")
        print(f"  雷达数据 -> {self.radar_file}")
        print(f"  光电数据 -> {self.optical_file}")
        print("  请让目标在视场内飞行，光电自动跟踪")
        
        if duration:
            threading.Timer(duration, self.stop_collection).start()
    
    def stop_collection(self):
        """停止采集"""
        self.is_collecting = False
        print(f"\n[采集] 停止采集")
        print(f"  雷达点迹数: {len(self.radar_buffer)}")
        print(f"  光电角度数: {len(self.optical_buffer)}")
    
    def match_points(self, time_window=0.2):
        """
        匹配雷达和光电数据点
        
        :param time_window: 时间匹配窗口（秒）
        :return: 匹配的点列表 [(radar_az, radar_pitch, radar_dist, opt_az, opt_pitch), ...]
        """
        matched_points = []
        
        # 读取文件中的完整数据
        radar_data = self._read_csv(self.radar_file)
        optical_data = self._read_csv(self.optical_file)
        
        if len(radar_data) == 0 or len(optical_data) == 0:
            print("[匹配] 没有数据，请先采集")
            return []
        
        print(f"[匹配] 雷达数据: {len(radar_data)}条, 光电数据: {len(optical_data)}条")
        
        for opt in optical_data:
            opt_time = opt['timestamp']
            
            # 找时间最接近的雷达点
            closest = min(radar_data, key=lambda x: abs(x['timestamp'] - opt_time))
            time_diff = abs(closest['timestamp'] - opt_time)
            
            if time_diff < time_window:
                matched_points.append((
                    closest['azimuth'],   # 雷达方位角
                    closest['pitch'],     # 雷达俯仰角
                    closest['distance'],  # 雷达距离
                    opt['azimuth'],       # 光电方位角
                    opt['pitch']          # 光电俯仰角
                ))
        
        print(f"[匹配] 匹配到 {len(matched_points)} 个点")
        
        # 去重和滤波
        matched_points = self._filter_outliers(matched_points)
        
        return matched_points
    
    def _read_csv(self, filename):
        """读取CSV文件"""
        data = []
        if not os.path.exists(filename):
            return data
        
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = True
            for row in reader:
                if header:
                    header = False
                    continue
                if len(row) >= 3:
                    data.append({
                        'timestamp': float(row[0]),
                        'azimuth': float(row[1]),
                        'pitch': float(row[2]),
                        'distance': float(row[3]) if len(row) > 3 else 0,
                        'target_id': row[4] if len(row) > 4 else None
                    })
        return data
    
    def _filter_outliers(self, points, sigma=2.0):
        """剔除异常点"""
        if len(points) < 3:
            return points
        
        arr = np.array(points)
        
        # 计算各维度的均值和标准差
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        
        filtered = []
        for p in points:
            z_scores = np.abs((np.array(p) - means) / (stds + 1e-6))
            if np.all(z_scores < sigma):
                filtered.append(p)
        
        if len(filtered) < len(points):
            print(f"[匹配] 剔除 {len(points)-len(filtered)} 个异常点")
        
        return filtered
    
    def get_summary(self):
        """获取采集摘要"""
        radar_data = self._read_csv(self.radar_file)
        optical_data = self._read_csv(self.optical_file)
        
        print(f"\n[采集摘要]")
        print(f"  雷达文件: {self.radar_file} ({len(radar_data)}条)")
        print(f"  光电文件: {self.optical_file} ({len(optical_data)}条)")
        
        if len(radar_data) > 0:
            times = [d['timestamp'] for d in radar_data]
            print(f"  雷达时间范围: {min(times):.1f} - {max(times):.1f}")
        if len(optical_data) > 0:
            times = [d['timestamp'] for d in optical_data]
            print(f"  光电时间范围: {min(times):.1f} - {max(times):.1f}")


class IntegratedCollector:
    """
    集成采集器 - 集成到你的MHT进程中
    当光电跟踪成功时自动记录数据
    """
    
    def __init__(self, optical_client, output_dir="calibration_data"):
        self.optical_client = optical_client
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.radar_file = os.path.join(output_dir, "radar_points.csv")
        self.optical_file = os.path.join(output_dir, "optical_angles.csv")
        
        # 是否在记录
        self.is_recording = False
        self.current_target_id = None
    
    def start_recording(self, target_id=None):
        """开始记录（在跟踪成功时调用）"""
        self.is_recording = True
        self.current_target_id = target_id
        print(f"[采集] 开始记录，目标ID: {target_id}")
    
    def stop_recording(self):
        """停止记录"""
        self.is_recording = False
        self.current_target_id = None
        print(f"[采集] 停止记录")
    
    def record_radar_point(self, timestamp, azimuth, pitch, distance, target_id=None):
        """记录雷达原始点迹"""
        if not self.is_recording:
            return
        
        with open(self.radar_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, azimuth, pitch, distance, target_id, self.current_target_id])
    
    def record_optical_angle(self, timestamp, azimuth, pitch):
        """记录光电角度"""
        if not self.is_recording:
            return
        
        with open(self.optical_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, azimuth, pitch, self.current_target_id])
    
    def init_files(self):
        """初始化CSV文件"""
        with open(self.radar_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'azimuth', 'pitch', 'distance', 'target_id', 'tracked_target_id'])
        
        with open(self.optical_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'azimuth', 'pitch', 'tracked_target_id'])