import math
import numpy as np
import json
import os
from scipy.optimize import least_squares
import socket
import struct
import time
import threading
from datetime import datetime
import uuid

# ==================== 雷达-光电坐标转换 ====================

class RadarOpticalCalibration:
    """雷达-光电标定转换器（支持多点标定）"""
    
    def __init__(self, radar_pos, optical_pos, calib_file="calibration.json"):
        self.radar_pos = np.array(radar_pos, dtype=np.float64)
        self.optical_pos = np.array(optical_pos, dtype=np.float64)
        self.calib_file = calib_file
        
        # 计算偏移向量（光电相对于雷达的位置差）
        self.offset = self.optical_pos - self.radar_pos
        
        # 偏差参数（直接角度偏差）
        self.yaw_bias = 0      # 水平角度偏差（度）
        self.pitch_bias = 0    # 俯仰角度偏差（度）
        
        # 尝试加载已保存的标定参数
        self._load_calibration()
    
    def _load_calibration(self):
        """从文件加载标定参数"""
        if os.path.exists(self.calib_file):
            try:
                with open(self.calib_file, 'r') as f:
                    data = json.load(f)
                    self.yaw_bias = data.get('yaw_bias', 0)
                    self.pitch_bias = data.get('pitch_bias', 0)
                    # 可选：加载位置信息
                    if 'radar_pos' in data:
                        self.radar_pos = np.array(data['radar_pos'])
                    if 'optical_pos' in data:
                        self.optical_pos = np.array(data['optical_pos'])
                        self.offset = self.optical_pos - self.radar_pos
                print(f"[标定] 加载成功: 水平偏差={self.yaw_bias:.3f}°, 俯仰偏差={self.pitch_bias:.3f}°")
            except Exception as e:
                print(f"[标定] 加载失败: {e}")
    
    def _save_calibration(self):
        """保存标定参数到文件"""
        with open(self.calib_file, 'w') as f:
            json.dump({
                'yaw_bias': self.yaw_bias,
                'pitch_bias': self.pitch_bias,
                'radar_pos': self.radar_pos.tolist(),
                'optical_pos': self.optical_pos.tolist(),
                'calibration_time': time.time()
            }, f, indent=2)
        print(f"[标定] 已保存到 {self.calib_file}")
    
    def calibrate_from_reference(self, radar_az, radar_pitch, radar_dist, optical_az, optical_pitch):
        """单点标定（向后兼容）"""
        calc_az, calc_pitch, _ = self._geometric_convert(radar_az, radar_pitch, radar_dist)
        
        self.yaw_bias = optical_az - calc_az
        self.pitch_bias = optical_pitch - calc_pitch
        
        # 归一化
        if self.yaw_bias > 180:
            self.yaw_bias -= 360
        if self.yaw_bias < -180:
            self.yaw_bias += 360
        
        self._save_calibration()
        print(f"[标定] 单点完成: 水平={self.yaw_bias:.3f}°, 俯仰={self.pitch_bias:.3f}°")
        return self.yaw_bias, self.pitch_bias
    
    def calibrate_from_multiple_points(self, points):
        """
        多点标定（推荐）
        
        :param points: 标定点列表，每个点为 (radar_az, radar_pitch, radar_dist, optical_az, optical_pitch)
        :return: (yaw_bias, pitch_bias, rms_error)
        """
        if len(points) < 3:
            print(f"[标定] 错误：至少需要3个点，当前{len(points)}个")
            return None
        
        print(f"[标定] 多点标定开始，使用 {len(points)} 个点")
        
        # 先用第一个点估算初始值
        radar_az0, radar_pitch0, radar_dist0, optical_az0, optical_pitch0 = points[0]
        calc_az0, calc_pitch0, _ = self._geometric_convert(radar_az0, radar_pitch0, radar_dist0)
        yaw_init = optical_az0 - calc_az0
        pitch_init = optical_pitch0 - calc_pitch0
        
        if yaw_init > 180:
            yaw_init -= 360
        if yaw_init < -180:
            yaw_init += 360
        
        # 优化目标函数
        def residuals(params):
            yaw, pitch = params
            total_residual = []
            
            for radar_az, radar_pitch, radar_dist, optical_az, optical_pitch in points:
                target_in_radar = self._radar_to_cartesian(radar_az, radar_pitch, radar_dist)
                target_in_optical = target_in_radar - self.offset
                
                pred_az = math.degrees(math.atan2(target_in_optical[0], target_in_optical[1]))
                if pred_az < 0:
                    pred_az += 360
                
                pred_pitch = math.degrees(math.atan2(
                    target_in_optical[2], 
                    math.sqrt(target_in_optical[0]**2 + target_in_optical[1]**2)
                ))
                
                pred_az += yaw
                pred_pitch += pitch
                pred_az = pred_az % 360
                
                az_diff = (optical_az - pred_az + 180) % 360 - 180
                pitch_diff = optical_pitch - pred_pitch
                
                total_residual.append(az_diff)
                total_residual.append(pitch_diff)
            
            return total_residual
        
        # 最小二乘法优化
        print("[标定] 正在优化求解...")
        result = least_squares(residuals, [yaw_init, pitch_init],
                               bounds=([-180, -90], [180, 90]), verbose=0)
        
        self.yaw_bias, self.pitch_bias = result.x
        
        # 计算残差
        final_residuals = residuals([self.yaw_bias, self.pitch_bias])
        residual_az = final_residuals[0::2]
        residual_pitch = final_residuals[1::2]
        
        rms_az = np.sqrt(np.mean(np.array(residual_az)**2))
        rms_pitch = np.sqrt(np.mean(np.array(residual_pitch)**2))
        rms_total = np.sqrt((rms_az**2 + rms_pitch**2) / 2)
        
        print(f"\n[标定] 完成！")
        print(f"  水平偏差: {self.yaw_bias:.3f}°")
        print(f"  俯仰偏差: {self.pitch_bias:.3f}°")
        print(f"  残差RMS: 方位={rms_az:.3f}°, 俯仰={rms_pitch:.3f}°, 总体={rms_total:.3f}°")
        
        # 打印每个点的误差
        print(f"\n  各点误差:")
        for i, (radar_az, radar_pitch, radar_dist, optical_az, optical_pitch) in enumerate(points):
            pred_az, pred_pitch, _ = self.convert(radar_az, radar_pitch, radar_dist)
            az_err = (optical_az - pred_az + 180) % 360 - 180
            pitch_err = optical_pitch - pred_pitch
            print(f"    点{i+1}: 方位误差={az_err:.3f}°, 俯仰误差={pitch_err:.3f}°")
        
        self._save_calibration()
        return self.yaw_bias, self.pitch_bias, rms_total
    
    def _radar_to_cartesian(self, radar_az, radar_pitch, radar_dist):
        """雷达极坐标转笛卡尔坐标"""
        az_rad = math.radians(radar_az)
        pitch_rad = math.radians(radar_pitch)
        
        x = radar_dist * math.cos(pitch_rad) * math.sin(az_rad)
        y = radar_dist * math.cos(pitch_rad) * math.cos(az_rad)
        z = radar_dist * math.sin(pitch_rad)
        
        return np.array([x, y, z])
    
    def _geometric_convert(self, radar_az, radar_pitch, radar_dist):
        """纯几何转换（不考虑系统误差）"""
        target_in_radar = self._radar_to_cartesian(radar_az, radar_pitch, radar_dist)
        target_in_optical = target_in_radar - self.offset
        
        azimuth = math.degrees(math.atan2(target_in_optical[0], target_in_optical[1]))
        if azimuth < 0:
            azimuth += 360
        
        distance = np.linalg.norm(target_in_optical)
        pitch = math.degrees(math.atan2(target_in_optical[2], 
                                        math.sqrt(target_in_optical[0]**2 + target_in_optical[1]**2)))
        
        return azimuth, pitch, distance
    
    def convert(self, radar_az, radar_pitch, radar_dist):
        """转换雷达目标到光电角度（自动应用标定偏差）"""
        azimuth, pitch, distance = self._geometric_convert(radar_az, radar_pitch, radar_dist)
        
        azimuth += self.yaw_bias
        pitch += self.pitch_bias
        
        azimuth = azimuth % 360
        if azimuth < 0:
            azimuth += 360
        pitch = max(-90, min(90, pitch))
        
        return azimuth, pitch, distance


class RadarOpticalTracker:
    """雷达引导光电跟踪器"""
    
    def __init__(self, radar_pos, optical_pos, calib_file="calibration.json"):
        self.calibrator = RadarOpticalCalibration(radar_pos, optical_pos, calib_file)
        self.is_calibrated = (self.calibrator.yaw_bias != 0 or self.calibrator.pitch_bias != 0)
        
        if self.is_calibrated:
            print(f"[转换器] 已加载标定参数")
        else:
            print(f"[转换器] 未标定，请先运行 calibrate.py")
    
    def calibrate(self, radar_az, radar_pitch, radar_dist, optical_az, optical_pitch):
        """单点标定"""
        self.calibrator.calibrate_from_reference(radar_az, radar_pitch, radar_dist, optical_az, optical_pitch)
        self.is_calibrated = True
    
    def calibrate_multiple(self, points):
        """多点标定"""
        result = self.calibrator.calibrate_from_multiple_points(points)
        if result:
            self.is_calibrated = True
        return result
    
    def track(self, radar_az, radar_pitch, radar_dist):
        """跟踪雷达目标"""
        if not self.is_calibrated:
            raise RuntimeError("请先进行标定！")
        return self.calibrator.convert(radar_az, radar_pitch, radar_dist)
    
    def get_optical_angle(self, radar_az, radar_pitch, radar_dist):
        """获取光电应转角度（别名）"""
        return self.track(radar_az, radar_pitch, radar_dist)


# ==================== 光电客户端 ====================

class OpticalTrackerClient:
    """光电跟踪客户端"""
    
    def __init__(self, optical_ip, local_ip, port=9966):
        self.optical_ip = optical_ip
        self.local_ip = local_ip
        self.port = port
        self.sock = None
        self.seq = 1
        self._connected = False
        
        # 协议常量
        self.START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
        self.STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
        self.PROTOCOL_VERSION = 9002
        
        # 命令字
        self.CMD_EXTENDED_ANGLE = 0x16
        self.CMD_RELEASE = 0x04
        self.CMD_LENS_CONTROL = 0x09
        
        # 接收线程
        self.receive_thread = None
        self.running = False
        self.latest_angle = (None, None)  # (azimuth, pitch)
        self.latest_status = None
    
    def connect(self):
        """建立UDP连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.local_ip, self.port))
            self.sock.settimeout(0.5)
            self._connected = True
            print(f"[光电] 连接成功 {self.optical_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[光电] 连接失败: {e}")
            return False
    
    def start_receive_thread(self):
        """启动接收线程（监听光电上报）"""
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        print("[光电] 接收线程已启动")
    
    def _receive_loop(self):
        """接收光电上报数据"""
        while self.running and self._connected:
            try:
                data, addr = self.sock.recvfrom(2048)
                self._parse_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[光电] 接收错误: {e}")
    
    def _parse_packet(self, data):
        """解析光电上报包"""
        if len(data) < 20:
            return
        
        try:
            # 跳过起始位(4)和协议号(4)，读取包长度
            packet_len = struct.unpack('<I', data[8:12])[0]
            cmd = struct.unpack('<I', data[12:16])[0]
            
            # 0x02: 方位俯仰信息包
            if cmd == 0x02 and len(data) >= 40:
                # 解析方位角和俯仰角
                azimuth = struct.unpack('<d', data[24:32])[0]
                pitch = struct.unpack('<d', data[32:40])[0]
                self.latest_angle = (azimuth, pitch)
            
            # 0x08: 状态扩展信息包（跟踪状态）
            elif cmd == 0x08 and len(data) >= 32:
                work_status = struct.unpack('<I', data[24:28])[0]
                self.latest_status = work_status  # 0空闲 1搜索 2跟踪
                
        except Exception as e:
            pass
    
    def get_current_angle(self):
        """获取最新光电角度"""
        return self.latest_angle
    
    def is_tracking(self):
        """是否处于跟踪状态"""
        return self.latest_status == 2
    
    def _send_packet(self, cmd, data):
        """发送数据包"""
        if not self._connected:
            return False
        
        timestamp = int(time.time() * 1000)
        
        packet = bytearray()
        packet.extend(self.START_BITS)
        packet.extend(struct.pack('<I', self.PROTOCOL_VERSION))
        packet.extend(struct.pack('<I', 20 + len(data)))
        packet.extend(struct.pack('<I', cmd))
        packet.extend(struct.pack('<Q', timestamp))
        packet.extend(data)
        packet.extend(struct.pack('<I', self.seq))
        packet.extend(struct.pack('<I', 0))
        packet.extend(self.STOP_BITS)
        
        try:
            self.sock.sendto(packet, (self.optical_ip, self.port))
            self.seq += 1
            return True
        except Exception as e:
            print(f"[光电] 发送失败: {e}")
            return False
    
    def goto_and_track(self, azimuth, pitch, distance, search_mode=3):
        """转到目标位置并搜索跟踪"""
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<d', 0.0)   # 经度
        data += struct.pack('<d', 0.0)   # 纬度
        data += struct.pack('<d', 0.0)   # 高度
        data += struct.pack('<I', int(distance))  # 显示距离
        data += struct.pack('<I', int(distance))  # 实际距离
        data += struct.pack('<d', float(azimuth))
        data += struct.pack('<d', float(pitch))
        data += struct.pack('<H', 0)     # 用户ID
        data += struct.pack('<B', 0)     # 引导模式
        data += struct.pack('<B', 0)     # 运动方向
        data += struct.pack('<I', search_mode)
        data += struct.pack('<I', 0)
        data += struct.pack('<I', 0)
        data += struct.pack('<I', 0)
        
        success = self._send_packet(self.CMD_EXTENDED_ANGLE, data)
        if success:
            print(f"[光电] 引导: 方位={azimuth:.1f}°, 俯仰={pitch:.1f}°, 距离={distance:.0f}m")
        return success
    
    def zoom_in(self, speed=100, duration=1.5):
        """调大焦距"""
        data = struct.pack('<I', 0)
        data += struct.pack('<I', 0)
        data += struct.pack('<Q', 0)
        data += struct.pack('<I', 0x04)  # 持续推远
        data += struct.pack('<I', speed)
        data += struct.pack('<i', 0)
        data += struct.pack('<i', 0)
        data += struct.pack('<I', 0)
        
        success = self._send_packet(self.CMD_LENS_CONTROL, data)
        if success:
            print(f"[光电] 调焦开始 (速度={speed})")
        
        if duration > 0:
            threading.Timer(duration, self.stop_zoom).start()
        
        return success
    
    def stop_zoom(self):
        """停止镜头运动"""
        data = struct.pack('<I', 0)
        data += struct.pack('<I', 0)
        data += struct.pack('<Q', 0)
        data += struct.pack('<I', 0x00)
        data += struct.pack('<I', 0)
        data += struct.pack('<i', 0)
        data += struct.pack('<i', 0)
        data += struct.pack('<I', 0)
        
        success = self._send_packet(self.CMD_LENS_CONTROL, data)
        if success:
            print("[光电] 调焦停止")
        return success
    
    def release_target(self):
        """释放目标"""
        data = struct.pack('<I', 0)
        data += struct.pack('<Q', 0)
        data += struct.pack('<I', 3)  # 释放
        data += struct.pack('<I', 0)
        data += struct.pack('<I', 0)
        data += struct.pack('<i', 0)
        data += struct.pack('<i', 0)
        data += struct.pack('<I', 0)
        
        return self._send_packet(self.CMD_RELEASE, data)
    
    def close(self):
        """关闭连接"""
        self.running = False
        if self.sock:
            self.sock.close()
            self.sock = None
        self._connected = False
        print("[光电] 连接已关闭")


# ==================== 跟踪会话 ====================

class TrackingSession:
    """跟踪会话管理器"""
    
    def __init__(self, target_id, track_id, target_type, azimuth, pitch, distance):
        self.session_id = str(uuid.uuid4())[:8]
        self.target_id = target_id
        self.track_id = track_id
        self.target_type = target_type
        self.start_time = time.time()
        self.start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        self.initial_azimuth = azimuth
        self.initial_pitch = pitch
        self.initial_distance = distance
        self.is_recording = False
        self.end_time = None
        self.end_reason = None
    
    def end_session(self, reason="released"):
        self.end_time = time.time()
        self.end_reason = reason
        self.is_recording = False
    
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'target_id': self.target_id,
            'track_id': self.track_id,
            'target_type': self.target_type,
            'start_time': self.start_time_str,
            'start_timestamp_ms': int(self.start_time * 1000),
            'initial_azimuth': self.initial_azimuth,
            'initial_pitch': self.initial_pitch,
            'initial_distance': self.initial_distance,
            'end_time': datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if self.end_time else None,
            'duration_sec': round(self.end_time - self.start_time, 2) if self.end_time else None,
            'end_reason': self.end_reason
        }