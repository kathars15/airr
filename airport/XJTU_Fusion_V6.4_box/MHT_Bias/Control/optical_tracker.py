# optical_tracker.py
"""
光电跟踪系统闭环控制模块

功能：
1. 绝对角度控制（带PID闭环）
2. 雷达引导光电跟踪
3. 自动跟踪目标
4. 实时位置反馈

使用示例：
    from optical_tracker import OpticalTracker
    
    tracker = OpticalTracker(device_ip="192.168.0.4", local_ip="192.168.0.9", port=9966)
    tracker.connect()
    
    # 转到指定角度（闭环控制）
    tracker.goto_closed_loop(azimuth=45.0, pitch=10.0, tolerance=0.1, timeout=10)
    
    # 雷达引导跟踪
    tracker.track_radar_target(radar_az=45.0, radar_pitch=10.0, radar_dist=500)
    
    tracker.close()
"""

import socket
import struct
import time
import threading
import math
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from enum import IntEnum


class ControlMode(IntEnum):
    """控制模式"""
    OPEN_LOOP = 0      # 开环（仅发送指令）
    CLOSED_LOOP = 1    # 闭环（PID控制）


class CommandType(IntEnum):
    """命令类型"""
    TRACK_CONTROL = 0x04   # 跟踪控制
    ABSOLUTE_ANGLE = 0x16  # 绝对角度设置


class TrackCommand(IntEnum):
    """跟踪指令"""
    SEARCH_AND_TRACK = 1   # 搜索并自动跟踪
    RELEASE = 3            # 释放目标


class PIDController:
    """PID控制器"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limit=None, integral_limit=None):
        """
        初始化PID控制器
        
        :param kp: 比例系数
        :param ki: 积分系数
        :param kd: 微分系数
        :param output_limit: 输出限制 (min, max)
        :param integral_limit: 积分项限制
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        
    def reset(self):
        """重置PID状态"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        
    def update(self, error, dt=None):
        """
        更新PID计算
        
        :param error: 当前误差（目标值 - 当前值）
        :param dt: 时间间隔（秒），如果不提供则自动计算
        :return: 控制输出
        """
        if dt is None:
            current_time = time.time()
            if self._prev_time is None:
                dt = 0.01
            else:
                dt = max(0.001, current_time - self._prev_time)
            self._prev_time = current_time
            
        # P项
        p_term = self.kp * error
        
        # I项
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral
        
        # D项
        d_term = 0.0
        if dt > 0:
            derivative = (error - self._prev_error) / dt
            d_term = self.kd * derivative
        
        # 总输出
        output = p_term + i_term + d_term
        
        # 输出限幅
        if self.output_limit is not None:
            output = max(self.output_limit[0], min(self.output_limit[1], output))
        
        self._prev_error = error
        return output


@dataclass
class OpticalState:
    """光电设备状态"""
    azimuth: float = 0.0      # 当前方位角（度）
    pitch: float = 0.0        # 当前俯仰角（度）
    timestamp: float = 0.0    # 时间戳
    is_tracking: bool = False  # 是否正在跟踪
    target_azimuth: float = 0.0   # 目标方位角
    target_pitch: float = 0.0     # 目标俯仰角


class OpticalTracker:
    """光电跟踪控制器 - 闭环控制版本"""
    
    # 协议固定值
    START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
    STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
    PROTOCOL_VERSION = 9002
    
    def __init__(
        self,
        # device_ip: str = "192.168.0.4",
        # local_ip: str = "192.168.0.9",
        device_ip: str = "127.0.0.1",
        local_ip: str = "127.0.0.1",
        port: int = 9966,
        control_mode: ControlMode = ControlMode.CLOSED_LOOP,
        auto_release: bool = True
    ):
        """
        初始化光电跟踪器
        
        :param device_ip: 光电设备IP
        :param local_ip: 本地IP
        :param port: 端口
        :param control_mode: 控制模式（开环/闭环）
        :param auto_release: 是否自动释放目标
        """
        self.device_ip = device_ip
        self.local_ip = local_ip
        self.port = port
        self.control_mode = control_mode
        self.auto_release = auto_release
        
        self.sock = None
        self.seq = 1
        self.running = False
        
        # 状态
        self.state = OpticalState()
        
        # PID控制器（方位和俯仰独立）
        self._pid_azimuth = PIDController(kp=1.0, ki=0.1, kd=0.5, 
                                          output_limit=(-30, 30),
                                          integral_limit=50)
        self._pid_pitch = PIDController(kp=1.0, ki=0.05, kd=0.3,
                                        output_limit=(-20, 20),
                                        integral_limit=30)
        
        # 回调函数
        self.on_position_update: Optional[Callable[[float, float], None]] = None
        self.on_target_reached: Optional[Callable[[float, float], None]] = None
        
        # 锁
        self._state_lock = threading.Lock()
        
    def connect(self) -> bool:
        """建立连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.local_ip, self.port))
            self.sock.settimeout(0.5)
            
            self.running = True
            self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.recv_thread.start()
            
            print(f"[连接成功] {self.local_ip}:{self.port} -> {self.device_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[连接失败] {e}")
            return False
            
    def close(self):
        """关闭连接"""
        self.running = False
        if self.sock:
            self.sock.close()
            self.sock = None
        print("[已断开]")
        
    def _receive_loop(self):
        """接收反馈线程"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                self._parse_feedback(data)
            except socket.timeout:
                pass
            except Exception as e:
                if self.running:
                    print(f"[接收错误] {e}")
                    
    def _parse_feedback(self, data: bytes):
        """
        解析反馈数据 - 协议2.2 方位俯仰信息包
        """
        if len(data) < 24:
            return
            
        cmd = struct.unpack('<I', data[12:16])[0]
        
        # 0x02: 方位俯仰信息包
        if cmd == 0x02:
            content = data[24:-8]
            if len(content) >= 56:
                offset = 12
                with self._state_lock:
                    self.state.azimuth = struct.unpack('<d', content[offset:offset+8])[0]
                    self.state.pitch = struct.unpack('<d', content[offset+8:offset+16])[0]
                    self.state.timestamp = time.time()
                
                # 触发回调
                if self.on_position_update:
                    self.on_position_update(self.state.azimuth, self.state.pitch)
                    
    def _send_packet(self, cmd: int, data: bytes):
        """
        发送数据包 - 协议1.3
        """
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
        
        if self.sock:
            self.sock.sendto(packet, (self.device_ip, self.port))
            
        self.seq += 1
        if self.seq > 65535:
            self.seq = 1
            
    def release_target(self):
        """释放目标"""
        data = struct.pack('<I', 0)   # 光电编号
        data += struct.pack('<Q', 0)  # 系统下发时间戳
        data += struct.pack('<I', TrackCommand.RELEASE)  # 释放指令
        data += struct.pack('<I', 0) * 5
        
        self._send_packet(CommandType.TRACK_CONTROL, data)
        time.sleep(0.2)
        print("[释放] 目标已释放")
        
    def start_track(self):
        """开始自动跟踪"""
        data = struct.pack('<I', 0)
        data += struct.pack('<Q', 0)
        data += struct.pack('<I', TrackCommand.SEARCH_AND_TRACK)
        data += struct.pack('<I', 0) * 5
        
        self._send_packet(CommandType.TRACK_CONTROL, data)
        with self._state_lock:
            self.state.is_tracking = True
        print("[跟踪] 开始自动跟踪")
        
    def stop_track(self):
        """停止跟踪"""
        self.release_target()
        with self._state_lock:
            self.state.is_tracking = False
            
    def _send_goto(self, azimuth: float, pitch: float, distance: float = 0):
        """
        发送绝对角度指令（开环）
        
        :param azimuth: 目标方位角（度）
        :param pitch: 目标俯仰角（度）
        :param distance: 目标距离（米）
        """
        # 协议2.4格式
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 系统下发时间戳
        data += struct.pack('<d', 0.0)   # 目标经度
        data += struct.pack('<d', 0.0)   # 目标纬度
        data += struct.pack('<d', 0.0)   # 目标高度
        data += struct.pack('<I', 0)     # 显示距离
        data += struct.pack('<I', int(distance))  # 实际距离
        data += struct.pack('<d', float(azimuth))   # 水平角度
        data += struct.pack('<d', float(pitch))     # 俯仰角度
        data += struct.pack('<H', 0)     # 用户ID
        data += struct.pack('<B', 0)     # 引导模式
        data += struct.pack('<B', 0)     # 目标运动方向
        data += struct.pack('<I', 0)     # 搜索模式
        data += struct.pack('<I', 0)     # 左右搜索视场角
        data += struct.pack('<I', 0)     # 上下搜索视场角
        data += struct.pack('<I', 0)     # 保留
        
        self._send_packet(CommandType.ABSOLUTE_ANGLE, data)
        
        with self._state_lock:
            self.state.target_azimuth = azimuth
            self.state.target_pitch = pitch
            
        print(f"[指令] 目标: 方位={azimuth:.1f}°, 俯仰={pitch:.1f}°, 距离={distance:.0f}m")
        
    def goto_open_loop(self, azimuth: float, pitch: float, distance: float = 0):
        """
        开环控制：只发送一次指令，不进行闭环调节
        
        :param azimuth: 目标方位角（度）
        :param pitch: 目标俯仰角（度）
        :param distance: 目标距离（米）
        """
        self._send_goto(azimuth, pitch, distance)
        
    def goto_closed_loop(
        self,
        azimuth: float,
        pitch: float,
        distance: float = 0,
        tolerance: float = 0.1,
        timeout: float = 15,
        check_interval: float = 0.1,
        max_corrections: int = 50
    ) -> bool:
        """
        闭环控制：持续调节直到到达目标位置
        
        :param azimuth: 目标方位角（度）
        :param pitch: 目标俯仰角（度）
        :param distance: 目标距离（米）
        :param tolerance: 到达容差（度）
        :param timeout: 超时时间（秒）
        :param check_interval: 检查间隔（秒）
        :param max_corrections: 最大修正次数
        :return: 是否成功到达
        """
        print(f"\n[闭环控制] 目标: 方位={azimuth:.1f}°, 俯仰={pitch:.1f}°")
        print(f"           容差={tolerance}°, 超时={timeout}s")
        
        # 重置PID
        self._pid_azimuth.reset()
        self._pid_pitch.reset()
        
        # 发送初始指令
        self._send_goto(azimuth, pitch, distance)
        
        start_time = time.time()
        correction_count = 0
        
        # 等待稳定
        time.sleep(1.0)
        
        while time.time() - start_time < timeout:
            # 获取当前位置
            with self._state_lock:
                current_az = self.state.azimuth
                current_pitch = self.state.pitch
            
            # 计算误差（考虑角度环绕）
            error_az = self._normalize_angle(azimuth - current_az)
            error_pitch = pitch - current_pitch
            
            # 检查是否到达
            if abs(error_az) <= tolerance and abs(error_pitch) <= tolerance:
                print(f"[闭环控制] ✅ 已到达: 方位={current_az:.2f}°, 俯仰={current_pitch:.2f}°")
                if self.on_target_reached:
                    self.on_target_reached(current_az, current_pitch)
                return True
            
            # 计算PID控制量
            control_az = self._pid_azimuth.update(error_az)
            control_pitch = self._pid_pitch.update(error_pitch)
            
            # 计算修正后的目标角度
            corrected_az = azimuth + control_az
            corrected_pitch = pitch + control_pitch
            
            # 归一化
            corrected_az = self._normalize_angle(corrected_az)
            corrected_pitch = max(-45, min(45, corrected_pitch))
            
            # 发送修正指令
            if correction_count < max_corrections:
                self._send_goto(corrected_az, corrected_pitch, distance)
                correction_count += 1
            
            # 打印进度
            print(f"  位置: 方位={current_az:.2f}°, 俯仰={current_pitch:.2f}°, "
                  f"Δ方位={error_az:+.2f}°, Δ俯仰={error_pitch:+.2f}°, "
                  f"修正#{correction_count}")
            
            time.sleep(check_interval)
        
        print(f"[闭环控制] ⚠️ 超时未到达: 最终误差 Δ方位={error_az:+.2f}°, Δ俯仰={error_pitch:+.2f}°")
        return False
    
    def goto_adaptive(
        self,
        azimuth: float,
        pitch: float,
        distance: float = 0,
        tolerance: float = 0.1,
        timeout: float = 15
    ) -> bool:
        """
        自适应控制：根据当前控制模式选择开环或闭环
        
        :param azimuth: 目标方位角（度）
        :param pitch: 目标俯仰角（度）
        :param distance: 目标距离（米）
        :param tolerance: 到达容差（度）
        :param timeout: 超时时间（秒）
        :return: 是否成功
        """
        if self.control_mode == ControlMode.CLOSED_LOOP:
            return self.goto_closed_loop(azimuth, pitch, distance, tolerance, timeout)
        else:
            self.goto_open_loop(azimuth, pitch, distance)
            return True
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到 [0, 360)"""
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle
    
    def get_current_position(self) -> Tuple[float, float]:
        """获取当前位置"""
        with self._state_lock:
            return self.state.azimuth, self.state.pitch
    
    def wait_for_position(
        self,
        target_az: float,
        target_pitch: float,
        tolerance: float = 0.5,
        timeout: float = 10,
        callback: Optional[Callable[[float, float, float, float], None]] = None
    ) -> bool:
        """
        等待到达目标位置（只等待，不发送控制指令）
        
        :param target_az: 目标方位角（度）
        :param target_pitch: 目标俯仰角（度）
        :param tolerance: 容差（度）
        :param timeout: 超时时间（秒）
        :param callback: 回调函数，参数为(当前方位, 当前俯仰, 误差方位, 误差俯仰)
        :return: 是否到达
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_az, current_pitch = self.get_current_position()
            
            error_az = self._normalize_angle(target_az - current_az)
            error_pitch = target_pitch - current_pitch
            
            if abs(error_az) <= tolerance and abs(error_pitch) <= tolerance:
                print(f"[等待] ✅ 已到达: 方位={current_az:.2f}°, 俯仰={current_pitch:.2f}°")
                return True
            
            if callback:
                callback(current_az, current_pitch, error_az, error_pitch)
            
            time.sleep(0.2)
        
        print(f"[等待] ⚠️ 超时: Δ方位={error_az:+.2f}°, Δ俯仰={error_pitch:+.2f}°")
        return False
    
    def set_pid_params(self, kp_az: float = None, ki_az: float = None, kd_az: float = None,
                       kp_pitch: float = None, ki_pitch: float = None, kd_pitch: float = None):
        """
        设置PID参数
        
        :param kp_az: 方位角比例系数
        :param ki_az: 方位角积分系数
        :param kd_az: 方位角微分系数
        :param kp_pitch: 俯仰角比例系数
        :param ki_pitch: 俯仰角积分系数
        :param kd_pitch: 俯仰角微分系数
        """
        if kp_az is not None:
            self._pid_azimuth.kp = kp_az
        if ki_az is not None:
            self._pid_azimuth.ki = ki_az
        if kd_az is not None:
            self._pid_azimuth.kd = kd_az
            
        if kp_pitch is not None:
            self._pid_pitch.kp = kp_pitch
        if ki_pitch is not None:
            self._pid_pitch.ki = ki_pitch
        if kd_pitch is not None:
            self._pid_pitch.kd = kd_pitch
            
        print(f"[PID] 方位: Kp={self._pid_azimuth.kp}, Ki={self._pid_azimuth.ki}, Kd={self._pid_azimuth.kd}")
        print(f"[PID] 俯仰: Kp={self._pid_pitch.kp}, Ki={self._pid_pitch.ki}, Kd={self._pid_pitch.kd}")
    
    def set_control_mode(self, mode: ControlMode):
        """设置控制模式"""
        self.control_mode = mode
        print(f"[模式] {'闭环控制' if mode == ControlMode.CLOSED_LOOP else '开环控制'}")
    
    def reset(self):
        """重置控制器状态"""
        self._pid_azimuth.reset()
        self._pid_pitch.reset()


class RadarGuidedTracker:
    """雷达引导光电跟踪器"""
    
    def __init__(
        self,
        optical_tracker: OpticalTracker,
        offset_x: float = 0,
        offset_y: float = 0,
        offset_z: float = 0,
        az_bias: float = 0,
        pitch_bias: float = 0
    ):
        """
        初始化雷达引导跟踪器
        
        :param optical_tracker: 光电跟踪器实例
        :param offset_x: 光电相对于雷达的东向偏移（米）
        :param offset_y: 光电相对于雷达的北向偏移（米）
        :param offset_z: 光电相对于雷达的高度偏移（米）
        :param az_bias: 水平角度偏差（度）
        :param pitch_bias: 俯仰角度偏差（度）
        """
        self.optical = optical_tracker
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.az_bias = az_bias
        self.pitch_bias = pitch_bias
        
    def set_calibration(
        self,
        offset_x: float = None,
        offset_y: float = None,
        offset_z: float = None,
        az_bias: float = None,
        pitch_bias: float = None
    ):
        """设置标定参数"""
        if offset_x is not None:
            self.offset_x = offset_x
        if offset_y is not None:
            self.offset_y = offset_y
        if offset_z is not None:
            self.offset_z = offset_z
        if az_bias is not None:
            self.az_bias = az_bias
        if pitch_bias is not None:
            self.pitch_bias = pitch_bias
            
        print(f"[标定] 偏移({self.offset_x},{self.offset_y},{self.offset_z})m, "
              f"偏差({self.az_bias},{self.pitch_bias})°")
    
    def radar_to_optical(self, radar_az: float, radar_pitch: float, radar_dist: float) -> Tuple[float, float]:
        """
        雷达坐标转光电角度
        
        :param radar_az: 雷达方位角（度）
        :param radar_pitch: 雷达俯仰角（度）
        :param radar_dist: 雷达距离（米）
        :return: (光电方位角, 光电俯仰角)
        """
        # 雷达坐标系下的目标位置
        radar_az_rad = math.radians(radar_az)
        radar_pitch_rad = math.radians(radar_pitch)
        
        radar_x = radar_dist * math.cos(radar_pitch_rad) * math.sin(radar_az_rad)
        radar_y = radar_dist * math.cos(radar_pitch_rad) * math.cos(radar_az_rad)
        radar_z = radar_dist * math.sin(radar_pitch_rad)
        
        # 转换到光电坐标系
        opt_x = radar_x - self.offset_x
        opt_y = radar_y - self.offset_y
        opt_z = radar_z - self.offset_z
        
        # 计算光电角度
        opt_az = math.degrees(math.atan2(opt_x, opt_y))
        if opt_az < 0:
            opt_az += 360
            
        opt_dist = math.sqrt(opt_x**2 + opt_y**2)
        opt_pitch = math.degrees(math.atan2(opt_z, opt_dist))
        
        # 加上偏差
        opt_az += self.az_bias
        opt_pitch += self.pitch_bias
        
        # 归一化
        opt_az = opt_az % 360
        opt_pitch = max(-45, min(45, opt_pitch))
        
        return opt_az, opt_pitch
    
    def track_radar_target(
        self,
        radar_az: float,
        radar_pitch: float,
        radar_dist: float,
        tolerance: float = 0.1,
        timeout: float = 15
    ) -> bool:
        """
        跟踪雷达目标（闭环控制）
        
        :param radar_az: 雷达方位角（度）
        :param radar_pitch: 雷达俯仰角（度）
        :param radar_dist: 雷达距离（米）
        :param tolerance: 到达容差（度）
        :param timeout: 超时时间（秒）
        :return: 是否成功
        """
        print(f"\n[雷达引导] 雷达目标: 方位={radar_az:.1f}°, 俯仰={radar_pitch:.1f}°, 距离={radar_dist:.0f}m")
        
        # 坐标转换
        opt_az, opt_pitch = self.radar_to_optical(radar_az, radar_pitch, radar_dist)
        print(f"[雷达引导] 光电目标: 方位={opt_az:.1f}°, 俯仰={opt_pitch:.1f}°")
        
        # 释放当前目标
        if self.optical.auto_release:
            self.optical.release_target()
            time.sleep(0.5)
        
        # 闭环控制到目标位置
        success = self.optical.goto_closed_loop(opt_az, opt_pitch, radar_dist, tolerance, timeout)
        
        if success:
            # 开始自动跟踪
            time.sleep(0.5)
            self.optical.start_track()
            
        return success


# ========== 使用示例 ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="光电跟踪控制器")
    parser.add_argument("--mode", type=str, choices=["test", "track", "calibrate"], 
                        default="calibrate", help="运行模式")
    parser.add_argument("--device_ip", type=str, default="127.0.0.1", help="光电设备IP")
    parser.add_argument("--local_ip", type=str, default="127.0.0.1", help="本地IP")
    parser.add_argument("--port", type=int, default=9966, help="端口")
    
    args = parser.parse_args()
    
    # 创建跟踪器
    tracker = OpticalTracker(
        device_ip=args.device_ip,
        local_ip=args.local_ip,
        port=args.port,
        control_mode=ControlMode.CLOSED_LOOP
    )
    
    def on_position_update(az, pitch):
        """位置更新回调"""
        pass  # 可以在这里添加日志或显示
    
    def on_target_reached(az, pitch):
        """到达目标回调"""
        print(f"🎯 已到达目标位置: ({az:.2f}°, {pitch:.2f}°)")
    
    tracker.on_position_update = on_position_update
    tracker.on_target_reached = on_target_reached
    try:
        if not tracker.connect():
            print("连接失败")
            exit(1)
            
        time.sleep(1)
        
        if args.mode == "test":
            # 测试模式：转到几个固定角度
            print("\n" + "=" * 60)
            print("测试模式 - 闭环控制")
            print("=" * 60)
            
            test_targets = [
                (0, 0, 0, "原点"),
                (45, 10, 0, "东北方向"),
                (90, 20, 0, "正东方向"),
                (135, 30, 0, "东南方向"),
                (180, 0, 0, "正南方向"),
                (225, -10, 0, "西南方向"),
                (270, -20, 0, "正西方向"),
                (315, -30, 0, "西北方向"),
                (0, 0, 0, "回到原点"),
            ]
            
            for az, pitch, dist, desc in test_targets:
                input(f"\n按 Enter 转到 {desc} (方位={az}°, 俯仰={pitch}°)...")
                tracker.goto_closed_loop(az, pitch, dist, tolerance=0.1, timeout=10)
                
        elif args.mode == "track":
            # 跟踪模式：雷达引导
            guided = RadarGuidedTracker(tracker)
            
            # 设置标定参数（根据实际情况修改）
            guided.set_calibration(
                offset_x=0, offset_y=0, offset_z=0,
                az_bias=0, pitch_bias=0
            )
            
            # 模拟雷达目标
            test_targets = [
                (45, 10, 500),
                (90, 20, 300),
                (135, 30, 600),
                (180, 40, 500),
                (225, 50, 400),
                (270, 60, 300),
                (315, 70, 200),
                (0, 0, 100),
            ]
            
            for az, pitch, dist in test_targets:
                input(f"\n按 Enter 跟踪雷达目标: 方位={az}°, 俯仰={pitch}°, 距离={dist}m...")
                guided.track_radar_target(az, pitch, dist, tolerance=0.2, timeout=15)
                
        elif args.mode == "calibrate":
            # 校准模式
            print("\n" + "=" * 60)
            print("校准模式 - 手动校准")
            print("=" * 60)
            print("请将光电对准已知目标，输入目标角度进行校准")
            
            while True:
                cmd = input("\n输入命令 (a:设置方位偏差, p:设置俯仰偏差, q:退出): ")
                
                if cmd == 'q':
                    break
                elif cmd == 'a':
                    target = float(input("请输入目标方位角: "))
                    current = tracker.get_current_position()[0]
                    bias = target - current
                    print(f"当前方位: {current:.2f}°, 目标: {target:.2f}°, 建议偏差: {bias:.2f}°")
                elif cmd == 'p':
                    target = float(input("请输入目标俯仰角: "))
                    current = tracker.get_current_position()[1]
                    bias = target - current
                    print(f"当前俯仰: {current:.2f}°, 目标: {target:.2f}°, 建议偏差: {bias:.2f}°")
                    
        print("\n按 Ctrl+C 退出...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        tracker.close()