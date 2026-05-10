# op.py
"""
光电跟踪测试程序
功能：手动输入目标角度、距离，先转到位置，到位后再开始搜索跟踪
"""

import socket
import struct
import time
import math
import threading
import os
import cv2
from datetime import datetime

from core.optical_measurement_log import append_optical_measurement

# ==================== 协议常量 ====================
START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
PROTOCOL_VERSION = 9002

# 命令字
CMD_ABSOLUTE_ANGLE = 0x16   # 设置光电目址扩展信息包
CMD_GET_STATUS = 0x01       # 获取设备状态
CMD_GET_ANGLE = 0x02        # 获取方位俯仰信息
CMD_SEARCH_TRACK = 0x04     # 搜索跟踪命令

# 搜索模式
SEARCH_MODE_NONE = 0        # 不开启搜索
SEARCH_MODE_LEFT_RIGHT = 1  # 左右搜索并自动锁定
SEARCH_MODE_UP_DOWN = 2     # 上下搜索并自动锁定
SEARCH_MODE_CURRENT = 3     # 在当前视场搜索并自动锁定
# ==================== 录制类  ====================
class RTSPRecorderController:
    def __init__(self, rtsp_urls, save_dir, show_window=False):
        self.rtsp_urls = rtsp_urls
        self.save_dir = save_dir
        self.show_window = show_window

        self.recording_event = threading.Event()
        self.stop_event = threading.Event()
        self.recording_lock = threading.Lock()

        self.current_tag = "auto"
        self.threads = []

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    def ensure_dir(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def open_rtsp(self, url: str):
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    def create_writer(self, save_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    def start(self):
        self.ensure_dir(self.save_dir)

        for item in self.rtsp_urls:
            t = threading.Thread(
                target=self._record_rtsp,
                args=(item["name"], item["url"]),
                daemon=True
            )
            t.start()
            self.threads.append(t)

            # print("[录像] 所有 RTSP 线程已启动")

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2)

    def start_recording(self, tag="track"):
        with self.recording_lock:
            self.current_tag = str(tag)
            if not self.recording_event.is_set():
                # print(f"[录像] 开始录制, tag={self.current_tag}")
                self.recording_event.set()

    def stop_recording(self):
        with self.recording_lock:
            if self.recording_event.is_set():
                # print("[录像] 停止录制")
                self.recording_event.clear()

    def is_recording(self):
        return self.recording_event.is_set()

    def _record_rtsp(self, stream_name: str, rtsp_url: str):
        # print(f"[{stream_name}] 启动线程, RTSP={rtsp_url}")

        cap = None
        writer = None
        width, height, fps = None, None, None
        recording_active = False
        current_save_path = None

        while not self.stop_event.is_set():
            if cap is None or not cap.isOpened():
                # print(f"[{stream_name}] 正在连接 RTSP...")
                cap = self.open_rtsp(rtsp_url)

                if not cap.isOpened():
                    # print(f"[{stream_name}] RTSP 打开失败，1秒后重试")
                    time.sleep(1)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    # print(f"[{stream_name}] 首帧读取失败，重新连接")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue

                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 1 or fps > 120:
                    fps = 25.0

                if self.show_window:
                    cv2.imshow(stream_name, frame)

            ok, frame = cap.read()
            if not ok or frame is None:
                # print(f"[{stream_name}] 读帧失败，准备重连")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1)
                continue

            should_record = self.recording_event.is_set()

            if should_record and not recording_active:
                file_name = f"{stream_name}_{self.current_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                current_save_path = os.path.join(self.save_dir, file_name)
                writer = self.create_writer(current_save_path, fps, width, height)

                if writer.isOpened():
                    recording_active = True
                    # print(f"[{stream_name}] 开始写文件: {current_save_path}")
                else:
                    # print(f"[{stream_name}] VideoWriter 打开失败")
                    writer = None
                    recording_active = False

            elif not should_record and recording_active:
                # print(f"[{stream_name}] 结束写文件: {current_save_path}")
                if writer is not None:
                    writer.release()
                    writer = None
                recording_active = False
                current_save_path = None

            current_h, current_w = frame.shape[:2]
            if current_w != width or current_h != height:
                width, height = current_w, current_h
                # print(f"[{stream_name}] 分辨率变化，切换为 {width}x{height}")

                if recording_active:
                    if writer is not None:
                        writer.release()

                    file_name = f"{stream_name}_{self.current_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    current_save_path = os.path.join(self.save_dir, file_name)
                    writer = self.create_writer(current_save_path, fps, width, height)

                    if writer.isOpened():
                        # print(f"[{stream_name}] 分辨率变化后切换文件: {current_save_path}")
                        pass
                    else:
                        # print(f"[{stream_name}] writer 重建失败")
                        writer = None
                        recording_active = False
                        current_save_path = None

            if recording_active and writer is not None:
                writer.write(frame)

            if self.show_window:
                cv2.imshow(stream_name, frame)

        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if self.show_window:
            try:
                cv2.destroyWindow(stream_name)
            except Exception:
                pass

# ==================== 光电跟踪器 ====================
class OpticalTracker:
    def __init__(self, device_ip="192.168.0.4", local_ip="192.168.0.9", port=9966):
        self.device_ip = device_ip
        self.local_ip = local_ip
        self.port = port
        self.sock = None
        self.seq = 1
        
        self.monitor_running = False
        self.monitor_thread = None
        self.latest_azimuth = None
        self.latest_pitch = None
        self.latest_range = None
        self.current_status = None
        self.latest_targets = []
        self.lock = threading.Lock()
        self.connected = False

        self.recorder = None
        self.record_status_last = None
    
    def init_recorder(self, save_dir=r"D:\video", show_window=False):
        rtsp_urls = [
            {
                "name": "stream0",
                "url": f"rtsp://{self.device_ip}:554/channel=0,stream=0"
            }
        ]

        self.recorder = RTSPRecorderController(
            rtsp_urls=rtsp_urls,
            save_dir=save_dir,
            show_window=show_window
        )
        self.recorder.start()
        print("[录像] 录像控制器已初始化（仅录制 stream0）")


    def update_recording_by_status(self, work_status):
        """
        0=空闲, 1=搜索, 2=跟踪
        - 进入2开始录制
        - 从2回到1或0停止录制
        """
        if self.recorder is None:
            return

        if self.record_status_last == work_status:
            return

        prev_status = self.record_status_last
        self.record_status_last = work_status

        if work_status == 2:
            tag = f"status2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.recorder.start_recording(tag=tag)
        elif prev_status == 2 and work_status in (0, 1):
            self.recorder.stop_recording()

    def connect(self):
        """建立连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.local_ip, self.port))
            self.sock.settimeout(0.5)
            self.connected = True
            print(f"[连接] 成功绑定 {self.local_ip}:{self.port}")
            print(f"[连接] 目标设备 {self.device_ip}:{self.port}")
            return True
        except Exception as e:
            self.connected = False
            print(f"[连接] 失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        if self.recorder is not None:
            self.recorder.stop()

        if self.sock:
            self.sock.close()
            self.sock = None
        self.connected = False
        print("[关闭] 连接已断开")
    
    def _send_packet(self, cmd, data):
        """发送数据包"""
        timestamp = int(time.time() * 1000)
        
        packet = bytearray()
        packet.extend(START_BITS)
        packet.extend(struct.pack('<I', PROTOCOL_VERSION))
        packet.extend(struct.pack('<I', 20 + len(data)))
        packet.extend(struct.pack('<I', cmd))
        packet.extend(struct.pack('<Q', timestamp))
        packet.extend(data)
        packet.extend(struct.pack('<I', self.seq))
        packet.extend(struct.pack('<I', 0))
        packet.extend(STOP_BITS)
        
        self.sock.sendto(packet, (self.device_ip, self.port))
        self.seq += 1
        if self.seq > 65535:
            self.seq = 1
    
    def goto_position(self, azimuth, pitch, distance):
        """
        只转到目标位置，不搜索（search_mode=0）
        """
        print(f"\n[步骤1] 转到目标位置")
        print(f"       方位角: {azimuth:.1f}°")
        print(f"       俯仰角: {pitch:.1f}°")
        print(f"       距离: {distance:.0f}m")
        
        # 构建数据包，search_mode=0
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 系统下发时间戳
        data += struct.pack('<d', 0.0)   # 目标经度
        data += struct.pack('<d', 0.0)   # 目标纬度
        data += struct.pack('<d', 0.0)   # 目标高度
        data += struct.pack('<I', int(distance))  # 显示距离
        data += struct.pack('<I', int(distance))  # 实际距离
        data += struct.pack('<d', float(azimuth)) # 水平角度
        data += struct.pack('<d', float(pitch))   # 俯仰角度
        data += struct.pack('<H', 0)     # 用户ID
        data += struct.pack('<B', 0)     # 引导模式
        data += struct.pack('<B', 0)     # 目标运动方向
        data += struct.pack('<I', 0)     # search_mode=0 不搜索
        data += struct.pack('<I', 0)     # 左右搜索视场角大小
        data += struct.pack('<I', 0)     # 上下搜索视场角大小
        data += struct.pack('<I', 0)     # 保留
        
        self._send_packet(CMD_ABSOLUTE_ANGLE, data)
        print(f"[发送] 0x16命令（仅转到位置）")
    
    def start_search(self, search_mode=0):
        """
        开始搜索并自动跟踪（0x04命令，指令=1）
        
        根据协议2.5节：
        指控指令=1: 搜索并自动跟踪
        搜索方式: 0=在当前视场搜索, 1=向上移动再搜索, 2=向下移动再搜索,
                3=向左移动再搜索, 4=向右移动再搜索, 7=从左顺时针搜索,
                8=从上顺时针搜索, 9=从右顺时针搜索, 10=从下顺时针搜索,
                11=按指定范围搜索, 12=按指定范围指定速度搜索
        
        :param search_mode: 搜索方式（推荐使用7=顺时针扫描）
        """
        print(f"\n[步骤2] 开始搜索并自动跟踪 (搜索方式={search_mode})")
        
        # 0x04命令，指令=1（搜索并自动跟踪）
        # 根据协议2.5节格式
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<I', 1)     # 指控指令=1（搜索并自动跟踪）
        data += struct.pack('<I', 0)     # 水平搜索开始角度
        data += struct.pack('<I', 0)     # 水平搜索结束角度
        data += struct.pack('<i', 0)     # 俯仰搜索开始角度
        data += struct.pack('<i', 0)     # 俯仰搜索结束角度
        data += struct.pack('<I', search_mode)  # 搜索方式（预留字段）
        
        self._send_packet(CMD_SEARCH_TRACK, data)
        print(f"[发送] 搜索命令已发送 (指令=1, 搜索方式={search_mode})")
    
    def wait_for_position(self, target_azimuth, target_pitch, timeout=10.0, tolerance=3):
        """
        等待光电转到目标位置 - 使用监控线程缓存的角度
        
        :param target_azimuth: 目标方位角
        :param target_pitch: 目标俯仰角
        :param timeout: 超时时间（秒）
        :param tolerance: 允许误差（度）
        :return: 是否到位
        """
        # print(f"\n[等待] 等待转台到位 (超时={timeout}秒, 允许误差={tolerance}°)")
        
        start_time = time.time()
        last_print_az = None
        
        while time.time() - start_time < timeout:
            # 从监控线程缓存获取最新角度
            with self.lock:
                current_az = self.latest_azimuth
                current_pitch = self.latest_pitch
            # 只在角度变化时打印
            if current_az is not None and current_az != last_print_az:
                az_error = abs(current_az - target_azimuth)
                pitch_error = abs(current_pitch - target_pitch) if current_pitch is not None else 999
                # print(f"       当前位置: 方位={current_az:.1f}°, 俯仰={current_pitch:.1f}°")
                # print(f"       误差: 方位={az_error:.1f}°, 俯仰={pitch_error:.1f}°")
                last_print_az = current_az
                
                # 检查是否到位
                if az_error <= tolerance and pitch_error <= tolerance:
                    # print("[到位] 转台已到位")
                    return True
            
            time.sleep(0.3)  # 每0.3秒检查一次
        
        # print("[超时] 转台未到位，继续执行搜索")
        return False
    
    def get_current_angle(self):
        """
        获取当前光电角度（0x02命令）
        
        根据协议，完整数据包结构：
        偏移0-3: 起始位
        偏移4-7: 协议号
        偏移8-11: 包长度
        偏移12-15: 命令字 (0x02)
        偏移16-23: 时间戳
        偏移24-31: 信息内容开始
        
        根据图片中的表格（信息内容偏移）：
        偏移12-19: 水平角度 (相对于信息内容起始)
        偏移20-27: 俯仰角度 (相对于信息内容起始)
        
        所以绝对偏移 = 24 + 12 = 36 开始是水平角度
                     = 24 + 20 = 44 开始是俯仰角度
        """
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 时间戳
        
        self._send_packet(CMD_GET_ANGLE, data)
        
        # 等待回复
        try:
            recv_data, _ = self.sock.recvfrom(1024)
            if len(recv_data) < 16:
                return None, None
            
            cmd = struct.unpack('<I', recv_data[12:16])[0]
            
            # 只处理0x02包
            if cmd != 0x02:
                return None, None
            
            if len(recv_data) >= 52:  # 需要足够长度
                # 信息内容从偏移24开始
                # 水平角度：信息内容偏移12，绝对偏移 = 24+12 = 36
                # 俯仰角度：信息内容偏移20，绝对偏移 = 24+20 = 44
                azimuth = struct.unpack('<d', recv_data[36:44])[0]
                pitch = struct.unpack('<d', recv_data[44:52])[0]
                
                # 过滤异常值
                if -180 <= azimuth <= 360 and -90 <= pitch <= 90:
                    return azimuth, pitch
                else:
                    print(f"[警告] 角度异常: 方位={azimuth}, 俯仰={pitch}")
                    return None, None
                    
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[获取角度错误] {e}")
        
        return None, None
    
    def goto_and_search(self, azimuth, pitch, distance, search_mode=3, wait_time=1.5):
        """
        完整流程：先转到位置，等待到位，再开始搜索
        
        :param azimuth: 目标方位角
        :param pitch: 目标俯仰角
        :param distance: 目标距离
        :param search_mode: 搜索模式（1=左右扫描, 2=上下扫描, 3=当前视场）
        :param wait_time: 等待时间（秒），如果无法查询角度则用固定时间
        """
        print("\n" + "=" * 50)
        print("开始跟踪流程")
        print("=" * 50)
        time.sleep(0.3)

        # ========== 获取当前状态 ==========
        current_status = None
        with self.lock:
            current_status = self.current_status
        
        # ========== 根据状态决定是否释放 ==========
        # 状态: 0=空闲, 1=搜索, 2=跟踪
        if current_status == 1:  # 正在搜索
            print("[状态] 当前处于搜索状态，先释放目标")
            self.release_target()
            time.sleep(0.3)  # 等待释放完成
        elif current_status == 2:  # 正在跟踪
            print("[状态] 当前处于跟踪状态，直接转到新位置（不释放）")
            # 跟踪状态不需要释放，直接转台即可
        else:
            print(f"[状态] 当前状态: {current_status} (0=空闲,1=搜索,2=跟踪)")

        # 1. 转到位置（不搜索）
        self.goto_position(azimuth, pitch, distance)
        
        # 2. 等待转台到位     
        # 尝试查询角度等待到位
        self.wait_for_position(azimuth, pitch, timeout=wait_time)        

        # 3. 开始搜索
        self.start_search()
        print("\n[提示] 光电正在搜索目标...")
    
    def release_target(self):
        """释放目标（停止跟踪）"""
        print(f"\n[命令] 释放目标")
        
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<I', 3)     # 指控指令=3（释放）
        data += struct.pack('<I', 0)     # 水平搜索开始角度
        data += struct.pack('<I', 0)     # 水平搜索结束角度
        data += struct.pack('<i', 0)     # 俯仰搜索开始角度
        data += struct.pack('<i', 0)     # 俯仰搜索结束角度
        data += struct.pack('<I', 0)     # 预留
        
        self._send_packet(CMD_SEARCH_TRACK, data)
        print(f"[发送] 释放命令已发送")
        if self.recorder is not None:
            self.recorder.stop_recording()

    def reset_zoom(self,zoom = 100, channel=0):
        """
        Zoom 归零 - 跳转到 zoom=0 的位置
        
        根据协议 2.12 节：
        控制命令 0x03: 跳转到指定 zoom 和 focus 位置
        
        :param channel: 0=可见光, 1=热像
        """
        print(f"\n[光电] Zoom 归零")
        
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 系统下发时间戳
        data += struct.pack('<I', 0x01)  # 控制命令=0x03 (跳转到指定zoom和focus位置)
        data += struct.pack('<I', zoom)     # 物理焦距/倍数 (保留)
        data += struct.pack('<i', 0)     # zoom位置 = 0
        data += struct.pack('<i', 0)     # focus位置 = 0 (归零)
        data += struct.pack('<I', channel)  # 通道号
        
        self._send_packet(0x09, data)
        print("[光电] Zoom 已归零")


    def start_monitor(self):
        """启动接收监控线程"""
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("[监控] 已启动")

    def _monitor_loop(self):
        """接收并打印所有数据包"""
        while self.monitor_running:
            try:
                data, addr = self.sock.recvfrom(4096)
                self._print_packet_info(data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.monitor_running:
                    print(f"[监控错误] {e}")

    def _print_packet_info(self, data, addr):
        """打印数据包信息并缓存"""
        if len(data) < 16:
            return
        
        cmd = struct.unpack('<I', data[12:16])[0]
        
        # 更新最新角度（使用正确的偏移）
        if cmd == 0x02 and len(data) >= 52:
            with self.lock:
                self.latest_azimuth = struct.unpack('<d', data[36:44])[0]
                self.latest_pitch = struct.unpack('<d', data[44:52])[0]
                current_status = self.current_status
                latest_range = self.latest_range
                latest_azimuth = self.latest_azimuth
                latest_pitch = self.latest_pitch
            if current_status == 2 and latest_azimuth is not None:
                append_optical_measurement(
                    time.time(), latest_azimuth, latest_pitch, current_status, latest_range
                )

        # ========== 缓存设备状态（0x08包） ==========
        if cmd == 0x08 and len(data) >= 20:
            # 工作状态在偏移12-15（相对于包起始）
            # 包结构：起始位(4) + 协议号(4) + 包长度(4) + 命令字(4) + 时间戳(8) + 信息内容
            # 工作状态在信息内容偏移12-15，即绝对偏移 = 4+4+4+4+8+12 = 36
            if len(data) >= 40:
                work_status = struct.unpack('<I', data[36:40])[0]
                with self.lock:
                    self.current_status = work_status
                self.update_recording_by_status(work_status)
                # print(f"[状态] 工作状态: {work_status} (0=空闲,1=搜索,2=跟踪)")

        # ========== 新增：缓存目标信息 ==========
        if cmd == 0x0B:
            targets = self._parse_target_info(data)
            with self.lock:
                self.latest_targets = targets
            
                # 如果有目标，更新最新距离（取第一个目标的距离）
                if targets and len(targets) > 0:
                    self.latest_range = targets[0].get('target_dist', 0)

            # print(f"\n[收到] 目标上报 | {len(targets)} 个目标")  # 可选打印
        
        # 其他包的打印可以保留或注释
        # if cmd == 0x02:
        #     print(f"\n[收到] 方位俯仰 | 方位={self.latest_azimuth:.1f}°, 俯仰={self.latest_pitch:.1f}°")


    def _parse_target_info(self, data):
        """解析0x0B包，返回目标列表"""
        targets = []
        
        if len(data) < 36:
            return targets
        
        try:
            target_count = struct.unpack('<I', data[28:32])[0]
            
            if target_count == 0 or target_count > 20:
                return targets
            
            offset = 36  # 第一个目标从偏移36开始
            
            for i in range(target_count):
                if offset + 60 > len(data):
                    break
                
                target = {
                    'target_id': struct.unpack('<I', data[offset:offset+4])[0],
                    'target_type': struct.unpack('<I', data[offset+4:offset+8])[0],
                    'similarity': struct.unpack('<I', data[offset+8:offset+12])[0] / 100.0,
                    'width': struct.unpack('<I', data[offset+12:offset+16])[0],
                    'height': struct.unpack('<I', data[offset+16:offset+20])[0],
                    'phys_width': struct.unpack('<I', data[offset+20:offset+24])[0] / 100.0,
                    'phys_height': struct.unpack('<I', data[offset+24:offset+28])[0] / 100.0,
                    'movement_dir': data[offset+28] if offset+28 < len(data) else 0,
                    'ai_template': data[offset+29] if offset+29 < len(data) else 0,
                    'pos_x': struct.unpack('<H', data[offset+36:offset+38])[0] if offset+38 <= len(data) else 0,
                    'pos_y': struct.unpack('<H', data[offset+38:offset+40])[0] if offset+40 <= len(data) else 0,
                    'target_az': struct.unpack('<d', data[offset+40:offset+48])[0] if offset+48 <= len(data) else 0,
                    'target_pitch': struct.unpack('<d', data[offset+48:offset+56])[0] if offset+56 <= len(data) else 0,
                    'target_dist': struct.unpack('<I', data[offset+56:offset+60])[0] if offset+60 <= len(data) else 0,
                }
                targets.append(target)
                offset += 60
                
        except Exception as e:
            print(f"解析目标错误: {e}")
        
        return targets

    def print_target_info(self, data):
        """打印目标上报包的详细信息"""
        if len(data) < 36:
            return
        
        try:
            # 信息内容从偏移24开始
            timestamp = struct.unpack('<I', data[24:28])[0]
            target_count = struct.unpack('<I', data[28:32])[0]
            
            
            # 每个目标60字节（不是56！）
            TARGET_SIZE = 60
            offset = 36  # 第一个目标从偏移36开始
            
            for i in range(target_count):
                if offset + TARGET_SIZE > len(data):
                    print(f"  数据不足，只解析了{i}个目标")
                    break
                
                # 目标编号 (0-3)
                target_id = struct.unpack('<I', data[offset:offset+4])[0]
                
                # 目标类型 (4-7)
                target_type = struct.unpack('<I', data[offset+4:offset+8])[0]
                
                # 相似度 (8-11)
                similarity = struct.unpack('<I', data[offset+8:offset+12])[0] / 100.0
                
                # 宽度 (12-15)
                width = struct.unpack('<I', data[offset+12:offset+16])[0]
                
                # 高度 (16-19)
                height = struct.unpack('<I', data[offset+16:offset+20])[0]
                
                # 物理宽度 (20-23)
                phys_width = struct.unpack('<I', data[offset+20:offset+24])[0] / 100.0
                
                # 物理高度 (24-27)
                phys_height = struct.unpack('<I', data[offset+24:offset+28])[0] / 100.0
                
                # 运动方向 (28)
                movement_dir = data[offset+28] if offset+28 < len(data) else 0
                
                # AI模板类型 (29)
                ai_template = data[offset+29] if offset+29 < len(data) else 0
                
                # 位置X (36-37)
                pos_x = struct.unpack('<H', data[offset+36:offset+38])[0] if offset+38 <= len(data) else 0
                
                # 位置Y (38-39)
                pos_y = struct.unpack('<H', data[offset+38:offset+40])[0] if offset+40 <= len(data) else 0
                
                # 目标方位角 (40-47)
                target_az = struct.unpack('<d', data[offset+40:offset+48])[0] if offset+48 <= len(data) else 0
                
                # 目标俯仰角 (48-55)
                target_pitch = struct.unpack('<d', data[offset+48:offset+56])[0] if offset+56 <= len(data) else 0
                
                # 目标距离 (56-59) ← 索引56，4字节
                target_dist = struct.unpack('<I', data[offset+56:offset+60])[0] if offset+60 <= len(data) else 0
                # print(f"  目标{i+1}: ID={target_id}, 类型={target_type}, 相似度={similarity:.0f}%, ai模板={ai_template}, 运动方向={movement_dir}, 物理尺寸={phys_width:.2f}m x {phys_height:.2f}m, "
                #         f"尺寸={width}x{height}, 位置=({pos_x},{pos_y}), 距离={target_dist}m")
                    
                offset += TARGET_SIZE
                
        except Exception as e:
            print(f"  解析错误: {e}")

    def set_report_destination(self, host_ip, port=9966):
        """
        设置上报目标IP和端口（0x19命令）
        根据协议文档第42页
        """
        print(f"\n[命令] 设置上报目标: {host_ip}:{port}")
        
        # 1. 光电编号 (4字节)
        data = struct.pack('<I', 0)
        
        # 2. 时间戳 (8字节，预留)
        data += struct.pack('<Q', 0)
        
        # 3. 参数长度 (4字节) - 后续参数的总长度
        # IP地址(32) + 端口(2) + 8个上报频率(各2字节) = 32 + 2 + 16 = 50
        param_len = 32 + 2 + 16
        data += struct.pack('<I', param_len)
        
        # 4. 指控平台IP地址 (32字节，字符串格式)
        # 例如 "10.129.41.89" 需要补齐到32字节
        ip_bytes = host_ip.encode('utf-8') + b'\x00' * (32 - len(host_ip))
        data += ip_bytes
        
        # 5. 指控平台端口 (2字节)
        data += struct.pack('<H', port)
        
        # 6. 各上报频率（每个2字节，共8个）
        # 顺序：0x01, 0x02, 0x08, 0x0B, 0x0C, 0x0F, 0x15, 0x18
        frequencies = [500, 100, 100, 100, 1000, 40, 1000, 100]
        for freq in frequencies:
            data += struct.pack('<H', freq)
        
        self._send_packet(0x19, data)
        print(f"[发送] 上报目标已设置: {host_ip}:{port}")
        print(f"       上报频率: 0x01=500ms, 0x0B=100ms, 0x02=100ms")

    def switch_to_thermal(self,mode=1):
        """切换到热成像通道"""
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<I', mode)     # 1=热像
        self._send_packet(0x0E, data)
        if  mode == 1:
            print("[切换] 已切换到热成像")
        else:
            print("[切换] 已切换到可见光")

    def set_ai_template(self, template_type):
        """
        切换AI模板（0x12命令）
        
        根据协议2.21节：
        参数配置命令=1: AI模板切换
        参数长度=4
        参数值: 1=对天3分类, 2=对地80分类, 4=20分类
        
        :param template_type: AI模板类型
            1 - 对天3分类（无人机、飞机、固定翼）
            2 - 对地80分类（人、车、自行车、鸟等80种）
            4 - 20分类（飞机、无人机、车、鸟、人等20种）
        """
        print(f"\n[命令] 切换AI模板: {template_type}")
        
        # 模板名称映射
        template_names = {
            1: "对天3分类（无人机、飞机、固定翼）",
            2: "对地80分类（人、车、自行车、鸟等）",
            4: "20分类（飞机、无人机、车、鸟、人等）"
        }
        
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<I', 1)     # 参数配置命令=1（AI模板切换）
        data += struct.pack('<I', 4)     # 参数长度=4
        data += struct.pack('<I', template_type)  # 模板类型
        
        self._send_packet(0x12, data)
        print(f"[发送] AI模板已切换: {template_names.get(template_type, '未知')}")

    def set_tracking_thresholds(self, zoom_in_threshold = 400, zoom_out_threshold = 600):
        """
        设置自动跟踪变倍阈值
        :param zoom_in_threshold: 放大阈值（目标尺寸小于此值放大）
        :param zoom_out_threshold: 缩小阈值（目标尺寸大于此值缩小）
        """
        data = struct.pack('<I', 0)          # 光电编号
        data += struct.pack('<I', 0)         # 系统编号
        data += struct.pack('<Q', 0)         # 时间戳
        data += struct.pack('<I', zoom_in_threshold)   # 放大阈值
        data += struct.pack('<I', zoom_out_threshold)  # 缩小阈值
        # 后续可能还有其它参数，通常填0
        data += struct.pack('<I', 0)         
        
        self._send_packet(0x14, data)
        print(f"[设置] 跟踪阈值: 放大={zoom_in_threshold}, 缩小={zoom_out_threshold}")
    

    def get_current_status(self):
        """
        获取当前光电工作状态（0x08命令）
        
        返回:
            int: 0=空闲, 1=搜索, 2=跟踪, None=获取失败
        """
        # 直接返回缓存的状态，避免每次都发命令
        with self.lock:
            return self.current_status
        
    def query_current_status(self):
        """
        主动查询当前光电工作状态（发送0x08命令获取回复）
        
        返回:
            int: 0=空闲, 1=搜索, 2=跟踪, None=获取失败
        """
        # 0x08是上报包，不能主动查询，需要等待设备主动上报
        # 所以这个方法实际上不需要实现，设备会主动上报0x08包
        # 我们只需要在_monitor_loop中缓存状态即可
        pass

# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("光电跟踪测试程序 - 先到位再搜索")
    print("=" * 60)
    

    device_ip = "10.129.41.98"

    local_ip = "10.129.41.8"
    
    tracker = OpticalTracker(device_ip=device_ip, local_ip=local_ip)
    
    if not tracker.connect():
        print("连接失败，请检查网络配置")
        return
    
    tracker.init_recorder(save_dir=r"D:\video", show_window=False)

    # 设置上报目标
    tracker.set_report_destination(local_ip, 9966)
    time.sleep(0.3)
    
    # # 启动监控（可选）
    # start_mon = input("\n是否开启数据监控？(y/n，默认n): ").strip().lower()
    # if start_mon == 'y':
    tracker.start_monitor()
    

    # if input("夜间模式？(y/n): ").lower() == 'y':
    #     tracker.switch_to_thermal(1)
    # else:
    #     tracker.switch_to_thermal(0)    
    # time.sleep(0.5)

    tracker.switch_to_thermal(0)   
    tracker.set_tracking_thresholds()
    print("\n" + "=" * 60)
    print("命令说明:")
    print("  t - 转到目标并搜索（先到位再搜索）")
    print("  r - 释放目标")
    print("  b - 显示目标信息")
    print("  q - 退出")
    print("=" * 60)
    
    while True:
        try:
            cmd = input("\n请输入命令: ").strip().lower()
            
            if cmd == 'q':
                break
            
            elif cmd == 'r':
                tracker.release_target()

            elif cmd == 'b':
                tracker.print_target_info()

            elif cmd == 't':
                # 输入距离
                while True:
                    try:
                        distance = float(input("  距离(米): "))
                        if distance > 0:
                            break
                        print("  距离必须大于0")
                    except ValueError:
                        print("  请输入数字")
                
                
                
                # 执行完整流程
                tracker.goto_and_search(25, 22, distance, 0, 3)
            
            else:
                print("未知命令，请输入 t/r/s/q")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
    
    tracker.close()
    print("\n程序退出")


if __name__ == "__main__":
    main()
