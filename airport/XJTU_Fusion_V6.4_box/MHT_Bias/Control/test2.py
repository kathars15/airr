# test2.py
"""
光电跟踪测试程序
功能：手动输入目标角度、距离，先转到位置，到位后再开始搜索跟踪
"""

import socket
import struct
import time
import math
import threading

# ==================== 协议常量 ====================
START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
PROTOCOL_VERSION = 9002

# 命令字
CMD_SET_POSITION = 0x03      # 设置光电目址信息包（0x03）
CMD_GET_STATUS = 0x01        # 获取设备状态
CMD_GET_ANGLE = 0x02         # 获取方位俯仰信息
CMD_SEARCH_TRACK = 0x04      # 搜索跟踪命令

# 搜索方式（0x04命令的预留字段）
SEARCH_WAY_CURRENT = 0       # 在当前视场搜索
SEARCH_WAY_UP = 1            # 向上移动再搜索
SEARCH_WAY_DOWN = 2          # 向下移动再搜索
SEARCH_WAY_LEFT = 3          # 向左移动再搜索
SEARCH_WAY_RIGHT = 4         # 向右移动再搜索
SEARCH_WAY_ZOOM_OUT = 5      # 镜头向后拉一倍再搜索
SEARCH_WAY_ZOOM_IN = 6       # 镜头向前推一倍再搜索
SEARCH_WAY_CLOCKWISE_L = 7   # 从左顺时针搜索
SEARCH_WAY_CLOCKWISE_U = 8   # 从上顺时针搜索
SEARCH_WAY_CLOCKWISE_R = 9   # 从右顺时针搜索
SEARCH_WAY_CLOCKWISE_D = 10  # 从下顺时针搜索
SEARCH_WAY_RANGE = 11        # 按指定范围搜索
SEARCH_WAY_RANGE_SPEED = 12  # 按指定范围指定速度搜索
SEARCH_WAY_LEFT_RIGHT = 13   # 左右搜索一次
SEARCH_WAY_UP_DOWN = 14      # 上下搜索一次


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
        self.lock = threading.Lock()

    def connect(self):
        """建立连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.local_ip, self.port))
            self.sock.settimeout(0.5)
            print(f"[连接] 成功绑定 {self.local_ip}:{self.port}")
            print(f"[连接] 目标设备 {self.device_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[连接] 失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        if self.sock:
            self.sock.close()
            self.sock = None
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
        转到目标位置（0x03命令）
        
        根据协议2.3节（第13页）：
        数据项: 光电编号(4) + 系统编号(4) + 时间戳(8) + 水平角度(8) + 俯仰角度(8) + 距离(8)
        """
        print(f"\n[步骤1] 转到目标位置")
        print(f"       方位角: {azimuth:.1f}°")
        print(f"       俯仰角: {pitch:.1f}°")
        print(f"       距离: {distance:.0f}m")
        
        # 构建0x03数据包（协议2.3格式）
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<d', (azimuth))  # 水平角度
        data += struct.pack('<d', float(pitch))    # 俯仰角度
        data += struct.pack('<d', float(distance)) # 距离
            
        # 调试：打印数据包内容
        print(f"[调试] 数据包长度: {len(data)} 字节")
        print(f"[调试] 十六进制: {data.hex()}")
        
        self._send_packet(CMD_SET_POSITION, data)
        print(f"[发送] 0x03命令（转到位置）")
    
    def start_search(self, search_way=7):
        """
        开始搜索并自动跟踪（0x04命令，指控指令=1）
        
        根据协议2.5节（第16-18页）：
        指控指令=1: 搜索并自动跟踪
        搜索方式(预留字段): 0-14 对应不同的搜索模式
        
        :param search_way: 搜索方式（推荐7=从左顺时针搜索）
        """
        print(f"\n[步骤2] 开始搜索并自动跟踪 (搜索方式={search_way})")
        
        # 0x04命令，指令=1（搜索并自动跟踪）
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<I', 1)     # 指控指令=1（搜索并自动跟踪）
        data += struct.pack('<I', 0)     # 水平搜索开始角度
        data += struct.pack('<I', 0)     # 水平搜索结束角度
        data += struct.pack('<i', 0)     # 俯仰搜索开始角度
        data += struct.pack('<i', 0)     # 俯仰搜索结束角度
        data += struct.pack('<I', search_way)  # 搜索方式（预留字段）
        
        self._send_packet(CMD_SEARCH_TRACK, data)
        
        # 打印搜索方式说明
        search_names = {
            0: "在当前视场搜索",
            1: "向上移动再搜索",
            2: "向下移动再搜索",
            3: "向左移动再搜索",
            4: "向右移动再搜索",
            5: "镜头向后拉一倍再搜索",
            6: "镜头向前推一倍再搜索",
            7: "从左顺时针搜索",
            8: "从上顺时针搜索",
            9: "从右顺时针搜索",
            10: "从下顺时针搜索",
            11: "按指定范围搜索",
            12: "按指定范围指定速度搜索",
            13: "左右搜索一次",
            14: "上下搜索一次",
        }
        print(f"[发送] 0x04命令（{search_names.get(search_way, '未知方式')}）")
    
    def wait_for_position(self, target_azimuth, target_pitch, timeout=10.0, tolerance=1):
        """
        等待光电转到目标位置 - 使用监控线程缓存的角度
        
        :param target_azimuth: 目标方位角
        :param target_pitch: 目标俯仰角
        :param timeout: 超时时间（秒）
        :param tolerance: 允许误差（度）
        :return: 是否到位
        """
        print(f"\n[等待] 等待转台到位 (超时={timeout}秒, 允许误差={tolerance}°)")
        
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
                print(f"       当前位置: 方位={current_az:.1f}°, 俯仰={current_pitch:.1f}°")
                print(f"       误差: 方位={az_error:.1f}°, 俯仰={pitch_error:.1f}°")
                last_print_az = current_az
                
                # 检查是否到位
                if az_error <= tolerance and pitch_error <= tolerance:
                    print("[到位] 转台已到位")
                    return True
            
            time.sleep(0.3)
        
        print("[超时] 转台未到位，继续执行搜索")
        return False
    
    def get_current_angle(self):
        """获取当前光电角度（0x02命令）"""
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 时间戳
        
        self._send_packet(CMD_GET_ANGLE, data)
        
        try:
            recv_data, _ = self.sock.recvfrom(1024)
            if len(recv_data) < 16:
                return None, None
            
            cmd = struct.unpack('<I', recv_data[12:16])[0]
            if cmd != 0x02:
                return None, None
            
            if len(recv_data) >= 52:
                azimuth = struct.unpack('<d', recv_data[36:44])[0]
                pitch = struct.unpack('<d', recv_data[44:52])[0]
                
                if -180 <= azimuth <= 360 and -90 <= pitch <= 90:
                    return azimuth, pitch
        except socket.timeout:
            pass
        
        return None, None
    
    def goto_and_search(self, azimuth, pitch, distance, search_way=7, wait_time=5.0):
        """
        完整流程：先转到位置，等待到位，再开始搜索
        
        :param azimuth: 目标方位角
        :param pitch: 目标俯仰角
        :param distance: 目标距离
        :param search_way: 搜索方式（推荐7=从左顺时针搜索）
        :param wait_time: 等待时间（秒）
        """
        print("\n" + "=" * 50)
        print("开始跟踪流程")
        print("=" * 50)
        
        # 1. 转到位置（0x03命令）
        self.goto_position(azimuth, pitch, distance)
        
        # 2. 等待转台到位
        time.sleep(1.0)  # 先给转台启动时间
        
        if not self.wait_for_position(azimuth, pitch, timeout=wait_time):
            print(f"[备用] 使用固定等待时间 {wait_time} 秒")
            time.sleep(wait_time)
        
        # 3. 开始搜索（0x04命令）
        self.start_search(search_way)
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
        """打印数据包信息"""
        if len(data) < 16:
            return
        
        cmd = struct.unpack('<I', data[12:16])[0]
        
        # 更新最新角度
        if cmd == 0x02 and len(data) >= 52:
            with self.lock:
                self.latest_azimuth = struct.unpack('<d', data[36:44])[0]
                self.latest_pitch = struct.unpack('<d', data[44:52])[0]

    def set_report_destination(self, host_ip, port=9966):
        """设置上报目标IP和端口（0x19命令）"""
        print(f"\n[命令] 设置上报目标: {host_ip}:{port}")
        
        param_len = 32 + 2 + 16
        data = struct.pack('<I', 0)
        data += struct.pack('<Q', 0)
        data += struct.pack('<I', param_len)
        
        ip_bytes = host_ip.encode('utf-8') + b'\x00' * (32 - len(host_ip))
        data += ip_bytes
        data += struct.pack('<H', port)
        
        frequencies = [500, 100, 100, 100, 1000, 40, 1000, 100]
        for freq in frequencies:
            data += struct.pack('<H', freq)
        
        self._send_packet(0x19, data)
        print(f"[发送] 上报目标已设置: {host_ip}:{port}")

    def switch_to_thermal(self, mode=1):
        """切换到热成像通道"""
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<I', mode)  # 1=热像
        self._send_packet(0x0E, data)
        if mode == 1:
            print("[切换] 已切换到热成像")
        else:
            print("[切换] 已切换到可见光")


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("光电跟踪测试程序 - 使用0x03和0x04")
    print("=" * 60)
    
    device_ip = "10.129.41.98"
    local_ip = "10.129.41.89"
    
    tracker = OpticalTracker(device_ip=device_ip, local_ip=local_ip)
    
    if not tracker.connect():
        print("连接失败，请检查网络配置")
        return
    
    # 设置上报目标
    tracker.set_report_destination(local_ip, 9966)
    time.sleep(0.3)
    
    # 启动监控
    tracker.start_monitor()
    
    # 切换到可见光
    tracker.switch_to_thermal(0)
    
    print("\n" + "=" * 60)
    print("命令说明:")
    print("  t - 转到目标并搜索（0x03到位 + 0x04搜索）")
    print("  r - 释放目标")
    print("  q - 退出")
    print("=" * 60)
    
    while True:
        try:
            cmd = input("\n请输入命令: ").strip().lower()
            
            if cmd == 'q':
                break
            
            elif cmd == 'r':
                tracker.release_target()
            
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
                
                # 选择搜索方式
                print("\n  搜索方式:")
                print("    0 - 在当前视场搜索")
                print("    7 - 从左顺时针搜索（推荐）")
                print("    8 - 从上顺时针搜索（推荐）")
                print("    13 - 左右搜索一次")
                print("    14 - 上下搜索一次")
                
                search_way = int(input("  请选择[默认7]: ") or "7")
                
                # 等待时间
                wait_time = float(input("  到位等待时间(秒，默认5): ") or "5")
                
                # 执行完整流程（方位角34°，俯仰角-3.22°）
                tracker.goto_and_search(34, -3.22, distance, search_way, wait_time)
            
            else:
                print("未知命令，请输入 t/r/q")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
    
    tracker.close()
    print("\n程序退出")


if __name__ == "__main__":
    main()