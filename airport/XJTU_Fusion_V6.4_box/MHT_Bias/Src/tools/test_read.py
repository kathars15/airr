import socket
import struct
import time

class OpticalMonitor:
    """光电状态监视器 - 修正版"""
    
    def __init__(self, local_ip="192.168.0.9", port=9966):
        self.local_ip = local_ip
        self.port = port
        self.sock = None
        
    def start(self):
        """启动监视"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.port))
        self.sock.settimeout(1.0)
        
        print(f"监听 {self.local_ip}:{self.port}")
        print("=" * 60)
        
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
                self.parse_packet(data)
            except socket.timeout:
                pass
            except KeyboardInterrupt:
                break
                
    def parse_packet(self, data):
        """解析数据包"""
        if len(data) < 24:
            return
            
        # 解析起始位 (4字节)
        start = data[0:4]
        
        # 解析协议号 (4字节)
        proto = struct.unpack('<I', data[4:8])[0]
        
        # 解析包长度 (4字节)
        pkg_len = struct.unpack('<I', data[8:12])[0]
        
        # 解析命令字 (4字节)
        cmd = struct.unpack('<I', data[12:16])[0]
        
        # 解析时间戳 (8字节)
        timestamp = struct.unpack('<Q', data[16:24])[0]
        
        # 信息内容从第24字节开始
        content = data[24:-8]  # 减去序列号(4) + 校验(4) + 停止位(4)
        
        # 根据命令字解析
        if cmd == 0x01:  # 光电设备状态信息包
            self.parse_status(content)
        elif cmd == 0x02:  # 光电设备方位俯仰信息包
            self.parse_position(content)
        elif cmd == 0x08:  # 光电设备状态扩展信息包
            self.parse_extended_status(content)
        elif cmd == 0x0B:  # 光电目标上报信息包
            self.parse_target(content)
        elif cmd == 0x0C:  # 光电镜头状态扩展信息包
            self.parse_lens(content)
        elif cmd == 0x0F:  # 脱靶量信息上报
            self.parse_offset(content)
        elif cmd == 0x15:  # 光电系统状态扩展信息包
            self.parse_system_status(content)
        elif cmd == 0x18:  # 目标上报扩展信息包
            self.parse_target_ext(content)
        else:
            # 打印前32字节用于调试
            print(f"未知命令: 0x{cmd:02X}, 长度: {len(data)}, 内容: {content[:32].hex()}")
            
    def parse_status(self, data):
        """解析设备状态信息包 (0x01) - 协议2.1"""
        if len(data) < 28:
            return
        # 光电编号 (4) + 时间戳(8) + 工作状态(4) + 故障编码(8) + 跟踪视频源(4)
        work_status = struct.unpack('<I', data[12:16])[0]
        status_names = {0: "异常", 1: "正常"}
        mode_names = {0: "空闲", 1: "搜索", 2: "跟踪"}
        mode = struct.unpack('<I', data[16:20])[0]
        print(f"[状态] 设备状态: {status_names.get(work_status, '未知')} | 工作模式: {mode_names.get(mode, '未知')}")
        
    def parse_position(self, data):
        """解析方位俯仰信息包 (0x02) - 协议2.2"""
        if len(data) < 56:
            print(f"位置数据长度不足: {len(data)}")
            return
            
        # 协议2.2格式:
        # 光电编号(4) + 时间戳(8) + 水平角度(8) + 俯仰角度(8) + 距离(8) + 
        # 水平角速度(8) + 俯仰角速度(8) + 目标高度(2) + 镜头倍率(1) + 保留(1)
        
        offset = 0
        # 跳过光电编号(4)和时间戳(8)
        offset += 12
        
        azimuth = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        
        pitch = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        
        distance = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        
        az_speed = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        
        pitch_speed = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        
        # 检查数值是否合理（方位角应该在0-360之间）
        if 0 <= azimuth <= 360:
            print(f"[位置] 方位={azimuth:.2f}° | 俯仰={pitch:.2f}° | 距离={distance:.1f}m | 速度=({az_speed:.1f}, {pitch_speed:.1f})°/s")
        else:
            print(f"[位置] 数据异常: 方位={azimuth} (可能解析偏移有误)")
            # 打印原始数据前64字节用于调试
            print(f"原始数据(hex): {data[:64].hex()}")
            
    def parse_extended_status(self, data):
        """解析扩展状态信息包 (0x08) - 协议2.10"""
        if len(data) < 16:
            return
        # 时间戳(4) + 工作状态(4) + 数据长度(4) + 外带数据...
        work_status = struct.unpack('<I', data[8:12])[0]
        status_names = {0: "空闲", 1: "搜索", 2: "跟踪"}
        print(f"[扩展状态] {status_names.get(work_status, '未知')}")
        
    def parse_target(self, data):
        """解析目标上报信息包 (0x0B) - 协议2.14"""
        if len(data) < 12:
            return
        # 时间戳(4) + 目标数量(4) + 数据长度(4)
        target_num = struct.unpack('<I', data[4:8])[0]
        print(f"[目标] 检测到 {target_num} 个目标")
        
    def parse_lens(self, data):
        """解析镜头状态扩展信息包 (0x0C) - 协议2.15"""
        print(f"[镜头] 状态数据")
        
    def parse_offset(self, data):
        """解析脱靶量信息上报 (0x0F) - 协议2.18"""
        if len(data) >= 20:
            # 跟踪视频源通道(4) + 水平脱靶量(4) + 俯仰脱靶量(4)
            channel = struct.unpack('<I', data[0:4])[0]
            offset_x = struct.unpack('<i', data[4:8])[0]
            offset_y = struct.unpack('<i', data[8:12])[0]
            channel_name = {0: "可见光", 1: "热像"}
            print(f"[脱靶量] 通道: {channel_name.get(channel, '未知')} | X偏移: {offset_x}像素 | Y偏移: {offset_y}像素")
        
    def parse_system_status(self, data):
        """解析光电系统状态扩展信息包 (0x15) - 协议2.24"""
        print(f"[系统状态] 扩展信息")
        
    def parse_target_ext(self, data):
        """解析目标上报扩展信息包 (0x18) - 协议2.26"""
        if len(data) >= 12:
            target_num = struct.unpack('<I', data[4:8])[0]
            print(f"[目标扩展] 检测到 {target_num} 个目标")



class OpticalController:
    """光电控制器 - 专门测试发送"""
    
    def __init__(self, device_ip="192.168.0.4", local_ip="192.168.0.9", port=9966):
        self.device_ip = device_ip
        self.local_ip = local_ip
        self.port = port
        self.sock = None
        self.seq = 1
        
    def connect(self):
        """建立发送socket"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不绑定端口，让系统自动分配
        print(f"发送socket已创建，目标: {self.device_ip}:{self.port}")
        
    def send_angle(self, azimuth, pitch):
        """发送角度指令 (协议2.3)"""
        # 协议固定值
        START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
        STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
        PROTOCOL_VERSION = 9002
        
        timestamp = int(time.time() * 1000)
        
        # 数据内容 (48字节)
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<d', 0.0)   # 目标经度
        data += struct.pack('<d', 0.0)   # 目标纬度
        data += struct.pack('<d', 0.0)   # 目标高度
        data += struct.pack('<d', azimuth)   # 水平角度
        data += struct.pack('<d', pitch)     # 俯仰角度
        data += struct.pack('<I', 0)     # 距离
        data += struct.pack('<B', 0)     # 目标运动方向
        data += struct.pack('<B', 0)     # 搜索模式
        data += struct.pack('<H', 0)     # 保留
        
        # 构建完整包
        packet = bytearray()
        packet.extend(START_BITS)
        packet.extend(struct.pack('<I', PROTOCOL_VERSION))
        packet.extend(struct.pack('<I', 20 + len(data)))
        packet.extend(struct.pack('<I', 0x03))  # 命令码 0x03
        packet.extend(struct.pack('<Q', timestamp))
        packet.extend(data)
        packet.extend(struct.pack('<I', self.seq))
        packet.extend(struct.pack('<I', 0))
        packet.extend(STOP_BITS)
        
        # 发送
        self.sock.sendto(packet, (self.device_ip, self.port))
        print(f"[发送] 方位={azimuth}°, 俯仰={pitch}°, 序列号={self.seq}")
        self.seq += 1
        
    def start_track(self):
        """开始搜索跟踪 (协议2.5)"""
        START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
        STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
        PROTOCOL_VERSION = 9002
        
        timestamp = int(time.time() * 1000)
        
        # 数据内容 (32字节)
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 系统下发当前时间戳
        data += struct.pack('<I', 1)     # 指控指令: 1-搜索并自动跟踪
        data += struct.pack('<I', 0)     # 水平搜索开始角度
        data += struct.pack('<I', 0)     # 水平搜索结束角度
        data += struct.pack('<i', 0)     # 俯仰搜索开始角度
        data += struct.pack('<i', 0)     # 俯仰搜索结束角度
        data += struct.pack('<I', 0)     # 预留/目标编号
        
        packet = bytearray()
        packet.extend(START_BITS)
        packet.extend(struct.pack('<I', PROTOCOL_VERSION))
        packet.extend(struct.pack('<I', 20 + len(data)))
        packet.extend(struct.pack('<I', 0x04))  # 命令码 0x04
        packet.extend(struct.pack('<Q', timestamp))
        packet.extend(data)
        packet.extend(struct.pack('<I', self.seq))
        packet.extend(struct.pack('<I', 0))
        packet.extend(STOP_BITS)
        
        self.sock.sendto(packet, (self.device_ip, self.port))
        print(f"[发送] 开始搜索跟踪, 序列号={self.seq}")
        self.seq += 1
        
    def release_target(self):
        """释放目标"""
        START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
        STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
        PROTOCOL_VERSION = 9002
        
        timestamp = int(time.time() * 1000)
        
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 系统下发当前时间戳
        data += struct.pack('<I', 3)     # 指控指令: 3-释放
        data += struct.pack('<I', 0)     # 水平搜索开始角度
        data += struct.pack('<I', 0)     # 水平搜索结束角度
        data += struct.pack('<i', 0)     # 俯仰搜索开始角度
        data += struct.pack('<i', 0)     # 俯仰搜索结束角度
        data += struct.pack('<I', 0)     # 预留/目标编号
        
        packet = bytearray()
        packet.extend(START_BITS)
        packet.extend(struct.pack('<I', PROTOCOL_VERSION))
        packet.extend(struct.pack('<I', 20 + len(data)))
        packet.extend(struct.pack('<I', 0x04))
        packet.extend(struct.pack('<Q', timestamp))
        packet.extend(data)
        packet.extend(struct.pack('<I', self.seq))
        packet.extend(struct.pack('<I', 0))
        packet.extend(STOP_BITS)
        
        self.sock.sendto(packet, (self.device_ip, self.port))
        print(f"[发送] 释放目标, 序列号={self.seq}")
        self.seq += 1


# ========== 测试 ==========
if __name__ == "__main__":
    controller = OpticalController(
        device_ip="192.168.0.4",
        local_ip="192.168.0.9",
        port=9966
    )
    controller.connect()
    
    print("\n开始测试...")
    
    # 测试1: 先释放当前目标
    print("\n[测试1] 释放目标")
    controller.release_target()
    time.sleep(1)
    
    # 测试2: 转动到方位0度
    print("\n[测试2] 转动到方位=0°, 俯仰=0°")
    controller.send_angle(10, 40)
    time.sleep(5)
    
    # 测试3: 转动到方位45度
    print("\n[测试3] 转动到方位=45°, 俯仰=10°")
    controller.send_angle(120, -30)
    time.sleep(5)
    
    # 测试4: 开始跟踪
    print("\n[测试4] 开始搜索跟踪")
    controller.start_track()
    
    print("\n测试完成！观察光电是否转动")

# # ========== 主程序 ==========
# if __name__ == "__main__":
#     monitor = OpticalMonitor()
#     print("正在接收光电数据...")
#     print("当前转台状态将实时显示")
#     print("=" * 60)
    
#     try:
#         monitor.start()
#     except KeyboardInterrupt:
#         print("\n退出")