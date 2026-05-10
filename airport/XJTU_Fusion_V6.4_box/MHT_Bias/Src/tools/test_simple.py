import socket
import struct
import time

class TestBothCommands:
    """测试0x03和0x16两种绝对角度命令"""
    
    def __init__(self, device_ip="192.168.0.4", local_ip="192.168.0.9", port=9966):
        self.device_ip = device_ip
        self.local_ip = local_ip
        self.port = port
        self.sock = None
        self.seq = 1
        
        self.START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
        self.STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
        self.PROTOCOL_VERSION = 9002
        
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.port))
        print(f"已连接: {self.local_ip}:{self.port} -> {self.device_ip}:{self.port}")
        
    def close(self):
        if self.sock:
            self.sock.close()
            
    def _send_packet(self, cmd, data):
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
        
        self.sock.sendto(packet, (self.device_ip, self.port))
        print(f"  发送 0x{cmd:02X}, 包长={len(packet)}")
        self.seq += 1
        
    def cmd_03(self, azimuth, pitch):
        """协议2.3 设置光电目址信息包"""
        print(f"\n[0x03] 转到 方位={azimuth}°, 俯仰={pitch}°")
        
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<Q', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 时间戳
        data += struct.pack('<d', 0.0)   # 经度
        data += struct.pack('<d', 0.0)   # 纬度
        data += struct.pack('<d', 0.0)   # 高度
        data += struct.pack('<d', float(azimuth))
        data += struct.pack('<d', float(pitch))
        data += struct.pack('<I', 0)     # 距离
        data += struct.pack('<B', 0)     # 运动方向
        data += struct.pack('<B', 0)     # 搜索模式
        data += struct.pack('<H', 0)     # 保留
        
        self._send_packet(0x03, data)
        
    def cmd_16(self, azimuth, pitch):
        """协议2.4 设置光电目址扩展信息包"""
        print(f"\n[0x16] 转到 方位={azimuth}°, 俯仰={pitch}°")
        
        data = struct.pack('<I', 0)      # 光电编号
        data += struct.pack('<I', 0)     # 系统编号
        data += struct.pack('<Q', 0)     # 系统下发时间戳
        data += struct.pack('<d', 0.0)   # 目标经度
        data += struct.pack('<d', 0.0)   # 目标纬度
        data += struct.pack('<d', 0.0)   # 目标高度
        data += struct.pack('<I', 0)     # 显示距离
        data += struct.pack('<I', 0)     # 实际距离
        data += struct.pack('<d', float(azimuth))   # 水平角度
        data += struct.pack('<d', float(pitch))     # 俯仰角度
        data += struct.pack('<H', 0)     # 用户ID
        data += struct.pack('<B', 0)     # 引导模式
        data += struct.pack('<B', 0)     # 目标运动方向
        data += struct.pack('<I', 0)     # 搜索模式
        data += struct.pack('<I', 0)     # 左右搜索视场角大小
        data += struct.pack('<I', 0)     # 上下搜索视场角大小
        data += struct.pack('<I', 0)     # 保留
        
        self._send_packet(0x16, data)
        
    def release(self):
        """释放目标"""
        print(f"\n[释放]")
        data = struct.pack('<I', 0)
        data += struct.pack('<Q', 0)
        data += struct.pack('<I', 3)
        data += struct.pack('<I', 0) * 5
        self._send_packet(0x04, data)
        
    def receive_feedback(self, duration=3):
        """接收反馈数据"""
        print(f"\n接收反馈 {duration} 秒...")
        self.sock.settimeout(0.5)
        start = time.time()
        while time.time() - start < duration:
            try:
                data, addr = self.sock.recvfrom(4096)
                cmd = struct.unpack('<I', data[12:16])[0]
                print(f"  收到命令: 0x{cmd:02X}, 长度={len(data)}")
            except socket.timeout:
                pass


# ========== 测试 ==========
if __name__ == "__main__":
    tester = TestBothCommands()
    tester.connect()
    
    print("\n" + "="*60)
    print("测试0x03和0x16命令")
    print("="*60)
    
    try:
        # 先释放
        tester.release()
        time.sleep(1)
        
        # 测试0x03命令
        tester.cmd_16(0, 0)
        tester.receive_feedback(3)
        
        # 再释放
        tester.release()
        time.sleep(1)
        
        # 测试0x16命令
        tester.cmd_16(45, 10)
        tester.receive_feedback(3)
        
        # 再释放
        tester.release()
        time.sleep(1)
        
        # 测试0x16命令
        tester.cmd_16(90, 20)
        tester.receive_feedback(3)

        print("\n" + "="*60)
        print("测试完成")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        tester.close()