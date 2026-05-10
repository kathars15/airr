# test_0x03.py
"""
简单的 0x03 命令测试程序
"""

import socket
import struct
import time
import math

# ==================== 协议常量 ====================
START_BITS = bytes([0x88, 0x89, 0x80, 0x8A])
STOP_BITS = bytes([0x89, 0x80, 0x8A, 0x8B])
PROTOCOL_VERSION = 9002

CMD_SET_POSITION = 0x03  # 设置光电目址信息包


class SimpleOpticalController:
    def __init__(self, device_ip="10.129.41.98", local_ip="10.129.41.89", port=9966):
        self.device_ip = device_ip
        self.local_ip = local_ip
        self.port = port
        self.sock = None
        self.seq = 1

    def connect(self):
        """建立连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.local_ip, self.port))
            self.sock.settimeout(2.0)
            print(f"[连接] 成功绑定 {self.local_ip}:{self.port}")
            print(f"[连接] 目标设备 {self.device_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[连接] 失败: {e}")
            return False

    def close(self):
        """关闭连接"""
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
        
        :param azimuth: 方位角（度）0-360
        :param pitch: 俯仰角（度）-90到90
        :param distance: 距离（米）
        """
        print(f"\n[命令] 0x03 转到位置")
        print(f"       方位角: {azimuth:.1f}°")
        print(f"       俯仰角: {pitch:.1f}°")
        print(f"       距离: {distance:.0f}m")
        
        # 构建0x03数据包
        # 格式：光电编号(4) + 系统编号(4) + 时间戳(8) + 水平角度(8) + 俯仰角度(8) + 距离(8)
        data = struct.pack('<I', 0)           # 光电编号
        data += struct.pack('<I', 0)          # 系统编号
        data += struct.pack('<Q', 0)          # 时间戳（预留）
        data += struct.pack('<d', float(azimuth))   # 水平角度
        data += struct.pack('<d', float(pitch))     # 俯仰角度
        data += struct.pack('<d', float(distance))  # 距离
        
        # 打印调试信息
        print(f"[调试] 数据长度: {len(data)} 字节")
        print(f"[调试] 十六进制: {data.hex()}")
        
        self._send_packet(CMD_SET_POSITION, data)
        print(f"[发送] 0x03命令已发送")

    def release(self):
        """释放目标"""
        print(f"\n[命令] 释放目标")
        data = struct.pack('<I', 0)
        data += struct.pack('<Q', 0)
        data += struct.pack('<I', 3)  # 释放
        data += struct.pack('<I', 0)
        data += struct.pack('<I', 0)
        data += struct.pack('<i', 0)
        data += struct.pack('<i', 0)
        data += struct.pack('<I', 0)
        self._send_packet(0x04, data)
        print(f"[发送] 释放命令已发送")


def main():
    print("=" * 60)
    print("0x03 命令测试程序")
    print("=" * 60)
    
    # 配置
    device_ip = input("光电设备IP [10.129.41.98]: ").strip()
    if not device_ip:
        device_ip = "10.129.41.98"
    
    local_ip = input("本机IP [10.129.41.89]: ").strip()
    if not local_ip:
        local_ip = "10.129.41.89"
    
    # 创建控制器
    controller = SimpleOpticalController(device_ip, local_ip)
    
    if not controller.connect():
        print("连接失败")
        return
    
    print("\n" + "=" * 60)
    print("命令说明:")
    print("  t - 发送0x03命令（转到指定位置）")
    print("  r - 释放目标")
    print("  q - 退出")
    print("=" * 60)
    
    while True:
        try:
            cmd = input("\n请输入命令: ").strip().lower()
            
            if cmd == 'q':
                break
            
            elif cmd == 'r':
                controller.release()
            
            elif cmd == 't':
                # 输入方位角
                azimuth = float(input("  方位角(度, 0-360) [34]: ") or "34")
                
                # 输入俯仰角
                pitch = float(input("  俯仰角(度, -90-90) [-3.2]: ") or "-3.2")
                
                # 输入距离
                distance = float(input("  距离(米) [300]: ") or "300")
                
                # 发送命令
                controller.goto_position(azimuth, pitch, distance)
                
                print("\n[提示] 观察光电是否转动到指定位置")
                print("       如果没反应，说明设备可能不支持0x03命令")
            
            else:
                print("未知命令，请输入 t/r/q")
                
        except KeyboardInterrupt:
            break
        except ValueError:
            print("请输入有效的数字")
        except Exception as e:
            print(f"错误: {e}")
    
    controller.close()
    print("\n程序退出")


if __name__ == "__main__":
    main()