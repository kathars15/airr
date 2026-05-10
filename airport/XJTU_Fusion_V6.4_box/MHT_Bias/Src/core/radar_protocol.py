# -*- coding: utf-8 -*-

import socket
import struct
import traceback

from core.app_config import (
    FRAME_HEAD_CONTROL, FRAME_HEAD_END, FRAME_HEAD_POINT, FRAME_HEAD_STATUS,
    FRAME_HEAD_TRACK, FRAME_TAIL_CONTROL, RADAR_IP, RADAR_PORT,
)
from core.console_utils import safe_print

def calculate_checksum_32bit(data, start=0, end=None):
    """计算32位校验和"""
    if end is None:
        end = len(data)
    checksum = 0
    for i in range(start, end, 4):
        if i + 4 <= len(data):
            value = struct.unpack('<I', data[i:i+4])[0]
            checksum += value
    return checksum & 0xFFFFFFFF

def parse_radar_status_packet(data):
    """解析系统状态包"""
    if len(data) < 128:
        return None
    
    result = {
        'frame_head': struct.unpack('<I', data[0:4])[0],
        'frame_len': struct.unpack('<I', data[4:8])[0],
        'center_computer_status': struct.unpack('<i', data[8:12])[0],
        'data_process_status': struct.unpack('<i', data[12:16])[0],
        'signal_process_status': struct.unpack('<i', data[16:20])[0],
        'fpga_temp': struct.unpack('<i', data[20:24])[0],
        'dsp1_temp': struct.unpack('<i', data[24:28])[0],
        'freq_synth_status': struct.unpack('<i', data[40:44])[0],
        'servo_status': struct.unpack('<i', data[44:48])[0],
        'beam_control_status': struct.unpack('<i', data[48:52])[0],
    }
    
    checksum_pos = 124
    result['checksum'] = struct.unpack('<I', data[checksum_pos:checksum_pos+4])[0]
    result['frame_tail'] = struct.unpack('<I', data[128:132])[0] if len(data) >= 132 else 0

    return result

def parse_radar_point_packet(data):
    """解析点迹数据包 - 动态计算偏移"""
    if len(data) < 12:
        return None, None
    
    frame_head = struct.unpack('<I', data[0:4])[0]
    if frame_head != FRAME_HEAD_POINT:
        return None, None
    
    frame_len = struct.unpack('<I', data[4:8])[0]
    point_num = struct.unpack('<I', data[8:12])[0]
    
    # safe_print(f"[点迹解析] 数据长度={len(data)}, 帧长度字段={frame_len}, 点迹数={point_num}")
    
    # ========== 动态计算惯导信息长度 ==========
    if point_num == 0:
        # 没有点迹数据，惯导信息一直到数据包末尾（减去帧尾4字节）
        ins_info_size = len(data) - 12 - 4  # 减去帧头(12)和帧尾(4)
    else:
        # 有点迹数据，每个点迹48字节，惯导信息长度 = 总长度 - 帧头(12) - 点迹总长度 - 帧尾(4) - 校验和(4)
        ins_info_size = len(data) - 12 - point_num * 48 - 8
    
    # safe_print(f"[点迹解析] 惯导信息长度={ins_info_size}字节")
    
    # ========== 解析惯导信息 ==========
    ins_info = {}
    if ins_info_size >= 20:
        offset = 12
        
        ins_info = {
            'dsp_time': struct.unpack('<I', data[offset+0:offset+4])[0] if ins_info_size >= 4 else 0,
            'gps_time': struct.unpack('<I', data[offset+4:offset+8])[0] if ins_info_size >= 8 else 0,
            'gps_lon': struct.unpack('<i', data[offset+8:offset+12])[0] / 100000.0 if ins_info_size >= 12 else 0,
            'gps_lat': struct.unpack('<i', data[offset+12:offset+16])[0] / 100000.0 if ins_info_size >= 16 else 0,
            'gps_alt_sat': struct.unpack('<I', data[offset+16:offset+20])[0] if ins_info_size >= 20 else 0,
            'radar_heading': struct.unpack('<I', data[offset+20:offset+24])[0] / 100.0 if ins_info_size >= 24 else 0,
            'radar_true_heading': struct.unpack('<I', data[offset+24:offset+28])[0] / 100.0 if ins_info_size >= 28 else 0,
            'roll_angle': struct.unpack('<i', data[offset+28:offset+32])[0] / 100.0 if ins_info_size >= 32 else 0,
            'pitch_angle': struct.unpack('<i', data[offset+32:offset+36])[0] / 100.0 if ins_info_size >= 36 else 0,
            'east_speed': struct.unpack('<i', data[offset+36:offset+40])[0] / 100.0 if ins_info_size >= 40 else 0,
            'north_speed': struct.unpack('<i', data[offset+40:offset+44])[0] / 100.0 if ins_info_size >= 44 else 0,
            'up_speed': struct.unpack('<i', data[offset+44:offset+48])[0] / 100.0 if ins_info_size >= 48 else 0,
            'ground_speed': struct.unpack('<i', data[offset+48:offset+52])[0] / 100.0 if ins_info_size >= 52 else 0,
            'frame_cnt': struct.unpack('<i', data[offset+60:offset+64])[0] if ins_info_size >= 64 else 0,
            'frame_time': struct.unpack('<i', data[offset+64:offset+68])[0] if ins_info_size >= 68 else 0,
        }
        ins_info['gps_altitude'] = (ins_info.get('gps_alt_sat', 0) >> 16) & 0xFFFF
        ins_info['satellite_num'] = ins_info.get('gps_alt_sat', 0) & 0xFFFF
        
        # safe_print(f"[点迹惯导] GPS时间: {ins_info.get('gps_time', 0)}, "
        #       f"雷达航向: {ins_info.get('radar_heading', 0):.1f}°, "
        #       f"帧计数: {ins_info.get('frame_cnt', 0)}")
    
    # ========== 解析点迹信息 ==========
    points = []
    
    if point_num > 0:
        # 点迹数据起始偏移 = 12 + ins_info_size
        point_offset = 12 + ins_info_size
        point_size = 48  # 每个点迹48字节（12个uint32）
        
        
        for i in range(point_num):
            offset = point_offset + i * point_size
            if offset + point_size > len(data):
                break
            
            # 解析各个字段
            target_id = struct.unpack('<I', data[offset:offset+4])[0]
            range_val = struct.unpack('<I', data[offset+4:offset+8])[0] / 10.0
            
            angle_info = struct.unpack('<I', data[offset+8:offset+12])[0]
            pitch = (angle_info >> 16) & 0xFFFF
            azimuth = angle_info & 0xFFFF
            if pitch > 32767:
                pitch = pitch - 65536
            pitch_deg = pitch / 10.0
            azimuth_deg = azimuth / 10.0
            
            # 速度/类型信息
            doppler_type_speed = struct.unpack('<I', data[offset+24:offset+28])[0]
            doppler = (doppler_type_speed >> 22) & 0x3FF
            target_type = (doppler_type_speed >> 19) & 0x7
            speed_dir = (doppler_type_speed >> 16) & 0x7
            speed_raw = doppler_type_speed & 0xFFFF
            if speed_raw > 32767:
                speed_raw = speed_raw - 65536
            speed_ms = speed_raw / 10.0
            
            # 标志位
            flags = struct.unpack('<I', data[offset+44:offset+48])[0]
            is_true_point = flags & 0x01
            
            point = {
                'target_id': target_id,
                'range': range_val,
                'azimuth': azimuth_deg,
                'pitch': pitch_deg,
                'speed': speed_ms,
                'speed_dir': speed_dir,
                'doppler': doppler,
                'target_type': target_type,
                'is_true_point': is_true_point,
            }
            points.append(point)
            
            # # 调试打印前3个点迹
            # if i < 6:
            #     safe_print(f"[点迹{i}] ID={target_id}, 距离={range_val:.1f}m, "
            #           f"方位={azimuth_deg:.1f}°, 速度={speed_ms:.1f}m/s")
    
    # safe_print(f"[点迹解析] 成功解析 {len(points)} 个点迹")
    return points, ins_info

def parse_radar_track_packet(data):
    """解析航迹数据包 - 动态计算偏移"""
    if len(data) < 12:
        return None, None
    
    frame_head = struct.unpack('<I', data[0:4])[0]
    if frame_head != FRAME_HEAD_TRACK:
        return None, None
    
    frame_len = struct.unpack('<I', data[4:8])[0]
    track_num = struct.unpack('<I', data[8:12])[0]
    
    
    # ========== 动态计算惯导信息长度 ==========
    if track_num == 0:
        # 没有航迹数据，惯导信息一直到数据包末尾（减去帧尾4字节）
        ins_info_size = len(data) - 12 - 4  # 减去帧头(12)和帧尾(4)
    else:
        # 有航迹数据，惯导信息长度 = 总长度 - 帧头(12) - 航迹总长度 - 帧尾(4) - 校验和(4)
        ins_info_size = len(data) - 12 - track_num * 64 - 8
    
    
    # ========== 解析惯导信息 ==========
    ins_info = {}
    if ins_info_size >= 20:
        # 惯导信息从偏移12开始
        offset = 12
        
        ins_info = {
            'dsp_time': struct.unpack('<I', data[offset+0:offset+4])[0] if ins_info_size >= 4 else 0,
            'gps_time': struct.unpack('<I', data[offset+4:offset+8])[0] if ins_info_size >= 8 else 0,
            'gps_lon': struct.unpack('<i', data[offset+8:offset+12])[0] / 100000.0 if ins_info_size >= 12 else 0,
            'gps_lat': struct.unpack('<i', data[offset+12:offset+16])[0] / 100000.0 if ins_info_size >= 16 else 0,
            'gps_alt_sat': struct.unpack('<I', data[offset+16:offset+20])[0] if ins_info_size >= 20 else 0,
            'radar_heading': struct.unpack('<I', data[offset+20:offset+24])[0] / 100.0 if ins_info_size >= 24 else 0,
            'radar_true_heading': struct.unpack('<I', data[offset+24:offset+28])[0] / 100.0 if ins_info_size >= 28 else 0,
            'roll_angle': struct.unpack('<i', data[offset+28:offset+32])[0] / 100.0 if ins_info_size >= 32 else 0,
            'pitch_angle': struct.unpack('<i', data[offset+32:offset+36])[0] / 100.0 if ins_info_size >= 36 else 0,
            'east_speed': struct.unpack('<i', data[offset+36:offset+40])[0] / 100.0 if ins_info_size >= 40 else 0,
            'north_speed': struct.unpack('<i', data[offset+40:offset+44])[0] / 100.0 if ins_info_size >= 44 else 0,
            'up_speed': struct.unpack('<i', data[offset+44:offset+48])[0] / 100.0 if ins_info_size >= 48 else 0,
            'ground_speed': struct.unpack('<i', data[offset+48:offset+52])[0] / 100.0 if ins_info_size >= 52 else 0,
            'frame_cnt': struct.unpack('<i', data[offset+60:offset+64])[0] if ins_info_size >= 64 else 0,
            'frame_time': struct.unpack('<i', data[offset+64:offset+68])[0] if ins_info_size >= 68 else 0,
        }
        ins_info['gps_altitude'] = (ins_info.get('gps_alt_sat', 0) >> 16) & 0xFFFF
        ins_info['satellite_num'] = ins_info.get('gps_alt_sat', 0) & 0xFFFF
        
        # safe_print(f"[惯导] GPS时间: {ins_info.get('gps_time', 0)}, "
        #       f"雷达航向: {ins_info.get('radar_heading', 0):.1f}°, "
        #       f"帧计数: {ins_info.get('frame_cnt', 0)}")
    
    # ========== 解析航迹信息 ==========
    tracks = []
    
    if track_num > 0:
        # 航迹数据起始偏移 = 12 + ins_info_size
        track_offset = 12 + ins_info_size
        # safe_print(f"[parse] 航迹数据起始偏移={track_offset}, 每个航迹64字节")
        
        for i in range(track_num):
            offset = track_offset + i * 64
            if offset + 64 > len(data):
                # safe_print(f"[parse] 航迹{i} 超出数据范围 (offset={offset}, len={len(data)})")
                break
            
            # 解析各个字段
            display_id = struct.unpack('<I', data[offset:offset+4])[0]
            absolute_id = struct.unpack('<I', data[offset+4:offset+8])[0]
            
            # 距离 (单位: 0.1米)
            range_raw = struct.unpack('<I', data[offset+8:offset+12])[0]
            range_val = range_raw / 10.0
            
            # 角度信息 (第4个字)
            angle_info = struct.unpack('<I', data[offset+12:offset+16])[0]
            azimuth = angle_info & 0xFFFF
            pitch_raw = (angle_info >> 16) & 0xFFFF
            if pitch_raw > 32767:
                pitch_raw = pitch_raw - 65536
            
            azimuth_relative = azimuth / 10.0
            pitch_deg = pitch_raw / 10.0
            
            # 获取雷达航向角（从惯导信息）
            radar_heading = ins_info.get('radar_heading', 0)
            
            # 转换为绝对方位角（真北方向）
            azimuth_absolute = azimuth_relative + radar_heading
            if azimuth_absolute >= 360:
                azimuth_absolute -= 360
            elif azimuth_absolute < 0:
                azimuth_absolute += 360
            
            # 速度和类型 (第7个字)
            type_speed = struct.unpack('<I', data[offset+28:offset+32])[0]
            target_type = (type_speed >> 19) & 0x7
            speed_raw = type_speed & 0xFFFF
            if speed_raw > 32767:
                speed_raw = speed_raw - 65536
            speed_ms = speed_raw / 10.0
            
            # 航迹标志 (第8个字)
            track_flag = struct.unpack('<I', data[offset+36:offset+40])[0]
            is_tas = (track_flag >> 30) & 0x3
            
            # 高度 (第9个字)
            height_raw = struct.unpack('<I', data[offset+60:offset+64])[0]
            height = height_raw & 0xFFFF
            if height > 32767:
                height = height - 65536
            
            # 只添加合理航迹
            if range_val > 0 and range_val < 100000:
                track = {
                    'display_id': display_id,
                    'absolute_id': absolute_id,
                    'range': range_val,
                    'azimuth': azimuth_absolute,
                    'pitch': pitch_deg,
                    'target_type': target_type,
                    'speed': speed_ms,
                    'is_tas': is_tas,
                    'height': height,
                }
                tracks.append(track)
                
    #             # 调试打印前6个航迹
    #             if i < 6:
    #                 safe_print(f"[航迹{i}] ID={display_id}, 距离={range_val:.1f}m, "
    #                       f"方位={azimuth_absolute:.1f}°, 速度={speed_ms:.1f}m/s")
    
    # safe_print(f"[parse] 成功解析 {len(tracks)} 个航迹")
    return tracks, ins_info

def parse_radar_end_packet(data):
    """解析结束标志包"""
    if len(data) < 32:
        return False
    frame_head = struct.unpack('<I', data[0:4])[0]
    return frame_head == FRAME_HEAD_END

def build_control_packet(radar_on=True, radiation_on=True, work_mode=1,
                         azimuth_scan_mode=4,  # 4=机扫周扫
                         azimuth_start=-180, azimuth_end=180,
                         azimuth_step=0,  # 0表示不使用步进（周扫）
                         pitch_scan_mode=0,  # 0=定向
                         pitch_start=0, pitch_end=0,
                         pitch_step=0,
                         frequency=16000,  # 工作频率 MHz
                         signal_tw=100,  # 信号时宽 us
                         stc=0, mgc=0,
                         cfar_type=0, cfar_ref=16, cfar_guard=8, cfar_threshold=5,
                         radar_height=0,  # 雷达架设高度 m
                         longitude=0, latitude=0):  # 雷达经纬度（度）
    """
    构建雷达控制参数包
    """
    # 控制参数共64个short (128字节)
    packet = []
    
    # 辅助函数：安全地将值转换为uint16
    def to_uint16(val):
        """安全转换为0-65535范围内的整数"""
        val_int = int(val)
        if val_int < 0:
            return 0
        if val_int > 65535:
            return 65535
        return val_int
    
    # 1. 帧头
    packet.append(FRAME_HEAD_CONTROL)  # 0xAA55
    
    # 2. 本帧信息长度 (64个short)
    packet.append(64)
    
    # 3. 辐射开关 (0=关, 1=开)
    packet.append(1 if radiation_on else 0)
    
    # 4. 雷达开关 (0=关, 1=开)
    packet.append(1 if radar_on else 0)
    
    # 5. 雷达工作模式 (1-6)
    packet.append(to_uint16(work_mode))
    
    # 6-7. 备份
    packet.extend([0, 0])
    
    # 8. 方位扫描模式
    packet.append(to_uint16(azimuth_scan_mode))
    
    # 9. 方位扫描起始角 (*10) - 范围[-1800, 1800]
    start_val = int(azimuth_start * 10)
    start_val = max(-1800, min(1800, start_val))
    if start_val < 0:
        start_val = 65536 + start_val  # 转为有符号
    packet.append(to_uint16(start_val))
    
    # 10. 方位扫描终止角 (*10)
    end_val = int(azimuth_end * 10)
    end_val = max(-1800, min(1800, end_val))
    if end_val < 0:
        end_val = 65536 + end_val
    packet.append(to_uint16(end_val))
    
    # 11. 方位波束步进角度 (*10)
    step_val = int(azimuth_step * 10)
    packet.append(to_uint16(step_val))
    
    # 12. 俯仰扫描模式
    packet.append(to_uint16(pitch_scan_mode))
    
    # 13. 俯仰扫描起始角 (*10) - 范围[-250, 250]
    pitch_start_val = int(pitch_start * 10)
    pitch_start_val = max(-250, min(250, pitch_start_val))
    if pitch_start_val < 0:
        pitch_start_val = 65536 + pitch_start_val
    packet.append(to_uint16(pitch_start_val))
    
    # 14. 俯仰扫描终止角 (*10)
    pitch_end_val = int(pitch_end * 10)
    pitch_end_val = max(-250, min(250, pitch_end_val))
    if pitch_end_val < 0:
        pitch_end_val = 65536 + pitch_end_val
    packet.append(to_uint16(pitch_end_val))
    
    # 15. 俯仰波束步进角度 (*10)
    pitch_step_val = int(pitch_step * 10)
    packet.append(to_uint16(pitch_step_val))
    
    # 16-20. 备份 (5个)
    packet.extend([0, 0, 0, 0, 0])
    
    # 21. AD采样延迟 (默认20us)
    packet.append(20)
    
    # 22-24. 备份 (3个)
    packet.extend([0, 0, 0])
    
    # 25. 工作频率 (MHz) - 范围0-65535
    packet.append(to_uint16(frequency))
    
    # 26. 信号时宽 (us)
    packet.append(to_uint16(signal_tw))
    
    # 27. STC
    packet.append(to_uint16(stc))
    
    # 28. MGC
    packet.append(to_uint16(mgc))
    
    # 29. CFAR参数
    packet.append(to_uint16(cfar_type))
    
    # 30. CFAR参考单元数
    packet.append(to_uint16(cfar_ref))
    
    # 31. CFAR保护单元数
    packet.append(to_uint16(cfar_guard))
    
    # 32. CFAR检测信噪比
    packet.append(to_uint16(cfar_threshold))
    
    # 33. 检测时扣掉的杂波通道数
    packet.append(3)
    
    # 34-56. 备份 (23个)
    packet.extend([0] * 23)
    
    # 57. 雷达零位指向角 (0-3599)
    packet.append(0)
    
    # 58. 用户高度输入/显示模式
    height_val = int(radar_height)
    height_val = max(0, min(8191, height_val))  # 13位最大值8191
    height_value = (height_val << 3) | 3
    packet.append(to_uint16(height_value))
    
    # 59-60. 经度输入值 (高16bits + 低16bits)
    lon_int = int(longitude * 10000000)
    lon_int = max(-2147483648, min(2147483647, lon_int))
    packet.append(to_uint16((lon_int >> 16) & 0xFFFF))
    packet.append(to_uint16(lon_int & 0xFFFF))
    
    # 61-62. 纬度输入值
    lat_int = int(latitude * 10000000)
    lat_int = max(-2147483648, min(2147483647, lat_int))
    packet.append(to_uint16((lat_int >> 16) & 0xFFFF))
    packet.append(to_uint16(lat_int & 0xFFFF))
    
    # 63. 备份
    packet.append(0)
    
    # 64. 帧尾
    packet.append(FRAME_TAIL_CONTROL)
    
    # 确保长度正确
    if len(packet) != 64:
        safe_print(f"[警告] 控制包长度不正确: {len(packet)}, 应该是64")
        # 补齐或截断
        while len(packet) < 64:
            packet.append(0)
        packet = packet[:64]
    
    # 转换为字节数组 (每个short转2字节，小端序)
    result = bytearray()
    for val in packet:
        try:
            result.extend(struct.pack('>H', val))
        except struct.error as e:
            safe_print(f"[错误] struct.pack失败: val={val}, 错误={e}")
            result.extend(struct.pack('>H', 0))
    
    return bytes(result)

def send_control_packet(radar_ip=RADAR_IP, radar_port=RADAR_PORT, **kwargs):
    """
    发送控制命令到雷达
    """
    try:
        # 验证IP地址
        if radar_ip == "10.129.41.99" and not any(
            [radar_ip.startswith("192.168."), radar_ip == "127.0.0.1", radar_ip.startswith("10.")]):
            safe_print(f"[控制] 警告: 雷达IP {radar_ip} 可能不可达")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        
        packet = build_control_packet(**kwargs)
        
        safe_print(f"[控制] 完整数据(hex): {packet.hex()}")
        
        sock.sendto(packet, (radar_ip, radar_port))
        
        # 可选：等待确认
        try:
            data, addr = sock.recvfrom(1024)
            safe_print(f"[控制] 收到响应: {len(data)}字节")
        except socket.timeout:
            safe_print("[控制] 无响应（正常，雷达可能不回复）")
        
        sock.close()
        return True
        
    except Exception as e:
        safe_print(f"[控制] 发送失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def init_radar_with_defaults():
    """使用默认参数初始化雷达"""
    safe_print("[初始化] 正在初始化雷达...")
    
    # 从配置文件读取雷达位置（如果有）
    try:
        from Sensor_Config.sensor_config import lla_original
        longitude = lla_original[1]  # 经度
        latitude = lla_original[0]   # 纬度
        height = lla_original[2]     # 高度
    except:
        longitude = 0
        latitude = 0
        height = 0
    
    success = send_control_packet(
        radar_on=True,
        radiation_on=True,
        work_mode=1,
        azimuth_scan_mode=4,  # 周扫
        azimuth_start=-180,
        azimuth_end=180,
        pitch_scan_mode=0,  # 定向
        pitch_start=0,
        frequency=16000,
        radar_height=int(height),
        longitude=longitude,
        latitude=latitude
    )
    
    if success:
        safe_print("[初始化] 雷达初始化完成")
    else:
        safe_print("[初始化] 雷达初始化失败，请检查网络连接")
    
    return success

