# -*- coding: utf-8 -*-
import enum
from math import nan
from operator import index
import string
import numpy as np
import sys, pickle, csv
import socket
import struct

from datetime import datetime, timedelta

"""
使用前
修改Classify/Initial_Params.py和本文件的路径
!!!!!!
"""
# sys.path.append('/home/wy/桌面/设备代码/XJTU_Fusion_V6.4_box/MHT_Bias/')
PROJECT_ROOT = r'D:\\desk\\air\\airport\\XJTU_Fusion_V6.4_box'
MHT_BIAS_PATH = r'D:\\desk\\air\\airport\\XJTU_Fusion_V6.4_box\\MHT_Bias'

# 添加路径到系统路径
sys.path.append(PROJECT_ROOT)
sys.path.append(MHT_BIAS_PATH)

import signal
from MHT.POMHT import POMHT_Bias
from copy import copy, deepcopy
import time, json, pickle
import scipy.io
import pandas as pd
from common.clusters import Clustering_Obs
from common.utlis import Angle_to_Rotation_3D, geodetic_to_enu, time_string_to_timestamp, enu_to_geodetic
from collections import deque
import multiprocessing
import keyboard
import os
import queue
from datetime import datetime
from common.Tracker import Get_H_k, Get_F_G_CV, Cal_Gate, KF_Prediction_CV, KF_Update
from Classify.TrackingClassify import TrackingClassify
import uuid
import threading


import datetime

def gps_time_to_datetime(gps_ms):
    """
    将GPS时间（毫秒）转换为 datetime 对象
    GPS时间通常从 2000-01-01 00:00:00 开始计算
    """
    # GPS纪元：2000年1月1日 00:00:00 UTC
    gps_epoch = datetime.datetime(2000, 1, 1, 0, 0, 0)
    
    # 转换为秒
    gps_seconds = gps_ms / 1000.0
    
    # 计算具体时间
    dt = gps_epoch + datetime.timedelta(seconds=gps_seconds)
    
    return dt

def format_gps_time(gps_ms):
    """格式化GPS时间为易读的字符串"""
    dt = gps_time_to_datetime(gps_ms)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 精确到毫秒

"""
UDP雷达通信：接收雷达数据，实时处理并UDP发送结果。
进程1：UDP接收雷达数据，解析并加入处理队列  
进程2：MHT跟踪，并UDP发送结果
"""
from Sensor_Config.sensor_config import Sensor_Config, SignalType2Name, lla_original, SingleFrameDt


# ==================== UDP通信配置 ====================
# 雷达配置（根据文档）
RADAR_IP = "10.129.41.99"  # 雷达IP
HOST_IP = "10.129.41.89"    # 主机IP

# RADAR_IP = "127.0.0.1"  # 雷达IP
# HOST_IP = "127.0.0.1"    # 主机IP

RADAR_PORT = 8080          # 雷达端口
HOST_PORT = 6000          # 主机监听端口

# 雷达协议帧头/帧尾定义
FRAME_HEAD_STATUS = 0xA0A0A7A7
FRAME_TAIL_STATUS = 0x7A7A0A0A

FRAME_HEAD_POINT = 0xA1A1A8A8
FRAME_TAIL_POINT = 0x8A8A1A1A

FRAME_HEAD_TRACK = 0xA3A3AAAA
FRAME_TAIL_TRACK = 0xAAAA3A3A

FRAME_HEAD_END = 0xA8A8AFAF
FRAME_TAIL_END = 0xFAFA8A8A


# ==================== 雷达协议解析函数 ====================

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
    
    print(f"[点迹解析] 数据长度={len(data)}, 帧长度字段={frame_len}, 点迹数={point_num}")
    
    # ========== 动态计算惯导信息长度 ==========
    if point_num == 0:
        # 没有点迹数据，惯导信息一直到数据包末尾（减去帧尾4字节）
        ins_info_size = len(data) - 12 - 4  # 减去帧头(12)和帧尾(4)
    else:
        # 有点迹数据，每个点迹48字节，惯导信息长度 = 总长度 - 帧头(12) - 点迹总长度 - 帧尾(4) - 校验和(4)
        ins_info_size = len(data) - 12 - point_num * 48 - 8
    
    print(f"[点迹解析] 惯导信息长度={ins_info_size}字节")
    
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
        
        print(f"[点迹惯导] GPS时间: {ins_info.get('gps_time', 0)}, "
              f"雷达航向: {ins_info.get('radar_heading', 0):.1f}°, "
              f"帧计数: {ins_info.get('frame_cnt', 0)}")
    
    # ========== 解析点迹信息 ==========
    points = []
    
    if point_num > 0:
        # 点迹数据起始偏移 = 12 + ins_info_size
        point_offset = 12 + ins_info_size
        point_size = 48  # 每个点迹48字节（12个uint32）
        
        print(f"[点迹解析] 点迹数据起始偏移={point_offset}, 每个点迹{point_size}字节")
        
        for i in range(point_num):
            offset = point_offset + i * point_size
            if offset + point_size > len(data):
                print(f"[点迹解析] 点迹{i} 超出数据范围 (offset={offset}, len={len(data)})")
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
            
            # 调试打印前3个点迹
            if i < 6:
                print(f"[点迹{i}] ID={target_id}, 距离={range_val:.1f}m, "
                      f"方位={azimuth_deg:.1f}°, 速度={speed_ms:.1f}m/s")
    
    print(f"[点迹解析] 成功解析 {len(points)} 个点迹")
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
        
        print(f"[惯导] GPS时间: {ins_info.get('gps_time', 0)}, "
              f"雷达航向: {ins_info.get('radar_heading', 0):.1f}°, "
              f"帧计数: {ins_info.get('frame_cnt', 0)}")
    
    # ========== 解析航迹信息 ==========
    tracks = []
    
    if track_num > 0:
        # 航迹数据起始偏移 = 12 + ins_info_size
        track_offset = 12 + ins_info_size
        print(f"[parse] 航迹数据起始偏移={track_offset}, 每个航迹64字节")
        
        for i in range(track_num):
            offset = track_offset + i * 64
            if offset + 64 > len(data):
                print(f"[parse] 航迹{i} 超出数据范围 (offset={offset}, len={len(data)})")
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
                
                # 调试打印前6个航迹
                if i < 6:
                    print(f"[航迹{i}] ID={display_id}, 距离={range_val:.1f}m, "
                          f"方位={azimuth_absolute:.1f}°, 速度={speed_ms:.1f}m/s")
    
    print(f"[parse] 成功解析 {len(tracks)} 个航迹")
    return tracks, ins_info


def parse_radar_end_packet(data):
    """解析结束标志包"""
    if len(data) < 32:
        return False
    frame_head = struct.unpack('<I', data[0:4])[0]
    return frame_head == FRAME_HEAD_END


# ==================== UDP雷达数据接收 ====================

def receive_radar_data(Data2Process_Buffer):
    """UDP接收雷达数据，解析并存入缓冲区"""
    from Sensor_Config.sensor_config import SignalType2Name, lla_original
    
    # UDP Socket初始化
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)
    
    try:
        sock.bind((HOST_IP, HOST_PORT))
        print(f"[UDP接收] 启动成功，监听 {HOST_IP}:{HOST_PORT}")
    except Exception as e:
        print(f"[UDP接收] 绑定端口失败: {e}")
        return
    
    # 初始化缓冲区
    Meas_Buffer = {}
    Times_Buffer = {}
    Infos_Buffer = {}
    Save_Initial = False
    Wait_Timestamps = {}
    Min_Num_In_Buffer = 5
    
    # 数据队列
    data_receive_queue = queue.Queue()
    
    # UDP接收线程
    receive_thread_running = True
    remaining_data = b''
    
    def udp_receive_thread():
        nonlocal remaining_data
        thread_packet_count = 0
        
        while receive_thread_running:
            try:
                data, addr = sock.recvfrom(65536)
                if data:
                    print(f"[UDP] 收到 {len(data)} 字节，来自 {addr}")
                    remaining_data += data
                    print(f"[UDP] 累积缓冲区: {len(remaining_data)} 字节")
                    
                    # 尝试解析完整包
                    while len(remaining_data) >= 12:  # 至少需要帧头+长度字段
                        # 查找有效帧头
                        if len(remaining_data) < 4:
                            break
                        
                        frame_head = struct.unpack('<I', remaining_data[0:4])[0]
                        
                        # 验证帧头
                        valid_heads = [FRAME_HEAD_STATUS, FRAME_HEAD_POINT, 
                                    FRAME_HEAD_TRACK, FRAME_HEAD_END]
                        
                        if frame_head not in valid_heads:
                            print(f"[UDP] 无效帧头 0x{frame_head:08X}，跳过1字节")
                            remaining_data = remaining_data[1:]
                            continue
                        
                        # 读取帧长度
                        if len(remaining_data) < 8:
                            break
                        frame_len = struct.unpack('<I', remaining_data[4:8])[0]
                        
                        # 合理性检查
                        if frame_len < 12 or frame_len > 10000:
                            print(f"[UDP] 异常帧长度 {frame_len}，跳过4字节")
                            remaining_data = remaining_data[4:]
                            continue
                        
                        # 检查数据是否完整
                        if frame_len > len(remaining_data):
                            print(f"[UDP] 等待更多数据: 需要{frame_len}字节，现有{len(remaining_data)}字节")
                            break  # 退出循环，等待下一个UDP包
                        
                        # 提取完整数据包
                        packet = remaining_data[:frame_len]
                        data_receive_queue.put(packet)
                        print(f"[UDP] 提取完整包: {frame_len}字节 (帧头: 0x{frame_head:08X})")
                        remaining_data = remaining_data[frame_len:]
                        thread_packet_count += 1
                        
            except socket.timeout:
                continue
            except Exception as e:
                if receive_thread_running:
                    print(f"[UDP接收线程] 错误: {e}")
                break
    
    # 启动接收线程
    receive_thread = threading.Thread(target=udp_receive_thread, daemon=True)
    receive_thread.start()
    print("[UDP接收] UDP接收线程已启动")
    
    # 统计变量
    packet_count = 0
    point_packet_count = 0
    track_packet_count = 0
    status_packet_count = 0
    end_packet_count = 0
    unknown_packet_count = 0
    parse_error_count = 0
    point_total = 0
    track_total = 0
    
    # 主循环
    Keep_Run = True
    last_print_time = time.time()
    
    print("[UDP接收] 开始主循环，按 'q' 键退出")
    
    while Keep_Run:
        # 检查退出条件
        try:
            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                print("\n[UDP接收] 收到退出信号")
                Keep_Run = False
                break
        except:
            pass
        
        # 获取数据
        try:
            recv_data = data_receive_queue.get(timeout=0.1)
        except queue.Empty:
            # 定期打印统计信息
            if time.time() - last_print_time > 10:
                print(f"\n[统计] 帧总数: {packet_count} | "
                      f"点迹包: {point_packet_count} ({point_total}点) | "
                      f"航迹包: {track_packet_count} ({track_total}航迹) | "
                      f"状态包: {status_packet_count} | "
                      f"结束包: {end_packet_count}")
                last_print_time = time.time()
            continue
        
        packet_count += 1
        
        # 检查数据长度
        if len(recv_data) < 4:
            parse_error_count += 1
            continue
        
        # 解析帧头
        try:
            frame_head = struct.unpack('<I', recv_data[0:4])[0]
        except Exception as e:
            parse_error_count += 1
            continue
        
        # ========== 打印包信息 ==========
        frame_type = ""
        if frame_head == FRAME_HEAD_STATUS:
            frame_type = "状态包"
        elif frame_head == FRAME_HEAD_POINT:
            frame_type = "点迹包"
        elif frame_head == FRAME_HEAD_TRACK:
            frame_type = "航迹包"
        elif frame_head == FRAME_HEAD_END:
            frame_type = "结束包"
        else:
            frame_type = "未知包"
        
        
        # 处理点迹数据包
        if frame_head == FRAME_HEAD_POINT:
            point_packet_count += 1
            
            try:
                points, ins_info = parse_radar_point_packet(recv_data)
            except Exception as e:
                print(f"[错误] 解析点迹包失败: {e}")
                parse_error_count += 1
                continue
            
            print(f"[点迹包] 点迹数量: {len(points) if points else 0}")
            
            if points is None or len(points) == 0:
                continue
            
            point_total += len(points)
            


            if points and len(points) > 0:
                import csv
                with open('radar_calibration_data.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    for point in points:
                        writer.writerow([
                            time.time(),           # 接收时间
                            point['azimuth'],      # 雷达方位角
                            point['pitch'],        # 雷达俯仰角  
                            point['range'],        # 雷达距离
                            point['target_id'],    # 目标ID
                            point.get('speed', 0)  # 速度
                        ])
                print(f"[保存] 已记录 {len(points)} 个点迹")
            # # 打印前3个点迹信息
            # for i, point in enumerate(points[:3]):
            #     print(f"  点迹{i+1}: ID={point['target_id']}, 距离={point['range']:.1f}m, "
            #           f"方位={point['azimuth']:.1f}°, 速度={point['speed']:.1f}m/s")
            
            # 坐标转换
            measurements = []
            timestamps = []
            extra_infos = []
            
            for point in points:
                azimuth_rad = np.radians(point['azimuth'])
                pitch_rad = np.radians(point['pitch'])
                r = point['range']
                
                x = r * np.cos(pitch_rad) * np.sin(azimuth_rad)
                y = r * np.cos(pitch_rad) * np.cos(azimuth_rad)
                z = r * np.sin(pitch_rad)
                measurements.append([x, y, z])
                
                if ins_info.get('gps_time', 0) > 0:
                    timestamp = ins_info['gps_time']
                else:
                    timestamp = ins_info.get('dsp_time', 0) / 1000.0
                
                timestamps.append(timestamp)
                
                extra_info = {
                    'target_id': point['target_id'],
                    'target_type': point['target_type'],
                    'speed': point['speed'],
                    'doppler': point['doppler'],
                    'azimuth': point['azimuth'],
                    'pitch': point['pitch'],
                    'range': point['range'],
                    'is_true_point': point['is_true_point'],
                    'signal_source_types': 1,
                    'droneName': None,
                    'direction': 0.0,
                    'pilotLongitude': 0.0,
                    'pilotLatitude': 0.0,
                    'signalPowerDidch1': None,
                    'deviceCode': 'RADAR',
                    'sn': None,
                    'home_lat': None,
                    'home_lng': None,
                    'reportTime': time.time(),
                    'object_type': None,
                }
                extra_infos.append(extra_info)
            
            if len(measurements) == 0:
                continue
            
            measurements_k = np.array(measurements).T
            timestamps_k = np.array(timestamps)
            sensor_name_k = 'Radar'
            
            # 存入缓冲池
            for it, time_ in enumerate(timestamps_k):
                mea_ = measurements_k[:, it].reshape(-1, 1)
                extra_info_ = extra_infos[it]
                
                if sensor_name_k not in Meas_Buffer:
                    Meas_Buffer[sensor_name_k] = []
                    Times_Buffer[sensor_name_k] = []
                    Infos_Buffer[sensor_name_k] = []
                
                Meas_Buffer[sensor_name_k].append(mea_)
                Times_Buffer[sensor_name_k].append(time_)
                Infos_Buffer[sensor_name_k].append(extra_info_)
            
            # 处理缓冲区数据
            process_buffer_data(Data2Process_Buffer, Meas_Buffer, Times_Buffer, Infos_Buffer, 
                               Save_Initial, Wait_Timestamps, Min_Num_In_Buffer, 
                               point_packet_count)
        
        # 处理航迹数据包
        elif frame_head == FRAME_HEAD_TRACK:
            track_packet_count += 1
            
            try:
                tracks, ins_info = parse_radar_track_packet(recv_data)
            except Exception as e:
                print(f"[错误] 解析航迹包失败: {e}")
                parse_error_count += 1
                continue
            
            print(f"[航迹包] 航迹数量: {len(tracks) if tracks else 0}")
            
            # ========== 打印惯导信息 ==========
            if ins_info:
                print(f"[惯导] GPS时间: {ins_info.get('gps_time', 0)}, "
                    f"雷达航向: {ins_info.get('radar_heading', 0):.1f}°, "
                    f"横滚角: {ins_info.get('roll_angle', 0):.1f}°, "
                    f"纵摇角: {ins_info.get('pitch_angle', 0):.1f}°, "
                    f"帧计数: {ins_info.get('frame_cnt', 0)}")
            
            if tracks is None or len(tracks) == 0:
                continue
            
            track_total += len(tracks)
            
            
            # ========== 先获取时间戳 ==========
            if ins_info and ins_info.get('gps_time', 0) > 0:
                timestamp = ins_info['gps_time']
            else:
                timestamp = time.time()
            
            # 转换为量测
            measurements = []
            timestamps = []
            extra_infos = []

            for track in tracks:
                azimuth_rad = np.radians(track['azimuth'])
                pitch_rad = np.radians(track['pitch'])
                r = track['range']
                
                x = r * np.cos(pitch_rad) * np.sin(azimuth_rad)
                y = r * np.cos(pitch_rad) * np.cos(azimuth_rad)
                z = r * np.sin(pitch_rad)
                
                measurements.append([x, y, z])
                timestamps.append(timestamp) 

                extra_info = {
                    'track_id': track['display_id'],
                    'absolute_id': track['absolute_id'],
                    'target_type': track['target_type'],
                    'speed': track['speed'],
                    'is_tas': track['is_tas'],
                    'signal_source_types': 2,
                    'droneName': None,
                    'direction': 0.0,
                    'pilotLongitude': 0.0,
                    'pilotLatitude': 0.0,
                    'signalPowerDidch1': None,
                    'deviceCode': 'RADAR',
                    'sn': None,
                    'home_lat': None,
                    'home_lng': None,
                    'reportTime': time.time(),
                    'object_type': None,
                    'target_id': track['display_id'],
                }
                extra_infos.append(extra_info)
            
            print(f"[航迹包] 转换后测量数: {len(measurements)}")

            if len(measurements) > 0:
                measurements_k = np.array(measurements).T  # (3, N)
                timestamps_k = np.array(timestamps)        # (N,)
                sensor_name_k = 'Radar_Track'
                
                # ========== 一次性存入整帧数据 ==========
                if sensor_name_k not in Meas_Buffer:
                    Meas_Buffer[sensor_name_k] = []
                    Times_Buffer[sensor_name_k] = []
                    Infos_Buffer[sensor_name_k] = []
                
                # 存入整批测量（而不是逐个）
                Meas_Buffer[sensor_name_k].append(measurements_k)      # 整批测量
                Times_Buffer[sensor_name_k].append(timestamps_k)       # 整批时间戳
                Infos_Buffer[sensor_name_k].append(extra_infos)        # 整批额外信息
                
                print(f"[航迹包] 已存入缓冲区，当前缓冲区大小={len(Meas_Buffer[sensor_name_k])}")
                
                # 处理缓冲区数据
                process_buffer_data(Data2Process_Buffer, Meas_Buffer, Times_Buffer, Infos_Buffer, 
                                Save_Initial, Wait_Timestamps, Min_Num_In_Buffer, 
                                track_packet_count)
        
        # 处理状态包
        elif frame_head == FRAME_HEAD_STATUS:
            status_packet_count += 1
            
            try:
                status = parse_radar_status_packet(recv_data)
            except Exception as e:
                print(f"[错误] 解析状态包失败: {e}")
                parse_error_count += 1
                continue
            
            if status:
                print(f"[状态包] 中心机状态: {status['center_computer_status']}, "
                      f"信号处理状态: {status['signal_process_status']}, "
                      f"FPGA温度: {status['fpga_temp']}°C")
        
        # 处理结束包
        elif frame_head == FRAME_HEAD_END:
            end_packet_count += 1
            print(f"[结束包] 第{end_packet_count}个")
        
        else:
            unknown_packet_count += 1
            print(f"[未知包] 帧头: 0x{frame_head:08X}")
    
    # 清理
    print("\n[UDP接收] 正在清理...")
    receive_thread_running = False
    receive_thread.join(timeout=2)
    
    print("[UDP接收] 处理剩余数据...")
    process_remaining_buffer(Data2Process_Buffer, Meas_Buffer, Times_Buffer, Infos_Buffer)
    
    Data2Process_Buffer.put(None)
    sock.close()
    
    print(f"\n[UDP接收] ========== 最终统计 ==========")
    print(f"总数据包: {packet_count}")
    print(f"点迹包: {point_packet_count} ({point_total} 个点迹)")
    print(f"航迹包: {track_packet_count} ({track_total} 个航迹)")
    print(f"状态包: {status_packet_count}")
    print(f"结束包: {end_packet_count}")
    print(f"[UDP接收] 退出")

def process_buffer_data(Data2Process_Buffer, Meas_Buffer, Times_Buffer, Infos_Buffer, 
                        Save_Initial, Wait_Timestamps, Min_Num_In_Buffer, 
                        frame_count=0):
    """处理缓冲区的数据，送入MHT处理队列"""
    sensor_names, min_timestamps, max_timestamps = [], [], []
    
    for sensor_name_, tmps in Times_Buffer.items():
        if len(tmps) == 0:
            continue
        sensor_names.append(sensor_name_)
        min_timestamps.append(tmps[0])
        max_timestamps.append(tmps[-1])
    
    if len(sensor_names) == 0:
        print("[DEBUG] sensor_names 为空，直接返回")  # ← 加这行
        return
    
    global_min_timestamp = min(min_timestamps)
    global_max_timestamp = max(max_timestamps)
    
    min_index = min_timestamps.index(global_min_timestamp)
    sensor_name_chosen = sensor_names[min_index]
    
    if len(Meas_Buffer[sensor_name_chosen]) == 0:
        return
    
    meas_chosen = deepcopy(Meas_Buffer[sensor_name_chosen][0])
    tmp_chosen = copy(Times_Buffer[sensor_name_chosen][0])
    info_chosen = deepcopy(Infos_Buffer[sensor_name_chosen][0])
    
    Data2Process_Buffer.put({
        'timestamp': tmp_chosen, 
        'meas': meas_chosen, 
        'sensor_name': sensor_name_chosen,
        'infos': info_chosen, 
        'global_max_timestamp': global_max_timestamp
    })
    
    del Meas_Buffer[sensor_name_chosen][0]
    del Times_Buffer[sensor_name_chosen][0]
    del Infos_Buffer[sensor_name_chosen][0]


def process_remaining_buffer(Data2Process_Buffer, Meas_Buffer, Times_Buffer, Infos_Buffer):
    """处理缓冲区剩余的所有数据"""
    processed_count = 0
    
    while True:
        sensor_names, min_timestamps, max_timestamps = [], [], []
        
        for sensor_name_, tmps in Times_Buffer.items():
            if len(tmps) == 0:
                continue
            sensor_names.append(sensor_name_)
            min_timestamps.append(tmps[0])
            max_timestamps.append(tmps[-1])
        
        if len(sensor_names) == 0:
            break
        
        global_min_timestamp = min(min_timestamps)
        min_index = min_timestamps.index(global_min_timestamp)
        sensor_name_chosen = sensor_names[min_index]
        
        if len(Meas_Buffer[sensor_name_chosen]) == 0:
            continue
        
        meas_chosen = deepcopy(Meas_Buffer[sensor_name_chosen][0])
        tmp_chosen = copy(Times_Buffer[sensor_name_chosen][0])
        info_chosen = deepcopy(Infos_Buffer[sensor_name_chosen][0])
        
        Data2Process_Buffer.put({
            'timestamp': tmp_chosen, 
            'meas': meas_chosen, 
            'sensor_name': sensor_name_chosen,
            'infos': info_chosen, 
            'global_max_timestamp': max(max_timestamps) if max_timestamps else tmp_chosen
        })
        
        del Meas_Buffer[sensor_name_chosen][0]
        del Times_Buffer[sensor_name_chosen][0]
        del Infos_Buffer[sensor_name_chosen][0]
        
        processed_count += 1
    
    if processed_count > 0:
        print(f"[缓冲区] 处理剩余数据 {processed_count} 条")


def mht_process_and_send(Data2Process_Buffer):
    """MHT跟踪并UDP发送结果，同时保存JSON"""
    print("[MHT进程] ========== MHT进程启动 ==========")
    print("[MHT进程] 等待接收雷达数据...")

    # 日志文件（文本格式）
    log_file = open('track_log.txt', 'w', encoding='utf-8')
    
    # JSON结果文件（每帧追加）
    result_json_file = 'track_results.json'
    # 如果是第一次运行，创建文件并写入数组开头
    import os
    if not os.path.exists(result_json_file):
        with open(result_json_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
    else:
        # 如果文件已存在且不为空，先读取内容，准备追加
        with open(result_json_file, 'r+', encoding='utf-8') as f:
            content = f.read()
            if content.endswith(']\n'):
                # 去掉结尾的]
                f.seek(0)
                f.write(content[:-2])
                f.truncate()
            f.write(',\n')
    
    Decided_Tree_All = []
    exit_flag = False
    
    def signal_handler(signum, frame):
        nonlocal exit_flag
        print("\n[MHT进程] 收到中断信号，准备退出...")
        exit_flag = True
        try:
            Data2Process_Buffer.put_nowait(None)
        except:
            pass
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    from Sensor_Config.sensor_config import Sensor_Config, Name2SignalType, lla_original
    
    Initial = False
    timestamp_last = -1
    frame_count = 0
    
    dim_d = 3
    Debug_Params = {'Debug': False, 'Begin_Frame': 30}
    MHT_Params = {
        'Lambda_NT': 1, 'Q_k': np.identity(dim_d) * 10.0,
        "Max_Vel": 100.0, 'N_Scan': 3, 'Pg': 0.999,
        'P_death': 1e-2, 'dim_d': dim_d,
        'Debug_Params': Debug_Params, 'Resolved_Time_Window': 2,
        'Resolved_Min_Detect': 1, 'max_detect_time': 20
    }
    
    base_sensor_name = 'Radar'
    base_sensor_appear = False
    
    Cluster_Params = {'Sigma': np.diag([10.0, 10.0, 10.0]), 'Distance': 50.0}
    
    label_id_map = {}
    label = 1
    
    Classify_Results = {}
    from Classify.Initial_Params import Initial_Classify_Params
    
    measurement_history = []
    estimation_history = []
    
    while True:
        try:
            data_k = Data2Process_Buffer.get(timeout=0.5)
        except queue.Empty:
            if exit_flag:
                break
            continue
        
        if data_k is None or exit_flag:
            print("[MHT进程] 收到结束信号，退出")
            break
        
        frame_count += 1
        print(f"\n[MHT进程] ========== 第 {frame_count} 帧 ==========")
        
        time_begin = time.time()
        meas_chosen = data_k['meas']
        sensor_name_chosen = data_k['sensor_name']
        tmp_chosen = data_k['timestamp']
        infos_chosen = data_k['infos']
        
        # 时间戳转换（毫秒转秒）
        if isinstance(tmp_chosen, np.ndarray):
            timestamp_sec = tmp_chosen[0] / 1000.0 if tmp_chosen[0] > 100000 else tmp_chosen[0]
        else:
            timestamp_sec = tmp_chosen / 1000.0 if tmp_chosen > 100000 else tmp_chosen
        
        # print(f"[测量值] 时间戳: {timestamp_sec:.3f}s, 传感器: {sensor_name_chosen}, 量测数: {meas_chosen.shape[1]}")
        
        # 量测聚类
        obs_k = []
        for ib in range(meas_chosen.shape[1]):
            obs_k.append(np.array(meas_chosen[:3, ib]).reshape(-1, 1))
        
        if len(obs_k) == 0:
            continue
        
        obs_clusters, obs_indexs = Clustering_Obs(
            obs_k=obs_k,
            Clustering_Type='DBSCAN',
            eps=Cluster_Params['Distance'],
            min_samples=1,
            Sigma=Cluster_Params['Sigma']
        )
        
        if len(obs_clusters) == 0:
            continue
        
        obs_k = []
        for obs_cluster in obs_clusters:
            obs_mean = np.mean(np.concatenate(obs_cluster, axis=1), axis=1).reshape(-1, 1)
            obs_k.append(obs_mean)
        
        # print(f"[聚类] 点迹数: {len(obs_k)}")
        
        measurement_history.append({
            'frame': frame_count,
            'timestamp': timestamp_sec,
            'measurements': [obs.copy() for obs in obs_k]
        })
        
        sensor_config = deepcopy(Sensor_Config.get(sensor_name_chosen, Sensor_Config['Radar']))
        
        if sensor_name_chosen == base_sensor_name:
            base_sensor_appear = True
        
        if not base_sensor_appear:
            sensor_config['Biased_Ignore'] = True
        
        # MHT跟踪
        if not Initial:
            print("[MHT] 初始化跟踪器...")
            Initial = True
            TOMHT = POMHT_Bias(
                Lambda_NT=MHT_Params['Lambda_NT'],
                obs_k=obs_k, timestamp=timestamp_sec,
                sensor_config=sensor_config,
                Q_k=MHT_Params['Q_k'],
                Max_Vel=MHT_Params['Max_Vel'],
                N_Scan=MHT_Params['N_Scan'],
                Pg=MHT_Params['Pg'],
                P_death=MHT_Params['P_death'],
                dim_d=MHT_Params['dim_d'],
                Debug_Params=MHT_Params['Debug_Params'],
                extra_infos=infos_chosen,
                Resolved_Time_Window=MHT_Params['Resolved_Time_Window'],
                Resolved_Min_Detect=MHT_Params["Resolved_Min_Detect"],
                max_detect_time=MHT_Params['max_detect_time']
            )
        else:
            if timestamp_last > timestamp_sec:
                print(f"[MHT] 时间戳乱序，跳过")
                continue
            timestamp_last = timestamp_sec
            TOMHT.forward(timestamp=timestamp_sec, obs_k=obs_k, 
                         sensor_config=sensor_config, extra_infos=infos_chosen)
        
        # 输出结果
        if hasattr(TOMHT, 'Output_Nodes') and len(TOMHT.Output_Nodes) > 0:
            Decided_Tree = deepcopy(TOMHT.Output_Nodes[-1])
            target_num = len(Decided_Tree)
            
            if target_num > 0:
                print(f"[MHT结果] 确认航迹数: {target_num}")
                timestamp_output = TOMHT.Timestamps[-1] if hasattr(TOMHT, 'Timestamps') and len(TOMHT.Timestamps) > 0 else timestamp_sec
                
                msg_result = {'timestamp': int(timestamp_output * 1000), 'result': []}
                
                # 用于日志记录
                log_targets = []
                json_targets = []  # 用于JSON保存
                
                type_map = {
                    -1: '无人机',
                    0: '未知',
                    1: '飞鸟',
                    # 如果还有其他类型继续添加
                }
                for node in Decided_Tree.values():
                    if node.label not in label_id_map:
                        label_id_map[node.label] = label
                        label += 1
                    
                    pos_enu = node.x_k_k[:3, :]
                    vel_enu = node.x_k_k[3:6, :]
                    speed = np.linalg.norm(vel_enu)
                    
                    track_lla = enu_to_geodetic(lla_original[0], lla_original[1], 
                                                lla_original[2], pos_enu)
                    
                        # node.obs_id 存储了该航迹关联的量测索引
                    if hasattr(node, 'obs_id') and node.obs_id and len(node.obs_id) > 0:
                        # 取最新的量测索引
                        last_obs_idx = node.obs_id[-1]
                        
                        # infos_chosen 是列表，索引对应量测
                        if infos_chosen and last_obs_idx < len(infos_chosen):

                            cluster_idx = node.obs_id[-1]
                            original_indices = obs_indexs[cluster_idx]  # 获取所有原始索引
                            all_ids = [infos_chosen[idx].get('track_id') for idx in original_indices]
                            raw_display_id = ','.join(map(str, all_ids))  # "23,26,27"
                            raw_info = infos_chosen[last_obs_idx]
                            # raw_display_id = raw_info.get('track_id') or raw_info.get('target_id')
                            raw_absolute_id = raw_info.get('absolute_id')
                    
                    # 打印对比信息到控制台
                    if raw_display_id is not None:
                        print(f"[ID映射] MHT输出: Radar-{label_id_map[node.label]} (label={node.label}) -> 原始雷达ID: {raw_display_id}")

                    # ========== 添加分类逻辑 ==========
                    # 初始化分类结果
                    if node.label not in Classify_Results.keys():
                        Classify_Results[node.label] = {
                            'Target_type': None, 
                            'Id': label_id_map[node.label], 
                            'Time_N': 1, 'T': 0, 
                            'Bird_Model_IMM': Initial_Classify_Params['Bird_Model_IMM'],
                            'Bird_Result_k': Initial_Classify_Params['Bird_Result_k'], 
                            'UAV_Model_IMM': Initial_Classify_Params['UAV_Model_IMM'],
                            'UAV_Result_k': Initial_Classify_Params['UAV_Result_k'], 
                            'Log_Likelihood_Ratio': Initial_Classify_Params['Log_Likelihood_Ratio'],
                            'Log_Likelihood_Ratio_all': []  
                        }
                    else:
                        Classify_Results[node.label]['Time_N'] += 1
                        if len(TOMHT.Timestamps) >= 2:
                            Classify_Results[node.label]['T'] = TOMHT.Timestamps[-1] - TOMHT.Timestamps[-2]
                    
                    # 获取量测用于分类
                    if len(node.obs_id) > 0:
                        z_k_ = TOMHT.obs_s[-1][node.obs_id[0]]
                        
                        # 目标分类计算
                        if Classify_Results[node.label]['Target_type'] == 0 or Classify_Results[node.label]['Target_type'] is None:
                            (Target_type, 
                            Classify_Results[node.label]['Log_Likelihood_Ratio'], 
                            Classify_Results[node.label]['Bird_Model_IMM'], 
                            Initial_Classify_Params['Bird_Qs'], 
                            Classify_Results[node.label]['Bird_Result_k'],
                            Classify_Results[node.label]['UAV_Model_IMM'], 
                            Initial_Classify_Params['UAV_Qs'], 
                            Classify_Results[node.label]['UAV_Result_k']) = TrackingClassify(
                                Classify_Results[node.label]['Time_N'], z_k_, Classify_Results[node.label]['T'],
                                Classify_Results[node.label]['Bird_Model_IMM'], 
                                Initial_Classify_Params['Bird_Qs'], Classify_Results[node.label]['Bird_Result_k'],
                                Classify_Results[node.label]['UAV_Model_IMM'], 
                                Initial_Classify_Params['UAV_Qs'], Classify_Results[node.label]['UAV_Result_k'],
                                Classify_Results[node.label]['Log_Likelihood_Ratio'], 
                                Initial_Classify_Params['ConstValue']
                            )
                            Classify_Results[node.label]['Target_type'] = Target_type
                            # ========== 提取标量值用于打印 ==========
                            lr_value = Classify_Results[node.label]['Log_Likelihood_Ratio']
                            if isinstance(lr_value, np.ndarray):
                                lr_value = lr_value.item() if lr_value.size == 1 else lr_value[0]
                            
                            type_name = type_map.get(Target_type, '未知')
                            print(f"[分类] label={node.label}, Time_N={Classify_Results[node.label]['Time_N']}, "
                                f"LR={lr_value:.3f}, type={Target_type} ({type_name})")
                    
                    # 使用分类结果
                    target_type = Classify_Results[node.label]['Target_type'] if node.label in Classify_Results else 3
                           
                    target_result = {
                        'track_id': f'Radar-{label_id_map[node.label]}',
                        'target_type': target_type,
                        'lat': track_lla[0, 0].item(),
                        'lon': track_lla[1, 0].item(),
                        'alt': track_lla[2, 0].item(),
                        'height': track_lla[2, 0].item(),
                        'speed': speed.item(),
                        'extra_info': {
                            'vel_x': vel_enu[0, 0].item(),
                            'vel_y': vel_enu[1, 0].item(),
                            'vel_z': vel_enu[2, 0].item(),
                            'fusion_time': timestamp_output,
                            'signal_source_types': 1,
                            'raw_display_id': raw_display_id,
                            'raw_absolute_id': raw_absolute_id,
                        }
                    }
                    msg_result['result'].append(target_result)
                    
                    # 保存用于日志
                    log_targets.append({
                        'track_id': target_result['track_id'],
                        'lat': target_result['lat'],
                        'lon': target_result['lon'],
                        'speed': target_result['speed']
                    })
                    
                    # 保存用于JSON（完整信息）
                    json_targets.append({
                        'track_id': target_result['track_id'],
                        'target_type': target_result['target_type'],
                        'lat': target_result['lat'],
                        'lon': target_result['lon'],
                        'alt': target_result['alt'],
                        'height': target_result['height'],
                        'speed': target_result['speed'],
                        'vel_x': target_result['extra_info']['vel_x'],
                        'vel_y': target_result['extra_info']['vel_y'],
                        'vel_z': target_result['extra_info']['vel_z'],
                        'fusion_time': target_result['extra_info']['fusion_time'],
                        'raw_display_id': target_result['extra_info'].get('raw_display_id'),
                        'raw_absolute_id': target_result['extra_info'].get('raw_absolute_id')
                    })
                
                # ========== 保存JSON结果 ==========
                json_frame_data = {
                    'frame': frame_count,
                    'timestamp': timestamp_output,
                    'timestamp_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp_output)),
                    'target_count': target_num,
                    'targets': json_targets
                }
                
                try:
                    with open('track_results.json', 'a', encoding='utf-8') as f:
                        json.dump(json_frame_data, f, ensure_ascii=False, indent=2)
                        f.write(',\n')
                    print(f"[JSON] 已保存第{frame_count}帧结果到 track_results.json")
                except Exception as e:
                    print(f"[JSON] 保存失败: {e}")
                
                # UDP发送结果
                try:
                    json_string = json.dumps(msg_result)
                    sock.sendto(json_string.encode(), (HOST_IP, 9999))
                    print(f"[发送] 已发送 {target_num} 个目标")
                except Exception as e:
                    print(f"[发送] UDP发送失败: {e}")
                
                # 写入文本日志文件
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | 第{frame_count}帧 | {target_num}个目标\n")
                for target in log_targets:
                    log_file.write(f"  {target['track_id']}: lat={target['lat']:.6f}, lon={target['lon']:.6f}, speed={target['speed']:.1f}m/s\n")
                log_file.flush()
            
            Decided_Tree_All.append(Decided_Tree)
        
        time_over = time.time()
        print(f"[性能] 耗时: {(time_over - time_begin)*1000:.2f}ms")
        
        if frame_count % 10 == 0:
            active_tracks = len(TOMHT.Output_Nodes[-1]) if len(TOMHT.Output_Nodes) > 0 else 0
            print(f"\n[统计] 已处理 {frame_count} 帧, 当前航迹数: {active_tracks}")
    
    # 关闭JSON文件（添加数组结尾）
    try:
        with open('track_results.json', 'r+', encoding='utf-8') as f:
            content = f.read()
            if content.endswith(',\n'):
                content = content[:-2] + '\n'
            f.seek(0)
            f.write(content + ']')
            f.truncate()
        print("[JSON] 文件已关闭")
    except:
        pass
    
    log_file.close()
    sock.close()
    print("[MHT进程] 退出")

# ==================== 主程序 ====================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    Data2Process_Buffer = multiprocessing.Queue()
    
    process_receive = multiprocessing.Process(target=receive_radar_data, 
                                              args=(Data2Process_Buffer,))
    process_mht = multiprocessing.Process(target=mht_process_and_send, 
                                          args=(Data2Process_Buffer,))
    
    process_receive.start()
    process_mht.start()
    
    print("UDP雷达处理程序启动")
    print("按 Ctrl+C 或 'q' 键退出")
    
    try:
        process_receive.join()
        process_mht.join()
    except KeyboardInterrupt:
        print("\n主程序收到中断信号")
        process_receive.terminate()
        process_mht.terminate()
    
    print("主程序结束")