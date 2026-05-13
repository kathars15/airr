# -*- coding: utf-8 -*-

import csv
import queue
import socket
import struct
import threading
import time
from copy import copy, deepcopy

import numpy as np

from core.app_config import (
    CALIBRATION_FILE, FRAME_HEAD_END, FRAME_HEAD_POINT, FRAME_HEAD_STATUS,
    FRAME_HEAD_TRACK, HOST_IP, HOST_PORT, RAW_TRACKS_FILE,
)
from core.console_utils import safe_print
from core.radar_protocol import (
    parse_radar_end_packet, parse_radar_point_packet, parse_radar_status_packet,
    parse_radar_track_packet,
)


def _timestamp_bound(value, reducer):
    """Return a scalar timestamp bound for scalar or packet-level timestamp arrays."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(reducer(value.astype(float).ravel()))

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_quit_key_pressed():
    """Best-effort console quit check without installing a global keyboard hook."""
    try:
        import msvcrt
    except ImportError:
        return False

    if not msvcrt.kbhit():
        return False

    key = msvcrt.getwch()
    return key.lower() == 'q' or key == '\x1b'


def _append_raw_tracks(timestamp, tracks):
    """Persist parsed radar TRACK records before MHT buffering."""
    if not tracks:
        return

    write_header = False
    try:
        import os
        write_header = not os.path.exists(RAW_TRACKS_FILE)
        with open(RAW_TRACKS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp', 'track_id', 'range', 'azimuth', 'pitch', 'speed', 'target_type', 'height'])
            for track in tracks:
                writer.writerow([
                    timestamp,
                    track.get('display_id'),
                    track.get('range'),
                    track.get('azimuth'),
                    track.get('pitch'),
                    track.get('speed'),
                    track.get('target_type'),
                    track.get('height'),
                ])
    except Exception as e:
        safe_print(f"[raw_tracks] write failed: {e}")


def receive_radar_data(Data2Process_Buffer, PointDebug_Buffer=None):
    """UDP接收雷达数据，解析并存入缓冲区"""
    from Sensor_Config.sensor_config import SignalType2Name, lla_original

    # UDP Socket初始化
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)
    
    try:
        sock.bind((HOST_IP, HOST_PORT))
        safe_print(f"[UDP接收] 启动成功，监听 {HOST_IP}:{HOST_PORT}")
    except Exception as e:
        safe_print(f"[UDP接收] 绑定端口失败: {e}")
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
    print_udp_stats = False
    udp_stat_counts = {'STATUS': 0, 'POINT': 0, 'TRACK': 0, 'END': 0, 'UNKNOWN': 0}
    last_udp_stat_print = time.time()
    
    def udp_receive_thread():
        nonlocal remaining_data, last_udp_stat_print
        thread_packet_count = 0
        
        while receive_thread_running:
            try:
                data, addr = sock.recvfrom(65536)
                if data:
                    # safe_print(f"[UDP] 收到 {len(data)} 字节，来自 {addr}")
                    remaining_data += data
                    # safe_print(f"[UDP] 累积缓冲区: {len(remaining_data)} 字节")
                    
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
                            udp_stat_counts['UNKNOWN'] += 1
                            safe_print(f"[UDP] 无效帧头 0x{frame_head:08X}，跳过1字节")
                            remaining_data = remaining_data[1:]
                            continue
                        
                        # 读取帧长度
                        if len(remaining_data) < 8:
                            break
                        frame_len = struct.unpack('<I', remaining_data[4:8])[0]
                        
                        # 合理性检查
                        if frame_len < 12 or frame_len > 10000:
                            safe_print(f"[UDP] 异常帧长度 {frame_len}，跳过4字节")
                            remaining_data = remaining_data[4:]
                            continue
                        
                        # 检查数据是否完整
                        if frame_len > len(remaining_data):
                            # safe_print(f"[UDP] 等待更多数据: 需要{frame_len}字节，现有{len(remaining_data)}字节")
                            break  # 退出循环，等待下一个UDP包
                        
                        # 提取完整数据包
                        packet = remaining_data[:frame_len]
                        data_receive_queue.put(packet)
                        if frame_head == FRAME_HEAD_STATUS:
                            udp_stat_counts['STATUS'] += 1
                        elif frame_head == FRAME_HEAD_POINT:
                            udp_stat_counts['POINT'] += 1
                        elif frame_head == FRAME_HEAD_TRACK:
                            udp_stat_counts['TRACK'] += 1
                        elif frame_head == FRAME_HEAD_END:
                            udp_stat_counts['END'] += 1
                        now = time.time()
                        if print_udp_stats and now - last_udp_stat_print >= 5.0:
                            safe_print(
                                "[UDP stats] "
                                f"STATUS={udp_stat_counts['STATUS']} "
                                f"POINT={udp_stat_counts['POINT']} "
                                f"TRACK={udp_stat_counts['TRACK']} "
                                f"END={udp_stat_counts['END']} "
                                f"UNKNOWN={udp_stat_counts['UNKNOWN']}"
                            )
                            last_udp_stat_print = now
                        # safe_print(f"[UDP] 提取完整包: {frame_len}字节 (帧头: 0x{frame_head:08X})")
                        remaining_data = remaining_data[frame_len:]
                        thread_packet_count += 1
                        
            except socket.timeout:
                continue
            except Exception as e:
                if receive_thread_running:
                    safe_print(f"[UDP接收线程] 错误: {e}")
                break
    
    # 启动接收线程
    receive_thread = threading.Thread(target=udp_receive_thread, daemon=True)
    receive_thread.start()
    safe_print("[UDP接收] UDP接收线程已启动")
    
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
    last_safe_print_time = time.time()
    
    safe_print("[UDP接收] 开始主循环，按 'q' 键退出")
    
    while Keep_Run:
        # 检查退出条件
        try:
            if _is_quit_key_pressed():
                safe_print("\n[UDP接收] 收到退出信号")
                Keep_Run = False
                break
        except:
            pass
        
        # 获取数据
        try:
            recv_data = data_receive_queue.get(timeout=0.1)
        except queue.Empty:
            # 定期打印统计信息
            if time.time() - last_safe_print_time > 10:
                # safe_print(f"\n[统计] 帧总数: {packet_count} | "
                #       f"点迹包: {point_packet_count} ({point_total}点) | "
                #       f"航迹包: {track_packet_count} ({track_total}航迹) | "
                #       f"状态包: {status_packet_count} | "
                #       f"结束包: {end_packet_count}")
                last_safe_print_time = time.time()
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
        
        # 处理点迹数据包
        if frame_head == FRAME_HEAD_POINT:
            point_packet_count += 1
            
            try:
                points, ins_info = parse_radar_point_packet(recv_data)
            except Exception as e:
                safe_print(f"[错误] 解析点迹包失败: {e}")
                parse_error_count += 1
                continue
            
            # safe_print(f"[点迹包] 点迹数量: {len(points) if points else 0}")
            
            if points is None or len(points) == 0:
                continue
            
            point_total += len(points)
            
            if points and len(points) > 0:
                import csv
                with open(CALIBRATION_FILE, 'a', newline='') as f:
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
                # safe_print(f"[保存] 已记录 {len(points)} 个点迹")
            
            # 坐标转换
            # Keep POINT packets for diagnostics only. MHT should use TRACK packets.
            continue

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
                safe_print(f"[错误] 解析航迹包失败: {e}")
                parse_error_count += 1
                continue
            
            # safe_print(f"[航迹包] 航迹数量: {len(tracks) if tracks else 0}")
            
            # # 打印惯导信息
            # if ins_info:
            #     safe_print(f"[惯导] GPS时间: {ins_info.get('gps_time', 0)}, "
            #         f"雷达航向: {ins_info.get('radar_heading', 0):.1f}°, "
            #         f"横滚角: {ins_info.get('roll_angle', 0):.1f}°, "
            #         f"纵摇角: {ins_info.get('pitch_angle', 0):.1f}°, "
            #         f"帧计数: {ins_info.get('frame_cnt', 0)}")
            
            if tracks is None or len(tracks) == 0:
                continue
            
            track_total += len(tracks)
            _append_raw_tracks(time.time(), tracks)
            
            # 获取时间戳
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
                    'range': track['range'],      # 添加距离
                    'azimuth': track['azimuth'],  # 添加方位角
                    'pitch': track['pitch'],      # 添加俯仰角
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
            
            # safe_print(f"[航迹包] 转换后测量数: {len(measurements)}")

            if len(measurements) > 0:
                measurements_k = np.array(measurements).T  # (3, N)
                timestamps_k = np.array(timestamps)        # (N,)
                sensor_name_k = 'Radar_Track'
                
                # 一次性存入整帧数据
                if sensor_name_k not in Meas_Buffer:
                    Meas_Buffer[sensor_name_k] = []
                    Times_Buffer[sensor_name_k] = []
                    Infos_Buffer[sensor_name_k] = []
                
                # 存入整批测量
                Meas_Buffer[sensor_name_k].append(measurements_k)
                Times_Buffer[sensor_name_k].append(timestamps_k)
                Infos_Buffer[sensor_name_k].append(extra_infos)
                
                # safe_print(f"[航迹包] 已存入缓冲区，当前缓冲区大小={len(Meas_Buffer[sensor_name_k])}")
                
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
                safe_print(f"[错误] 解析状态包失败: {e}")
                parse_error_count += 1
                continue
            
            # if status:
                # safe_print(f"[状态包] 中心机状态: {status['center_computer_status']}, "
                #       f"信号处理状态: {status['signal_process_status']}, "
                #       f"FPGA温度: {status['fpga_temp']}°C")
        
        # 处理结束包
        elif frame_head == FRAME_HEAD_END:
            end_packet_count += 1
            # safe_print(f"[结束包] 第{end_packet_count}个")
        
        else:
            unknown_packet_count += 1
            safe_print(f"[未知包] 帧头: 0x{frame_head:08X}")
    
    # 清理
    safe_print("\n[UDP接收] 正在清理...")
    receive_thread_running = False
    receive_thread.join(timeout=2)
    
    safe_print("[UDP接收] 处理剩余数据...")
    process_remaining_buffer(Data2Process_Buffer, Meas_Buffer, Times_Buffer, Infos_Buffer)
    
    Data2Process_Buffer.put(None)
    sock.close()

def process_buffer_data(Data2Process_Buffer, Meas_Buffer, Times_Buffer, Infos_Buffer, 
                        Save_Initial, Wait_Timestamps, Min_Num_In_Buffer, 
                        frame_count=0):
    """处理缓冲区的数据，送入MHT处理队列"""
    sensor_names, min_timestamps, max_timestamps = [], [], []
    
    def to_scalar(val):
            """将 numpy 数组或标量转为 Python 标量"""
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    return 0.0
                return float(val.flat[0])
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0
        
    for sensor_name_, tmps in Times_Buffer.items():
        if len(tmps) == 0:
            continue
        sensor_names.append(sensor_name_)
        min_timestamps.append(_timestamp_bound(tmps[0], np.min))
        max_timestamps.append(_timestamp_bound(tmps[-1], np.max))
    
    if len(sensor_names) == 0:
        safe_print("[DEBUG] sensor_names 为空，直接返回")
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
    def to_scalar(val):
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    return 0.0
                return float(val.flat[0])
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0
          
    while True:
        sensor_names, min_timestamps, max_timestamps = [], [], []
        
        for sensor_name_, tmps in Times_Buffer.items():
            if len(tmps) == 0:
                continue
            sensor_names.append(sensor_name_)
            min_timestamps.append(_timestamp_bound(tmps[0], np.min))
            max_timestamps.append(_timestamp_bound(tmps[-1], np.max))
        
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
        safe_print(f"[缓冲区] 处理剩余数据 {processed_count} 条")

