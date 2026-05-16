import os
import re

from core.app_config import TRACK_LOG_FILE


def print_available_tracks_from_log():
    """从日志文件打印当前航迹"""
    try:
        if not os.path.exists(TRACK_LOG_FILE):
            print("\n暂无航迹日志文件")
            return
        
        with open(TRACK_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找最后一帧
        last_frame_idx = -1
        for i in range(len(lines)-1, -1, -1):
            line = lines[i].strip()
            if '| 第' in line and '帧 |' in line:
                last_frame_idx = i
                break
        
        if last_frame_idx == -1:
            print("\n暂无航迹数据")
            return
        
        # 打印帧信息
        frame_line = lines[last_frame_idx].strip()
        print(f"\n{frame_line}")
        
        # 获取后面的航迹行（直到空行或下一帧）
        tracks = []
        for i in range(last_frame_idx + 1, len(lines)):
            line = lines[i]
            # 如果是空行，跳过
            if not line.strip():
                continue
            # 如果遇到新的帧行，停止
            if '| 第' in line and '帧 |' in line:
                break
            # 如果是航迹行（包含 "Radar-"）
            if 'Radar-' in line:
                tracks.append(line.strip())
        
        if tracks:
            print("\n当前可用航迹:")
            print("-" * 70)
            for line in tracks:
                # 解析格式: "Radar-1: 距离=1057.5m, 方位=288.9°, 俯仰=5.9°, 速度=5.8m/s, 位置=(...)"
                if ': ' in line:
                    parts = line.split(': ', 1)
                    track_id = parts[0].strip()
                    data_part = parts[1]
                    
                    # 提取参数
                    import re
                    range_match = re.search(r'距离=([\d.]+)m', data_part)
                    az_match = re.search(r'方位=([\d.]+)°', data_part)
                    pitch_match = re.search(r'俯仰=([\d.]+)°', data_part)
                    speed_match = re.search(r'速度=([\d.]+)m/s', data_part)
                    
                    if range_match:
                        range_val = float(range_match.group(1))
                        az_val = float(az_match.group(1)) if az_match else 0
                        pitch_val = float(pitch_match.group(1)) if pitch_match else 0
                        speed_val = float(speed_match.group(1)) if speed_match else 0
                        print(f"  {track_id}: 距离={range_val:.1f}m, "
                              f"方位={az_val:.1f}°, "
                              f"俯仰={pitch_val:.1f}°, "
                              f"速度={speed_val:.1f}m/s")
            print("-" * 70)
            print(f"共 {len(tracks)} 个航迹")
        else:
            print("\n当前无可用航迹")
            
    except Exception as e:
        print(f"\n读取日志失败: {e}")
        import traceback
        traceback.print_exc()


def get_track_by_id_from_log(track_id):
    """从日志文件获取最新航迹信息"""
    try:
        if not os.path.exists(TRACK_LOG_FILE):
            return None
        
        with open(TRACK_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找最后一帧
        last_frame_idx = -1
        for i in range(len(lines)-1, -1, -1):
            line = lines[i].strip()
            if '| 第' in line and '帧 |' in line:
                last_frame_idx = i
                break
        
        if last_frame_idx == -1:
            return None
        
        # 在最后一帧中查找目标ID
        for i in range(last_frame_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            if '| 第' in line and '帧 |' in line:
                break
            if track_id in line and 'Radar-' in line:
                import re
                range_match = re.search(r'距离=([\d.]+)m', line)
                az_match = re.search(r'方位=([\d.]+)°', line)
                pitch_match = re.search(r'俯仰=([\d.]+)°', line)
                speed_match = re.search(r'速度=([\d.]+)m/s', line)
                
                if range_match:
                    return {
                        'track_id': track_id,
                        'range': float(range_match.group(1)),
                        'azimuth': float(az_match.group(1)) if az_match else 0,
                        'pitch': float(pitch_match.group(1)) if pitch_match else 0,
                        'speed': float(speed_match.group(1)) if speed_match else 0
                    }
    except Exception as e:
        print(f"读取日志失败: {e}")
    return None
