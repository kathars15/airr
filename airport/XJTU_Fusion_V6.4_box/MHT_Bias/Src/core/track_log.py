# -*- coding: utf-8 -*-

import os
import re
import traceback

from core.app_config import TRACK_LOG_FILE
from core.console_utils import safe_print


def _find_last_frame(lines):
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if '| 第' in line and '帧 |' in line:
            return i
        if '| 绗' in line and '甯' in line:
            return i
    return -1


def _parse_track_line(line):
    range_match = re.search(r'(?:距离|璺濈)=([\d.]+)m', line)
    az_match = re.search(r'(?:方位|鏂逛綅)=([\d.]+)', line)
    pitch_match = re.search(r'(?:俯仰|淇话)=([\d.]+)', line)
    speed_match = re.search(r'(?:速度|閫熷害)=([\d.]+)m/s', line)

    if not range_match:
        return None

    display_id = line.split(':', 1)[0].strip()
    raw_match = re.match(r'^(Radar-\d+)(?:\(raw=([^)]+)\))?$', display_id)
    track_id = raw_match.group(1) if raw_match else display_id
    raw_display_id = raw_match.group(2) if raw_match else None
    return {
        'track_id': track_id,
        'display_id': display_id,
        'raw_display_id': raw_display_id,
        'range': float(range_match.group(1)),
        'azimuth': float(az_match.group(1)) if az_match else 0.0,
        'pitch': float(pitch_match.group(1)) if pitch_match else 0.0,
        'speed': float(speed_match.group(1)) if speed_match else 0.0,
    }


def get_all_tracks_from_log(track_log_file=TRACK_LOG_FILE):
    try:
        if not os.path.exists(track_log_file):
            return []

        with open(track_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        last_frame_idx = _find_last_frame(lines)
        if last_frame_idx == -1:
            return []

        tracks = []
        for i in range(last_frame_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            if _find_last_frame([line]) == 0:
                break
            if 'Radar-' not in line:
                continue

            track = _parse_track_line(line)
            if track:
                tracks.append(track)

        return tracks

    except Exception as e:
        safe_print(f"[自动跟踪] 读取最后一帧目标失败: {e}")
        return []


def get_nearest_target_from_log(track_log_file=TRACK_LOG_FILE):
    tracks = get_all_tracks_from_log(track_log_file)
    if not tracks:
        return None
    return min(tracks, key=lambda x: x['range'])


def get_track_by_id_from_log(track_id, track_log_file=TRACK_LOG_FILE):
    tracks = get_all_tracks_from_log(track_log_file)
    wanted = str(track_id)
    for track in tracks:
        if track['track_id'] == wanted or track.get('display_id') == wanted:
            return track
    return None


def safe_print_available_tracks_from_log(track_log_file=TRACK_LOG_FILE):
    try:
        if not os.path.exists(track_log_file):
            safe_print("\n暂无航迹日志文件")
            return

        with open(track_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        last_frame_idx = _find_last_frame(lines)
        if last_frame_idx == -1:
            safe_print("\n暂无航迹数据")
            return

        safe_print(f"\n{lines[last_frame_idx].strip()}")
        tracks = get_all_tracks_from_log(track_log_file)

        if tracks:
            safe_print("\n当前可用航迹:")
            safe_print("-" * 70)
            for track in tracks:
                safe_print(
                    f"  {track['track_id']}: 距离={track['range']:.1f}m, "
                    f"方位={track['azimuth']:.1f}°, "
                    f"俯仰={track['pitch']:.1f}°, "
                    f"速度={track['speed']:.1f}m/s"
                )
            safe_print("-" * 70)
            safe_print(f"共 {len(tracks)} 个航迹")
        else:
            safe_print("\n当前无可用航迹")

    except Exception as e:
        safe_print(f"\n读取日志失败: {e}")
        traceback.print_exc()
