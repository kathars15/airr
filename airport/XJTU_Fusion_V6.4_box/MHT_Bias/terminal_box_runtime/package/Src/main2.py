# -*- coding: utf-8 -*-
import json
import contextlib
import io
import multiprocessing
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy
import os
import csv
import numpy as np

from core import optical_service
from core.app_config import (
    DEBUG_POINT_MHT, FAKE_DIS, HOST_IP, MAX_RANGE, MHT_BIAS_PATH,
    POINT_RECORDS_FILE, POINT_TRACK_LOG_FILE, POINT_TRACK_RESULTS_FILE, POINT_VS_RAW_COMPARE_FILE,
    PROJECT_ROOT, RAW_TRACKS_FILE, TRACK_LOG_FILE, TRACK_RESULTS_FILE, OPTICAL_STATUS_FILE,
    CV_DETECTION_RESULTS_FILE,
    ENABLE_TRUE_POSITION_OUTPUT, TRUE_POSITION_CONFIRM_FRAMES,
    TRUE_POSITION_MAX_OPTICAL_AGE_SEC, TRUE_POSITION_PRINT_INTERVAL_SEC,
    TRUE_POSITION_REQUIRE_CURRENT_GUIDED_TARGET, TRUE_POSITION_MAX_RADAR_AGE_SEC,
    ENABLE_MANAGED_UDP_FANOUT, ENABLE_MANAGED_CV_DETECTION, CV_DETECTION_SCRIPT,
)
from core.calibration import calibrator
from core.console_utils import safe_print
from core.interactive_console import ConsoleExit, clear_data_dir, interactive_console
from core.optical_service import init_optical_tracker, send_to_optical
from core.radar_receiver import receive_radar_data
from core.radar_protocol import send_control_packet
from core.track_log import get_all_tracks_from_log, get_track_by_id_from_log
from core.true_position_estimator import TruePositionEstimator
from tools.compare_point_tracks_vs_raw import print_summary as print_point_debug_summary
from tools.compare_point_tracks_vs_raw import run_compare as run_point_debug_compare

sys.path.append(PROJECT_ROOT)
sys.path.append(MHT_BIAS_PATH)


def _popen_creationflags():
    if os.name == "nt":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


def start_managed_process(name, command, cwd=None, env=None, settle_sec=0.8):
    """Start an auxiliary process and keep it tied to the main program lifecycle."""
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            creationflags=_popen_creationflags(),
        )
    except Exception as exc:
        safe_print(f"[BOOT][WARN] {name} 启动失败: {exc}")
        return None

    time.sleep(settle_sec)
    if process.poll() is not None:
        safe_print(f"[BOOT][WARN] {name} 已退出，退出码: {process.returncode}")
        return None

    safe_print(f"[BOOT] {name} 已启动，PID={process.pid}")
    return process


def start_managed_udp_fanout():
    if not ENABLE_MANAGED_UDP_FANOUT:
        safe_print("[BOOT] managed UDP fanout disabled")
        return None

    script = os.path.join(MHT_BIAS_PATH, "Src", "tools", "network", "udp_fanout.py")
    if not os.path.exists(script):
        safe_print(f"[BOOT][WARN] UDP分发器脚本不存在: {script}")
        return None

    return start_managed_process(
        "UDP分发器",
        [sys.executable, script],
        cwd=os.path.dirname(script),
    )


def start_managed_cv_detection():
    if not ENABLE_MANAGED_CV_DETECTION:
        safe_print("[BOOT] managed CV detection disabled")
        return None

    script = CV_DETECTION_SCRIPT
    if not script or not os.path.exists(script):
        safe_print(f"[BOOT][WARN] 视频识别脚本不存在: {script}")
        return None

    stop_existing_cv_detection_processes(script)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.path.dirname(script))
    return start_managed_process(
        "光电视频识别窗口",
        [sys.executable, script],
        cwd=os.path.dirname(script),
        env=env,
        settle_sec=1.2,
    )


def stop_existing_cv_detection_processes(script):
    """Best-effort cleanup for stale RTSP detection processes from previous runs."""
    if os.name != "nt":
        return

    try:
        import subprocess as _subprocess

        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -match 'python' -and "
                f"$_.CommandLine -like '*{script.replace(chr(92), chr(92) + chr(92))}*' }} | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
            ),
        ]
        _subprocess.run(cmd, stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL, timeout=5)
    except Exception as exc:
        safe_print(f"[BOOT][WARN] 清理旧视频识别进程失败: {exc}")


def stop_managed_processes(processes):
    for name, process in reversed(processes):
        if process is None or process.poll() is not None:
            continue
        safe_print(f"[BOOT] 正在关闭{name} PID={process.pid}")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            safe_print(f"[BOOT][WARN] {name} 未及时退出，强制结束")
            process.kill()
            process.wait(timeout=3)
        except Exception as exc:
            safe_print(f"[BOOT][WARN] 关闭{name}失败: {exc}")


def read_latest_cv_detection(max_age_sec=1.0):
    try:
        with open(CV_DETECTION_RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return {
            "active": False,
            "reason": "cv_result_missing",
            "has_detection": False,
            "detections": [],
            "best_detection": None,
            "age_sec": None,
        }

    try:
        age_sec = time.time() - float(data.get("timestamp", 0.0))
    except (TypeError, ValueError):
        age_sec = None
    data["age_sec"] = age_sec
    if age_sec is None or age_sec > max_age_sec:
        data["active"] = False
        data["reason"] = f"cv_result_stale(age={age_sec})"
    if not data.get("active") or not data.get("has_detection"):
        data["best_detection"] = None
    return data


def write_true_position_record_csv(path, row):
    if not path:
        return
    fieldnames = [
        "host_time_text",
        "track_id",
        "raw_display_id",
        "east_m",
        "north_m",
        "up_m",
        "vel_east_mps",
        "vel_north_mps",
        "vel_up_mps",
        "speed_mps",
        "radar_range_m",
        "radar_class",
        "cv_class",
        "optical_confidence",
        "optical_status",
        "radar_age_sec",
        "optical_angle_age_sec",
        "cv_age_sec",
        "fusion_source",
    ]
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def write_true_position_record_jsonl(path, payload):
    if not path:
        return
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def enu_to_radar_polar(pos_enu, radar_heading_deg=None):
    """
    将ENU直角坐标转换为雷达极坐标
    
    参数:
        pos_enu: (3, 1) 数组，目标在ENU系中的位置 [x_east, y_north, z_up]
        radar_heading_deg: 雷达朝向（真北方向角，度），用于计算相对方位角
    
    返回:
        dict: 包含 range, azimuth, pitch, azimuth_relative
    """
    # 雷达位于ENU原点
    rel_pos = pos_enu.copy()
    
    # 距离
    range_m = np.linalg.norm(rel_pos)
    
    if range_m < 0.001:
        return {'range': 0, 'azimuth': 0, 'pitch': 0, 'azimuth_relative': 0}
    
    # 方位角（从北顺时针，0-360度）
    # atan2(x, y): x是东向位移，y是北向位移
    azimuth_deg = np.degrees(np.arctan2(rel_pos[0, 0], rel_pos[1, 0]))
    if azimuth_deg < 0:
        azimuth_deg += 360
    
    # 俯仰角
    pitch_deg = np.degrees(np.arcsin(rel_pos[2, 0] / range_m))
    
    # 相对方位角（相对于雷达朝向）
    azimuth_relative = azimuth_deg
    if radar_heading_deg is not None:
        azimuth_relative = azimuth_deg - radar_heading_deg
        azimuth_relative = azimuth_relative % 360  # 归一化到0-360
    
    return {
        'range': range_m,
        'azimuth': azimuth_deg,
        'pitch': pitch_deg,
        'azimuth_relative': azimuth_relative
    }


def _angle_delta_deg(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def match_raw_track_id(polar_coords, infos, max_score=120.0):
    """Best-effort mapping from an MHT output back to the raw radar track id."""
    best_score = None
    best_info = None
    for info in infos or []:
        if not isinstance(info, dict):
            continue
        raw_range = info.get('range')
        raw_az = info.get('azimuth')
        raw_pitch = info.get('pitch')
        if raw_range is None or raw_az is None or raw_pitch is None:
            continue
        try:
            score = (
                abs(float(polar_coords['range']) - float(raw_range)) * 0.10
                + _angle_delta_deg(polar_coords['azimuth'], raw_az) * 10.0
                + abs(float(polar_coords['pitch']) - float(raw_pitch)) * 10.0
            )
            if info.get('is_tas') is True:
                score -= 5.0
        except (TypeError, ValueError):
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_info = info
    if best_info is None or best_score is None or best_score > max_score:
        return None, None, best_score
    return best_info.get('track_id'), best_info.get('absolute_id'), best_score


def pick_preferred_track(tracks, exclude_track_id=None):
    """Prefer TAS tracks; fallback to nearest overall."""
    candidates = []
    for track in tracks or []:
        if exclude_track_id is not None and track.get('track_id') == exclude_track_id:
            continue
        candidates.append(track)
    if not candidates:
        return None

    tas_candidates = [track for track in candidates if track.get('is_tas') is True]
    if tas_candidates:
        return min(tas_candidates, key=lambda item: item['range'])
    return min(candidates, key=lambda item: item['range'])


def normalize_mht_timestamp(value):
    """Normalize radar timestamps without corrupting Unix wall-clock seconds."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        value = value.astype(float).ravel()[0]
    value = float(value)
    if value > 1_000_000_000:
        return value
    if value > 100_000:
        return value / 1000.0
    return value


def mht_process_and_send(
    Data2Process_Buffer,
    latest_tracks,
    calibration_queue,
    recording_state,
    auto_track_state_shared=None,
    optical_state_shared=None,
    result_json_file=TRACK_RESULTS_FILE,
    log_file_path=TRACK_LOG_FILE,
    process_label="MHT进程",
    update_shared_tracks=True,
    emit_udp=True,
):
    """MHT跟踪处理，同时按模式保存结果。"""
    safe_print(f"[{process_label}] ========== {process_label}启动 ==========")
    safe_print(f"[{process_label}] 等待接收雷达数据...")

    from core.calibration import calibrator
    safe_print(f"[{process_label}] 正在加载 MHT/聚类/分类依赖...")
    from MHT.POMHT import POMHT_Bias
    from common.clusters import Clustering_Obs
    from common.utlis import enu_to_geodetic
    from Sensor_Config.sensor_config import Sensor_Config, Name2SignalType, lla_original
    from Classify.TrackingClassify import TrackingClassify
    from Classify.Initial_Params import Initial_Classify_Params
    safe_print(f"[{process_label}] MHT/聚类/分类依赖加载完成")

    # 本地校准状态
    calibration_mode = False
    calibration_target = None
    true_position_estimator = TruePositionEstimator(
        confirm_frames=TRUE_POSITION_CONFIRM_FRAMES,
        max_optical_age_sec=TRUE_POSITION_MAX_OPTICAL_AGE_SEC,
        print_interval_sec=TRUE_POSITION_PRINT_INTERVAL_SEC,
    )

    PREDICT_SECONDS = 0

    # 创建发送socket
    ui_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def append_json_frame(frame_data):
        import os
        if not os.path.exists(result_json_file):
            with open(result_json_file, 'w', encoding='utf-8') as f:
                json.dump([frame_data], f, ensure_ascii=False, indent=2)
            return
        with open(result_json_file, 'r+', encoding='utf-8') as f:
            try:
                content = json.load(f)
                if not isinstance(content, list):
                    content = []
            except Exception:
                content = []
            content.append(frame_data)
            f.seek(0)
            json.dump(content, f, ensure_ascii=False, indent=2)
            f.truncate()

    def append_log_lines(lines):
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            for line in lines:
                log_file.write(line)
    
    Decided_Tree_All = []
    exit_flag = False
    
    def signal_handler(signum, frame):
        nonlocal exit_flag
        safe_print(f"\n[{process_label}] 收到中断信号，准备退出...")
        exit_flag = True
        try:
            Data2Process_Buffer.put_nowait(None)
        except:
            pass
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    Initial = False
    timestamp_last = -1
    frame_count = 0
    
    dim_d = 3
    Debug_Params = {'Debug': False, 'Begin_Frame': 30}
    MHT_Params = {
        'Lambda_NT': 1, 'Q_k': np.identity(dim_d) *0.1,
        "Max_Vel": 15.0, 'N_Scan': 1, 'Pg': 0.999,
        'P_death': 1e-2, 'dim_d': dim_d,
        'Debug_Params': Debug_Params, 'Resolved_Time_Window': 2,
        'Resolved_Min_Detect': 1, 'max_detect_time': 20
    }
    
    base_sensor_name = 'Radar_Track'
    base_sensor_appear = False
    
    Cluster_Params = {'Sigma': np.diag([10.0, 10.0, 10.0]), 'Distance': 20.0}
    
    label_id_map = {}
    label = 1
    
    Classify_Results = {}
    measurement_history = []
    estimation_history = []
    track_update_seq = {}
    
    # 获取雷达航向用于极坐标转换
    current_radar_heading = 0
    
    while True:

        try:
            cmd = calibration_queue.get_nowait()
            if cmd['type'] == 'start':
                calibration_mode = True
                calibration_target = cmd['target_id']
                # 也设置 calibrator 的状态（用于其他功能）
                calibrator.calibration_mode = True
                calibrator.current_target_id = cmd['target_id']
                safe_print(f"[{process_label}] 校准已开启，目标: {calibration_target}")
            elif cmd['type'] == 'stop':
                calibration_mode = False
                # 调用校准器计算
                calibrator.stop_calibration()
                safe_print(f"[{process_label}] 校准已停止")
        except queue.Empty:
            pass
        except Exception as e:
            safe_print(f"[{process_label}] 校准命令错误: {e}")
            
        try:
            data_k = Data2Process_Buffer.get(timeout=0.5)
        except queue.Empty:
            if exit_flag:
                break
            continue
        
        if data_k is None or exit_flag:
            safe_print(f"[{process_label}] 收到结束信号，退出")
            break
        
        frame_count += 1
        # safe_print(f"\n[MHT进程] ========== 第 {frame_count} 帧 ==========")
        
        time_begin = time.time()
        meas_chosen = data_k['meas']
        sensor_name_chosen = data_k['sensor_name']
        tmp_chosen = data_k['timestamp']
        infos_chosen = data_k['infos']
        if isinstance(infos_chosen, dict):
            infos_chosen = [infos_chosen]
        elif infos_chosen is None:
            infos_chosen = []
        
        timestamp_sec = normalize_mht_timestamp(tmp_chosen)
        
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
        

        #========== 在这里添加保存原始雷达数据的代码 ==========
        # 当传感器是 Radar_Track（雷达原始航迹）时，保存原始数据
        if (recording_state is None or recording_state.get('enabled', False)) and sensor_name_chosen == 'Radar_Track' and len(infos_chosen) > 0:
            import csv
            import os
            # 检查CSV文件是否存在，如果不存在则写入表头
            if not os.path.exists(RAW_TRACKS_FILE):
                with open(RAW_TRACKS_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'track_id', 'range', 'azimuth', 'pitch', 'speed', 'target_type'])
            
            # 保存每个原始航迹
            for i, obs in enumerate(obs_k):
                if i < len(infos_chosen):
                    raw_track = {
                        'timestamp': timestamp_sec,
                        'track_id': infos_chosen[i].get('track_id'),
                        'range': infos_chosen[i].get('range'),
                        'azimuth': infos_chosen[i].get('azimuth'),
                        'pitch': infos_chosen[i].get('pitch'),
                        'speed': infos_chosen[i].get('speed'),
                        'target_type': infos_chosen[i].get('target_type')
                    }
                    
                    # 保存到CSV文件
                    with open(RAW_TRACKS_FILE , 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            raw_track['timestamp'],
                            raw_track['track_id'],
                            raw_track['range'],
                            raw_track['azimuth'],
                            raw_track['pitch'],
                            raw_track['speed'],
                            raw_track['target_type']
                        ])
            
            # safe_print(f"[保存] 已保存 {len(obs_k)} 个原始航迹到 raw_tracks.csv")

            
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
            safe_print(f"[{process_label}] 初始化跟踪器...")
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
                safe_print(f"[{process_label}] 时间戳乱序，跳过")
                continue
            timestamp_last = timestamp_sec
            TOMHT.forward(timestamp=timestamp_sec, obs_k=obs_k, 
                         sensor_config=sensor_config, extra_infos=infos_chosen)
        
        # 输出结果
        if hasattr(TOMHT, 'Output_Nodes') and len(TOMHT.Output_Nodes) > 0:
            Decided_Tree = deepcopy(TOMHT.Output_Nodes[-1])
            target_num = len(Decided_Tree)
            
            if target_num > 0:
                # safe_print(f"[MHT结果] 确认航迹数: {target_num}")
                timestamp_output = TOMHT.Timestamps[-1] if hasattr(TOMHT, 'Timestamps') and len(TOMHT.Timestamps) > 0 else timestamp_sec
                
                msg_result = {'timestamp': int(timestamp_output * 1000), 'result': []}
                
                # 用于日志记录
                log_targets = []
                json_targets = []  # 用于JSON保存
                
                type_map = {
                    -1: '无人机',
                    0: '未知',
                    1: '飞鸟',
                }
                for node in Decided_Tree.values():
                    if node.label not in label_id_map:
                        label_id_map[node.label] = label
                        label += 1
                    
                    pos_enu = node.x_k_k[:3, :]
                    vel_enu = node.x_k_k[3:6, :]
                    speed = np.linalg.norm(vel_enu)
                    
                    track_id_str = f'Radar-{label_id_map[node.label]}'
                    track_update_seq[track_id_str] = track_update_seq.get(track_id_str, 0) + 1

                    if update_shared_tracks and latest_tracks is not None:
                        latest_tracks[track_id_str] = {
                            'valid': True,
                            'track_id': track_id_str,
                            'is_tas': infos_chosen[0].get('is_tas') if infos_chosen else None,
                            'track_mode': infos_chosen[0].get('track_mode') if infos_chosen else None,
                            'pos_enu': pos_enu.reshape(-1).tolist(),
                            'vel_enu': vel_enu.reshape(-1).tolist(),
                            'last_update_time': time.time(),
                            'update_seq': track_update_seq[track_id_str],
                        }

                    # =========================
                    # 2秒匀速预测
                    # =========================
                    pred_pos_enu = pos_enu + vel_enu * PREDICT_SECONDS

                    pred_track_lla = enu_to_geodetic(
                        lla_original[0], lla_original[1], lla_original[2], pred_pos_enu
                    )

                    pred_polar_coords = enu_to_radar_polar(pred_pos_enu, radar_heading_deg=None)
                    track_lla = enu_to_geodetic(lla_original[0], lla_original[1], 
                                                lla_original[2], pos_enu)


                    # ========== 添加：只在 MHT 输出时记录雷达数据 ==========
                    # 在记录雷达数据的地方，使用本地的 calibration_mode
                    # print(calibration_mode,track_id_str,calibration_target)
                    # if calibration_mode:
                    # Calibration now uses POINT packet sampling in main process.
                        
                    # 获取原始雷达ID
                    raw_display_id = None
                    raw_absolute_id = None
                    if hasattr(node, 'obs_id') and node.obs_id and len(node.obs_id) > 0:
                        last_obs_idx = node.obs_id[-1]
                        if infos_chosen and last_obs_idx < len(infos_chosen):
                            cluster_idx = node.obs_id[-1]
                            original_indices = obs_indexs[cluster_idx]
                            all_ids = [infos_chosen[idx].get('track_id') for idx in original_indices if idx < len(infos_chosen)]
                            if all_ids:
                                raw_display_id = ','.join(map(str, all_ids))
                            raw_info = infos_chosen[last_obs_idx]
                            raw_absolute_id = raw_info.get('absolute_id')
                    
                    # # 打印对比信息到控制台
                    # if raw_display_id is not None:
                    #     safe_print(f"[ID映射] MHT输出: Radar-{label_id_map[node.label]} (label={node.label}) -> 原始雷达ID: {raw_display_id}")
                    
                    # ========== 极坐标转换 ==========
                    # 获取雷达航向（从infos_chosen中获取）
                    radar_heading = 0
                    if infos_chosen and len(infos_chosen) > 0:
                        radar_heading = infos_chosen[0].get('radar_heading', 0)
                    
                    polar_coords = enu_to_radar_polar(pos_enu, radar_heading)
                    matched_raw_id, matched_absolute_id, raw_match_score = match_raw_track_id(polar_coords, infos_chosen)
                    if raw_display_id is None and matched_raw_id is not None:
                        raw_display_id = str(matched_raw_id)
                    if raw_absolute_id is None and matched_absolute_id is not None:
                        raw_absolute_id = matched_absolute_id

                    true_position_result = {
                        "used": False,
                        "skip_reason": "disabled",
                        "true_position_source": None,
                        "should_print": False,
                    }
                    if ENABLE_TRUE_POSITION_OUTPUT and optical_state_shared is not None:
                        now_for_true_pos = time.time()
                        try:
                            optical_state_snapshot = dict(optical_state_shared)
                        except Exception:
                            optical_state_snapshot = {}
                        try:
                            current_guided_id = (
                                auto_track_state_shared.get('current_track_id')
                                if auto_track_state_shared is not None else None
                            )
                        except Exception:
                            current_guided_id = None

                        is_current_guided_target = (
                            not TRUE_POSITION_REQUIRE_CURRENT_GUIDED_TARGET
                            or current_guided_id is None
                            or str(current_guided_id) == str(track_id_str)
                        )
                        if is_current_guided_target:
                            true_position_result = true_position_estimator.estimate(
                                track_id=track_id_str,
                                raw_display_id=raw_display_id,
                                range_m=polar_coords['range'],
                                optical_state=optical_state_snapshot,
                                fusion_time=timestamp_output,
                                now=now_for_true_pos,
                            )
                        else:
                            true_position_result = {
                                "used": False,
                                "skip_reason": "not_current_guided_target",
                                "true_position_source": None,
                                "should_print": False,
                            }
                     
                    # safe_print(f"[极坐标] track_Radar-{label_id_map[node.label]}: "
                    #       f"距离={polar_coords['range']:.1f}m, "
                    #       f"方位={polar_coords['azimuth']:.1f}°, "
                    #       f"俯仰={polar_coords['pitch']:.1f}°, "
                    #       f"相对方位={polar_coords['azimuth_relative']:.1f}°")
                    
                    # ========== 添加分类逻辑 ==========
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
                            lr_value = Classify_Results[node.label]['Log_Likelihood_Ratio']
                            if isinstance(lr_value, np.ndarray):
                                lr_value = lr_value.item() if lr_value.size == 1 else lr_value[0]
                            
                            type_name = type_map.get(Target_type, '未知')
                            # safe_print(f"[分类] label={node.label}, Time_N={Classify_Results[node.label]['Time_N']}, "
                            #     f"LR={lr_value:.3f}, type={Target_type} ({type_name})")
                    
                    target_type = Classify_Results[node.label]['Target_type'] if node.label in Classify_Results else 3
                           
                    target_result = {
                        'track_id': f'Radar-{label_id_map[node.label]}',
                        'target_type': target_type,

                        # 当前状态
                        'lat': track_lla[0, 0].item(),
                        'lon': track_lla[1, 0].item(),
                        'alt': track_lla[2, 0].item(),
                        'height': track_lla[2, 0].item(),
                        'speed': speed.item(),
                        'range': polar_coords['range'],
                        'azimuth': polar_coords['azimuth'],
                        'azimuth_relative': polar_coords['azimuth_relative'],
                        'pitch': polar_coords['pitch'],

                        # 2秒预测状态
                        'pred_2s': {
                            'lat': pred_track_lla[0, 0].item(),
                            'lon': pred_track_lla[1, 0].item(),
                            'alt': pred_track_lla[2, 0].item(),
                            'range': pred_polar_coords['range'],
                            'azimuth': pred_polar_coords['azimuth'],
                            'pitch': pred_polar_coords['pitch']
                        },

                        'extra_info': {
                            'vel_x': vel_enu[0, 0].item(),
                            'vel_y': vel_enu[1, 0].item(),
                            'vel_z': vel_enu[2, 0].item(),
                            'is_tas': infos_chosen[0].get('is_tas') if infos_chosen else None,
                            'track_mode': infos_chosen[0].get('track_mode') if infos_chosen else None,
                            'fusion_time': timestamp_output,
                            'predict_seconds': PREDICT_SECONDS,
                            'signal_source_types': 1,
                            'raw_display_id': raw_display_id,
                            'raw_absolute_id': raw_absolute_id,
                            'raw_match_score': raw_match_score,
                            'estimated_true_enu': true_position_result.get('estimated_true_enu'),
                            'estimated_true_velocity_enu': true_position_result.get('estimated_true_velocity_enu'),
                            'estimated_true_speed_mps': true_position_result.get('estimated_true_speed_mps'),
                            'true_position_source': true_position_result.get('true_position_source'),
                            'optical_tracking_confirmed': bool(true_position_result.get('used')),
                            'optical_angle_age_sec': true_position_result.get('optical_angle_age_sec'),
                            'true_position_skip_reason': true_position_result.get('skip_reason'),
                        }
                    }

                    msg_result['result'].append(target_result)

                    if true_position_result.get('used') and true_position_result.get('should_print'):
                        try:
                            true_log_active = bool(recording_state.get('true_position_active', False))
                        except Exception:
                            true_log_active = False

                        radar_state_age = 0.0
                        try:
                            radar_state_snapshot = latest_tracks.get(track_id_str, {}) if latest_tracks is not None else {}
                            radar_state_age = now_for_true_pos - float(radar_state_snapshot.get('last_update_time', now_for_true_pos))
                        except Exception:
                            radar_state_age = TRUE_POSITION_MAX_RADAR_AGE_SEC + 1.0
                        if true_log_active and radar_state_age <= TRUE_POSITION_MAX_RADAR_AGE_SEC:
                            true_enu = true_position_result.get('estimated_true_enu') or {}
                            true_vel = true_position_result.get('estimated_true_velocity_enu') or {}
                            speed_true = true_position_result.get('estimated_true_speed_mps')
                            speed_text = f"{speed_true:.2f}" if speed_true is not None else "NA"
                            raw_text = raw_display_id or "NA"
                            cv_result = read_latest_cv_detection()
                            best_det = cv_result.get("best_detection") or {}
                            cv_class = best_det.get("class_name") or "none"
                            cv_conf = best_det.get("confidence")
                            cv_conf_text = f"{cv_conf:.2f}" if cv_conf is not None else "NA"
                            cv_age_sec = cv_result.get("age_sec")
                            cv_age_text = f"{cv_age_sec:.2f}" if cv_age_sec is not None else "NA"
                            optical_status_value = optical_state_snapshot.get("current_status")
                            host_time = time.time()
                            host_time_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(host_time))
                            safe_print(
                                f"[真实位置记录] time={host_time_text} raw={raw_text} track={track_id_str} "
                                f"ENU=({true_enu.get('east_m', 0.0):.2f}, "
                                f"{true_enu.get('north_m', 0.0):.2f}, "
                                f"{true_enu.get('up_m', 0.0):.2f})m "
                                f"V=({true_vel.get('east_mps', 0.0):.2f}, "
                                f"{true_vel.get('north_mps', 0.0):.2f}, "
                                f"{true_vel.get('up_mps', 0.0):.2f})m/s "
                                f"speed={speed_text}m/s range={polar_coords['range']:.1f}m "
                                f"radar_class={target_result['target_type']} "
                                f"opt=({true_position_result.get('optical_azimuth_deg', 0.0):.2f}°, "
                                f"{true_position_result.get('optical_pitch_deg', 0.0):.2f}°) "
                                f"cv={cv_class}({cv_conf_text}) "
                                f"age=(radar:{radar_state_age:.2f}, optical:{true_position_result.get('optical_angle_age_sec', 0.0):.2f}, cv:{cv_age_text})"
                            )

                            record_payload = {
                                "host_time_text": host_time_text,
                                "track_id": track_id_str,
                                "raw_display_id": raw_text,
                                "east_m": true_enu.get('east_m'),
                                "north_m": true_enu.get('north_m'),
                                "up_m": true_enu.get('up_m'),
                                "vel_east_mps": true_vel.get('east_mps'),
                                "vel_north_mps": true_vel.get('north_mps'),
                                "vel_up_mps": true_vel.get('up_mps'),
                                "speed_mps": speed_true,
                                "radar_range_m": polar_coords['range'],
                                "radar_class": target_result['target_type'],
                                "cv_class": cv_class,
                                "optical_confidence": cv_conf,
                                "optical_status": optical_status_value,
                                "radar_age_sec": radar_state_age,
                                "optical_angle_age_sec": true_position_result.get('optical_angle_age_sec'),
                                "cv_age_sec": cv_age_sec,
                                "fusion_source": true_position_result.get('true_position_source'),
                            }
                            row = dict(record_payload)
                            try:
                                write_true_position_record_csv(recording_state.get('true_position_csv'), row)
                                write_true_position_record_jsonl(recording_state.get('true_position_jsonl'), record_payload)
                            except Exception as exc:
                                safe_print(f"[真实位置记录] 写日志失败: {exc}")
                     
                    # 保存用于日志
                    log_targets.append({
                        'track_id': target_result['track_id'],
                        'lat': target_result['lat'],
                        'lon': target_result['lon'],
                        'speed': target_result['speed'],
                        'raw_display_id': target_result['extra_info'].get('raw_display_id'),
                        'estimated_true_enu': target_result['extra_info'].get('estimated_true_enu'),
                        'estimated_true_velocity_enu': target_result['extra_info'].get('estimated_true_velocity_enu'),
                        'estimated_true_speed_mps': target_result['extra_info'].get('estimated_true_speed_mps'),
                        'optical_tracking_confirmed': target_result['extra_info'].get('optical_tracking_confirmed'),
                        'optical_angle_age_sec': target_result['extra_info'].get('optical_angle_age_sec'),

                        # 当前状态
                        'range': polar_coords['range'],
                        'azimuth': polar_coords['azimuth'],
                        'pitch': polar_coords['pitch'],
                        'is_tas': target_result['extra_info'].get('is_tas'),
                        'track_mode': target_result['extra_info'].get('track_mode'),

                        # 2秒预测状态
                        'pred_range': pred_polar_coords['range'],
                        'pred_azimuth': pred_polar_coords['azimuth'],
                        'pred_pitch': pred_polar_coords['pitch']
                    })

                    
                    # 保存用于JSON
                    json_targets.append({
                        'track_id': target_result['track_id'],
                        'target_type': target_result['target_type'],
                        'lat': target_result['lat'],
                        'lon': target_result['lon'],
                        'alt': target_result['alt'],
                        'height': target_result['height'],
                        'speed': target_result['speed'],
                        'range': polar_coords['range'],
                        'azimuth': polar_coords['azimuth'],
                        'azimuth_relative': polar_coords['azimuth_relative'],
                        'pitch': polar_coords['pitch'],
                        'is_tas': target_result['extra_info'].get('is_tas'),
                        'track_mode': target_result['extra_info'].get('track_mode'),
                        'vel_x': target_result['extra_info']['vel_x'],
                        'vel_y': target_result['extra_info']['vel_y'],
                        'vel_z': target_result['extra_info']['vel_z'],
                        'fusion_time': target_result['extra_info']['fusion_time'],
                        'raw_display_id': target_result['extra_info'].get('raw_display_id'),
                        'raw_absolute_id': target_result['extra_info'].get('raw_absolute_id'),
                        'raw_match_score': target_result['extra_info'].get('raw_match_score'),
                        'estimated_true_enu': target_result['extra_info'].get('estimated_true_enu'),
                        'estimated_true_velocity_enu': target_result['extra_info'].get('estimated_true_velocity_enu'),
                        'estimated_true_speed_mps': target_result['extra_info'].get('estimated_true_speed_mps'),
                        'true_position_source': target_result['extra_info'].get('true_position_source'),
                        'optical_tracking_confirmed': target_result['extra_info'].get('optical_tracking_confirmed'),
                        'optical_angle_age_sec': target_result['extra_info'].get('optical_angle_age_sec'),
                        'true_position_skip_reason': target_result['extra_info'].get('true_position_skip_reason'),
                    })
                
                # 保存JSON结果
                json_frame_data = {
                    'frame': frame_count,
                    'timestamp': timestamp_output,
                    'timestamp_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp_output)),
                    'target_count': target_num,
                    'targets': json_targets
                }
                
                try:
                    append_json_frame(json_frame_data)
                except Exception as e:
                    safe_print(f"[JSON] 保存失败: {e}")
                
                # UDP发送结果
                if emit_udp:
                    try:
                        json_string = json.dumps(msg_result)
                        sock.sendto(json_string.encode(), (HOST_IP, 9999))
                    except Exception as e:
                        safe_print(f"[{process_label}] UDP发送失败: {e}")
                
                # 写入文本日志文件
                frame_log_lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} | 第{frame_count}帧 | {target_num}个目标\n"]
                for target in log_targets:
                    if target.get('range', 0) <= MAX_RANGE:
                        raw_id = target.get('raw_display_id')
                        display_id = f"{target['track_id']}(raw={raw_id})" if raw_id else target['track_id']
                        target['track_id'] = display_id
                        true_enu = target.get('estimated_true_enu')
                        true_vel = target.get('estimated_true_velocity_enu')
                        true_text = ""
                        if target.get('optical_tracking_confirmed') and true_enu:
                            true_text = (
                                f", 估计ENU=({true_enu.get('east_m', 0.0):.1f},"
                                f"{true_enu.get('north_m', 0.0):.1f},"
                                f"{true_enu.get('up_m', 0.0):.1f})m"
                            )
                            if true_vel:
                                true_text += (
                                    f", 估计V=({true_vel.get('east_mps', 0.0):.1f},"
                                    f"{true_vel.get('north_mps', 0.0):.1f},"
                                    f"{true_vel.get('up_mps', 0.0):.1f})m/s"
                                )
                        frame_log_lines.append(
                            f"  {target['track_id']}: 距离={target['range']:.1f}m, "
                            f"方位={target['azimuth']:.1f}°, "
                            f"俯仰={target['pitch']:.1f}°, "
                            f"速度={target['speed']:.1f}m/s, "
                            f"位置=({target['lat']:.6f},{target['lon']:.6f})"
                            f"{true_text}\n"
                        )
                        
                        # log_file.write(f"  {target['track_id']}: 距离={target['pred_range']:.1f}m, "
                        #             f"方位={target['pred_azimuth']:.1f}°, "
                        #             f"俯仰={target['pred_pitch']:.1f}°, "
                        #             f"速度={target['speed']:.1f}m/s, "
                        #             f"位置=({target['lat']:.6f},{target['lon']:.6f})\n")
                append_log_lines(frame_log_lines)
            
            Decided_Tree_All.append(Decided_Tree)
        
        time_over = time.time()
        # safe_print(f"[性能] 耗时: {(time_over - time_begin)*1000:.2f}ms")
        
        if frame_count % 10 == 0:
            active_tracks = len(TOMHT.Output_Nodes[-1]) if len(TOMHT.Output_Nodes) > 0 else 0
            if process_label != "MHT进程":
                safe_print(f"[{process_label}] 已处理{frame_count}帧, 当前航迹数={active_tracks}")
    
    # JSON/log are appended frame-by-frame; nothing to close here.
    sock.close()
    safe_print(f"[{process_label}] 退出")


def control_console():
    """交互式控制台，用于发送控制命令"""
    safe_print("\n" + "="*50)
    safe_print("雷达控制台")
    safe_print("="*50)
    safe_print("命令列表:")
    safe_print("  1. 开机 (辐射开, 雷达开)")
    safe_print("  2. 待机 (辐射关, 雷达开)")
    safe_print("  3. 关机 (辐射关, 雷达关)")
    safe_print("  4. 设置周扫模式 (360度扫描)")
    safe_print("  5. 设置扇扫模式 (指定角度范围)")
    safe_print("  6. 设置俯仰扫描")
    safe_print("  7. 设置工作频率")
    safe_print("  8. 设置雷达位置")
    safe_print("  9. 发送自定义控制包")
    safe_print("  0. 退出")
    
    current_config = {
        'radar_on': True,
        'radiation_on': True,
        'work_mode': 1,
        'azimuth_scan_mode': 4,
        'azimuth_start': -180,
        'azimuth_end': 180,
        'azimuth_step': 0,
        'pitch_scan_mode': 0,
        'pitch_start': 0,
        'pitch_end': 0,
        'pitch_step': 0,
        'frequency': 16000,
        'radar_height': 0,
        'longitude': 0,
        'latitude': 0
    }
    
    while True:
        try:
            cmd = input("\n请输入命令编号: ").strip()
            
            if cmd == '0':
                break
            elif cmd == '1':
                current_config['radar_on'] = True
                current_config['radiation_on'] = True
                send_control_packet(**current_config)
                safe_print("[控制] 已发送开机命令")
                
            elif cmd == '2':
                current_config['radar_on'] = True
                current_config['radiation_on'] = False
                send_control_packet(**current_config)
                safe_print("[控制] 已发送待机命令")
                
            elif cmd == '3':
                current_config['radar_on'] = False
                current_config['radiation_on'] = False
                send_control_packet(**current_config)
                safe_print("[控制] 已发送关机命令")
                
            elif cmd == '4':
                current_config['azimuth_scan_mode'] = 4
                current_config['azimuth_start'] = -180
                current_config['azimuth_end'] = 180
                current_config['azimuth_step'] = 0
                send_control_packet(**current_config)
                safe_print("[控制] 已设置为周扫模式 (360度)")
                
            elif cmd == '5':
                try:
                    start = float(input("起始角(度, -180~180): "))
                    end = float(input("终止角(度, -180~180): "))
                    step = float(input("步进角(度, 0/1/2/3/4): "))
                    
                    current_config['azimuth_scan_mode'] = 3
                    current_config['azimuth_start'] = start
                    current_config['azimuth_end'] = end
                    current_config['azimuth_step'] = step
                    send_control_packet(**current_config)
                    safe_print(f"[控制] 已设置为扇扫模式: {start}° ~ {end}°, 步进{step}°")
                except ValueError:
                    safe_print("[错误] 输入无效")
                    
            elif cmd == '6':
                try:
                    mode = int(input("俯仰模式 (0=定向, 1=扫描, 2=多波束): "))
                    if mode == 0:
                        angle = float(input("固定角度(度, -25~25): "))
                        start, end = angle, angle
                    else:
                        start = float(input("起始角(度, -25~25): "))
                        end = float(input("终止角(度, -25~25): "))
                    step = float(input("步进角(度, 0/2/4/6/8): "))
                    
                    current_config['pitch_scan_mode'] = mode
                    current_config['pitch_start'] = start
                    current_config['pitch_end'] = end
                    current_config['pitch_step'] = step
                    send_control_packet(**current_config)
                    safe_print("[控制] 已设置俯仰参数")
                except ValueError:
                    safe_print("[错误] 输入无效")
                    
            elif cmd == '7':
                try:
                    freq = int(input("工作频率(MHz, 例如16000): "))
                    current_config['frequency'] = freq
                    send_control_packet(**current_config)
                    safe_print(f"[控制] 已设置工作频率: {freq} MHz")
                except ValueError:
                    safe_print("[错误] 输入无效")
                    
            elif cmd == '8':
                try:
                    lon = float(input("经度(度, 例如108.12345): "))
                    lat = float(input("纬度(度, 例如34.12345): "))
                    height = float(input("海拔高度(米): "))
                    
                    current_config['longitude'] = lon
                    current_config['latitude'] = lat
                    current_config['radar_height'] = int(height)
                    send_control_packet(**current_config)
                    safe_print(f"[控制] 已设置雷达位置: ({lon}, {lat}, {height}m)")
                except ValueError:
                    safe_print("[错误] 输入无效")
                    
            elif cmd == '9':
                safe_print("使用当前配置发送...")
                send_control_packet(**current_config)
                
            else:
                safe_print("无效命令，请重新输入")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            safe_print(f"[错误] {e}")

def get_current_track_motion(track_id):
    """从共享状态中获取当前目标运动信息"""
    global latest_tracks,track_lock  

    with track_lock:
        motion = latest_tracks.get(track_id)

    if motion is None:
        return None

    try:
        pos_enu = np.array(motion['pos_enu'], dtype=float).reshape(3, 1)
        vel_enu = np.array(motion['vel_enu'], dtype=float).reshape(3, 1)

        return {
            'valid': motion.get('valid', False),
            'track_id': motion.get('track_id'),
            'is_tas': motion.get('is_tas'),
            'track_mode': motion.get('track_mode'),
            'pos_enu': pos_enu,
            'vel_enu': vel_enu,
            'last_update_time': float(motion.get('last_update_time', 0.0)),
            'update_seq': int(motion.get('update_seq', 0)),
        }
    except Exception as e:
        safe_print(f"[共享状态] 解析 {track_id} 运动信息失败: {e}")
        return None



RADAR_SCAN_PERIOD = 4.0          # 雷达扫描周期 4 秒
OPTICAL_SEND_INTERVAL = 1.8      # 光电重发周期
FOLLOW_PREDICT_LEAD = 0.0        # 额外前馈预测，可先设 0


def write_optical_status_file(status_payload):
    """Write optical status for the CV viewer without failing on transient Windows file locks."""
    tmp_file = f"{OPTICAL_STATUS_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(status_payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        for _ in range(5):
            try:
                os.replace(tmp_file, OPTICAL_STATUS_FILE)
                return True
            except PermissionError:
                time.sleep(0.02)

        # Some Windows readers can briefly block atomic replace. Fall back to
        # direct overwrite; the CV viewer already tolerates occasional JSON
        # read failures and will retry on the next frame.
        with open(OPTICAL_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status_payload, f, ensure_ascii=False)
        return True
    except Exception as e:
        safe_print(f"[光电状态] 写入失败: {e}")
        return False
    finally:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except OSError:
            pass


def optical_data_monitor():
    """Record optical measurements for calibration."""
    global optical_state_shared
    last_status_write_time = 0.0
    while True:
        try:
            tracker = optical_service.tracker
            if tracker and tracker.connected:
                with tracker.lock:
                    current_az = tracker.latest_azimuth
                    current_pitch = tracker.latest_pitch
                    current_range = tracker.latest_range
                    current_status = tracker.current_status
                    latest_angle_host_time = tracker.latest_angle_host_time
                    latest_status_host_time = tracker.latest_status_host_time
                    latest_target_host_time = tracker.latest_target_host_time
                    latest_targets = list(tracker.latest_targets)

                now = time.time()
                if optical_state_shared is not None:
                    optical_state_shared['latest_azimuth'] = current_az
                    optical_state_shared['latest_pitch'] = current_pitch
                    optical_state_shared['latest_range'] = current_range
                    optical_state_shared['current_status'] = current_status
                    optical_state_shared['latest_angle_host_time'] = latest_angle_host_time
                    optical_state_shared['latest_status_host_time'] = latest_status_host_time
                    optical_state_shared['latest_target_host_time'] = latest_target_host_time
                    optical_state_shared['host_update_time'] = now
                if now - last_status_write_time >= 0.2:
                    status_payload = {
                        'timestamp': now,
                        'latest_azimuth': current_az,
                        'latest_pitch': current_pitch,
                        'latest_range': current_range,
                        'current_status': current_status,
                        'latest_angle_host_time': latest_angle_host_time,
                        'latest_status_host_time': latest_status_host_time,
                        'latest_target_host_time': latest_target_host_time,
                        'true_position_recording_active': bool(recording_state.get('true_position_active', False)),
                        'true_position_recording_target': recording_state.get('true_position_target'),
                        'target_count': len(latest_targets),
                        'latest_targets': [
                            {
                                'target_id': t.get('target_id'),
                                'target_type': t.get('target_type'),
                                'similarity': t.get('similarity'),
                                'width': t.get('width'),
                                'height': t.get('height'),
                                'pos_x': t.get('pos_x'),
                                'pos_y': t.get('pos_y'),
                                'target_az': t.get('target_az'),
                                'target_pitch': t.get('target_pitch'),
                                'target_dist': t.get('target_dist'),
                                'packet_timestamp': t.get('packet_timestamp'),
                            }
                            for t in latest_targets[:5]
                        ],
                    }
                    if write_optical_status_file(status_payload):
                        last_status_write_time = now
                if current_status == 2 and current_az is not None:
                    optical_device_timestamp = None
                    if latest_targets:
                        raw_ts = latest_targets[0].get('packet_timestamp')
                        if raw_ts is not None:
                            try:
                                optical_device_timestamp = float(raw_ts)
                            except (TypeError, ValueError):
                                optical_device_timestamp = None
                    if calibrator.calibration_mode:
                        calibrator.add_optical_measurement(
                            current_az,
                            current_pitch,
                            now,
                            current_status,
                            current_range,
                            device_timestamp=optical_device_timestamp,
                        )
            time.sleep(0.2)

        except Exception as e:
            safe_print(f"[????] ??: {e}")
            time.sleep(1)


def calibration_radar_log_sampler():
    """Sample calibration radar data from the latest confirmed track log."""
    last_signature = None

    while True:
        try:
            if not calibrator.calibration_mode or not calibrator.current_target_id:
                last_signature = None
                time.sleep(0.2)
                continue

            target = get_track_by_id_from_log(calibrator.current_target_id)
            if target is None:
                time.sleep(0.5)
                continue

            signature = (
                target.get('track_id'),
                round(float(target.get('range', 0.0)), 1),
                round(float(target.get('azimuth', 0.0)), 1),
                round(float(target.get('pitch', 0.0)), 1),
                round(float(target.get('speed', 0.0)), 1),
            )
            if signature == last_signature:
                time.sleep(0.2)
                continue
            last_signature = signature

            calibrator.add_radar_measurement(
                target['track_id'],
                target['azimuth'],
                target['pitch'],
                target['range'],
                time.time(),
                device_timestamp=target.get('fusion_time'),
            )
        except Exception as e:
            safe_print(f"[cal] radar log sampler error: {e}")
            time.sleep(1.0)

        time.sleep(0.2)


def auto_follow_loop():
    """自动跟踪循环（受 auto_track_config['enabled'] 控制）"""
    global auto_track_config, auto_track_lock, auto_track_state
    global OPTICAL_SEND_INTERVAL, RADAR_SCAN_PERIOD, FAKE_DIS, FOLLOW_PREDICT_LEAD

    last_target_id = None
    last_successful_send = 0.0
    last_sent_update_seq = None
    LOST_TIMEOUT = 5.0

    while True:
        try:
            tracker = optical_service.tracker
            if not auto_track_config['enabled']:
                last_target_id = None
                last_sent_update_seq = None
                time.sleep(0.2)
                continue

            with auto_track_lock:
                current_id = auto_track_state.get('current_track_id')
                # 自动模式下，如果处于手动锁定状态，跳过
                if auto_track_state.get('manual_locked', False):
                    time.sleep(0.2)
                    continue

            if current_id is None:
                if last_target_id is not None:
                    last_target_id = None
                    last_sent_update_seq = None
                time.sleep(0.2)
                continue

            if current_id != last_target_id:
                last_target_id = current_id
                last_sent_update_seq = None
                last_successful_send = time.time()

            elapsed_since_success = time.time() - last_successful_send
            if elapsed_since_success > LOST_TIMEOUT:
                if tracker and tracker.connected:
                    tracker.release_target()
                    tracker.reset_zoom(125)
                last_target_id = None
                last_sent_update_seq = None
                time.sleep(1)
                continue

            motion = get_current_track_motion(current_id)
            if motion is None or not motion.get('valid', False):
                time.sleep(0.1)
                continue

            update_seq = motion.get('update_seq', 0)
            if update_seq == last_sent_update_seq:
                time.sleep(0.2)
                continue

            now = time.time()
            dt = now - motion['last_update_time']
            if dt < 0:
                dt = 0.0
            dt = dt + FOLLOW_PREDICT_LEAD

            pred_pos_enu = motion['pos_enu'] + motion['vel_enu'] * dt
            pred_polar = enu_to_radar_polar(pred_pos_enu, radar_heading_deg=None)

            send_distance = pred_polar['range']
            if send_distance < 0:
                send_distance = 0.0

            # Calibration radar samples now come from POINT packet sampler.

            send_to_optical(
                current_id,
                pred_polar['azimuth'],
                pred_polar['pitch'],
                send_distance
            )

            if True:  # 成功时
                last_successful_send = time.time()
                last_sent_update_seq = update_seq
                
            time.sleep(0.2)

        except Exception as e:
            safe_print(f"[自动跟踪] 线程异常: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)


def manual_follow_loop():
    """手动跟踪循环（独立运行，不受 auto_track_config['enabled'] 影响）"""
    global auto_track_lock, auto_track_state
    global OPTICAL_SEND_INTERVAL, FAKE_DIS, FOLLOW_PREDICT_LEAD

    last_target_id = None
    last_successful_send = 0.0
    last_sent_update_seq = None
    LOST_TIMEOUT = 5.0  # 手动模式下目标丢失超时（秒）

    while True:
        try:
            tracker = optical_service.tracker
            with auto_track_lock:
                current_id = auto_track_state.get('current_track_id')
                manual_locked = auto_track_state.get('manual_locked', False)

            # 只在手动锁定模式下工作
            if not manual_locked or current_id is None:
                # 没有手动锁定时，重置状态
                if last_target_id is not None:
                    safe_print("[手动跟踪] 已退出手动模式")
                    last_target_id = None
                    last_successful_send = 0.0
                    last_sent_update_seq = None
                time.sleep(0.2)
                continue

            # 目标刚切换或首次锁定
            if current_id != last_target_id:
                last_target_id = current_id
                last_sent_update_seq = None
                last_successful_send = time.time()
                safe_print(f"[手动跟踪] 开始跟随目标 {current_id}")

            # 检查目标是否丢失
            elapsed_since_success = time.time() - last_successful_send
            if elapsed_since_success > LOST_TIMEOUT:
                safe_print(f"[手动跟踪] 目标 {current_id} 丢失超过 {LOST_TIMEOUT}s，释放目标")
                if tracker and tracker.connected:
                    tracker.release_target()
                # 清除手动锁定状态
                with auto_track_lock:
                    auto_track_state['current_track_id'] = None
                    auto_track_state['current_target'] = None
                    auto_track_state['manual_locked'] = False
                last_target_id = None
                last_sent_update_seq = None
                time.sleep(1)
                continue

            # 获取目标运动状态
            motion = get_current_track_motion(current_id)
            if motion is None or not motion.get('valid', False):
                safe_print(f"[手动跟踪] 等待目标 {current_id} 数据...")
                time.sleep(0.2)
                continue

            update_seq = motion.get('update_seq', 0)
            if update_seq == last_sent_update_seq:
                time.sleep(0.2)
                continue

            # 计算预测位置
            now = time.time()
            dt = now - motion['last_update_time']
            if dt < 0:
                dt = 0.0
            dt = dt + FOLLOW_PREDICT_LEAD

            dt = 0.0  #不用预测

            pred_pos_enu = motion['pos_enu'] + motion['vel_enu'] * dt
            pred_polar = enu_to_radar_polar(pred_pos_enu, radar_heading_deg=None)

            send_distance = pred_polar['range'] - FAKE_DIS
            if send_distance < 0:
                send_distance = 0.0

            # 发送到光电
            with contextlib.redirect_stdout(io.StringIO()):
                ok = send_to_optical(
                    current_id,
                    pred_polar['azimuth'],
                    pred_polar['pitch'],
                    send_distance
                )

            if ok:
                last_successful_send = time.time()
                last_sent_update_seq = update_seq
                safe_print(f"[手动跟踪] {current_id} | "
                      f"距离={send_distance:.1f}m, "
                      f"方位={pred_polar['azimuth']:.1f}°, "
                      f"俯仰={pred_polar['pitch']:.1f}°")
            else:
                safe_print(f"[手动跟踪] 发送失败，目标 {current_id}")

            time.sleep(0.2)

        except Exception as e:
            safe_print(f"[手动跟踪] 线程异常: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)




def auto_track_loop():
    """自动选择最近目标并维护状态，不直接发送光电命令"""
    global auto_track_config
    global auto_track_state, auto_track_lock

    # 在函数开头初始化这个变量！
    last_lost_notification = 0  # 防止重复通知

    while True:
        try:
            tracker = optical_service.tracker
            if not auto_track_config['enabled']:
                time.sleep(0.5)
                continue

            now = time.time()

            with auto_track_lock:
                current_track_id = auto_track_state['current_track_id']
                lock_start_time = auto_track_state['lock_start_time']
                last_seen_time = auto_track_state['last_seen_time']
                manual_locked = auto_track_state.get('manual_locked', False)

            # 先读取最后一帧所有目标
            tracks = get_all_tracks_from_log()
            if not tracks:
                # 如果当前有锁定目标，但已经长时间没有看到任何目标
                if current_track_id is not None and (now - last_seen_time > auto_track_config['lost_timeout']):
                    if now - last_lost_notification > 2:  # 避免频繁打印
                        # safe_print(f"[自动跟踪] 当前目标 {current_track_id} 丢失超过 {auto_track_config['lost_timeout']}s，释放光电")
                        # 释放光电并重置焦距
                        if tracker and tracker.connected:
                            tracker.release_target()
                            tracker.reset_zoom(38)
                        last_lost_notification = now
                    
                    with auto_track_lock:
                        auto_track_state['current_track_id'] = None
                        auto_track_state['current_target'] = None
                        auto_track_state['manual_locked'] = False
                time.sleep(0.2)
                continue

            # 重置丢失通知计时器（有目标时）
            last_lost_notification = 0

            # =========================
            # 手动锁定模式：只维护该目标，不自动切换
            # =========================
            if manual_locked:
                current_target = None
                for t in tracks:
                    if t['track_id'] == current_track_id:
                        current_target = t
                        break

                if current_target is not None:
                    with auto_track_lock:
                        auto_track_state['last_seen_time'] = now
                        auto_track_state['current_target'] = current_target
                else:
                    if now - last_seen_time > auto_track_config['lost_timeout']:
                        safe_print(f"[自动跟踪] 手动指定目标 {current_track_id} 丢失")
                        with auto_track_lock:
                            auto_track_state['current_track_id'] = None
                            auto_track_state['current_target'] = None
                            auto_track_state['manual_locked'] = False

                time.sleep(0.2)
                continue

            # =========================
            # 自动模式：选择最近目标
            # =========================
            if not tracks:
                time.sleep(0.2)
                continue
                
            nearest = pick_preferred_track(tracks)

            # 1. 当前没有锁定目标 -> 直接锁定最近目标
            if current_track_id is None:
                with auto_track_lock:
                    auto_track_state['current_track_id'] = nearest['track_id']
                    auto_track_state['lock_start_time'] = now
                    auto_track_state['last_seen_time'] = now
                    auto_track_state['current_target'] = nearest
                    auto_track_state['manual_locked'] = False

                # safe_print(f"[自动跟踪] 已锁定最近目标: {nearest['track_id']} | 距离={nearest['range']:.1f}m")
                time.sleep(0.2)
                continue

            # 2. 当前有锁定目标，检查它是否还在
            current_target = None
            for t in tracks:
                if t['track_id'] == current_track_id:
                    current_target = t
                    break

            if current_target is not None:
                with auto_track_lock:
                    auto_track_state['last_seen_time'] = now
                    auto_track_state['current_target'] = current_target
            else:
                # 当前目标这一帧没出现
                if now - last_seen_time > auto_track_config['lost_timeout']:
                    safe_print(f"[自动跟踪] 当前目标 {current_track_id} 丢失超过 {auto_track_config['lost_timeout']}s，重选最近目标")
                    tracker.release_target()
                    tracker.reset_zoom(125)
                    with auto_track_lock:
                        auto_track_state['current_track_id'] = None
                        auto_track_state['current_target'] = None
                    time.sleep(0.2)
                    continue

            # 3. 没到保持时间，不允许切换
            if now - lock_start_time < auto_track_config['hold_seconds']:
                time.sleep(0.2)
                continue

            # 4. 到了保持时间
            if now - lock_start_time >= auto_track_config['hold_seconds']:
                # 找出第二近的目标（不是当前目标）
                next_target = pick_preferred_track(tracks, exclude_track_id=current_track_id)
                
                if next_target is not None:
                    # 切换到第二近的目标
                    should_switch = True
                    nearest = next_target
                    safe_print(f"[自动跟踪] 保持时间已到，强制切换到: {nearest['track_id']}")
                else:
                    # 没有其他目标，释放当前目标
                    safe_print(f"[自动跟踪] 保持时间已到，无其他目标，释放 {current_track_id}")
                    if tracker and tracker.connected:
                        tracker.release_target()
                    with auto_track_lock:
                        auto_track_state['current_track_id'] = None
                        auto_track_state['current_target'] = None
                    continue

            if should_switch:
                with auto_track_lock:
                    auto_track_state['current_track_id'] = nearest['track_id']
                    auto_track_state['lock_start_time'] = now
                    auto_track_state['last_seen_time'] = now
                    auto_track_state['current_target'] = nearest
                    auto_track_state['manual_locked'] = False

                safe_print(f"[自动跟踪] 切换到最近目标: {nearest['track_id']} | 距离={nearest['range']:.1f}m")
            else:
                # 当前目标继续保留，重新开始一个保持周期
                with auto_track_lock:
                    auto_track_state['lock_start_time'] = now

            time.sleep(0.2)

        except Exception as e:
            safe_print(f"[自动跟踪] 线程异常: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)



# ==================== 主程序 ====================

def _stop_process(process, name, timeout=5):
    if process is None:
        return

    if not process.is_alive():
        process.join(timeout=0.1)
        return

    safe_print(f"[退出] 等待{name}退出...")
    process.join(timeout=timeout)
    if process.is_alive():
        safe_print(f"[退出] {name}未及时退出，强制结束")
        process.terminate()
        process.join(timeout=2)


def shutdown_processes(
    process_receive,
    process_mht,
    data_queue,
    point_debug_process=None,
    point_debug_queue=None,
    clear_data=False,
):
    safe_print("[退出] 正在关闭光电连接...")
    optical_service.close_tracker()

    safe_print("[退出] 正在停止雷达接收进程...")
    _stop_process(process_receive, "雷达接收进程", timeout=3)

    safe_print("[退出] 正在停止MHT进程...")
    try:
        data_queue.put(None)
    except Exception:
        pass
    _stop_process(process_mht, "MHT进程", timeout=8)

    if point_debug_process is not None:
        safe_print("[退出] 正在停止点迹调试MHT进程...")
        try:
            if point_debug_queue is not None:
                point_debug_queue.put(None)
        except Exception:
            pass
        _stop_process(point_debug_process, "点迹调试MHT进程", timeout=8)

    if DEBUG_POINT_MHT:
        try:
            summary, _matches = run_point_debug_compare(
                RAW_TRACKS_FILE,
                POINT_TRACK_RESULTS_FILE,
                POINT_VS_RAW_COMPARE_FILE,
                time_window=0.5,
            )
            safe_print("[点迹调试] 自动对比完成")
            print_point_debug_summary(summary)
            safe_print(f"[点迹调试] 对比明细已保存: {POINT_VS_RAW_COMPARE_FILE}")
        except Exception as exc:
            safe_print(f"[点迹调试] 自动对比失败: {exc}")

    if clear_data:
        deleted, failed = clear_data_dir()
        safe_print(f"[数据] 已清空 data 目录，删除 {deleted} 项")
        for path, exc in failed:
            safe_print(f"[数据] 删除失败: {path} ({exc})")


if __name__ == "__main__":
    print("[BOOT] entering main startup", flush=True)
    multiprocessing.freeze_support()
    managed_processes = []
    fanout_process = start_managed_udp_fanout()
    if fanout_process is not None:
        managed_processes.append(("UDP分发器", fanout_process))
    
    # 用于存储最新的航迹数据
    print("[BOOT] creating multiprocessing manager and queues", flush=True)
    try:
        manager = multiprocessing.Manager()
    except PermissionError as exc:
        print(
            "[BOOT][ERROR] multiprocessing.Manager 启动失败，Windows 拒绝创建进程通信管道。"
            "请尝试用管理员权限启动 VSCode，或在普通 cmd/PowerShell 中直接运行 python main2.py。",
            flush=True,
        )
        raise
    latest_tracks = manager.dict()
    track_lock = threading.Lock()
    auto_track_config = {
        'enabled': False,
        'hold_seconds': 10,
        'lost_timeout': 3,
        'switch_margin': 100.0,
    }

    auto_track_state = manager.dict({
        'current_track_id': None,
        'lock_start_time': 0.0,
        'last_seen_time': 0.0,
        'current_target': None,
        'manual_locked': False,
    })
    optical_state_shared = manager.dict({
        'latest_azimuth': None,
        'latest_pitch': None,
        'latest_range': None,
        'current_status': None,
        'latest_angle_host_time': None,
        'latest_status_host_time': None,
        'latest_target_host_time': None,
        'host_update_time': 0.0,
    })

    auto_track_lock = threading.Lock()

    
    calibration_queue = multiprocessing.Queue()
    point_debug_queue = multiprocessing.Queue() if DEBUG_POINT_MHT else None
    recording_state = manager.dict()
    recording_state['enabled'] = False

    # 启动数据处理进程
    Data2Process_Buffer = multiprocessing.Queue()
    
    process_receive = multiprocessing.Process(target=receive_radar_data, 
                                              args=(Data2Process_Buffer, point_debug_queue, recording_state))
    process_mht = multiprocessing.Process(
        target=mht_process_and_send,
        args=(
            Data2Process_Buffer,
            latest_tracks,
            calibration_queue,
            recording_state,
            auto_track_state,
            optical_state_shared,
        )
    )
    point_debug_process = None
    if DEBUG_POINT_MHT:
        point_debug_process = multiprocessing.Process(
            target=mht_process_and_send,
            args=(
                point_debug_queue,
                None,
                multiprocessing.Queue(),
                recording_state,
                None,
                None,
                POINT_TRACK_RESULTS_FILE,
                POINT_TRACK_LOG_FILE,
                "点迹调试MHT进程",
                False,
                False,
            ),
        )
    
    print("[BOOT] starting radar receiver and MHT processes", flush=True)
    process_receive.start()
    process_mht.start()
    if point_debug_process is not None:
        safe_print("[点迹调试] DEBUG_POINT_MHT=ON，启动点迹调试链路")
        point_debug_process.start()
    print("[BOOT] initializing optical tracker", flush=True)
    init_optical_tracker()
    print("[BOOT] optical initialization step finished", flush=True)

   # ========== 启动两个跟踪线程 ==========
    # 手动跟踪线程（始终运行，但只在 manual_locked=True 时工作）
    manual_thread = threading.Thread(target=manual_follow_loop, daemon=True)
    manual_thread.start()
    safe_print("[手动跟踪] 线程已启动")

    # ========== 启动光电监控线程 ==========
    optical_monitor_thread = threading.Thread(target=optical_data_monitor, daemon=True)
    optical_monitor_thread.start()
    calibration_radar_thread = threading.Thread(target=calibration_radar_log_sampler, daemon=True)
    calibration_radar_thread.start()
    safe_print("[cal] radar log sampler started")
    safe_print("[光电监控] 线程已启动，每0.2秒记录一次光电数据")

    cv_process = start_managed_cv_detection()
    if cv_process is not None:
        managed_processes.append(("光电视频识别窗口", cv_process))

    # 自动跟踪线程（由 auto_track_config['enabled'] 控制）
    auto_follow_thread = threading.Thread(target=auto_follow_loop, daemon=True)
    auto_follow_thread.start()
    safe_print(f"[自动跟踪] 线程已启动，当前状态: {'开启' if auto_track_config['enabled'] else '关闭'}")

    # follow_thread = threading.Thread(target=optical_follow_loop, daemon=True)
    # follow_thread.start()
    # safe_print("[光电跟随] 线程已启动")

    auto_track_thread = threading.Thread(target=auto_track_loop, daemon=True)
    auto_track_thread.start()
        
    safe_print(f"[自动跟踪] 已启动：最近目标自动指派，保持 {auto_track_config['hold_seconds']}s，丢失超时 {auto_track_config['lost_timeout']}s")

    safe_print("UDP雷达处理程序启动")
    safe_print("数据处理中，稍后可使用交互控制台...")
    
    # 等待一下让数据处理启动
    time.sleep(2)
    
    # 启动交互控制台（在主线程中运行）
    clear_data_on_exit = False
    try:
        interactive_console(
            tracker_getter=lambda: optical_service.tracker,
            auto_track_config=auto_track_config,
            auto_track_state=auto_track_state,
            auto_track_lock=auto_track_lock,
            calibration_queue=calibration_queue,
            get_current_track_motion=get_current_track_motion,
            recording_state=recording_state,
        )
    except ConsoleExit as exc:
        clear_data_on_exit = exc.clear_data
    finally:
        shutdown_processes(
            process_receive,
            process_mht,
            Data2Process_Buffer,
            point_debug_process=point_debug_process,
            point_debug_queue=point_debug_queue,
            clear_data=clear_data_on_exit,
        )
        stop_managed_processes(managed_processes)

