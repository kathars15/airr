# -*- coding: utf-8 -*-
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy

import torch

from core.app_config import (
    CV_DETECTION_RESULTS_FILE,
    CV_DETECTION_SCRIPT,
    ENABLE_LOCAL_CV_WHEN_CPU_ONLY,
    ENABLE_LOCAL_CV_WHEN_GPU_AVAILABLE,
    ENABLE_MANAGED_UDP_FANOUT,
    ENABLE_TERMINAL_BOX_MODE,
    HOST_IP,
    MHT_BIAS_PATH,
    OPTICAL_IP,
    OPTICAL_STATUS_FILE,
    TERMINAL_PRINT_INTERVAL_SEC,
    TERMINAL_RESULT_LOG_DIR,
    TERMINAL_TARGET_MATCH_MAX_AZ_DIFF_DEG,
    TERMINAL_TARGET_MATCH_MAX_RADAR_AGE_SEC,
)
from core.console_utils import safe_print
from core.optical_service import init_optical_tracker
from core.radar_receiver import receive_radar_data
from main2 import _popen_creationflags, mht_process_and_send, write_optical_status_file
from core import optical_service


def start_managed_process(name, command, cwd=None, env=None, settle_sec=0.8):
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            creationflags=_popen_creationflags(),
        )
    except Exception as exc:
        safe_print(f"[BOX][WARN] {name} 启动失败: {exc}")
        return None
    time.sleep(settle_sec)
    if process.poll() is not None:
        safe_print(f"[BOX][WARN] {name} 已退出，退出码: {process.returncode}")
        return None
    safe_print(f"[BOX] {name} 已启动，PID={process.pid}")
    return process


def stop_managed_processes(processes):
    for name, process in reversed(processes):
        if process is None or process.poll() is not None:
            continue
        safe_print(f"[BOX] 正在关闭{name} PID={process.pid}")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            safe_print(f"[BOX][WARN] {name} 未及时退出，强制结束")
            process.kill()
            process.wait(timeout=3)
        except Exception as exc:
            safe_print(f"[BOX][WARN] 关闭{name}失败: {exc}")


def stop_process(process, name, timeout=5):
    if process is None:
        return
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    safe_print(f"[BOX] 等待{name}退出...")
    process.join(timeout=timeout)
    if process.is_alive():
        safe_print(f"[BOX][WARN] {name}未及时退出，强制结束")
        process.terminate()
        process.join(timeout=2)


def start_managed_udp_fanout():
    if not ENABLE_MANAGED_UDP_FANOUT:
        return None
    script = os.path.join(MHT_BIAS_PATH, "Src", "tools", "network", "udp_fanout.py")
    if not os.path.exists(script):
        safe_print(f"[BOX][WARN] UDP分发器脚本不存在: {script}")
        return None
    return start_managed_process(
        "UDP分发器",
        [sys.executable, script],
        cwd=os.path.dirname(script),
    )


def write_cv_state(payload):
    tmp_file = f"{CV_DETECTION_RESULTS_FILE}.{os.getpid()}.tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_file, CV_DETECTION_RESULTS_FILE)
    except Exception as exc:
        safe_print(f"[BOX][WARN] 写入CV状态失败: {exc}")
    finally:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except OSError:
            pass


def start_box_cv_detection():
    has_gpu = torch.cuda.is_available()
    if has_gpu and not ENABLE_LOCAL_CV_WHEN_GPU_AVAILABLE:
        write_cv_state({
            "active": False,
            "reason": "gpu_cv_disabled_by_config",
            "scope": None,
            "has_detection": False,
            "detections": [],
            "best_detection": None,
            "timestamp": time.time(),
        })
        safe_print("[融合降级] reason=gpu_cv_disabled_by_config")
        return None
    if (not has_gpu) and (not ENABLE_LOCAL_CV_WHEN_CPU_ONLY):
        write_cv_state({
            "active": False,
            "reason": "disabled_no_gpu",
            "scope": None,
            "has_detection": False,
            "detections": [],
            "best_detection": None,
            "timestamp": time.time(),
        })
        safe_print("[融合降级] reason=disabled_no_gpu")
        return None

    script = CV_DETECTION_SCRIPT
    if not script or not os.path.exists(script):
        safe_print(f"[BOX][WARN] 视频识别脚本不存在: {script}")
        return None

    env = os.environ.copy()
    env["AIRR_CV_DETECT_REQUIRE_RECORDING"] = "0"
    env["AIRR_CV_SHOW_WINDOW"] = "0"
    env.setdefault("PYTHONPATH", os.path.dirname(script))
    return start_managed_process(
        "终端盒子CV识别",
        [sys.executable, script],
        cwd=os.path.dirname(script),
        env=env,
        settle_sec=1.2,
    )


def angle_delta_deg(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def match_optical_tracking_target(tracks_snapshot, optical_state):
    current_status = optical_state.get("current_status")
    optical_az = optical_state.get("latest_azimuth")
    angle_time = optical_state.get("latest_angle_host_time")
    now = time.time()

    if current_status != 2:
        return None, "optical_not_tracking", None
    if optical_az is None or angle_time is None:
        return None, "missing_optical_angle", None
    try:
        optical_az = float(optical_az)
        angle_age = max(0.0, now - float(angle_time))
    except (TypeError, ValueError):
        return None, "invalid_optical_angle", None

    best_track_id = None
    best_score = None
    for track_id, info in tracks_snapshot.items():
        if not isinstance(info, dict) or not info.get("valid", False):
            continue
        try:
            last_update_time = float(info.get("last_update_time", 0.0))
        except (TypeError, ValueError):
            continue
        radar_age = now - last_update_time
        if radar_age > TERMINAL_TARGET_MATCH_MAX_RADAR_AGE_SEC:
            continue
        try:
            pos = info.get("pos_enu") or [0.0, 0.0, 0.0]
            east = float(pos[0])
            north = float(pos[1])
            azimuth = (float((180.0 / 3.141592653589793) * __import__("math").atan2(east, north)) + 360.0) % 360.0
        except Exception:
            continue
        score = angle_delta_deg(azimuth, optical_az)
        if best_score is None or score < best_score:
            best_score = score
            best_track_id = track_id

    if best_track_id is None:
        return None, "radar_target_not_found", angle_age
    if best_score is None or best_score > TERMINAL_TARGET_MATCH_MAX_AZ_DIFF_DEG:
        return None, "radar_target_not_matched", angle_age
    return best_track_id, None, angle_age


def optical_data_monitor(recording_state, optical_state_shared, stop_event):
    last_status_write_time = 0.0
    while not stop_event.is_set():
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
                optical_state_shared["latest_azimuth"] = current_az
                optical_state_shared["latest_pitch"] = current_pitch
                optical_state_shared["latest_range"] = current_range
                optical_state_shared["current_status"] = current_status
                optical_state_shared["latest_angle_host_time"] = latest_angle_host_time
                optical_state_shared["latest_status_host_time"] = latest_status_host_time
                optical_state_shared["latest_target_host_time"] = latest_target_host_time
                optical_state_shared["host_update_time"] = now

                if now - last_status_write_time >= 0.2:
                    status_payload = {
                        "timestamp": now,
                        "latest_azimuth": current_az,
                        "latest_pitch": current_pitch,
                        "latest_range": current_range,
                        "current_status": current_status,
                        "latest_angle_host_time": latest_angle_host_time,
                        "latest_status_host_time": latest_status_host_time,
                        "latest_target_host_time": latest_target_host_time,
                        "true_position_recording_active": True,
                        "true_position_recording_target": recording_state.get("true_position_target"),
                        "target_count": len(latest_targets),
                        "latest_targets": [
                            {
                                "target_id": t.get("target_id"),
                                "target_type": t.get("target_type"),
                                "similarity": t.get("similarity"),
                                "width": t.get("width"),
                                "height": t.get("height"),
                                "pos_x": t.get("pos_x"),
                                "pos_y": t.get("pos_y"),
                                "target_az": t.get("target_az"),
                                "target_pitch": t.get("target_pitch"),
                                "target_dist": t.get("target_dist"),
                                "packet_timestamp": t.get("packet_timestamp"),
                            }
                            for t in latest_targets[:5]
                        ],
                    }
                    if write_optical_status_file(status_payload):
                        last_status_write_time = now
            time.sleep(0.2)
        except Exception as exc:
            safe_print(f"[BOX][光电状态] {exc}")
            time.sleep(1.0)


def target_selector_loop(latest_tracks, optical_state_shared, auto_track_state, stop_event):
    last_reason = None
    last_selected = None
    last_print_time = 0.0
    tracking_frames = 0
    while not stop_event.is_set():
        try:
            tracks_snapshot = {key: deepcopy(value) for key, value in latest_tracks.items()}
            optical_snapshot = dict(optical_state_shared)
            selected_track_id, reason, angle_age = match_optical_tracking_target(tracks_snapshot, optical_snapshot)
            if selected_track_id != last_selected:
                tracking_frames = 0
                last_selected = selected_track_id
            if selected_track_id is not None:
                tracking_frames += 1
                auto_track_state["current_track_id"] = selected_track_id
                auto_track_state["current_target"] = tracks_snapshot.get(selected_track_id)
                auto_track_state["last_seen_time"] = time.time()
                auto_track_state["match_reason"] = None
                auto_track_state["optical_angle_age_sec"] = angle_age
                auto_track_state["tracking_frames"] = tracking_frames
                if tracking_frames < 3:
                    reason = "tracking_not_confirmed"
            else:
                auto_track_state["current_track_id"] = None
                auto_track_state["current_target"] = None
                auto_track_state["match_reason"] = reason
                auto_track_state["optical_angle_age_sec"] = angle_age
                auto_track_state["tracking_frames"] = 0

            now = time.time()
            if reason and (reason != last_reason or now - last_print_time >= TERMINAL_PRINT_INTERVAL_SEC):
                safe_print(f"[融合等待] reason={reason}")
                last_reason = reason
                last_print_time = now
            elif selected_track_id and tracking_frames >= 3:
                last_reason = None
            time.sleep(0.2)
        except Exception as exc:
            safe_print(f"[BOX][目标匹配] {exc}")
            time.sleep(1.0)


def main():
    safe_print("[BOX] terminal box mode startup")
    multiprocessing.freeze_support()
    manager = multiprocessing.Manager()

    latest_tracks = manager.dict()
    optical_state_shared = manager.dict({
        "latest_azimuth": None,
        "latest_pitch": None,
        "latest_range": None,
        "current_status": None,
        "latest_angle_host_time": None,
        "latest_status_host_time": None,
        "latest_target_host_time": None,
        "host_update_time": 0.0,
    })
    auto_track_state = manager.dict({
        "current_track_id": None,
        "current_target": None,
        "last_seen_time": 0.0,
        "match_reason": None,
        "optical_angle_age_sec": None,
        "tracking_frames": 0,
    })
    recording_state = manager.dict()
    timestamp_text = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    recording_state["enabled"] = True
    recording_state["true_position_active"] = True
    recording_state["true_position_started_at"] = time.time()
    recording_state["true_position_target"] = "auto_matched"
    recording_state["true_position_csv"] = os.path.join(TERMINAL_RESULT_LOG_DIR, f"fusion_output_{timestamp_text}.csv")
    recording_state["true_position_jsonl"] = os.path.join(TERMINAL_RESULT_LOG_DIR, f"fusion_output_{timestamp_text}.jsonl")

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    managed_processes = []
    fanout_process = start_managed_udp_fanout()
    if fanout_process is not None:
        managed_processes.append(("UDP分发器", fanout_process))

    cv_process = start_box_cv_detection()
    if cv_process is not None:
        managed_processes.append(("终端盒子CV识别", cv_process))

    data_queue = multiprocessing.Queue()
    process_receive = multiprocessing.Process(target=receive_radar_data, args=(data_queue, None, recording_state))
    process_mht = multiprocessing.Process(
        target=mht_process_and_send,
        args=(
            data_queue,
            latest_tracks,
            multiprocessing.Queue(),
            recording_state,
            auto_track_state,
            optical_state_shared,
            None,
            None,
            "终端盒子MHT进程",
            True,
            False,
        ),
    )
    process_receive.start()
    process_mht.start()

    init_optical_tracker()

    optical_thread = threading.Thread(target=optical_data_monitor, args=(recording_state, optical_state_shared, stop_event), daemon=True)
    optical_thread.start()
    selector_thread = threading.Thread(target=target_selector_loop, args=(latest_tracks, optical_state_shared, auto_track_state, stop_event), daemon=True)
    selector_thread.start()

    safe_print(f"[BOX] optical_ip={OPTICAL_IP}")
    safe_print(f"[BOX] 日志CSV: {recording_state['true_position_csv']}")
    safe_print(f"[BOX] 日志JSONL: {recording_state['true_position_jsonl']}")
    safe_print("[BOX] 运行中，等待光电跟踪与融合输出...")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        safe_print("[BOX] shutting down...")
        optical_service.close_tracker()
        try:
            data_queue.put(None)
        except Exception:
            pass
        stop_process(process_receive, "雷达接收进程", timeout=3)
        stop_process(process_mht, "终端盒子MHT进程", timeout=8)
        stop_managed_processes(managed_processes)


if __name__ == "__main__":
    if not ENABLE_TERMINAL_BOX_MODE:
        safe_print("[BOX][WARN] ENABLE_TERMINAL_BOX_MODE=False，仍继续按终端盒子入口运行。")
    main()
