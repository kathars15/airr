# -*- coding: utf-8 -*-

import time

from core.app_config import (
    OPTICAL_AI_TEMPLATE,
    OPTICAL_IP,
    OPTICAL_LOCAL_IP,
    OPTICAL_PORT,
)
from core.calibration import calibrator
from core.console_utils import safe_print
from core.opti import OpticalTracker

tracker = None


def init_optical_tracker():
    global tracker

    if tracker is None:
        safe_print("[光电] 初始化连接...")
        tracker = OpticalTracker(
            device_ip=OPTICAL_IP,
            local_ip=OPTICAL_LOCAL_IP,
            port=OPTICAL_PORT,
        )
        safe_print("[光电] 绑定本地 UDP...")
        if tracker.connect():
            safe_print("[光电] UDP 连接已建立，启动录像线程...")
            tracker.init_recorder(save_dir=r"D:\\video_new", show_window=False)
            safe_print("[光电] 设置目标上报地址...")
            tracker.set_report_destination(OPTICAL_LOCAL_IP, OPTICAL_PORT)
            safe_print(f"[光电] 切换 AI 模板: {OPTICAL_AI_TEMPLATE}")
            tracker.set_ai_template(OPTICAL_AI_TEMPLATE)
            time.sleep(0.3)
            safe_print("[光电] 启动状态监控线程...")
            tracker.start_monitor()
            safe_print("[光电] 初始化完成，保持连接")
            return True

        safe_print("[光电] 初始化失败")
        return False

    return True

def close_tracker():
    global tracker
    if tracker:
        tracker.close()


def send_to_optical(track_id, azimuth, pitch, distance):
    global tracker

    calibrated_az, calibrated_pitch, calibrated_range = calibrator.apply_calibration(
        azimuth, pitch, distance
    )

    print(f"\n[目标] 航迹ID: {track_id}")
    print(f"       原始方位角: {azimuth:.1f}° -> 校准后: {calibrated_az:.1f}°")
    print(f"       原始俯仰角: {pitch:.1f}° -> 校准后: {calibrated_pitch:.1f}°")
    if calibrated_range is not None:
        print(f"       原始距离: {distance:.0f}m -> 校准后距离: {calibrated_range:.0f}m")
    else:
        print(f"       距离: {distance:.0f}m")

    if tracker is None or not tracker.connected:
        print("[光电] 重新连接...")
        if not init_optical_tracker():
            return False

    try:
        if calibrated_range is not None:
            tracker.goto_and_search(calibrated_az, calibrated_pitch, calibrated_range)
        else:
            tracker.goto_and_search(calibrated_az, calibrated_pitch, distance)
        return True
    except Exception as e:
        print(f"[光电] 发送失败: {e}")
        return False
