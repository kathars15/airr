# -*- coding: utf-8 -*-

import time

from core.calibration import calibrator
from core.console_utils import safe_print

def start_calibration(track_id):
    """开始校准"""
    if calibrator.start_calibration(track_id):
        safe_print(f"[校准] 开始校准目标 {track_id}")
        safe_print(f"[校准] 提示: 确保目标稳定，雷达扫描周期4秒，建议采集10-20组数据")
        safe_print(f"[校准] 输入 'cal_stop' 停止校准并计算参数")
        return True
    return False

def stop_calibration():
    """停止校准"""
    return calibrator.stop_calibration()

def get_calibration_status():
    """获取校准状态"""
    status = calibrator.get_status()
    if status['mode']:
        safe_print(f"[校准] 进行中: 目标={status['target_id']}, 已采集={status['radar_samples']}组")
    else:
        safe_print(f"[校准] 未进行, 已有校准参数: {'是' if status['has_calibration'] else '否'}")
    return status

def show_calibration_result():
    """显示校准结果"""
    result = calibrator.calibration_result
    if result['sample_count'] > 0:
        safe_print("\n" + "=" * 50)
        safe_print("当前校准参数:")
        safe_print(f"  方位角偏移: {result['azimuth_offset']:.2f}°")
        safe_print(f"  俯仰角偏移: {result['pitch_offset']:.2f}°")
        safe_print(f"  样本数量: {result['sample_count']}")
        safe_print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))}")
        safe_print("=" * 50)
    else:
        safe_print("暂无校准参数")

def clear_calibration():
    """清除校准参数"""
    calibrator.calibration_result = {
        'azimuth_offset': 0.0,
        'pitch_offset': 0.0,
        'azimuth_scale': 1.0,
        'pitch_scale': 1.0,
        'timestamp': 0,
        'sample_count': 0
    }
    calibrator.save_calibration()
    safe_print("[校准] 已清除校准参数")

