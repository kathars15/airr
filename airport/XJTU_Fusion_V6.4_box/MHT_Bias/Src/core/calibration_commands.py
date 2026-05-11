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
    position = calibrator.position_offset
    if result['sample_count'] > 0:
        safe_print("\n" + "=" * 50)
        safe_print("当前校准参数:")
        safe_print(f"  方位角偏移: {result['azimuth_offset']:.2f}°")
        safe_print(f"  俯仰角偏移: {result['pitch_offset']:.2f}°")
        safe_print(f"  样本数量: {result['sample_count']}")
        if result.get('method'):
            safe_print(f"  角度算法: {result['method']}")
        safe_print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))}")
        if position.get('sample_count', 0) > 0:
            safe_print("")
            safe_print("位置偏移参数:")
            safe_print(f"  dx(东向): {position['dx']:.2f}m")
            safe_print(f"  dy(北向): {position['dy']:.2f}m")
            safe_print(f"  dz(天向): {position['dz']:.2f}m")
            safe_print(f"  是否启用: {'是' if position.get('use_position', False) else '否'}")
            safe_print(f"  位置算法: {position.get('method', 'unknown')}")
            if 'mean_error_deg' in position:
                safe_print(
                    f"  重投影误差: 均值={position['mean_error_deg']:.3f}°, "
                    f"最大={position.get('max_error_deg', 0.0):.3f}°"
                )
        safe_print("=" * 50)
    else:
        safe_print("暂无校准参数")

def clear_calibration():
    """清除校准参数"""
    calibrator.clear_calibration()

