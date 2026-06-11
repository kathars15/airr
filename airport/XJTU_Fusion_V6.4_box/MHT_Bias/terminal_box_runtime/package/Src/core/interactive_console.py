# -*- coding: utf-8 -*-

import os
import shutil
import time

import numpy as np

from core.app_config import (
    DATA_DIR, FAKE_DIS, FLIGHT_RUNS_DIR, TRUE_POSITION_LOG_DIR,
    TRUE_POSITION_MAX_RADAR_AGE_SEC,
)
from core.calibration import calibrator
from core.calibration_commands import (
    clear_calibration,
    get_calibration_status,
    show_calibration_result,
)
from core.console_utils import safe_print
from core.optical_service import close_tracker, send_to_optical
from core.track_log import (
    get_track_by_id_from_log,
    get_track_by_raw_display_id_from_log,
    safe_print_available_tracks_from_log,
)


class ConsoleExit(BaseException):
    def __init__(self, clear_data=False):
        super().__init__("interactive console requested exit")
        self.clear_data = clear_data


def clear_data_dir():
    """清空 Src/data 目录中的运行数据。"""
    deleted = 0
    failed = []

    if not os.path.isdir(DATA_DIR):
        return deleted, failed

    for entry in os.scandir(DATA_DIR):
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
            deleted += 1
        except OSError as exc:
            failed.append((entry.path, exc))

    return deleted, failed


def archive_current_run():
    """Copy current data files into one timestamped flight run directory."""
    if not os.path.isdir(DATA_DIR):
        return None, []

    timestamp_text = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = os.path.join(FLIGHT_RUNS_DIR, f"flight_run_{timestamp_text}")
    os.makedirs(run_dir, exist_ok=True)

    copied = []
    for entry in os.scandir(DATA_DIR):
        if not entry.is_file():
            continue
        target_path = os.path.join(run_dir, entry.name)
        shutil.copy2(entry.path, target_path)
        copied.append(target_path)

    return run_dir, copied


def exit_program(clear_data=False):
    raise ConsoleExit(clear_data=clear_data)


def _safe_filename_text(value):
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def start_true_position_recording(
    tracker,
    auto_track_state,
    auto_track_lock,
    recording_state,
    get_current_track_motion,
):
    if not tracker or not tracker.connected:
        safe_print("[真实位置记录] 光电未连接，不能开始记录")
        return False

    with tracker.lock:
        optical_status = tracker.current_status

    if optical_status != 2:
        safe_print(f"[真实位置记录] 光电尚未tracking，当前状态={optical_status}，请等稳定后再输入 j")
        return False

    with auto_track_lock:
        current_track_id = auto_track_state.get('current_track_id')

    if current_track_id is None:
        safe_print("[真实位置记录] 当前没有锁定/引导目标，请先 t <ID> 或等待自动锁定")
        return False

    motion = get_current_track_motion(current_track_id)
    if motion is None or not motion.get('valid', False):
        safe_print(f"[真实位置记录] 当前锁定目标没有有效雷达/MHT状态: {current_track_id}")
        return False

    try:
        pos_enu = motion.get('pos_enu')
        radar_range = float(np.linalg.norm(pos_enu))
        radar_age = time.time() - float(motion.get('last_update_time', 0.0))
    except Exception:
        safe_print(f"[真实位置记录] 雷达/MHT状态解析失败: {current_track_id}")
        return False

    if radar_range <= 0.0:
        safe_print(f"[真实位置记录] 雷达/MHT距离无效: range={radar_range:.2f}m")
        return False

    if radar_age > TRUE_POSITION_MAX_RADAR_AGE_SEC:
        safe_print(
            f"[真实位置记录] 雷达/MHT数据过旧: age={radar_age:.2f}s "
            f"> {TRUE_POSITION_MAX_RADAR_AGE_SEC:.2f}s"
        )
        return False

    if recording_state.get('true_position_active', False):
        safe_print(f"[真实位置记录] 已在记录中: {recording_state.get('true_position_csv')}")
        return True

    timestamp_text = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    target_text = _safe_filename_text(current_track_id)
    base_name = f"true_position_{timestamp_text}_{target_text}"
    os.makedirs(TRUE_POSITION_LOG_DIR, exist_ok=True)

    recording_state['true_position_active'] = True
    recording_state['true_position_started_at'] = time.time()
    recording_state['true_position_target'] = current_track_id
    recording_state['true_position_csv'] = os.path.join(TRUE_POSITION_LOG_DIR, f"{base_name}.csv")
    recording_state['true_position_jsonl'] = os.path.join(TRUE_POSITION_LOG_DIR, f"{base_name}.jsonl")
    recording_state['true_position_start_radar_range_m'] = radar_range
    recording_state['true_position_start_radar_age_sec'] = radar_age

    safe_print(f"[真实位置记录] 已开始: target={current_track_id}")
    safe_print(f"[真实位置记录] 雷达/MHT有效: range={radar_range:.1f}m, age={radar_age:.2f}s")
    safe_print(f"[真实位置记录] CSV: {recording_state['true_position_csv']}")
    safe_print(f"[真实位置记录] JSONL: {recording_state['true_position_jsonl']}")
    return True


def stop_true_position_recording(recording_state):
    if not recording_state.get('true_position_active', False):
        return False

    csv_path = recording_state.get('true_position_csv')
    jsonl_path = recording_state.get('true_position_jsonl')
    target = recording_state.get('true_position_target')
    started_at = recording_state.get('true_position_started_at')
    try:
        duration_text = f"{time.time() - float(started_at):.1f}s"
    except (TypeError, ValueError):
        duration_text = "unknown"

    recording_state['true_position_active'] = False
    safe_print(f"[??????] ???: target={target}, duration={duration_text}")
    if csv_path:
        safe_print(f"[??????] CSV???: {csv_path}")
    if jsonl_path:
        safe_print(f"[??????] JSONL???: {jsonl_path}")
    recording_state['true_position_target'] = None
    recording_state['true_position_csv'] = None
    recording_state['true_position_jsonl'] = None
    recording_state['true_position_started_at'] = None
    return True


def resolve_track_for_true_position_command(raw_target_text):
    raw_target_text = str(raw_target_text or "").strip()
    if not raw_target_text:
        return None

    if raw_target_text.lower().startswith("radar-"):
        return get_track_by_id_from_log(raw_target_text)

    track = get_track_by_raw_display_id_from_log(raw_target_text)
    if track is not None:
        return track

    if raw_target_text.isdigit():
        return get_track_by_id_from_log(f"Radar-{raw_target_text}")

    return None


def lock_target_for_follow(track_id, auto_track_state, auto_track_lock):
    import time

    target = get_track_by_id_from_log(track_id)
    if target is None:
        safe_print(f"[手动锁定] 目标不存在: {track_id}")
        return False

    now = time.time()
    with auto_track_lock:
        auto_track_state['current_track_id'] = track_id
        auto_track_state['lock_start_time'] = now
        auto_track_state['last_seen_time'] = now
        auto_track_state['current_target'] = target
        auto_track_state['manual_locked'] = True

    safe_print(f"[手动锁定] 已指定目标进入自动跟随流程: {track_id}")
    return True


def _show_optical_status(tracker):
    if tracker and tracker.connected:
        with tracker.lock:
            safe_print(f"[光电] 当前位置: 方位={tracker.latest_azimuth:.1f}°, 俯仰={tracker.latest_pitch:.1f}°")
            safe_print(f"[光电] 工作状态: {tracker.current_status} (0=空闲,1=搜索,2=跟踪)")
            safe_print(f"[光电] 目标数量: {len(tracker.latest_targets)}")
    else:
        safe_print("[光电] 未连接")


# ========== 新增：光电距离查询函数 ==========
def _show_optical_distance_only(tracker):
    """
    仅显示光电跟踪目标的距离，不涉及雷达数据
    """
    if not tracker or not tracker.connected:
        safe_print("[光电] 未连接")
        return

    with tracker.lock:
        optical_status = tracker.current_status
        optical_range = tracker.latest_range
        optical_az = tracker.latest_azimuth
        optical_pitch = tracker.latest_pitch
        latest_targets = tracker.latest_targets

    safe_print("\n" + "=" * 50)
    safe_print("[光电目标距离]")
    safe_print("-" * 50)

    # 显示工作状态
    status_text = {0: '空闲', 1: '搜索中', 2: '跟踪中'}.get(optical_status, '未知')
    safe_print(f"光电工作状态: {status_text}")

    if optical_status == 2:
        # 跟踪状态 - 显示距离信息
        if optical_range is not None and optical_range > 0:
            safe_print(f"\n  【目标距离】: {optical_range:.1f} 米")
            safe_print(f"  方位角: {optical_az:.1f}°")
            safe_print(f"  俯仰角: {optical_pitch:.1f}°")
        else:
            safe_print(f"  方位角: {optical_az:.1f}°")
            safe_print(f"  俯仰角: {optical_pitch:.1f}°")
            safe_print("\n  【目标距离】: 未知 (未收到测距数据)")
            safe_print("  可能原因:")
            safe_print("    1. 光电无激光测距功能")
            safe_print("    2. 激光测距未触发")
            safe_print("    3. 目标距离超出测距范围")

        # 显示详细目标信息
        if latest_targets:
            safe_print(f"\n[目标详细信息]")
            for i, t in enumerate(latest_targets[:3]):  # 最多显示3个
                target_dist = t.get('target_dist', 0)
                target_id = t.get('target_id', 'N/A')
                target_type = t.get('target_type', 'N/A')
                similarity = t.get('similarity', 0)
                safe_print(f"  目标{i+1}: ID={target_id}, 类型={target_type}, "
                          f"相似度={similarity}%, 距离={target_dist}m")

    elif optical_status == 1:
        safe_print("\n  【状态】: 光电正在搜索目标，尚未锁定")
        safe_print("  请等待光电进入跟踪状态后重试")
    else:
        safe_print("\n  【状态】: 光电空闲，没有跟踪目标")
        safe_print("  请先用 t <ID> 命令锁定目标")

    safe_print("=" * 50)

def _show_radar_optical_compare(tracker, auto_track_state, auto_track_lock, get_current_track_motion):
    if not tracker or not tracker.connected:
        safe_print("[光电] 未连接")
        return

    with auto_track_lock:
        current_track_id = auto_track_state.get('current_track_id')

    if current_track_id is None:
        safe_print("[提示] 当前没有跟踪目标，请先用 t 命令锁定目标")
        return

    motion = get_current_track_motion(current_track_id)
    with tracker.lock:
        optical_az = tracker.latest_azimuth
        optical_pitch = tracker.latest_pitch
        optical_range = tracker.latest_range
        optical_status = tracker.current_status

    # 添加状态说明
    status_text = {0: '空闲', 1: '搜索中', 2: '跟踪中'}.get(optical_status, '未知')
    safe_print(f"[光电] 工作状态: {status_text}")

    if optical_status != 2:
        safe_print("[提示] 光电未进入跟踪状态，请等待几秒后再试 p 命令")

    safe_print("\n" + "=" * 60)
    safe_print(f"目标: {current_track_id}")
    safe_print("-" * 60)

    radar_range = radar_az = radar_pitch = None
    if motion is not None and motion.get('valid', False):
        pos_enu = motion['pos_enu']
        vel_enu = motion['vel_enu']
        radar_range = np.linalg.norm(pos_enu)
        radar_az = np.degrees(np.arctan2(pos_enu[0, 0], pos_enu[1, 0]))
        if radar_az < 0:
            radar_az += 360
        radar_pitch = np.degrees(np.arcsin(pos_enu[2, 0] / radar_range)) if radar_range > 0 else 0

        safe_print("[雷达测量]")
        safe_print(f"  距离: {radar_range:.1f}m")
        safe_print(f"  方位角: {radar_az:.1f}°")
        safe_print(f"  俯仰角: {radar_pitch:.1f}°")
        safe_print(f"  速度: {np.linalg.norm(vel_enu):.1f}m/s")
    else:
        safe_print("[雷达测量] 无数据")

    safe_print("-" * 60)

    if optical_az is not None:
        safe_print("[光电测量]")
        safe_print(f"  距离: {optical_range if optical_range else '未知'}m")
        safe_print(f"  方位角: {optical_az:.1f}°")
        safe_print(f"  俯仰角: {optical_pitch:.1f}°")
        status_text = {0: '空闲', 1: '搜索', 2: '跟踪'}.get(optical_status, '未知')
        safe_print(f"  工作状态: {status_text}")

        if radar_range is not None and optical_range:
            range_diff = radar_range - optical_range
            az_diff = radar_az - optical_az
            if az_diff > 180:
                az_diff -= 360
            elif az_diff < -180:
                az_diff += 360
            pitch_diff = radar_pitch - optical_pitch

            safe_print("-" * 60)
            safe_print("[差值] (雷达 - 光电)")
            safe_print(f"  距离差: {range_diff:.1f}m")
            safe_print(f"  方位差: {az_diff:.1f}°")
            safe_print(f"  俯仰差: {pitch_diff:.1f}°")
    else:
        safe_print("[光电测量] 无数据")

    safe_print("=" * 60)


def interactive_console(
    tracker_getter,
    auto_track_config,
    auto_track_state,
    auto_track_lock,
    calibration_queue,
    get_current_track_motion,
    recording_state,
):
    safe_print("\n" + "=" * 60)
    safe_print("  l / list        - 列出当前所有航迹")
    safe_print("  t <ID>          - 手动跟踪指定航迹")
    safe_print("  a on/off        - 开启/关闭自动跟踪")
    safe_print("  auto            - 查看自动跟踪状态")
    safe_print("  r               - 释放当前目标")
    safe_print("  am              - 恢复自动跟踪")
    safe_print("\n校准命令:")
    safe_print("  cal <ID>        - 开始校准并自动跟随指定目标")
    safe_print("  done            - 停止校准并计算参数")
    safe_print("  cdone           - 基于已保存可用历史样本重新计算参数")
    safe_print("  cstat           - 查看校准状态")
    safe_print("  cres            - 显示校准参数")
    safe_print("  cclear          - 清除校准参数")
    safe_print("  ctest           - 测试校准效果")
    safe_print("\n  q / quit        - 退出程序并清空 data")
    safe_print("  qq              - 退出程序但保留 data")
    safe_print("=" * 60)

    while True:
        try:
            cmd_input = input("\n> ").strip()
            if not cmd_input:
                continue

            cmd_parts = cmd_input.split()
            cmd = cmd_parts[0].lower()

            if cmd == 'qq':
                safe_print("正在退出...（保留 data 文件）")
                exit_program(clear_data=False)

            if cmd in ['q', 'quit', 'exit']:
                safe_print("正在退出...")
                exit_program(clear_data=True)

            if cmd == 'n':
                deleted, failed = clear_data_dir()
                recording_state['enabled'] = True
                safe_print(f"[记录] 已开始新一组记录，清空 data {deleted} 项")
                for path, exc in failed:
                    safe_print(f"[数据] 删除失败: {path} ({exc})")
                continue

            if cmd.startswith('j'):
                raw_target_text = cmd_input[1:].strip()
                if raw_target_text:
                    target = resolve_track_for_true_position_command(raw_target_text)
                    if target is None:
                        safe_print(f"[??????] ?????ID/??ID????: {raw_target_text}")
                        continue

                    track_id = target.get('track_id')
                    ok = lock_target_for_follow(track_id, auto_track_state, auto_track_lock)
                    if not ok:
                        safe_print(f"[??????] ??????: {track_id}")
                        continue

                    if recording_state.get('true_position_active', False):
                        current_record_target = recording_state.get('true_position_target')
                        if str(current_record_target) != str(track_id):
                            stop_true_position_recording(recording_state)

                start_true_position_recording(
                    tracker_getter(),
                    auto_track_state,
                    auto_track_lock,
                    recording_state,
                    get_current_track_motion,
                )
                continue

            if cmd in ['l', 'list']:
                safe_print_available_tracks_from_log()

            # ========== 新增：光电距离查询命令 ==========
            elif cmd in ['gd', 'od', 'opt_dist', 'optical_distance']:
                tracker = tracker_getter()
                _show_optical_distance_only(tracker)

            elif cmd == 'auto':
                with auto_track_lock:
                    safe_print(f"[自动跟踪] 开关状态: {'开启' if auto_track_config['enabled'] else '关闭'}")
                    safe_print(f"[自动跟踪] 当前目标: {auto_track_state['current_track_id']}")
                    safe_print(f"[自动跟踪] 保持时间: {auto_track_config['hold_seconds']}s")
                    safe_print(f"[自动跟踪] 丢失超时: {auto_track_config['lost_timeout']}s")

            elif cmd in ['cal', 'cstt']:
                if len(cmd_parts) < 2:
                    safe_print("用法: cal <航迹ID> (例如: cal 3)")
                    continue

                track_id = cmd_parts[1]
                if track_id.isdigit():
                    track_id = f"Radar-{track_id}"

                if get_track_by_id_from_log(track_id) is None:
                    safe_print(f"[校准] 目标不存在: {track_id}")
                    continue

                if calibrator.calibration_mode:
                    result = calibrator.switch_calibration_target(track_id)
                    if result == 'switched':
                        calibration_queue.put({'type': 'start', 'target_id': track_id})
                        safe_print(f"[calibration] switched to {track_id}, keeping existing samples")
                    else:
                        safe_print(f"[calibration] target still {track_id}, continue current session")
                else:
                    calibrator.start_calibration(track_id)
                    calibration_queue.put({'type': 'start', 'target_id': track_id})
                    safe_print("[cal] sampling only; optical auto-search is not started. Use t <ID> separately if needed.")
                    safe_print(f"[calibration] started: {track_id}, keep target stable and optical tracking on")

            elif cmd in ['done', 'cstp']:
                calibration_queue.put({'type': 'stop'})
                ok = calibrator.stop_calibration()
                safe_print("[校准] 已停止并计算参数" if ok else "[校准] 停止失败，样本可能不足")

            elif cmd == 'cdone':
                ok = calibrator.recalculate_from_history()
                safe_print("[calibration] recomputed from saved usable history" if ok else "[calibration] recompute from history failed")

            elif cmd in ['cstat', 'css']:
                get_calibration_status()

            elif cmd in ['cres', 'cs']:
                show_calibration_result()

            elif cmd in ['cclear', 'cc']:
                clear_calibration()

            elif cmd in ['ctest', 'ct']:
                try:
                    az = float(input("  雷达方位角(度): "))
                    pitch = float(input("  雷达俯仰角(度): "))
                    cal_az, cal_pitch = calibrator.apply_calibration(az, pitch)[:2]
                    safe_print(f"  原始: ({az:.1f}°, {pitch:.1f}°)")
                    safe_print(f"  校准后: ({cal_az:.1f}°, {cal_pitch:.1f}°)")
                    safe_print(f"  偏移: 方位={cal_az - az:.2f}°, 俯仰={cal_pitch - pitch:.2f}°")
                except ValueError:
                    safe_print("输入无效")

            elif cmd == 'y':
                recording_state['enabled'] = False
                time.sleep(0.3)
                run_dir, copied = archive_current_run()
                if run_dir is None:
                    safe_print("[????] data ?????")
                else:
                    safe_print(f"[????] ??? {len(copied)} ????: {run_dir}")
                    deleted, failed = clear_data_dir()
                    safe_print(f"[??] ??? data ????? {deleted} ?")
                    for path, exc in failed:
                        safe_print(f"[??] ????: {path} ({exc})")
                recording_state['enabled'] = True
                continue

            elif cmd == 'a':
                if len(cmd_parts) < 2:
                    safe_print("用法: a on / a off")
                    continue

                sub_cmd = cmd_parts[1].lower()
                if sub_cmd == 'on':
                    auto_track_config['enabled'] = True
                    safe_print("[自动跟踪] 已开启")
                elif sub_cmd == 'off':
                    auto_track_config['enabled'] = False
                    safe_print("[自动跟踪] 已关闭")
                else:
                    safe_print("用法: a on / a off")

            elif cmd == 'cp':
                result = calibrator.position_offset
                if result['sample_count'] > 0:
                    safe_print("\n位置偏移校准参数:")
                    safe_print(f"  光电相对雷达: 东偏移={result['dx']:.2f}m")
                    safe_print(f"               北偏移={result['dy']:.2f}m")
                    safe_print(f"               高偏移={result['dz']:.2f}m")
                    safe_print(f"  样本数量: {result['sample_count']}")
                    safe_print(f"  是否使用: {'是' if result.get('use_position', False) else '否'}")
                    safe_print(f"  算法: {result.get('method', 'unknown')}")
                    if 'mean_error_deg' in result:
                        safe_print(
                            f"  重投影误差: 均值={result['mean_error_deg']:.3f}°, "
                            f"最大={result.get('max_error_deg', 0.0):.3f}°"
                        )
                else:
                    safe_print("暂无位置偏移校准参数")

            elif cmd == 'am':
                with auto_track_lock:
                    auto_track_state['manual_locked'] = False
                    auto_track_state['current_track_id'] = None
                    auto_track_state['current_target'] = None
                safe_print("[自动跟踪] 已退出手动锁定，恢复自动模式")

            elif cmd == 'r':
                stop_true_position_recording(recording_state)
                tracker = tracker_getter()
                if tracker:
                    tracker.release_target()
                    tracker.reset_zoom(125)

                with auto_track_lock:
                    auto_track_state['current_track_id'] = None
                    auto_track_state['current_target'] = None
                    auto_track_state['lock_start_time'] = 0.0
                    auto_track_state['last_seen_time'] = 0.0
                    auto_track_state['manual_locked'] = False

                safe_print("[自动跟踪] 已清除当前跟踪目标")

            elif cmd == 'f':
                try:
                    tracker = tracker_getter()
                    zoom_value = int(cmd_parts[1]) if len(cmd_parts) >= 2 else int(input("请输入 reset_zoom 参数: ").strip())
                    tracker.reset_zoom(zoom_value)
                    safe_print(f"[光电] 已执行 reset_zoom({zoom_value})")
                except ValueError:
                    safe_print("[错误] 参数必须是数字")
                except Exception as e:
                    safe_print(f"[错误] reset_zoom 执行失败: {e}")

            elif cmd == 'status':
                _show_optical_status(tracker_getter())

            elif cmd == 'p':
                _show_radar_optical_compare(
                    tracker_getter(),
                    auto_track_state,
                    auto_track_lock,
                    get_current_track_motion,
                )

            elif cmd == 't':
                if len(cmd_parts) < 2:
                    safe_print("用法: t <航迹ID> (例如: t Radar-1 或 t 1)")
                    continue

                track_id = cmd_parts[1]
                if track_id.isdigit():
                    track_id = f"Radar-{track_id}"

                ok = lock_target_for_follow(track_id, auto_track_state, auto_track_lock)
                if not ok:
                    safe_print(f"航迹不存在: {track_id}")
                    safe_print("可用命令: l 查看所有航迹")
                else:
                    target = get_track_by_id_from_log(track_id)
                    if target is not None:
                        send_to_optical(
                            track_id,
                            target.get('azimuth', 0),
                            target.get('pitch', 0),
                            target.get('range', 0) - FAKE_DIS,
                        )

            else:
                safe_print(f"未知命令: {cmd}")
                safe_print("可用命令: l, t <ID>, q")

        except KeyboardInterrupt:
            safe_print("\n正在退出...")
            exit_program(clear_data=False)
        except EOFError:
            safe_print("[控制台] 标准输入已关闭，交互控制台退出。")
            exit_program(clear_data=False)
        except Exception as e:
            safe_print(f"错误: {e}")
