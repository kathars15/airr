# Tools

这里放在线主程序之外的离线工具。

按当前用途，建议这样理解：

## 雷达主线相关

- `network/`
  - UDP 分发、联机检查、启动脚本。
  - 需要同时喂给 `main2` 和上位机软件时，优先看这里。

- `point_mht/`
  - 点迹、原始 TRACK、MHT 对比回放。
  - 用于分析雷达处理链路效果。

- `radar_debug/`
  - 雷达协议、UDP、量程模式等诊断工具。

- `simulation/`
  - 融合算法离线仿真。
  - 当前重点子目录：`simulation/mht_requirement_study/`。

## 标定相关

- `calibration/`
  - 离线标定回放、配对分析、参数求解、报告生成。

## 辅助与兼容

- `manual_control/`
  - 一些控制或辅助脚本。

- `legacy_debug/`
  - 历史一次性实验，保留参考，不建议作为当前入口。

- `cal_offset.py`
  - 被 `core/calibration.py` 直接使用，不能随意移动。

- `compare_point_tracks_vs_raw.py`
  - 被 `main2.py` 使用，不能随意移动。

- `replay_points_mht_compare.py`
  - 顶层兼容入口。

- `replay_calibration_session.py`
  - 顶层兼容入口。

## 当前建议

如果你现在主要做雷达数据处理，优先看：

1. `network/`
2. `point_mht/`
3. `radar_debug/`
4. `simulation/`

标定功能保留，但放第二优先级。
