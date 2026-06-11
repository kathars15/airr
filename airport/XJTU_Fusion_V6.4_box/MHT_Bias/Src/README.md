# Src 目录说明

`Src/` 是当前在线系统主工作区。

## 主入口

- `main2.py`
  - 在线主程序入口。

## 核心目录

- `core/`
  - 在线运行核心模块。

- `tools/`
  - 离线分析、诊断、仿真、网络辅助脚本。

- `data/`
  - 在线临时输出。

- `flight_data_runs/`
  - 运行归档结果。

- `calibration_data/`
  - 标定数据与报告。

## 现在做雷达处理时看哪里

1. `main2.py`
2. `core/radar_receiver.py`
3. `core/radar_protocol.py`
4. `../MHT/`
5. `core/interactive_console.py`

## 现在做标定时看哪里

1. `core/calibration.py`
2. `core/calibration_commands.py`
3. `tools/calibration/`

