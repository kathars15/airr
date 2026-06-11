# Core 在线核心模块

这里是 `Src/main2.py` 直接依赖的在线核心代码。

## 雷达主线优先看

- `app_config.py`
  - 统一配置、路径、开关、端口。

- `radar_protocol.py`
  - 雷达协议解析、控制报文构造。

- `radar_receiver.py`
  - 雷达 UDP 接收、POINT/TRACK 处理、解析诊断。

- `interactive_console.py`
  - 控制台命令、跟踪控制、运行交互。

- `track_log.py`
  - 航迹日志查询与输出。

- `track_smoothing.py`
  - 小型平滑辅助。

## 标定相关

- `calibration.py`
  - 在线标定状态、样本收集、参数保存。

- `calibration_commands.py`
  - 控制台标定命令。

## 光电相关

- `opti.py`
  - 光电设备接入与状态缓存。

- `optical_service.py`
  - 光电辅助服务。

- `optical_measurement_log.py`
  - 光电测量日志。

- `true_position_estimator.py`
  - 光电方向 + 雷达距离 的目标位置估计。

## 通用辅助

- `console_utils.py`
- `time_utils.py`

## 整理原则

这里先不做大搬家。
原因：`main2.py`、多进程、联机逻辑都直接引用这些模块，保持导入路径稳定更重要。
