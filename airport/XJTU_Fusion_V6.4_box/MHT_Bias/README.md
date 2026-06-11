# MHT_Bias 工程说明

当前工程主目标：`雷达数据接收 -> 点迹/航迹处理 -> MHT 融合 -> 目标跟踪输出`。

标定功能保留，但现在建议把它视为辅助模块，不作为日常主入口。

## 建议阅读顺序

1. 在线主程序：`Src/main2.py`
2. 在线核心模块：`Src/core/`
3. 雷达相关离线工具：`Src/tools/`
4. 标定相关工具：`Src/tools/calibration/`
5. 仿真评估：`Src/tools/simulation/mht_requirement_study/`
6. 光电视频识别：`CV/`

## 目录分工

- `Src/main2.py`
  - 在线运行主入口。
  - 负责拉起雷达接收、MHT、光电服务、控制台命令、日志输出。

- `Src/core/`
  - 在线运行核心代码。
  - 目前最重要，日常改雷达处理逻辑优先看这里。

- `Src/tools/`
  - 离线工具集合。
  - 已按用途拆成 `calibration / network / point_mht / radar_debug / simulation / manual_control`。

- `Calibration/`
  - 历史标定相关资源或外部配套代码。
  - 保留，不作为当前雷达处理主线入口。

- `MHT/`
  - MHT 核心实现。
  - 若非必须，不建议频繁直接改内核。

- `Sensor_Config/`
  - 传感器配置与参数。

- `CV/`
  - 光电视频识别相关代码。
  - 属于辅助链路，不影响纯雷达主流程。

- `calibration_data/`
  - 标定样本、配对结果、报告等数据。

- `flight_data_runs/`
  - 运行过程归档结果，适合长期保留。

- `data/`
  - 在线临时输出目录。
  - 适合放运行期中间文件，不建议放最终成果。

## 当前推荐主线

如果你现在主要做雷达数据处理，优先只看下面几块：

1. `Src/main2.py`
2. `Src/core/radar_receiver.py`
3. `Src/core/radar_protocol.py`
4. `MHT/`
5. `Src/core/interactive_console.py`
6. `Src/core/track_log.py`

如果要联动光电，再补看：

1. `Src/core/opti.py`
2. `Src/core/optical_service.py`
3. `Src/core/true_position_estimator.py`

如果要做标定，再补看：

1. `Src/core/calibration.py`
2. `Src/core/calibration_commands.py`
3. `Src/tools/calibration/`

## 当前整理原则

- 保留标定功能
- 不打断现有在线主程序导入关系
- 主入口尽量收敛到 `Src/main2.py`
- 离线试验、历史脚本、重复副本尽量归档，不混在主线旁边

