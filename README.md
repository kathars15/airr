# AIRR 雷达-光电融合与标定系统

本仓库用于雷达、光电设备的联动跟踪、数据融合与现场标定。项目包含雷达 UDP 数据接收与协议解析、MHT 多目标跟踪、目标分类、光电转台控制、雷达-光电角度/位置标定、运行日志记录以及若干现场调试工具。

## 项目目录

```text
.
├── airport/XJTU_Fusion_V6.4_box/MHT_Bias/   主程序与算法代码
│   ├── Src/main2.py                         当前主入口程序
│   ├── Src/core/                            核心配置、协议、接收、校准与日志模块
│   ├── Src/tools/                           现场调试、探测、离线标定等工具脚本
│   ├── MHT/                                 多假设跟踪算法
│   ├── Classify/                            目标运动模型与分类模块
│   ├── common/                              坐标转换、滤波、聚类、绘图等公共函数
│   ├── Control/                             光电/控制相关脚本
│   ├── UI/                                  雷达显示与视频接收界面
│   └── 标定操作手册.md                       现场标定操作说明
├── calibration_data/                        标定结果或外部校准数据
└── docu/                                    雷达、光电等设备协议文档
```

## 主要功能

- 接收雷达转发的 UDP 数据，解析点迹、航迹和状态包。
- 使用 MHT 进行多目标跟踪融合，并输出目标列表与 JSON 结果。
- 根据目标航迹控制光电设备进行手动或自动跟踪。
- 支持雷达与光电的角度标定、位置偏移估计和离线标定。
- 记录雷达原始数据、光电测量数据、融合结果和运行日志，便于复盘。

## 运行环境

建议使用 Python 3.10 或更高版本。核心程序和工具脚本会用到以下常见依赖：

```powershell
pip install numpy scipy pandas scikit-learn cvxpy matplotlib pyproj rasterio opencv-python PyQt5
```

说明：

- `numpy`、`scipy`、`scikit-learn`、`cvxpy` 主要用于跟踪、聚类、优化和分类算法。
- `opencv-python` 用于光电视频相关处理。
- `PyQt5` 用于 `UI` 目录下的可视化界面。
- `pyproj`、`rasterio` 用于部分坐标转换和地理数据处理。

## 网络配置

主配置文件位于：

```text
airport/XJTU_Fusion_V6.4_box/MHT_Bias/Src/core/app_config.py
```

常用字段：

```python
RADAR_IP = "192.168.0.99"
HOST_IP = "127.0.0.1"
RADAR_PORT = 8080
HOST_PORT = 9000

OPTICAL_IP = "192.168.0.98"
OPTICAL_LOCAL_IP = "192.168.0.9"
OPTICAL_PORT = 9966
```

现场运行时通常需要根据设备实际网段修改 `RADAR_IP`、`HOST_IP`、`OPTICAL_IP` 和 `OPTICAL_LOCAL_IP`。如果雷达软件将数据转发到本机，建议确认转发 IP 与 `HOST_IP:HOST_PORT` 一致；需要监听所有本机 IPv4 地址时，可将 `HOST_IP` 设置为 `0.0.0.0`。

## 启动主程序

进入主程序目录：

```powershell
cd D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\Src
```

运行：

```powershell
python .\main2.py
```

启动后会进入交互控制台，可查看目标、锁定目标、启停自动跟踪和执行标定。

常用命令：

```text
l / list     查看当前目标列表
t <ID>       手动跟踪指定目标
a on/off     开启或关闭自动跟踪
auto         查看自动跟踪状态
r            释放当前目标
cal <ID>     开始标定并跟踪指定目标
done         停止标定并计算参数
cstat        查看标定状态
cres         显示标定结果
cclear       清除标定参数
q / quit     退出程序
```

## 数据文件

运行数据默认写入：

```text
airport/XJTU_Fusion_V6.4_box/MHT_Bias/Src/data/
```

主要文件：

- `radar_calibration_data.csv`：雷达 POINT 点迹记录。
- `raw_tracks.csv`：雷达 TRACK 航迹记录。
- `optical_measurements.csv`：光电测量记录。
- `track_log.txt`：MHT 融合后的目标列表日志。
- `track_results.json`：MHT 融合结果 JSON。

## 调试工具

常用工具位于：

```text
airport/XJTU_Fusion_V6.4_box/MHT_Bias/Src/tools/
```

示例：

```powershell
python .\tools\radar_udp_probe.py --host 0.0.0.0 --port 9000 --seconds 20
python .\tools\diagnose_radar_range_mode.py
python .\tools\offline_calibrate_from_logs.py
```

其中 `radar_udp_probe.py` 可用于确认雷达转发端口是否收到数据，是现场排查网络和协议问题时优先使用的脚本。

## 标定说明

详细现场流程见：

```text
airport/XJTU_Fusion_V6.4_box/MHT_Bias/标定操作手册.md
```

标定相关代码主要位于：

- `Src/core/calibration.py`
- `Src/core/calibration_commands.py`
- `Src/core/interactive_console.py`
- `Src/tools/offline_calibrate_from_logs.py`

## 注意事项

- 运行前请确认本机 IP、雷达转发地址、光电设备地址和端口配置一致。
- 主程序依赖真实设备或雷达转发数据；没有 UDP 输入时，目标列表和融合结果可能为空。
- `Src/data` 中的日志文件会随着运行更新，提交代码前请确认是否需要保留现场数据。
- 若在 Windows 下遇到权限或多进程相关错误，可尝试使用普通 PowerShell 或管理员 PowerShell 重新运行。
