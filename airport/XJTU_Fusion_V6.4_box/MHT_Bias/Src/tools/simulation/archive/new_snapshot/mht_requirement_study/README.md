# MHT 低空融合输入性能需求仿真

## 先看这里：路径设置和使用方法

这套仿真现在已经整理成可单独打包的目录。**直接把整个 `mht_requirement_study` 文件夹发过去即可**，不要只发单个脚本。

建议目录结构保持如下：

```text
mht_requirement_study/
├─ run_requirement_sim.py
├─ README.md
├─ MHT/
│  ├─ __init__.py
│  └─ POMHT.py
├─ common/
│  ├─ __init__.py
│  ├─ Tracker.py
│  ├─ clusters.py
│  ├─ Plots.py
│  └─ utlis.py
└─ outputs/
```

### 运行环境

需要 Python 环境里至少有这些库：

```text
numpy
scipy
matplotlib
cvxpy
scikit-learn
rasterio
pyproj
```

如果用的是 `conda`，可以先在终端里测试：

```powershell
python -c "import numpy, scipy, matplotlib, cvxpy, sklearn, rasterio, pyproj; print('ok')"
```

### 如何运行

进入这个文件夹后直接运行：

```powershell
python run_requirement_sim.py --quick
```

完整默认网格：

```powershell
python run_requirement_sim.py
```

如果想只跑某一小段参数范围：

```powershell
python run_requirement_sim.py --mc-runs 10 --sensor-count 4 --sensor-pd 0.8,0.9 --sensor-pos-std 30 --sensor-fa-prob 0.2,0.25,0.3
```

### 输出在哪里

所有结果默认写到当前文件夹下的 `outputs/`：

- `summary.csv`
- `summary.json`
- `requirement_boundary.csv`
- `heatmap_detection.png`
- `heatmap_rmse.png`
- `heatmap_false_alarm.png`
- `pass_fail_map.png`

### 路径说明

当前脚本已经改成：

1. 优先使用当前文件夹里的 `MHT/` 和 `common/`
2. 如果你仍然放在原工程里运行，再回退到上层工程路径

所以正常情况下，**别人拿到整个文件夹后，不需要再改 Python 导入路径**。

如果对方非要把里面的文件拆开，导致 `MHT/` 或 `common/` 不在同级目录，脚本就会导入失败。

## 这个仿真是做什么的

这个文件夹用于离线评估当前 MHT 低空融合算法对输入传感器性能的要求。

核心问题是：

```text
如果融合输出需要达到：
  检测率 >= 95%
  三维位置跟踪 RMSE <= 15 m
  输出虚警率 <= 5%

那么输入侧大概需要：
  多少个传感器？
  单个传感器检测率要多高？
  单个传感器 XYZ 位置探测精度要多好？
  单个传感器虚警率能容忍到多少？
```

这套脚本直接调用现有 `POMHT_Bias`，但不会修改在线代码、传感器配置文件、标定文件或 MHT 内核。

## 文件说明

- `run_requirement_sim.py`：主仿真脚本
- `MHT/`：打包进来的 MHT 核心代码
- `common/`：打包进来的公共数学与聚类模块
- `outputs/`：自动生成的 CSV、JSON 和图表输出目录

## 默认仿真假设

- 传感器直接输出 ENU/XYZ 三维坐标量测
- 多个传感器同步输出，扫描周期为 `2 s`
- 同一扫描周期内的多传感器量测会先做 DBSCAN 聚类，再对每个聚类求均值送入 MHT
- 单传感器初始位置量测误差默认按 XYZ 三轴独立零均值高斯误差建模，标准差为 `30 m`
- 输入虚警率重点测试 `20%-30%`
- 单个输入虚警表示：每个传感器在每个扫描周期内，以指定概率产生一个随机虚假 XYZ 点
- 第一版不建模时间不同步、传感器系统偏差、标定误差、极坐标角度异常、雷达俯仰突变等问题

## 默认真实目标

默认仿真一共生成 `5` 个真实目标，对应脚本中的：

```text
TARGET_COUNT = 5
```

这 5 个目标按固定类型循环生成：

- `multirotor_uav`：多旋翼无人机
- `fixed_wing_small`：小型固定翼目标
- `bird_slow`：鸟类 / 慢速小目标

默认场景大致可以理解成：

```text
2 个多旋翼无人机 + 2 个小型固定翼目标 + 1 个鸟类/慢速小目标
```

目标真实状态统一为：

```text
[E, N, U, VE, VN, VU]
```

其中 `E/N/U` 表示东、北、天方向位置，`VE/VN/VU` 表示对应速度。

## 真实目标运动生成方式

仿真中的真实目标不是静止点，也不是每帧随机重采样位置，而是按连续运动模型生成真值轨迹。

每个目标都会独立生成：

- 初始 `E/N/U` 位置
- 初始水平运动方向
- 初始水平速度
- 初始垂直速度
- 转弯率 `turn_rate`
- 相位参数 `phase`

每一帧都按 `2 s` 扫描周期推进：

```text
p_k = p_{k-1} + v_k * T
T = 2 s
```

速度并不是固定不变，而是根据目标类型叠加不同程度的航向、速度和高度扰动。

### 三类目标特征

- `multirotor_uav`
  速度范围约 `3-18 m/s`
  可短暂停留或低速盘旋
  航向缓慢变化，叠加轻微随机扰动

- `fixed_wing_small`
  速度范围约 `18-45 m/s`
  不悬停，速度变化更平滑
  转弯率较小，高度变化较缓

- `bird_slow`
  速度范围约 `2-15 m/s`
  航向和速度扰动更强
  垂直速度起伏更明显

### 边界处理

- 如果目标飞出东西/南北仿真区域，会做反弹处理
- 如果高度低于 `30 m` 或高于 `300 m`，同样做高度反弹

这样可以保证目标在较长仿真时间内持续存在，便于统计检测率、跟踪精度和虚警率。

## 运行方式

快速冒烟测试：

```powershell
python run_requirement_sim.py --quick
```

完整默认网格：

```powershell
python run_requirement_sim.py
```

只跑某个重点参数范围：

```powershell
python run_requirement_sim.py --mc-runs 10 --sensor-count 4 --sensor-pd 0.8,0.9 --sensor-pos-std 30 --sensor-fa-prob 0.2,0.25,0.3
```

## 默认扫描参数

脚本顶部默认配置包括：

```text
SCAN_PERIOD_SEC = 2.0
TARGET_COUNT = 5
SENSOR_COUNTS = [1, 2, 3, 4, 5, 6]
SENSOR_PD_VALUES = [0.60, 0.70, 0.80, 0.90, 0.95]
SENSOR_POS_STD_VALUES_M = [10, 15, 20, 30, 40, 50]
SENSOR_FA_PROB_VALUES = [0.20, 0.25, 0.30]
DEFAULT_SENSOR_POS_STD_M = 30.0
```

完整网格会比较耗时，因为每个参数组合都会运行真实 MHT 假设树和整数规划。建议先用 `--quick` 验证，再围绕可疑参数范围做局部扫描。

## 输出文件

输出文件生成在 `outputs/`：

- `summary.csv`：每组输入参数对应一行结果
- `summary.json`：完整配置、随机种子和仿真结果
- `requirement_boundary.csv`：在已测试网格中找到的最小达标边界
- `heatmap_detection.png`：输出检测率热力图
- `heatmap_rmse.png`：三维位置 RMSE 热力图
- `heatmap_false_alarm.png`：输出虚警率热力图
- `pass_fail_map.png`：达标 / 不达标图

## 指标公式

输出检测率：

```text
P_D,out = sum_t sum_i d_{t,i} / sum_t N_t
```

其中 `N_t` 表示时刻 `t` 的真实目标数量。若真实目标 `i` 在时刻 `t` 被融合输出航迹匹配，且三维位置误差在匹配门限内，则 `d_{t,i}=1`，否则 `d_{t,i}=0`。

输出虚警率：

```text
P_FA,out = N_false_track / N_output_track
```

其中 `N_output_track` 表示统计窗口内融合输出航迹总数，`N_false_track` 表示无法匹配任何真实目标的输出航迹数。

三维位置误差：

```text
e_{t,i} = sqrt((E_hat_{t,i}-E_{t,i})^2 + (N_hat_{t,i}-N_{t,i})^2 + (U_hat_{t,i}-U_{t,i})^2)
```

三维位置均方根误差：

```text
RMSE_pos = sqrt(1 / M * sum_{k=1}^{M} e_k^2)
```

其中 `M` 表示成功匹配的“输出航迹-真实目标”样本数量。

识别率：

```text
P_cls = N_correct_class / N_classified
```

其中 `N_classified` 表示被系统输出类别的目标数量，`N_correct_class` 表示类别与真值一致的目标数量。当前这套 MHT 输入性能需求仿真保留该公式用于文档一致性，但第一版不把图像识别类别强行纳入 MHT 性能扫描。

## 达标条件

```text
P_D,out >= 0.95
RMSE_pos <= 15 m
P_FA,out <= 0.05
```

只有三项同时满足时，该输入传感器配置才记为 `PASS`。

## 结果解读

控制台会打印每组参数的结果，例如：

```text
[仿真] 1/1 N=3, Pd=0.95, sigma=30.0m, Pfa_in=0.25
[仿真]   -> Pd_out=0.974, RMSE=26.90m, Pfa_out=0.122, FAIL
```

这表示在该仿真条件下：

- 输出检测率已经达到 95% 要求
- 位置 RMSE 超过 15 m
- 输出虚警率超过 5%
- 因此该配置整体不达标

## 注意事项

- 仿真结果是对当前 MHT 框架和当前指标口径的定量评估，不代表真实雷达硬件的绝对性能
- 如果后续要模拟雷达俯仰角突变、距离段异常、系统偏差或时间不同步，需要在本脚本基础上继续扩展输入误差模型
- 如果要给论文或汇报使用，建议以 `summary.csv` 和 `requirement_boundary.csv` 为主，再配合热力图说明趋势
