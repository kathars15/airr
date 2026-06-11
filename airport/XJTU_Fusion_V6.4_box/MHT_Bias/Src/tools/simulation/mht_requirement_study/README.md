# MHT 输入性能需求仿真

## 作用

这里用于离线回答一个问题：

当融合输出要达到给定指标时，输入传感器至少需要什么性能。

当前脚本围绕下面几类输入量做扫描：

- 传感器数量 `sensor_count`
- 单传感器检测率 `sensor_pd`
- 单传感器位置误差 `sensor_pos_std_m`
- 单传感器输入虚警率 `sensor_fa_prob`

输出重点看三项：

- `Pd_out`
- `RMSE`
- `Pfa_out`

## 当前主入口

- `run_requirement_sim.py`
  - 主仿真脚本。

- `build_combined_impact_plots.py`
  - 固定一组基线条件后，分别扫多个输入参数，自动生成三张综合影响图。

## 当前目录建议保留内容

- `run_requirement_sim.py`
- `build_combined_impact_plots.py`
- `common/`
- `MHT/`
- `outputs/`
- `impact_outputs/`

其他脚本或旧输出如果不再使用，建议放入 `archive/`。

## 运行方式

### 1. 直接跑当前主场景

```powershell
python run_requirement_sim.py
```

### 2. 快速检查

```powershell
python run_requirement_sim.py --quick
```

### 3. 单变量纵向扫描

```powershell
python run_requirement_sim.py --sweep-field sensor_pd --sweep-values 0.70,0.80,0.90,0.95 --sensor-count 3 --sensor-pos-std 19 --sensor-fa-prob 0.20 --mc-runs 3
```

### 4. 生成综合影响图

先改 `build_combined_impact_plots.py` 顶部：

- `BASELINE`
- `SWEEPS`

再运行：

```powershell
python build_combined_impact_plots.py
```

## 场景说明

当前仿真采用典型低空目标场景，核心状态统一为：

```text
[E, N, U, VE, VN, VU]
```

此前主要按“多旋翼无人机为主、直线飞行为主、带轻微机动转弯”方向做测试。

## 输出文件

### `outputs/`

主仿真结果，例如：

- `simulation_summary.json`
- `simulation_summary_report.md`
- `trajectory_compare_3d.png`
- `trajectory_compare_timeseries.png`

### `impact_outputs/`

综合影响图输出目录。

当前重点只关心：

- `combined_pd_out.png`
- `combined_rmse.png`
- `combined_pfa_out.png`

## 注意

- 当前建模仍以同步量测、统一扫描周期为主。
- 当前 `sensor_pos_std_m` 仍按三轴同方差位置噪声处理。
- 如果后面要改成异步时间戳、取消聚类、重新定义输入虚警率，需要再单独改模型，不属于这版整理范围。
