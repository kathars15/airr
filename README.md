# AIRR 总仓库说明

这个仓库当前主要承载低空目标雷达/光电融合项目，核心工程位于：

- [MHT_Bias](/D:/desk/airr/airport/XJTU_Fusion_V6.4_box/MHT_Bias)

## 当前仓库主要内容

- `airport/`
  - 主工程代码。
- `docu/`
  - 项目说明、报告、部署指北等文档材料。
- `calibration_data/`
  - 仓库根层历史数据说明目录。
- `README.md`
  - 仓库级说明。

## 当前主工程入口

研发/桌面版主入口：

- [main2.py](/D:/desk/airr/airport/XJTU_Fusion_V6.4_box/MHT_Bias/Src/main2.py)

终端盒子版主入口：

- [terminal_box_main.py](/D:/desk/airr/airport/XJTU_Fusion_V6.4_box/MHT_Bias/Src/terminal_box_main.py)

## 终端盒子最小部署目录

仓库内会额外提供一个最小部署目录：

- [terminal_box_runtime](/D:/desk/airr/airport/XJTU_Fusion_V6.4_box/MHT_Bias/terminal_box_runtime)

它面向“直接拷到终端盒子运行”的场景，只保留：

- 雷达接收
- MHT 跟踪
- 光电状态接收
- 跟踪后位置/速度融合输出
- 雷达识别结果输出
- 光电 YOLO 识别结果输出

## 不提交的本机/临时内容

以下目录视为本机环境或临时文件，不纳入项目版本管理：

- `.agents/`
- `.vscode/`
- `tmp_scientific_agent_skills/`
- `tmp_torch_wheels/`
- `logs/`

此外，运行生成目录默认不纳入版本管理，例如：

- `Src/data/`
- `Src/flight_data_runs/`
- `Src/calibration_data/`
- `CV/code_image/runs/`

## 终端盒子主目录里常见文件是什么

你在盒子里看到的这些目录中，只有少数是项目相关：

- `airport/`
  - 项目代码目录，和本仓库对应。
- `Yolo/`
  - 如果盒子上单独放过视觉项目或模型代码，通常属于部署/实验目录，不一定是当前主工程必需。
- `post_install/`
  - 常见为安装后脚本或环境初始化目录，需要按盒子实际内容判断是否保留。

这些通常是系统/用户环境内容，不是本项目代码：

- `.cache/`
- `.conda/`
- `.config/`
- `.local/`
- `miniconda3/`
- `.bashrc`
- `.bash_history`

这些不应作为“工程代码”去整理或拷贝。

