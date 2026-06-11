#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于当前低空 MHT 融合算法的蒙特卡洛输入性能需求仿真。

这个脚本回答一个很具体的问题：
如果融合输出要求达到 95% 检测率、15 m 三维位置 RMSE 和 5% 输出虚警率，
那么输入侧需要多少个传感器、单传感器检测率要多高、位置精度要多好、
虚警水平要控制在什么范围。

当前版本的主要假设：
- 每个传感器每 2 s 同步输出一次 ENU/XYZ 位置量测；
- 单传感器位置误差服从 E/N/U 三轴零均值高斯分布；
- 输入虚警概率表示每个传感器在一次扫描中按该概率产生一个随机虚假量测；
- 第一版不建模传感器系统偏差、时间不同步、标定误差和角度异常。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
from scipy.optimize import linear_sum_assignment


#SCRIPT_DIR = Path(__file__).resolve().parent
#LOCAL_PACKAGE_ROOT = SCRIPT_DIR
#SRC_DIR = SCRIPT_DIR.parents[2]
#MHT_BIAS_DIR = SRC_DIR.parent
#WORKSPACE_DIR = MHT_BIAS_DIR.parents[2]

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGE_ROOT = SCRIPT_DIR

# 建立一个安全获取父目录的函数，如果撞到盘符根目录就停止向上，防止越界
def safe_get_parent(path: Path, levels: int) -> Path:
    current = path
    for _ in range(levels):
        if current.parent == current:  # 代表已经到了磁盘根目录（如 D:\）
            break
        current = current.parent
    return current

SRC_DIR = safe_get_parent(SCRIPT_DIR, 3)
MHT_BIAS_DIR = SRC_DIR.parent
WORKSPACE_DIR = safe_get_parent(MHT_BIAS_DIR, 3)

# 优先加载当前打包目录内的 MHT/common 副本；如果用户仍在原工程里运行，
# 再回退到上层工程路径。
for path in (str(LOCAL_PACKAGE_ROOT), str(SRC_DIR), str(MHT_BIAS_DIR), str(WORKSPACE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from MHT.POMHT import POMHT_Bias  # noqa: E402
from common.clusters import Clustering_Obs  # noqa: E402


class SimulationPOMHT(POMHT_Bias):
    """离线参数扫描用的轻量兼容包装层。

    现有 MHT 代码里的部分 score 表达式有时会变成 1x1 的 numpy 数组。
    在线链路不一定会触发严格的标量转换，但离线仿真在做整数规划时需要
    明确的 Python float。这里把兼容处理限定在仿真脚本内部，不改生产内核。
    """

    @staticmethod
    def _as_float(value: object) -> float:
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return float(value)  # type: ignore[arg-type]

    def _normalize_tree_scores(self) -> None:
        for layer in getattr(self, "Tree", []):
            for node in layer.values():
                node.score = self._as_float(node.score)

    def Best_hypotheses(self) -> None:  # noqa: N802 - 保持上游方法名不变
        self._normalize_tree_scores()
        if len(self.Tree) != self.Tree_maxlen:
            self.hyp_num = 0
            return

        hyp_num = len(self.Tree[-1])
        if hyp_num <= 0:
            self.hyp_num = 0
            return

        self.hyp_num = hyp_num
        for il in range(-2, -self.Tree_maxlen - 1, -1):
            parent_nodes = self.Tree[il]
            children_nodes = self.Tree[il + 1]
            for _, parent_node in parent_nodes.items():
                parent_node.hyp_ids = []
                for child_id in parent_node.children_ids:
                    if child_id in children_nodes:
                        parent_node.hyp_ids.extend(children_nodes[child_id].hyp_ids)

        assign = cp.Variable(hyp_num, boolean=True)
        cp_constraints = []

        track_num = len(self.Tree[0])
        if track_num > 0:
            constraint_matrix = np.zeros((track_num, hyp_num), dtype=np.uint64)
            for row_idx, (_, root_node) in enumerate(self.Tree[0].items()):
                if root_node.hyp_ids:
                    constraint_matrix[row_idx, root_node.hyp_ids] = 1
            cp_constraints.append(constraint_matrix @ assign == np.ones((constraint_matrix.shape[0],)))

        obs_constraint_matrices = []
        for layer_id in range(1, self.Tree_maxlen):
            obs_num = len(self.obs_s[layer_id])
            if obs_num <= 0:
                continue
            constraint_matrix = np.zeros((obs_num, hyp_num), dtype=np.uint64)
            for _, node in self.Tree[layer_id].items():
                if len(node.obs_id) > 0 and len(node.hyp_ids) > 0:
                    for obs_id in node.obs_id:
                        constraint_matrix[obs_id, node.hyp_ids] = 1
            obs_constraint_matrices.append(constraint_matrix)

        if obs_constraint_matrices:
            obs_constraints = np.concatenate(obs_constraint_matrices)
            if obs_constraints.shape[0] > 0:
                cp_constraints.append(obs_constraints @ assign == np.ones((obs_constraints.shape[0],)))

        score_list = np.array(
            [float(leaf_node.score) for _, leaf_node in self.Tree[-1].items()],
            dtype=float,
        ).reshape(1, -1)
        score_list = np.clip(score_list, -1e10, 1e10)

        if not cp_constraints:
            self.assignment = np.zeros((hyp_num,), dtype=float)
            self.assignment[int(np.argmax(score_list.reshape(-1)))] = 1.0
            self.score_sum = float(np.max(score_list))
            return

        prob = cp.Problem(cp.Maximize(score_list @ assign), cp_constraints)
        prob.solve(solver="GLPK_MI")
        self.score_sum = prob.value
        self.assignment = assign.value


# =========================
# 面向使用者的仿真配置
# =========================
SCAN_PERIOD_SEC = 2.0
SIM_DURATION_SEC = 400.0
WARMUP_SEC = 10.0
MONTE_CARLO_RUNS = 50
RANDOM_SEED = 20260601

TARGET_COUNT = 2
SENSOR_COUNTS = [1, 2, 3, 4, 5, 6]
SENSOR_PD_VALUES = [0.60, 0.70, 0.80, 0.90, 0.95]
SENSOR_POS_STD_VALUES_M = [10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
SENSOR_FA_PROB_VALUES = [0.20, 0.25, 0.30]
DEFAULT_SENSOR_POS_STD_M = 30.0

REQUIRED_DETECTION_RATE = 0.90
REQUIRED_RMSE_M = 15.0
REQUIRED_FALSE_ALARM_RATE = 0.05

MATCH_GATE_ROOT_SCALE = 1.0
# 评估匹配门限缩放系数。
# 匹配门限按 sqrt(3) * sigma * 系数 自动计算，不再使用固定米数。
CLUSTER_EPS_SIGMA = 3.0
CLUSTER_MIN_DISTANCE_M = 20.0
AREA_EAST_LIMIT_M = (-1500.0, 1500.0)
AREA_NORTH_LIMIT_M = (-1500.0, 1500.0)
ALTITUDE_LIMIT_M = (30.0, 300.0)
FALSE_ALARM_ALTITUDE_LIMIT_M = (20.0, 350.0)

# MHT 参数。这里沿用现有在线/回放链路的风格，但仅在本脚本内部生效，
# 这样仿真研究不会影响线上行为。
MHT_LAMBDA_NT = 1.0
MHT_MAX_VEL_MPS = 60.0
MHT_N_SCAN = 1
MHT_PG = 0.999
MHT_P_DEATH = 1e-2
MHT_RESOLVED_TIME_WINDOW_SEC = 4.0
MHT_RESOLVED_MIN_DETECT = 1
MHT_MAX_DETECT_TIME_SEC = 20.0
MHT_Q_SCALE = 0.05 #0.5
MHT_VOLUME_M3 = (
    (AREA_EAST_LIMIT_M[1] - AREA_EAST_LIMIT_M[0])
    * (AREA_NORTH_LIMIT_M[1] - AREA_NORTH_LIMIT_M[0])
    * (FALSE_ALARM_ALTITUDE_LIMIT_M[1] - FALSE_ALARM_ALTITUDE_LIMIT_M[0])
)

OUTPUT_DIR = SCRIPT_DIR / "outputs"


@dataclass
class ScenarioConfig:
    sensor_count: int
    sensor_pd: float
    sensor_pos_std_m: float
    sensor_fa_prob: float


@dataclass
class TargetState:
    target_id: int
    target_type: str
    pos: np.ndarray
    vel: np.ndarray
    phase: float
    turn_rate: float
    turn_bias: float
    start_pos: Optional[np.ndarray] = None


@dataclass
class RunMetrics:
    detection_rate: float
    false_alarm_rate: float
    rmse_m: Optional[float]
    mean_error_m: Optional[float]
    p95_error_m: Optional[float]
    matched_samples: int
    truth_samples: int
    output_tracks: int
    false_tracks: int


@dataclass
class TraceFrame:
    timestamp: float
    truth_positions: List[List[float]]
    output_positions: List[List[float]]
    matched_truth_positions: List[Optional[List[float]]]
    matched_output_positions: List[Optional[List[float]]]
    matched_errors: List[Optional[float]]


def _float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def clamp(value: float, limits: Tuple[float, float]) -> float:
    return min(max(float(value), limits[0]), limits[1])


def compute_match_gate_m(sensor_pos_std_m: float) -> float:
    """根据单传感器位置标准差计算输出评估时的匹配门限。"""
    sigma = max(float(sensor_pos_std_m), 1e-6)
    return MATCH_GATE_ROOT_SCALE * math.sqrt(3.0) * sigma


def reflect_position(pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = pos.copy()
    vel = vel.copy()
    for idx, limits in enumerate((AREA_EAST_LIMIT_M, AREA_NORTH_LIMIT_M, ALTITUDE_LIMIT_M)):
        if pos[idx] < limits[0]:
            pos[idx] = limits[0] + (limits[0] - pos[idx])
            vel[idx] *= -1.0
        elif pos[idx] > limits[1]:
            pos[idx] = limits[1] - (pos[idx] - limits[1])
            vel[idx] *= -1.0
        pos[idx] = clamp(pos[idx], limits)
    return pos, vel


def random_unit_xy(rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def generate_initial_targets(rng: np.random.Generator, target_count: int) -> List[TargetState]:
    # target_types = ["multirotor_uav", "fixed_wing_small", "bird_slow"]
    target_types = ["multirotor_uav"]
    targets: List[TargetState] = []
    square_side = 1000.0  # 正方形边长 1000 米
    for target_id in range(target_count):
        target_type = target_types[target_id % len(target_types)]
        pos = np.array(
            [
                rng.uniform(*AREA_EAST_LIMIT_M) * 0.75,
                rng.uniform(*AREA_NORTH_LIMIT_M) * 0.75,
                rng.uniform(*ALTITUDE_LIMIT_M),
            ],
            dtype=float,
        )
        direction = random_unit_xy(rng)
        if target_type == "multirotor_uav":
            speed = rng.uniform(3.0, 18.0)
            vz = rng.uniform(-1.5, 1.5)
            turn_rate = rng.uniform(-0.06, 0.06)
        elif target_type == "fixed_wing_small":
            speed = rng.uniform(18.0, 45.0)
            vz = rng.uniform(-2.0, 2.0)
            turn_rate = rng.uniform(-0.025, 0.025)
        else:
            speed = rng.uniform(2.0, 15.0)
            vz = rng.uniform(-2.5, 2.5)
            turn_rate = rng.uniform(-0.10, 0.10)

        vel = np.array([direction[0] * speed, direction[1] * speed, vz], dtype=float)
        targets.append(
            TargetState(
                target_id=target_id,
                target_type=target_type,
                pos=pos,
                vel=vel,
                phase=rng.uniform(0.0, 2.0 * math.pi),
                turn_rate=turn_rate,
                turn_bias=rng.uniform(-0.008, 0.008) if target_type == "multirotor_uav" else 0.0,
            )
        )
    return targets


def propagate_targets(targets: Sequence[TargetState], dt: float, rng: np.random.Generator, step_idx: int) -> List[TargetState]:
    updated: List[TargetState] = []

    start_step = int(WARMUP_SEC / SCAN_PERIOD_SEC)  # 默认 5
    end_step = int(SIM_DURATION_SEC / SCAN_PERIOD_SEC)  # 默认 175
    total_sim_steps = end_step - start_step  # 正式仿真总步数

    for target in targets:
        pos = target.pos.copy()
        vel = target.vel.copy()
        phase = target.phase + 0.15 * dt

        # 安全初始化 turn_bias
        turn_bias = target.turn_bias

        # 1. 锁定第一帧的初始位置作为基准起点
        start_pos = target.start_pos
        if step_idx == start_step:
            start_pos = pos.copy()
            target.start_pos = start_pos

        # 2. 进入正式仿真闭环期
        if step_idx >= start_step and start_pos is not None:
            current_sim_step = step_idx - start_step
            remaining_steps = end_step - step_idx

            # 将总时间均匀切分为 4 个边（阶段 0, 1, 2, 3）
            stage_length = total_sim_steps / 4.0
            current_stage = min(int(current_sim_step / stage_length), 3)

            # 3. 规划口字型航线的 4 个理论顶点 (基于起点和各机型基准速度)
            base_speed = 12.0 if target.target_type == "multirotor_uav" else (
                30.0 if target.target_type == "fixed_wing_small" else 8.0)
            side_len = base_speed * (stage_length * dt)

            corner0 = start_pos.copy()  # 左上
            corner1 = start_pos + np.array([side_len, 0.0, 0.0])  # 右上
            corner2 = start_pos + np.array([side_len, -side_len, 0.0])  # 右下
            corner3 = start_pos + np.array([0.0, -side_len, 0.0])  # 左下

            # 定义当前阶段的航线【起点 A】与【终点 B】，以及标准航向
            if current_stage == 0:
                p_A, p_B = corner0, corner1
                base_heading = 0.0  # 东
            elif current_stage == 1:
                p_A, p_B = corner1, corner2
                base_heading = -math.pi / 2  # 南
            elif current_stage == 2:
                p_A, p_B = corner2, corner3
                base_heading = math.pi  # 西
            else:
                p_A, p_B = corner3, corner0  # 北 (最终回到起点)
                base_heading = math.pi / 2

            # 4. 动态过程噪声（降低高频硬噪声，让物理机动更丝滑）
            speed_xy = base_speed
            if target.target_type == "multirotor_uav":
                if step_idx % 20 == 0:
                    turn_bias = clamp(turn_bias + rng.normal(0.0, 0.002), (-0.01, 0.01))
                # 降低帧与帧之间的纯白噪声（从0.005降到0.001），主要依靠 turn_bias 展现缓慢波动机动
                heading = base_heading + turn_bias + rng.normal(0.0, 0.001)
                speed_xy = clamp(base_speed + rng.normal(0.0, 0.1), (4.0, 16.0))
                vel[2] = clamp(0.12 * math.sin(phase * 0.35) + rng.normal(0.0, 0.01), (-0.3, 0.3))
            elif target.target_type == "fixed_wing_small":
                heading = base_heading + 0.008 * math.sin(phase) + rng.normal(0.0, 0.001)
                speed_xy = clamp(base_speed + rng.normal(0.0, 0.3), (16.0, 48.0))
                vel[2] = clamp(0.15 * math.sin(phase * 0.35) + rng.normal(0.0, 0.01), (-0.4, 0.4))
            else:  # bird_slow
                heading = base_heading + rng.normal(0.0, 0.005)
                speed_xy = clamp(base_speed + rng.normal(0.0, 0.2), (1.5, 18.0))
                vel[2] = clamp(0.18 * math.sin(phase) + rng.normal(0.0, 0.01), (-0.4, 0.4))

            # 重新合成物理速度
            vel[0] = speed_xy * math.cos(heading)
            vel[1] = speed_xy * math.sin(heading)

            # 物理位置更新推进
            pos = pos + vel * dt

            # 5. 【核心优化：点到线段的航迹纠偏法 (Cross-Track Error Correction)】
            # 计算当前位置在当前航线段 AB 上的投影，只纠正横向偏航，不强拉纵向进度
            AB = p_B - p_A
            AB_unit = AB / np.linalg.norm(AB)
            AP = pos - p_A

            # 计算沿航线方向的投影距离，并限制在线段 AB 内部
            current_t = clamp(float(np.dot(AP, AB_unit)), (0.0, float(np.linalg.norm(AB))))
            # 得到航线上的理论标准点
            standard_pos = p_A + AB_unit * current_t

            # 柔和纠偏：以 15% 的温和反馈力度将飞机拉回基准航线，既保留波浪机动，又防止“切圆角”
            pos[0] += (standard_pos[0] - pos[0]) * 0.15
            pos[1] += (standard_pos[1] - pos[1]) * 0.15

            # Z 轴高度控制：同样柔和回归初始高度
            pos[2] += (start_pos[2] - pos[2]) * 0.15

            # 6. 【终点绝对闭环】仅在全剧终最后 3 步时，强行收敛到绝对起点 corner0，确保完美闭环
            if remaining_steps <= 3:
                pos += (corner0 - pos) / (remaining_steps + 1)

        else:
            # 预热期维持物理线性传播
            pos = pos + vel * dt

        # 边界安全兜底
        pos, vel = reflect_position(pos, vel)

        updated.append(
            TargetState(
                target_id=target.target_id,
                target_type=target.target_type,
                pos=pos,
                vel=vel,
                phase=phase,
                turn_rate=target.turn_rate,
                turn_bias=turn_bias,
                start_pos=start_pos,# 确保起点状态能够持续往下传递
            )
        )

    return updated


def generate_measurements(
    targets: Sequence[TargetState],
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    obs_k: List[np.ndarray] = []
    infos: List[Dict[str, object]] = []
    for sensor_idx in range(config.sensor_count):
        for target in targets:
            if rng.random() <= config.sensor_pd:
                noise = rng.normal(0.0, config.sensor_pos_std_m, size=3)
                meas = target.pos + noise
                obs_k.append(meas.reshape(3, 1).astype(float))
                infos.append(
                    {
                        "kind": "truth",
                        "sensor_idx": sensor_idx,
                        "target_id": target.target_id,
                        "target_type": target.target_type,
                    }
                )

        if rng.random() <= config.sensor_fa_prob:
            false_pos = np.array(
                [
                    rng.uniform(*AREA_EAST_LIMIT_M),
                    rng.uniform(*AREA_NORTH_LIMIT_M),
                    rng.uniform(*FALSE_ALARM_ALTITUDE_LIMIT_M),
                ],
                dtype=float,
            )
            obs_k.append(false_pos.reshape(3, 1))
            infos.append({"kind": "false_alarm", "sensor_idx": sensor_idx, "target_id": None})

    return obs_k, infos


def cluster_measurements(
    obs_k: List[np.ndarray],
    infos: List[Dict[str, object]],
    config: ScenarioConfig,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    if not obs_k:
        return [], []

    sigma = max(float(config.sensor_pos_std_m), 1e-6)
    obs_clusters, obs_indices = Clustering_Obs(
        obs_k=obs_k,
        Clustering_Type="DBSCAN",
        eps=max(CLUSTER_MIN_DISTANCE_M, CLUSTER_EPS_SIGMA * sigma),
        min_samples=1,
        Sigma=np.identity(3),
    )
    clustered_obs: List[np.ndarray] = []
    clustered_infos: List[Dict[str, object]] = []
    for cluster, indices in zip(obs_clusters, obs_indices):
        mean_obs = np.mean(np.concatenate(cluster, axis=1), axis=1).reshape(3, 1)
        clustered_obs.append(mean_obs)
        source_infos = [infos[idx] for idx in indices if idx < len(infos)]
        truth_ids = sorted(
            {
                int(item["target_id"])
                for item in source_infos
                if item.get("kind") == "truth" and item.get("target_id") is not None
            }
        )
        false_count = sum(1 for item in source_infos if item.get("kind") == "false_alarm")
        clustered_infos.append(
            {
                "kind": "cluster",
                "cluster_size": len(indices),
                "truth_target_ids": truth_ids,
                "false_count": false_count,
            }
        )
    return clustered_obs, clustered_infos


def build_sensor_config(config: ScenarioConfig) -> Dict[str, object]:
    # 把“单传感器虚警概率”映射成 POMHT 打分侧需要的期望虚警数量。
    # 这里保留一个很小的下限，避免出现 log(0)。
    fa_num = max(config.sensor_count * config.sensor_fa_prob, 1e-6)
    sensor_config: Dict[str, object] = {
        "Name": "Sim_XYZ",
        "Position": np.array([[0.0, 0.0, 0.0]]).T,
        "Meas_Type": "Position",
        "R": np.power(np.diag([config.sensor_pos_std_m] * 3), 2),
        "P_D": float(config.sensor_pd),
        "Is_Biased": False,
        "Sensor_Yaw": 0.0,
        "Sensor_Pitch": 0.0,
        "Sensor_Roll": 0.0,
        "Max_Range": 7000.0,
        "Max_Pitch": 90.0,
        "Max_Yaw": 360.0,
        "Volume": float(MHT_VOLUME_M3),
        "FA_Num": float(fa_num),
    }
    sensor_config["lambda_death"] = -math.log(1.0 - 0.01) / (2.0 * SCAN_PERIOD_SEC)
    sensor_config["mu_detect"] = -math.log(max(1.0 - config.sensor_pd, 1e-6)) / (2.0 * SCAN_PERIOD_SEC)
    return sensor_config


def init_tracker(timestamp: float, obs_k: List[np.ndarray], sensor_config: Dict[str, object], extra_infos: List[Dict[str, object]]) -> SimulationPOMHT:
    dim_d = 3
    match_gate_m = compute_match_gate_m(float(sensor_config["R"][0, 0]) ** 0.5)
    return SimulationPOMHT(
        Lambda_NT=MHT_LAMBDA_NT,
        obs_k=obs_k,
        timestamp=timestamp,
        sensor_config=deepcopy(sensor_config),
        Q_k=np.identity(dim_d) * MHT_Q_SCALE,
        Max_Vel=MHT_MAX_VEL_MPS,
        N_Scan=MHT_N_SCAN,
        Pg=MHT_PG,
        P_death=MHT_P_DEATH,
        dim_d=dim_d,
        Debug_Params={"Debug": False},
        extra_infos=extra_infos,
        Resolved_Time_Window=MHT_RESOLVED_TIME_WINDOW_SEC,
        Resolved_Min_Detect=MHT_RESOLVED_MIN_DETECT,
        max_detect_time=MHT_MAX_DETECT_TIME_SEC,
        Merge_Threshold=max(20.0, match_gate_m * 0.8),
    )


def get_output_positions(tracker: POMHT_Bias) -> List[np.ndarray]:
    if not hasattr(tracker, "Output_Nodes") or not tracker.Output_Nodes:
        return []
    positions: List[np.ndarray] = []
    for node in deepcopy(tracker.Output_Nodes[-1]).values():
        if getattr(node, "hyp_type", "") != "detect":
            continue
        pos = np.asarray(node.x_k_k[:3, :], dtype=float).reshape(3)
        if np.all(np.isfinite(pos)):
            positions.append(pos)
    return positions


def align_outputs_to_truth(
    output_positions: Sequence[np.ndarray],
    targets: Sequence[TargetState],
    match_gate_m: float,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[float]]]:
    matched_positions: List[Optional[np.ndarray]] = [None] * len(targets)
    matched_errors: List[Optional[float]] = [None] * len(targets)
    if len(output_positions) == 0 or len(targets) == 0:
        return matched_positions, matched_errors

    cost = np.zeros((len(output_positions), len(targets)), dtype=float)
    for i, out_pos in enumerate(output_positions):
        for j, target in enumerate(targets):
            cost[i, j] = float(np.linalg.norm(out_pos - target.pos))

    row_ind, col_ind = linear_sum_assignment(cost)
    for row, col in zip(row_ind, col_ind):
        err = float(cost[row, col])
        if err <= match_gate_m:
            matched_positions[int(col)] = np.asarray(output_positions[int(row)], dtype=float).reshape(3)
            matched_errors[int(col)] = err
    return matched_positions, matched_errors


def match_outputs_to_truth(
    output_positions: Sequence[np.ndarray],
    targets: Sequence[TargetState],
    match_gate_m: float,
) -> Tuple[int, int, List[float]]:
    if len(output_positions) == 0:
        return 0, 0, []
    if len(targets) == 0:
        return 0, len(output_positions), []

    cost = np.zeros((len(output_positions), len(targets)), dtype=float)
    for i, out_pos in enumerate(output_positions):
        for j, target in enumerate(targets):
            cost[i, j] = float(np.linalg.norm(out_pos - target.pos))

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_errors: List[float] = []
    matched_outputs = set()
    for row, col in zip(row_ind, col_ind):
        err = float(cost[row, col])
        if err <= match_gate_m:
            matched_outputs.add(int(row))
            matched_errors.append(err)

    false_tracks = len(output_positions) - len(matched_outputs)
    return len(matched_errors), false_tracks, matched_errors


def run_single_trial(
    config: ScenarioConfig,
    seed: int,
    record_trace: bool = False,
) -> Tuple[RunMetrics, Optional[List[TraceFrame]]]:
    rng = np.random.default_rng(seed)
    targets = generate_initial_targets(rng, TARGET_COUNT)
    sensor_config = build_sensor_config(config)
    match_gate_m = compute_match_gate_m(config.sensor_pos_std_m)
    tracker: Optional[SimulationPOMHT] = None

    matched_samples = 0
    truth_samples = 0
    output_tracks = 0
    false_tracks = 0
    errors: List[float] = []
    trace_frames: List[TraceFrame] = []

    frame_count = int(SIM_DURATION_SEC / SCAN_PERIOD_SEC) + 1
    for frame_idx in range(frame_count):
        timestamp = frame_idx * SCAN_PERIOD_SEC
        if frame_idx > 0:
            targets = propagate_targets(targets, SCAN_PERIOD_SEC, rng, frame_idx)

        raw_obs_k, raw_infos = generate_measurements(targets, config, rng)
        obs_k, infos = cluster_measurements(raw_obs_k, raw_infos, config)
        if tracker is None:
            tracker = init_tracker(timestamp, obs_k, sensor_config, infos)
        else:
            tracker.forward(
                timestamp=timestamp,
                obs_k=obs_k,
                sensor_config=deepcopy(sensor_config),
                extra_infos=infos,
            )

        if timestamp < WARMUP_SEC:
            continue

        outputs = get_output_positions(tracker)
        frame_matched, frame_false, frame_errors = match_outputs_to_truth(outputs, targets, match_gate_m)
        aligned_outputs, aligned_errors = align_outputs_to_truth(outputs, targets, match_gate_m)
        matched_samples += frame_matched
        truth_samples += len(targets)
        output_tracks += len(outputs)
        false_tracks += frame_false
        errors.extend(frame_errors)
        if record_trace:
            trace_frames.append(
                TraceFrame(
                    timestamp=float(timestamp),
                    truth_positions=[target.pos.astype(float).tolist() for target in targets],
                    output_positions=[np.asarray(pos, dtype=float).reshape(3).tolist() for pos in outputs],
                    matched_truth_positions=[target.pos.astype(float).tolist() for target in targets],
                    matched_output_positions=[
                        None if pos is None else np.asarray(pos, dtype=float).reshape(3).tolist()
                        for pos in aligned_outputs
                    ],
                    matched_errors=[None if err is None else float(err) for err in aligned_errors],
                )
            )

    detection_rate = matched_samples / truth_samples if truth_samples > 0 else 0.0
    false_alarm_rate = false_tracks / output_tracks if output_tracks > 0 else 0.0
    if errors:
        errors_np = np.asarray(errors, dtype=float)
        rmse_m = float(np.sqrt(np.mean(np.square(errors_np))))
        mean_error_m = float(np.mean(errors_np))
        p95_error_m = float(np.percentile(errors_np, 95))
    else:
        rmse_m = None
        mean_error_m = None
        p95_error_m = None

    return RunMetrics(
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        rmse_m=rmse_m,
        mean_error_m=mean_error_m,
        p95_error_m=p95_error_m,
        matched_samples=matched_samples,
        truth_samples=truth_samples,
        output_tracks=output_tracks,
        false_tracks=false_tracks,
    ), (trace_frames if record_trace else None)


def summarize_trials(config: ScenarioConfig, trial_metrics: Sequence[RunMetrics]) -> Dict[str, object]:
    def mean_attr(name: str, none_as: Optional[float] = None) -> Optional[float]:
        values = []
        for item in trial_metrics:
            value = getattr(item, name)
            if value is None:
                if none_as is None:
                    continue
                value = none_as
            values.append(float(value))
        return float(np.mean(values)) if values else None

    total_matched = int(sum(item.matched_samples for item in trial_metrics))
    total_truth = int(sum(item.truth_samples for item in trial_metrics))
    total_outputs = int(sum(item.output_tracks for item in trial_metrics))
    total_false = int(sum(item.false_tracks for item in trial_metrics))

    detection_rate = total_matched / total_truth if total_truth > 0 else 0.0
    false_alarm_rate = total_false / total_outputs if total_outputs > 0 else 0.0
    rmse_m = mean_attr("rmse_m")
    mean_error_m = mean_attr("mean_error_m")
    p95_error_m = mean_attr("p95_error_m")
    pass_requirement = (
        detection_rate >= REQUIRED_DETECTION_RATE
        and rmse_m is not None
        and rmse_m <= REQUIRED_RMSE_M
        and false_alarm_rate <= REQUIRED_FALSE_ALARM_RATE
    )
    return {
        "sensor_count": config.sensor_count,
        "sensor_pd": config.sensor_pd,
        "sensor_pos_std_m": config.sensor_pos_std_m,
        "sensor_fa_prob": config.sensor_fa_prob,
        "monte_carlo_runs": len(trial_metrics),
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "rmse_m": _float_or_none(rmse_m),
        "mean_error_m": _float_or_none(mean_error_m),
        "p95_error_m": _float_or_none(p95_error_m),
        "matched_samples": total_matched,
        "truth_samples": total_truth,
        "output_tracks": total_outputs,
        "false_tracks": total_false,
        "pass_requirement": bool(pass_requirement),
        "trial_detection_rate_mean": mean_attr("detection_rate"),
        "trial_false_alarm_rate_mean": mean_attr("false_alarm_rate"),
        "trial_rmse_m_mean": mean_attr("rmse_m"),
    }


def iter_configs(args: argparse.Namespace) -> Iterable[ScenarioConfig]:
    if args.quick:
        sensor_counts = [1, 3]
        sensor_pd_values = [0.80, 0.95]
        sensor_pos_values = [15.0, DEFAULT_SENSOR_POS_STD_M]
        sensor_fa_values = [0.20, 0.30]
    else:
        sensor_counts = SENSOR_COUNTS
        sensor_pd_values = SENSOR_PD_VALUES
        sensor_pos_values = SENSOR_POS_STD_VALUES_M
        sensor_fa_values = SENSOR_FA_PROB_VALUES

    if args.sensor_count:
        sensor_counts = args.sensor_count
    if args.sensor_pd:
        sensor_pd_values = args.sensor_pd
    if args.sensor_pos_std:
        sensor_pos_values = args.sensor_pos_std
    if args.sensor_fa_prob:
        sensor_fa_values = args.sensor_fa_prob

    for sensor_count, sensor_pd, sensor_pos_std_m, sensor_fa_prob in product(
        sensor_counts,
        sensor_pd_values,
        sensor_pos_values,
        sensor_fa_values,
    ):
        yield ScenarioConfig(
            sensor_count=int(sensor_count),
            sensor_pd=float(sensor_pd),
            sensor_pos_std_m=float(sensor_pos_std_m),
            sensor_fa_prob=float(sensor_fa_prob),
        )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_requirement_boundary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    passed = [row for row in rows if row.get("pass_requirement")]
    boundary_rows: List[Dict[str, object]] = []
    if not passed:
        return boundary_rows

    by_fa: Dict[float, List[Dict[str, object]]] = {}
    for row in passed:
        by_fa.setdefault(float(row["sensor_fa_prob"]), []).append(row)

    for fa_prob, fa_rows in sorted(by_fa.items()):
        best = sorted(
            fa_rows,
            key=lambda row: (
                int(row["sensor_count"]),
                float(row["sensor_pd"]),
                float(row["sensor_pos_std_m"]),
                float(row["rmse_m"] if row["rmse_m"] is not None else 1e9),
            ),
        )[0]
        boundary_rows.append(
            {
                "sensor_fa_prob": fa_prob,
                "min_sensor_count": best["sensor_count"],
                "min_sensor_pd_in_pass_set": best["sensor_pd"],
                "sensor_pos_std_m": best["sensor_pos_std_m"],
                "detection_rate": best["detection_rate"],
                "rmse_m": best["rmse_m"],
                "false_alarm_rate": best["false_alarm_rate"],
            }
        )

    best_overall = sorted(
        passed,
        key=lambda row: (
            int(row["sensor_count"]),
            float(row["sensor_pd"]),
            float(row["sensor_pos_std_m"]),
            -float(row["sensor_fa_prob"]),
        ),
    )[0]
    boundary_rows.append(
        {
            "sensor_fa_prob": "overall",
            "min_sensor_count": best_overall["sensor_count"],
            "min_sensor_pd_in_pass_set": best_overall["sensor_pd"],
            "sensor_pos_std_m": best_overall["sensor_pos_std_m"],
            "detection_rate": best_overall["detection_rate"],
            "rmse_m": best_overall["rmse_m"],
            "false_alarm_rate": best_overall["false_alarm_rate"],
        }
    )
    return boundary_rows


def choose_available_slice(
    rows: Sequence[Dict[str, object]],
    desired_fa: float = 0.25,
    desired_sigma: float = DEFAULT_SENSOR_POS_STD_M,
) -> Tuple[Optional[float], Optional[float]]:
    pairs = sorted({(float(row["sensor_fa_prob"]), float(row["sensor_pos_std_m"])) for row in rows})
    if not pairs:
        return None, None
    best = min(pairs, key=lambda item: (abs(item[0] - desired_fa), abs(item[1] - desired_sigma)))
    return best


def _metric_grid(
    rows: Sequence[Dict[str, object]],
    metric: str,
    sensor_fa_prob: float,
    sensor_pos_std_m: float,
) -> Tuple[List[float], List[int], np.ndarray]:
    filtered = [
        row
        for row in rows
        if abs(float(row["sensor_fa_prob"]) - sensor_fa_prob) < 1e-9
        and abs(float(row["sensor_pos_std_m"]) - sensor_pos_std_m) < 1e-9
    ]
    x_values = sorted({float(row["sensor_pd"]) for row in filtered})
    y_values = sorted({int(row["sensor_count"]) for row in filtered})
    grid = np.full((len(y_values), len(x_values)), np.nan)
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    for row in filtered:
        value = row.get(metric)
        if value is None:
            continue
        grid[y_index[int(row["sensor_count"])], x_index[float(row["sensor_pd"])]] = float(value)
    return x_values, y_values, grid


def plot_heatmap(
    rows: Sequence[Dict[str, object]],
    metric: str,
    title: str,
    output_path: Path,
    cmap: str,
    sensor_fa_prob: float = 0.25,
    sensor_pos_std_m: float = DEFAULT_SENSOR_POS_STD_M,
) -> None:
    x_values, y_values, grid = _metric_grid(rows, metric, sensor_fa_prob, sensor_pos_std_m)
    if not x_values or not y_values:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(x_values)), [f"{value:.2f}" for value in x_values])
    ax.set_yticks(range(len(y_values)), [str(value) for value in y_values])
    ax.set_xlabel("Single-sensor Pd")
    ax.set_ylabel("Sensor count")
    ax.set_title(f"{title}\ninput FA={sensor_fa_prob:.0%}, input sigma={sensor_pos_std_m:.0f}m")
    for y in range(len(y_values)):
        for x in range(len(x_values)):
            value = grid[y, x]
            label = "NA" if np.isnan(value) else (f"{value:.2f}" if metric != "rmse_m" else f"{value:.1f}")
            ax.text(x, y, label, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pass_fail(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    # 展示最接近默认压力场景的一组切片。
    fa_slice, sigma_slice = choose_available_slice(rows)
    if fa_slice is None or sigma_slice is None:
        return
    x_values, y_values, grid = _metric_grid(
        rows,
        "pass_requirement",
        sensor_fa_prob=fa_slice,
        sensor_pos_std_m=sigma_slice,
    )
    if not x_values or not y_values:
        return
    grid = np.nan_to_num(grid, nan=0.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(x_values)), [f"{value:.2f}" for value in x_values])
    ax.set_yticks(range(len(y_values)), [str(value) for value in y_values])
    ax.set_xlabel("Single-sensor Pd")
    ax.set_ylabel("Sensor count")
    ax.set_title(f"Pass/fail map\ninput FA={fa_slice:.0%}, input sigma={sigma_slice:.0f}m")
    for y in range(len(y_values)):
        for x in range(len(x_values)):
            ax.text(x, y, "PASS" if grid[y, x] >= 0.5 else "FAIL", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _series_from_trace(
    trace_frames: Sequence[TraceFrame],
    target_idx: int,
    truth_dim: int,
    matched: bool,
) -> List[float]:
    values: List[float] = []
    for frame in trace_frames:
        series = frame.matched_output_positions if matched else frame.matched_truth_positions
        item = series[target_idx] if target_idx < len(series) else None
        values.append(float("nan") if item is None else float(item[truth_dim]))
    return values


def write_trace_csv(path: Path, trace_frames: Sequence[TraceFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for frame in trace_frames:
        target_count = len(frame.matched_truth_positions)
        for target_idx in range(target_count):
            truth = frame.matched_truth_positions[target_idx]
            matched = frame.matched_output_positions[target_idx]
            rows.append(
                {
                    "timestamp_sec": frame.timestamp,
                    "target_id": target_idx,
                    "truth_e_m": None if truth is None else truth[0],
                    "truth_n_m": None if truth is None else truth[1],
                    "truth_u_m": None if truth is None else truth[2],
                    "track_e_m": None if matched is None else matched[0],
                    "track_n_m": None if matched is None else matched[1],
                    "track_u_m": None if matched is None else matched[2],
                    "match_error_m": frame.matched_errors[target_idx],
                }
            )
    write_csv(path, rows)


def plot_trace_comparison_3d(
    trace_frames: Sequence[TraceFrame],
    config: ScenarioConfig,
    metrics: RunMetrics,
    output_path: Path,
) -> None:
    if not trace_frames:
        return
    target_count = len(trace_frames[0].matched_truth_positions)
    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    truth_colors = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
    track_colors = ["tab:red", "tab:orange", "tab:pink", "tab:gray", "gold", "black"]
    for target_idx in range(target_count):
        e_truth = _series_from_trace(trace_frames, target_idx, 0, matched=False)
        n_truth = _series_from_trace(trace_frames, target_idx, 1, matched=False)
        u_truth = _series_from_trace(trace_frames, target_idx, 2, matched=False)
        e_track = _series_from_trace(trace_frames, target_idx, 0, matched=True)
        n_track = _series_from_trace(trace_frames, target_idx, 1, matched=True)
        u_track = _series_from_trace(trace_frames, target_idx, 2, matched=True)
        ax.plot(e_truth, n_truth, u_truth, color=truth_colors[target_idx % len(truth_colors)], linewidth=2.0, label=f"Truth T{target_idx}")
        ax.plot(
            e_track,
            n_track,
            u_track,
            color=track_colors[target_idx % len(track_colors)],
            linestyle="--",
            linewidth=1.7,
            label=f"MHT T{target_idx}",
        )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title(
        "3D trajectory comparison\n"
        f"N={config.sensor_count}, Pd={config.sensor_pd:.2f}, sigma={config.sensor_pos_std_m:.1f}m, "
        f"Pfa={config.sensor_fa_prob:.2f}, RMSE={metrics.rmse_m:.2f}m"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trace_comparison_timeseries(
    trace_frames: Sequence[TraceFrame],
    config: ScenarioConfig,
    metrics: RunMetrics,
    output_path: Path,
) -> None:
    if not trace_frames:
        return
    target_count = len(trace_frames[0].matched_truth_positions)
    times = [frame.timestamp for frame in trace_frames]
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True)
    dim_names = ["East", "North", "Up"]
    truth_colors = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
    track_colors = ["tab:red", "tab:orange", "tab:pink", "tab:gray", "gold", "black"]
    for dim_idx, ax in enumerate(axes):
        for target_idx in range(target_count):
            truth_values = _series_from_trace(trace_frames, target_idx, dim_idx, matched=False)
            track_values = _series_from_trace(trace_frames, target_idx, dim_idx, matched=True)
            ax.plot(times, truth_values, color=truth_colors[target_idx % len(truth_colors)], linewidth=2.0, label=f"Truth T{target_idx}")
            ax.plot(
                times,
                track_values,
                color=track_colors[target_idx % len(track_colors)],
                linestyle="--",
                linewidth=1.5,
                label=f"MHT T{target_idx}",
            )
        ax.set_ylabel(f"{dim_names[dim_idx]} (m)")
        ax.grid(True, alpha=0.25)
    axes[0].set_title(
        "Truth vs MHT output by time\n"
        f"N={config.sensor_count}, Pd={config.sensor_pd:.2f}, sigma={config.sensor_pos_std_m:.1f}m, "
        f"Pfa={config.sensor_fa_prob:.2f}, Pd_out={metrics.detection_rate:.3f}, Pfa_out={metrics.false_alarm_rate:.3f}"
    )
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_representative_trace(
    output_dir: Path,
    config: ScenarioConfig,
    seed: int,
) -> None:
    metrics, trace_frames = run_single_trial(config, seed, record_trace=True)
    if not trace_frames:
        return
    trace_json = {
        "config": {
            "sensor_count": config.sensor_count,
            "sensor_pd": config.sensor_pd,
            "sensor_pos_std_m": config.sensor_pos_std_m,
            "sensor_fa_prob": config.sensor_fa_prob,
            "seed": seed,
        },
        "metrics": {
            "detection_rate": metrics.detection_rate,
            "false_alarm_rate": metrics.false_alarm_rate,
            "rmse_m": metrics.rmse_m,
            "mean_error_m": metrics.mean_error_m,
            "p95_error_m": metrics.p95_error_m,
            "matched_samples": metrics.matched_samples,
            "truth_samples": metrics.truth_samples,
            "output_tracks": metrics.output_tracks,
            "false_tracks": metrics.false_tracks,
        },
        "trace_frames": [
            {
                "timestamp": frame.timestamp,
                "truth_positions": frame.truth_positions,
                "output_positions": frame.output_positions,
                "matched_truth_positions": frame.matched_truth_positions,
                "matched_output_positions": frame.matched_output_positions,
                "matched_errors": frame.matched_errors,
            }
            for frame in trace_frames
        ],
    }
    write_json(output_dir / "representative_trace.json", trace_json)
    write_trace_csv(output_dir / "representative_trace.csv", trace_frames)
    plot_trace_comparison_3d(trace_frames, config, metrics, output_dir / "trajectory_compare_3d.png")
    plot_trace_comparison_timeseries(trace_frames, config, metrics, output_dir / "trajectory_compare_timeseries.png")


def make_plots(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    fa_slice, sigma_slice = choose_available_slice(rows)
    if fa_slice is None or sigma_slice is None:
        return
    plot_heatmap(
        rows,
        "detection_rate",
        "Output detection rate",
        output_dir / "heatmap_detection.png",
        "YlGnBu",
        sensor_fa_prob=fa_slice,
        sensor_pos_std_m=sigma_slice,
    )
    plot_heatmap(
        rows,
        "rmse_m",
        "3D position RMSE (m)",
        output_dir / "heatmap_rmse.png",
        "magma_r",
        sensor_fa_prob=fa_slice,
        sensor_pos_std_m=sigma_slice,
    )
    plot_heatmap(
        rows,
        "false_alarm_rate",
        "Output false-alarm rate",
        output_dir / "heatmap_false_alarm.png",
        "OrRd",
        sensor_fa_prob=fa_slice,
        sensor_pos_std_m=sigma_slice,
    )
    plot_pass_fail(rows, output_dir / "pass_fail_map.png")


def format_metric(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def run_study(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = list(iter_configs(args))
    mc_runs = int(args.mc_runs if args.mc_runs is not None else (2 if args.quick else MONTE_CARLO_RUNS))
    base_seed = int(args.seed)

    print(f"[仿真] 输出目录: {output_dir}")
    print(f"[仿真] 参数组合数: {len(configs)}, 蒙特卡洛次数: {mc_runs}, 随机种子: {base_seed}")
    print(
        "[仿真] 输出指标要求: "
        f"Pd_out>={REQUIRED_DETECTION_RATE:.2f}, RMSE<={REQUIRED_RMSE_M:.1f}m, "
        f"Pfa_out<={REQUIRED_FALSE_ALARM_RATE:.2f}"
    )

    rows: List[Dict[str, object]] = []
    started = time.time()
    for cfg_idx, config in enumerate(configs, start=1):
        match_gate_m = compute_match_gate_m(config.sensor_pos_std_m)
        print(
            f"[仿真] {cfg_idx}/{len(configs)} "
            f"N={config.sensor_count}, Pd={config.sensor_pd:.2f}, "
            f"sigma={config.sensor_pos_std_m:.1f}m, Pfa_in={config.sensor_fa_prob:.2f}, "
            f"gate={match_gate_m:.1f}m"
        )
        trial_metrics: List[RunMetrics] = []
        for run_idx in range(mc_runs):
            seed = (
                base_seed
                + cfg_idx * 100000
                + run_idx * 1009
                + config.sensor_count * 17
                + int(config.sensor_pd * 1000) * 19
                + int(config.sensor_pos_std_m * 10) * 23
                + int(config.sensor_fa_prob * 1000) * 29
            )
            metrics, _ = run_single_trial(config, seed)
            trial_metrics.append(metrics)

        row = summarize_trials(config, trial_metrics)
        rows.append(row)
        status = "PASS" if row["pass_requirement"] else "FAIL"
        print(
            f"[仿真]   -> Pd_out={format_metric(row['detection_rate'])}, "
            f"RMSE={format_metric(row['rmse_m'], 2)}m, "
            f"Pfa_out={format_metric(row['false_alarm_rate'])}, {status}"
        )

        # 每跑完一组参数就落盘一次，这样长时间仿真中断后也能保留部分结果。
        write_csv(output_dir / "summary.csv", rows)
        write_json(
            output_dir / "summary.json",
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "settings": build_settings_dict(mc_runs, base_seed),
                "rows": rows,
            },
        )

    boundary_rows = build_requirement_boundary(rows)
    write_csv(output_dir / "requirement_boundary.csv", boundary_rows)
    make_plots(rows, output_dir)
    if boundary_rows:
        overall = boundary_rows[-1]
        trace_config = ScenarioConfig(
            sensor_count=int(overall["min_sensor_count"]),
            sensor_pd=float(overall["min_sensor_pd_in_pass_set"]),
            sensor_pos_std_m=float(overall["sensor_pos_std_m"]),
            sensor_fa_prob=float(
                boundary_rows[0]["sensor_fa_prob"] if overall["sensor_fa_prob"] == "overall" else overall["sensor_fa_prob"]
            ),
        )
        trace_seed = (
            base_seed
            + trace_config.sensor_count * 17
            + int(trace_config.sensor_pd * 1000) * 19
            + int(trace_config.sensor_pos_std_m * 10) * 23
            + int(trace_config.sensor_fa_prob * 1000) * 29
        )
        export_representative_trace(output_dir, trace_config, trace_seed)

    elapsed = time.time() - started
    print(f"[仿真] 完成，总耗时 {elapsed:.1f}s")
    if boundary_rows:
        overall = boundary_rows[-1]
        print("[仿真] 推荐的保守边界:")
        print(
            f"  传感器数 >= {overall['min_sensor_count']}, "
            f"单传感器 Pd >= {float(overall['min_sensor_pd_in_pass_set']):.2f}, "
            f"sigma <= {float(overall['sensor_pos_std_m']):.1f}m, "
            f"输入虚警率测试到 {overall['sensor_fa_prob']}"
        )
    else:
        print("[仿真] 当前已测试配置中，没有组合同时满足全部指标要求。")
    return rows, boundary_rows


def build_settings_dict(mc_runs: int, seed: int) -> Dict[str, object]:
    return {
        "scan_period_sec": SCAN_PERIOD_SEC,
        "sim_duration_sec": SIM_DURATION_SEC,
        "warmup_sec": WARMUP_SEC,
        "monte_carlo_runs": mc_runs,
        "random_seed": seed,
        "target_count": TARGET_COUNT,
        "sensor_counts": SENSOR_COUNTS,
        "sensor_pd_values": SENSOR_PD_VALUES,
        "sensor_pos_std_values_m": SENSOR_POS_STD_VALUES_M,
        "sensor_fa_prob_values": SENSOR_FA_PROB_VALUES,
        "default_sensor_pos_std_m": DEFAULT_SENSOR_POS_STD_M,
        "required_detection_rate": REQUIRED_DETECTION_RATE,
        "required_rmse_m": REQUIRED_RMSE_M,
        "required_false_alarm_rate": REQUIRED_FALSE_ALARM_RATE,
        "match_gate_rule": "MATCH_GATE_M = 1.0 * sqrt(3) * sigma",
        "match_gate_root_scale": MATCH_GATE_ROOT_SCALE,
        "cluster_eps_sigma": CLUSTER_EPS_SIGMA,
        "cluster_min_distance_m": CLUSTER_MIN_DISTANCE_M,
        "target_models": [
            "multirotor_uav: 3-18 m/s，可悬停，轻微机动",
            "fixed_wing_small: 18-45 m/s，转弯和高度变化较平滑",
            "bird_slow: 2-15 m/s，航向和速度扰动更强",
        ],
        "input_false_alarm_model": "每个传感器每次扫描以指定概率产生一个随机虚假 XYZ 点",
        "position_error_model": "ENU/XYZ 三轴独立零均值高斯误差",
    }


def parse_float_list(values: Optional[Sequence[str]]) -> Optional[List[float]]:
    if not values:
        return None
    parsed: List[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(float(part))
    return parsed


def parse_int_list(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    floats = parse_float_list(values)
    return [int(value) for value in floats] if floats else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 MHT 输入性能需求蒙特卡洛仿真。")
    parser.add_argument("--quick", action="store_true", help="运行一个较小的快速冒烟网格。")
    parser.add_argument("--mc-runs", type=int, default=None, help="覆盖每组参数的蒙特卡洛试验次数。")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="基础随机种子。")
    parser.add_argument("--output-dir", default="", help="自定义输出目录。")
    parser.add_argument("--sensor-count", action="append", help="覆盖传感器数量，可重复传入或逗号分隔。")
    parser.add_argument("--sensor-pd", action="append", help="覆盖单传感器检测率。")
    parser.add_argument("--sensor-pos-std", action="append", help="覆盖单传感器 XYZ 标准差，单位米。")
    parser.add_argument("--sensor-fa-prob", action="append", help="覆盖输入虚警概率。")
    args = parser.parse_args()
    args.sensor_count = parse_int_list(args.sensor_count)
    args.sensor_pd = parse_float_list(args.sensor_pd)
    args.sensor_pos_std = parse_float_list(args.sensor_pos_std)
    args.sensor_fa_prob = parse_float_list(args.sensor_fa_prob)
    return args


def main() -> int:
    args = parse_args()
    run_study(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
