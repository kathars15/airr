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


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGE_ROOT = SCRIPT_DIR
SRC_DIR = SCRIPT_DIR.parents[2]
MHT_BIAS_DIR = SRC_DIR.parent
WORKSPACE_DIR = MHT_BIAS_DIR.parents[2]

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
# 每次扫描之间的时间间隔，单位秒。
# 这个值越大，目标在相邻两帧之间位移越大，对关联和滤波更不友好。

SIM_DURATION_SEC = 380.0
# 单次仿真的总时长，单位秒。
# 越大，统计越稳定，但总运行时间也越长。

WARMUP_SEC = 10.0
# 预热时间，单位秒。
# 这段时间内的输出不计入检测率/RMSE/虚警率统计，用来给 MHT 充足时间完成航迹起始和稳定。

MONTE_CARLO_RUNS = 3
# 每组参数重复仿真的次数。
# 这个值越大，结果越稳健，但完整网格运行会明显变慢。

RANDOM_SEED = 20260605
# 基础随机种子。
# 不同参数组合会在这个基础上派生自己的种子，因此同一份代码和同一组参数是可复现的。

TARGET_COUNT = 2
# 每次场景中真实目标的数量。
# 改大后会增加多目标关联压力，也会更容易暴露虚警和串轨问题。

SENSOR_COUNT_RANGE = (5, 5)
SENSOR_COUNT_STEPS = 1
# 需要扫描的“传感器数量”列表。
# 完整网格会依次测试这些传感器数量配置。

SENSOR_PD_RANGE = (0.7, 0.95)
SENSOR_PD_STEPS = 5
# 需要扫描的“单传感器检测率”列表。
# 例如 0.80 表示每个真实目标在单个传感器上有 80% 概率被探测到。

SENSOR_POS_STD_RANGE_M = (15.0, 15.0)
SENSOR_POS_STD_STEPS = 1
# 需要扫描的“单传感器三维位置量测标准差”列表，单位米。
# 这里存的是标准差，不是方差；协方差矩阵 R 会在代码里自动平方后构造。

SENSOR_FA_PROB_RANGE = (0.25, 0.25)
SENSOR_FA_PROB_STEPS = 1
# 需要扫描的“单传感器输入虚警概率”列表。
# 例如 0.20 表示一个传感器在一帧内以 20% 概率产生一个随机虚假 XYZ 点。

DEFAULT_SENSOR_POS_STD_M = 30.0
# 默认展示用的单传感器位置误差标准差，单位米。
# 主要用于选取默认热力图切片、默认报告说明和代表性结果展示。

REQUIRED_DETECTION_RATE = 0.9
# 融合输出检测率的达标门槛。

REQUIRED_RMSE_M = 15.0
# 融合输出三维位置 RMSE 的达标门槛，单位米。

REQUIRED_FALSE_ALARM_RATE = 0.05
# 融合输出虚警率的达标门槛。

MATCH_GATE_ROOT_SCALE = 0.9
# 输出航迹与真实目标做评估匹配时的门限缩放系数。
# 评估门限不再写死为固定米数，而是按 “sqrt(3) * sigma * 系数” 自动计算。
# 这里取 0.90，表示门限略小于三维位置误差的典型根尺度。

CLUSTER_EPS_SIGMA = 3.5
# 聚类半径与量测标准差的比例系数。
# 实际聚类半径会取 max(CLUSTER_MIN_DISTANCE_M, CLUSTER_EPS_SIGMA * sigma)。

CLUSTER_MIN_DISTANCE_M = 20.0
# 聚类的最小距离门限，单位米。
# 即使量测误差很小，也不会把 DBSCAN 半径设得比这个值更小。

AREA_EAST_LIMIT_M = (-1500.0, 1500.0)
AREA_NORTH_LIMIT_M = (-1500.0, 1500.0)
ALTITUDE_LIMIT_M = (30.0, 300.0)
# 真实目标活动区域的东西/南北/高度范围，单位米。
# 目标超出边界后会做“反弹”处理，避免太快飞出仿真区域。

FALSE_ALARM_ALTITUDE_LIMIT_M = (20.0, 350.0)
# 输入虚假点允许出现的高度范围，单位米。
# 可以略宽于真实目标高度范围，用于模拟杂波/误检。

MULTIROTOR_LOOP_SIDE_M = 960.0
# 多旋翼“口”字回环的正方形边长，单位米。
# 当前 180s 仿真时长下，如果边长过大（例如 700m），目标通常跑不完一整圈，就不会呈现闭环。

MULTIROTOR_CRUISE_SPEED_RANGE_MPS = (7.0, 12.0)
# 多旋翼沿正方形边飞行时的巡航速度范围，单位米每秒。

MULTIROTOR_WAYPOINT_SWITCH_RADIUS_M = 200.0
# 接近顶点多少米时切换到下一个顶点。
# 值越大，转弯更早开始；值越小，轨迹更贴近几何顶点。

MULTIROTOR_MAX_TURN_PER_STEP_RAD = 0.22
# 每个仿真步允许的最大转向量，单位弧度。
# 值越大，拐角更容易拐过去；值越小，更平滑但也更容易拐不过去。

MULTIROTOR_ALTITUDE_WAVE_AMPLITUDE_M = 0.08
# 多旋翼高度小幅起伏幅度，单位米。

MULTIROTOR_ALTITUDE_SPEED_LIMIT_MPS = 0.12
# 多旋翼垂直速度上限，单位米每秒。

PLOT_Z_RANGE_EXPAND_FACTOR = 2.0
# 3D 轨迹图中 Z 轴的视觉压缩系数。
# 值越小，Z 轴在图上越扁，更接近俯视 XY 平面图；值越大，Z 轴看起来越高。

# MHT 参数。这里沿用现有在线/回放链路的风格，但仅在本脚本内部生效，
# 这样仿真研究不会影响线上行为。
MHT_LAMBDA_NT = 0.01
# 新航迹先验强度。
# 改大后更容易起新轨，也更容易把杂波解释成新目标。

MHT_MAX_VEL_MPS = 30.0
# 目标最大速度先验，单位米每秒。
# 这个值越大，初始速度协方差越宽，关联包络通常也会更松。

MHT_N_SCAN = 1
# N-scan 剪枝深度。
# 值越大，假设树保留更久，但计算量会明显增长。

MHT_PG = 0.90
# 关联门限对应的概率参数。
# 越大，马氏距离门限越宽，越容易关联上远一点的量测。

MHT_P_DEATH = 1e-2
# 航迹死亡先验概率。
# 越大，漏检后航迹更容易被判定为结束。

MHT_RESOLVED_TIME_WINDOW_SEC = 6.0
# 航迹确认时间窗，单位秒。
# 在这个时间窗内满足一定检测次数后，航迹才会进入确认状态。

MHT_RESOLVED_MIN_DETECT = 3
# 航迹确认所需的最小检测次数。
# 当前设为 1，表示更偏向快速确认；如果改大，会更保守。

MHT_MAX_DETECT_TIME_SEC = 3.0
# 允许航迹最长连续“无有效检测”的保留时间，单位秒。

MHT_Q_SCALE = 0.22
# 目标运动过程噪声强度。
# 越大，滤波对机动更宽容，但轨迹会更发散；越小，轨迹更平滑但更容易跟丢机动目标。

MHT_VOLUME_M3 = (
    (AREA_EAST_LIMIT_M[1] - AREA_EAST_LIMIT_M[0])
    * (AREA_NORTH_LIMIT_M[1] - AREA_NORTH_LIMIT_M[0])
    * (FALSE_ALARM_ALTITUDE_LIMIT_M[1] - FALSE_ALARM_ALTITUDE_LIMIT_M[0])
)
# 量测体积，用于把虚警强度归一到空间体积中。
# 一般不需要频繁改，除非你明显扩大或缩小了仿真区域。

OUTPUT_DIR = SCRIPT_DIR / "outputs"
# 默认输出目录。
# 所有 CSV、JSON、热力图、代表性轨迹图都会写到这里。


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
    start_pos: np.ndarray
    loop_vertices: List[np.ndarray]
    waypoint_idx: int
    cruise_speed: float
    base_altitude: float


@dataclass
class RunMetrics:
    detection_rate: float
    false_alarm_rate: float
    rmse_m: Optional[float]
    rmse_z_m: Optional[float]
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


SWEEP_FIELD_TO_ROW_KEY = {
    "sensor_count": "sensor_count",
    "sensor_pd": "sensor_pd",
    "sensor_pos_std": "sensor_pos_std_m",
    "sensor_pos_std_m": "sensor_pos_std_m",
    "sensor_fa_prob": "sensor_fa_prob",
}

SWEEP_FIELD_TO_ATTR = {
    "sensor_count": "sensor_count",
    "sensor_pd": "sensor_pd",
    "sensor_pos_std": "sensor_pos_std_m",
    "sensor_pos_std_m": "sensor_pos_std_m",
    "sensor_fa_prob": "sensor_fa_prob",
}

SWEEP_FIELD_LABELS = {
    "sensor_count": "Sensor count",
    "sensor_pd": "Single-sensor Pd",
    "sensor_pos_std_m": "Single-sensor sigma (m)",
    "sensor_fa_prob": "Input false-alarm probability",
}


def _float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def clamp(value: float, limits: Tuple[float, float]) -> float:
    return min(max(float(value), limits[0]), limits[1])


def compute_match_gate_m(sensor_pos_std_m: float) -> float:
    """根据单传感器位置标准差计算评估匹配门限。

    三维独立同方差误差下，欧式误差的典型根尺度可近似看作 sqrt(3) * sigma。
    这里再乘一个略小于 1 的系数，让评估门限比该尺度稍严格一点。
    """
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


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def generate_initial_targets(rng: np.random.Generator, target_count: int) -> List[TargetState]:
    # target_types = ["multirotor_uav", "fixed_wing_small", "bird_slow"]
    target_types = ["multirotor_uav"]
    targets: List[TargetState] = []
    for target_id in range(target_count):
        target_type = target_types[target_id % len(target_types)]
        if target_type == "multirotor_uav":
            # 使用正方形回环，保证在当前 180s 仿真时长内能明显形成闭环“口”字轨迹。
            loop_side = MULTIROTOR_LOOP_SIDE_M
            start_pos = np.array(
                [
                    rng.uniform(AREA_EAST_LIMIT_M[0] + 120.0, AREA_EAST_LIMIT_M[1] - loop_side - 120.0),
                    rng.uniform(AREA_NORTH_LIMIT_M[0] + 120.0, AREA_NORTH_LIMIT_M[1] - loop_side - 120.0),
                    rng.uniform(120.0, 240.0),
                ],
                dtype=float,
            )
            loop_vertices = [
                start_pos.copy(),
                start_pos + np.array([loop_side, 0.0, 0.0], dtype=float),
                start_pos + np.array([loop_side, loop_side, 0.0], dtype=float),
                start_pos + np.array([0.0, loop_side, 0.0], dtype=float),
            ]
            cruise_speed = rng.uniform(*MULTIROTOR_CRUISE_SPEED_RANGE_MPS)
            heading0 = math.atan2(
                loop_vertices[1][1] - start_pos[1],
                loop_vertices[1][0] - start_pos[0],
            )
            speed = cruise_speed
            vz = 0.0
            turn_rate = 0.12
            turn_bias = 0.0
        elif target_type == "fixed_wing_small":
            start_pos = np.array(
                [
                    rng.uniform(*AREA_EAST_LIMIT_M) * 0.75,
                    rng.uniform(*AREA_NORTH_LIMIT_M) * 0.75,
                    rng.uniform(*ALTITUDE_LIMIT_M),
                ],
                dtype=float,
            )
            direction = random_unit_xy(rng)
            speed = rng.uniform(18.0, 45.0)
            vz = rng.uniform(-2.0, 2.0)
            turn_rate = rng.uniform(-0.025, 0.025)
            turn_bias = 0.0
            loop_vertices = [start_pos.copy()]
            cruise_speed = speed
            heading0 = math.atan2(direction[1], direction[0])
        else:
            start_pos = np.array(
                [
                    rng.uniform(*AREA_EAST_LIMIT_M) * 0.75,
                    rng.uniform(*AREA_NORTH_LIMIT_M) * 0.75,
                    rng.uniform(*ALTITUDE_LIMIT_M),
                ],
                dtype=float,
            )
            direction = random_unit_xy(rng)
            speed = rng.uniform(2.0, 15.0)
            vz = rng.uniform(-2.5, 2.5)
            turn_rate = rng.uniform(-0.10, 0.10)
            turn_bias = 0.0
            loop_vertices = [start_pos.copy()]
            cruise_speed = speed
            heading0 = math.atan2(direction[1], direction[0])

        vel = np.array([math.cos(heading0) * speed, math.sin(heading0) * speed, vz], dtype=float)
        targets.append(
            TargetState(
                target_id=target_id,
                target_type=target_type,
                pos=start_pos.copy(),
                vel=vel,
                phase=rng.uniform(0.0, 2.0 * math.pi),
                turn_rate=turn_rate,
                turn_bias=turn_bias,
                start_pos=start_pos.copy(),
                loop_vertices=loop_vertices,
                waypoint_idx=1 if target_type == "multirotor_uav" else 0,
                cruise_speed=cruise_speed,
                base_altitude=float(start_pos[2]),
            )
        )
    return targets


def propagate_targets(targets: Sequence[TargetState], dt: float, rng: np.random.Generator, step_idx: int) -> List[TargetState]:
    updated: List[TargetState] = []
    for target in targets:
        pos = target.pos.copy()
        vel = target.vel.copy()
        phase = target.phase + 0.15 * dt
        speed_xy = float(np.linalg.norm(vel[:2]))
        heading = math.atan2(vel[1], vel[0])
        turn_bias = target.turn_bias

        if target.target_type == "multirotor_uav":
            # 多旋翼按四个顶点依次飞行，形成“口”字形回环轨迹。
            current_wp = target.loop_vertices[target.waypoint_idx]
            delta_xy = current_wp[:2] - pos[:2]
            distance_xy = float(np.linalg.norm(delta_xy))

            waypoint_idx = target.waypoint_idx
            if distance_xy < MULTIROTOR_WAYPOINT_SWITCH_RADIUS_M:
                waypoint_idx = (target.waypoint_idx + 1) % len(target.loop_vertices)
                current_wp = target.loop_vertices[waypoint_idx]
                delta_xy = current_wp[:2] - pos[:2]
                distance_xy = float(np.linalg.norm(delta_xy))

            desired_heading = math.atan2(delta_xy[1], delta_xy[0])
            heading_error = wrap_angle_rad(desired_heading - heading)
            max_turn_per_step = MULTIROTOR_MAX_TURN_PER_STEP_RAD
            heading += clamp(heading_error, (-max_turn_per_step, max_turn_per_step)) + rng.normal(0.0, 0.001)

            speed_xy = clamp(
                speed_xy + 0.35 * (target.cruise_speed - speed_xy) + rng.normal(0.0, 0.04),
                (6.0, 10.0),
            )
            desired_z = target.base_altitude + MULTIROTOR_ALTITUDE_WAVE_AMPLITUDE_M * math.sin(phase * 0.18)
            z_error = desired_z - pos[2]
            vel[2] = clamp(
                0.18 * z_error + rng.normal(0.0, 0.015),
                (-MULTIROTOR_ALTITUDE_SPEED_LIMIT_MPS, MULTIROTOR_ALTITUDE_SPEED_LIMIT_MPS),
            )
            turn_bias = 0.0
        elif target.target_type == "fixed_wing_small":
            waypoint_idx = target.waypoint_idx
            heading += target.turn_rate * dt + 0.004 * math.sin(phase)
            speed_xy = clamp(speed_xy + rng.normal(0.0, 0.5), (16.0, 48.0))
            vel[2] = clamp(1.5 * math.sin(phase * 0.35), (-2.5, 2.5))
        else:
            waypoint_idx = target.waypoint_idx
            heading += target.turn_rate * dt + rng.normal(0.0, 0.04)
            speed_xy = clamp(speed_xy + rng.normal(0.0, 1.2), (1.5, 18.0))
            vel[2] = clamp(1.6 * math.sin(phase) + rng.normal(0.0, 0.4), (-3.0, 3.0))

        vel[0] = speed_xy * math.cos(heading)
        vel[1] = speed_xy * math.sin(heading)
        pos = pos + vel * dt
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
                start_pos=target.start_pos.copy(),
                loop_vertices=[vertex.copy() for vertex in target.loop_vertices],
                waypoint_idx=waypoint_idx,
                cruise_speed=target.cruise_speed,
                base_altitude=target.base_altitude,
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
        min_samples=2,
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
    z_errors: List[float] = []
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
        for target, aligned_output in zip(targets, aligned_outputs):
            if aligned_output is not None:
                z_errors.append(float(aligned_output[2] - target.pos[2]))
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

    if z_errors:
        z_errors_np = np.asarray(z_errors, dtype=float)
        rmse_z_m = float(np.sqrt(np.mean(np.square(z_errors_np))))
    else:
        rmse_z_m = None

    return RunMetrics(
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        rmse_m=rmse_m,
        rmse_z_m=rmse_z_m,
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
    rmse_z_m = mean_attr("rmse_z_m")
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
        "rmse_z_m": _float_or_none(rmse_z_m),
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
        "trial_rmse_z_m_mean": mean_attr("rmse_z_m"),
    }


def iter_configs(args: argparse.Namespace) -> Iterable[ScenarioConfig]:
    # quick ??????????????????????????
    if args.quick:
        sensor_counts = [1, 3]
        sensor_pd_values = [0.80, 0.95]
        sensor_pos_values = [15.0, DEFAULT_SENSOR_POS_STD_M]
        sensor_fa_values = [0.20, 0.30]
    else:
        sensor_counts = expand_range_to_values(SENSOR_COUNT_RANGE, SENSOR_COUNT_STEPS, as_int=True)
        sensor_pd_values = expand_range_to_values(SENSOR_PD_RANGE, SENSOR_PD_STEPS, as_int=False)
        sensor_pos_values = expand_range_to_values(SENSOR_POS_STD_RANGE_M, SENSOR_POS_STD_STEPS, as_int=False)
        sensor_fa_values = expand_range_to_values(SENSOR_FA_PROB_RANGE, SENSOR_FA_PROB_STEPS, as_int=False)

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


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_sort_value(row: Dict[str, object], field: str) -> float:
    return float(row[SWEEP_FIELD_TO_ROW_KEY[field]])


def score_fail_case(row: Dict[str, object]) -> float:
    return (
        abs(float(row["detection_rate"]) - REQUIRED_DETECTION_RATE)
        + abs((float(row["rmse_m"]) if row["rmse_m"] is not None else REQUIRED_RMSE_M * 3) - REQUIRED_RMSE_M) / 15.0
        + abs(float(row["false_alarm_rate"]) - REQUIRED_FALSE_ALARM_RATE) * 4.0
    )


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


def summarize_stability_intervals(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    passed = [row for row in rows if row.get("pass_requirement")]
    summary: Dict[str, object] = {
        "total_cases": len(rows),
        "pass_cases": len(passed),
        "pass_ratio": (len(passed) / len(rows)) if rows else 0.0,
        "sensor_count_range": None,
        "sensor_pd_range": None,
        "sensor_pos_std_range_m": None,
        "sensor_fa_prob_range": None,
    }
    if not passed:
        return summary

    def _range(field: str) -> Tuple[float, float]:
        values = [float(row[field]) for row in passed]
        return min(values), max(values)

    sensor_counts = sorted({int(row["sensor_count"]) for row in passed})
    summary["sensor_count_range"] = (min(sensor_counts), max(sensor_counts))
    summary["sensor_pd_range"] = _range("sensor_pd")
    summary["sensor_pos_std_range_m"] = _range("sensor_pos_std_m")
    summary["sensor_fa_prob_range"] = _range("sensor_fa_prob")
    return summary


def build_failure_reason_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    summary = {
        "fail_detection_only": 0,
        "fail_rmse_only": 0,
        "fail_false_alarm_only": 0,
        "fail_multiple_conditions": 0,
        "pass": 0,
    }
    for row in rows:
        cond_detection = float(row["detection_rate"]) >= REQUIRED_DETECTION_RATE
        cond_rmse = row["rmse_m"] is not None and float(row["rmse_m"]) <= REQUIRED_RMSE_M
        cond_false_alarm = float(row["false_alarm_rate"]) <= REQUIRED_FALSE_ALARM_RATE
        fail_count = sum([not cond_detection, not cond_rmse, not cond_false_alarm])
        if fail_count == 0:
            summary["pass"] += 1
        elif fail_count >= 2:
            summary["fail_multiple_conditions"] += 1
        elif not cond_detection:
            summary["fail_detection_only"] += 1
        elif not cond_rmse:
            summary["fail_rmse_only"] += 1
        elif not cond_false_alarm:
            summary["fail_false_alarm_only"] += 1
    return summary


def choose_case(
    rows: Sequence[Dict[str, object]],
    pass_required: bool,
    sort_key,
) -> Optional[Dict[str, object]]:
    candidates = [row for row in rows if bool(row["pass_requirement"]) is pass_required]
    if not candidates:
        return None
    return sorted(candidates, key=sort_key)[0]


def choose_trace_cases(rows: Sequence[Dict[str, object]]) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    pass_case = choose_case(
        rows,
        True,
        lambda row: (
            int(row["sensor_count"]),
            -float(row["sensor_fa_prob"]),
            float(row["sensor_pos_std_m"]),
            -float(row["sensor_pd"]),
        ),
    )
    if pass_case is not None:
        return pass_case, choose_case(rows, False, score_fail_case)

    fail_rows = [row for row in rows if not bool(row["pass_requirement"])]
    if not fail_rows:
        return None, None
    closest_fail = sorted(fail_rows, key=score_fail_case)[0]
    worst_fail = sorted(
        fail_rows,
        key=lambda row: (
            float(row["detection_rate"]),
            -(float(row["rmse_m"]) if row["rmse_m"] is not None else REQUIRED_RMSE_M * 3),
            -float(row["false_alarm_rate"]),
        ),
    )[0]
    return closest_fail, worst_fail


def write_summary_report(
    output_dir: Path,
    rows: Sequence[Dict[str, object]],
    boundary_rows: Sequence[Dict[str, object]],
) -> None:
    stability = summarize_stability_intervals(rows)
    failure_summary = build_failure_reason_summary(rows)
    recommended = boundary_rows[-1] if boundary_rows else None
    pass_case = choose_case(
        rows,
        True,
        lambda row: (
            int(row["sensor_count"]),
            -float(row["sensor_fa_prob"]),
            float(row["sensor_pos_std_m"]),
            -float(row["sensor_pd"]),
        ),
    )
    fail_case = choose_case(
        rows,
        False,
        lambda row: (
            abs(float(row["detection_rate"]) - REQUIRED_DETECTION_RATE)
            + abs((float(row["rmse_m"]) if row["rmse_m"] is not None else REQUIRED_RMSE_M * 3) - REQUIRED_RMSE_M) / 15.0
            + abs(float(row["false_alarm_rate"]) - REQUIRED_FALSE_ALARM_RATE) * 4.0
        ),
    )

    def fmt_range(value: Optional[Tuple[float, float]], unit: str = "") -> str:
        if value is None:
            return "无"
        suffix = f" {unit}" if unit else ""
        return f"{value[0]} ~ {value[1]}{suffix}"

    max_pass_fa = None
    passed = [row for row in rows if row.get("pass_requirement")]
    if passed:
        max_pass_fa = max(float(row["sensor_fa_prob"]) for row in passed)

    lines = [
        "# MHT 输入性能需求测试总结报告",
        "",
        "## 1. 输出指标要求",
        "",
        f"- 检测率要求：`Pd_out >= {REQUIRED_DETECTION_RATE:.2f}`",
        f"- 三维位置精度要求：`RMSE <= {REQUIRED_RMSE_M:.1f} m`",
        f"- 输出虚警率要求：`Pfa_out <= {REQUIRED_FALSE_ALARM_RATE:.2f}`",
        "",
        "## 2. 当前仿真运动场景",
        "",
        f"- 扫描周期：`{SCAN_PERIOD_SEC:.1f} s`",
        f"- 单次仿真时长：`{SIM_DURATION_SEC:.1f} s`",
        f"- 真实目标数量：`{TARGET_COUNT}`",
        f"- 当前默认目标类型：`multirotor_uav`",
        f"- 初始水平速度范围：`3.0 ~ 18.0 m/s`",
        f"- 垂直速度典型范围：约 `-0.8 ~ 0.8 m/s`",
        f"- 真实目标高度范围：`{ALTITUDE_LIMIT_M[0]:.0f} ~ {ALTITUDE_LIMIT_M[1]:.0f} m`",
        f"- 东向活动范围：`{AREA_EAST_LIMIT_M[0]:.0f} ~ {AREA_EAST_LIMIT_M[1]:.0f} m`",
        f"- 北向活动范围：`{AREA_NORTH_LIMIT_M[0]:.0f} ~ {AREA_NORTH_LIMIT_M[1]:.0f} m`",
        f"- 输入虚假点高度范围：`{FALSE_ALARM_ALTITUDE_LIMIT_M[0]:.0f} ~ {FALSE_ALARM_ALTITUDE_LIMIT_M[1]:.0f} m`",
        "- 运动方式：低空缓转、多帧连续推进、越界后做反弹处理。",
        "",
        "## 3. 总体结论",
        "",
        f"- 总测试组合数：`{stability['total_cases']}`",
        f"- 通过组合数：`{stability['pass_cases']}`",
        f"- 通过比例：`{stability['pass_ratio']:.2%}`",
    ]

    if recommended:
        lines.extend(
            [
                f"- 推荐保守配置：`N >= {recommended['min_sensor_count']}`，`Pd >= {float(recommended['min_sensor_pd_in_pass_set']):.2f}`，`sigma <= {float(recommended['sensor_pos_std_m']):.1f}m`",
                f"- 在当前通过组合中，最高验证通过的输入虚警率：`{max_pass_fa:.2f}`" if max_pass_fa is not None else "- 当前没有通过组合，因此无法给出输入虚警率上限。",
            ]
        )
    else:
        lines.append("- 当前测试范围内没有找到同时满足三项输出指标的输入组合。")

    lines.extend(
        [
            "",
            "## 4. 稳定通过区间总结",
            "",
            f"- 传感器数量区间：`{fmt_range(stability['sensor_count_range'])}`",
            f"- 单传感器检测率区间：`{fmt_range(stability['sensor_pd_range'])}`",
            f"- 单传感器位置误差标准差区间：`{fmt_range(stability['sensor_pos_std_range_m'], 'm')}`",
            f"- 单传感器输入虚警率区间：`{fmt_range(stability['sensor_fa_prob_range'])}`",
            "",
            "## 5. 不通过原因分布",
            "",
            f"- 仅检测率不达标：`{failure_summary['fail_detection_only']}` 组",
            f"- 仅 RMSE 不达标：`{failure_summary['fail_rmse_only']}` 组",
            f"- 仅虚警率不达标：`{failure_summary['fail_false_alarm_only']}` 组",
            f"- 多项同时不达标：`{failure_summary['fail_multiple_conditions']}` 组",
            f"- 通过：`{failure_summary['pass']}` 组",
            "",
            "## 6. 代表性案例",
            "",
        ]
    )

    if pass_case:
        lines.extend(
            [
                "### 通过案例",
                "",
                f"- 输入：`N={pass_case['sensor_count']}`，`Pd={float(pass_case['sensor_pd']):.2f}`，`sigma={float(pass_case['sensor_pos_std_m']):.1f}m`，`Pfa_in={float(pass_case['sensor_fa_prob']):.2f}`",
                f"- 输出：`Pd_out={float(pass_case['detection_rate']):.3f}`，`RMSE={float(pass_case['rmse_m']):.2f}m`，`RMSE_Z={float(pass_case['rmse_z_m']):.2f}m`，`Pfa_out={float(pass_case['false_alarm_rate']):.3f}`",
                "",
            ]
        )

    if fail_case:
        lines.extend(
            [
                "### 临界失败案例",
                "",
                f"- 输入：`N={fail_case['sensor_count']}`，`Pd={float(fail_case['sensor_pd']):.2f}`，`sigma={float(fail_case['sensor_pos_std_m']):.1f}m`，`Pfa_in={float(fail_case['sensor_fa_prob']):.2f}`",
                f"- 输出：`Pd_out={float(fail_case['detection_rate']):.3f}`，`RMSE={float(fail_case['rmse_m'] if fail_case['rmse_m'] is not None else float('nan')):.2f}m`，`RMSE_Z={float(fail_case['rmse_z_m'] if fail_case['rmse_z_m'] is not None else float('nan')):.2f}m`，`Pfa_out={float(fail_case['false_alarm_rate']):.3f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 7. 敏感性结论",
            "",
            "- 检测率主要受传感器数量和单传感器检测率影响。",
            "- 三维位置 RMSE 主要受单传感器位置误差标准差影响。",
            "- 输出虚警率主要受输入虚警率和 MHT 起轨/确认参数影响。",
            "- 在当前调参版本里，最容易首先失效的指标通常是 `Pfa_out`。",
            "",
            "## 8. 交付建议",
            "",
            "- 对外交付时建议重点保留本总结报告和轨迹图。",
            "- 轨迹图文件为：`trajectory_compare_3d.png`、`trajectory_compare_timeseries.png`。",
            "- 如果需要复现实验，可直接重新运行当前目录中的 `run_requirement_sim.py`。",
            "",
        ]
    )

    (output_dir / "simulation_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    x_all: List[float] = []
    y_all: List[float] = []
    z_all: List[float] = []
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
        x_all.extend([value for value in e_truth if not np.isnan(value)])
        x_all.extend([value for value in e_track if not np.isnan(value)])
        y_all.extend([value for value in n_truth if not np.isnan(value)])
        y_all.extend([value for value in n_track if not np.isnan(value)])
        z_all.extend([value for value in u_truth if not np.isnan(value)])
        z_all.extend([value for value in u_track if not np.isnan(value)])
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    if x_all and y_all and z_all:
        x_span = max(max(x_all) - min(x_all), 1.0)
        y_span = max(max(y_all) - min(y_all), 1.0)
        z_span = max(max(z_all) - min(z_all), 1.0)
        # 仅压缩显示比例，让 Z 方向看起来不过度夸张；不改变任何真实数据。
        # 仅调整 3D 图中的 Z 轴视觉比例，不改变真实轨迹数据。
        ax.set_box_aspect((x_span, y_span, z_span * max(PLOT_Z_RANGE_EXPAND_FACTOR, 1e-3)))
    ax.set_title(
        "3D trajectory comparison\n"
        f"N={config.sensor_count}, Pd={config.sensor_pd:.2f}, sigma={config.sensor_pos_std_m:.1f}m, "
        f"Pfa={config.sensor_fa_prob:.2f}, RMSE={metrics.rmse_m:.2f}m, RMSE_Z={metrics.rmse_z_m:.2f}m"
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
        f"Pfa={config.sensor_fa_prob:.2f}, Pd_out={metrics.detection_rate:.3f}, Pfa_out={metrics.false_alarm_rate:.3f}, RMSE_Z={metrics.rmse_z_m:.2f}m"
    )
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_z_error_timeseries(
    trace_frames: Sequence[TraceFrame],
    config: ScenarioConfig,
    metrics: RunMetrics,
    output_path: Path,
) -> None:
    if not trace_frames:
        return
    target_count = len(trace_frames[0].matched_truth_positions)
    times = [frame.timestamp for frame in trace_frames]
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.5), sharex=True)
    truth_colors = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]
    track_colors = ["tab:red", "tab:orange", "tab:pink", "tab:gray", "gold", "black"]

    for target_idx in range(target_count):
        z_truth = _series_from_trace(trace_frames, target_idx, 2, matched=False)
        z_track = _series_from_trace(trace_frames, target_idx, 2, matched=True)
        z_error = []
        for truth_value, track_value in zip(z_truth, z_track):
            if np.isnan(truth_value) or np.isnan(track_value):
                z_error.append(float("nan"))
            else:
                z_error.append(float(track_value - truth_value))

        axes[0].plot(times, z_truth, color=truth_colors[target_idx % len(truth_colors)], linewidth=2.0, label=f"Truth T{target_idx}")
        axes[0].plot(
            times,
            z_track,
            color=track_colors[target_idx % len(track_colors)],
            linestyle="--",
            linewidth=1.6,
            label=f"MHT T{target_idx}",
        )
        axes[1].plot(
            times,
            z_error,
            color=track_colors[target_idx % len(track_colors)],
            linewidth=1.8,
            label=f"Z error T{target_idx}",
        )

    axes[0].set_ylabel("Up (m)")
    axes[0].set_title(
        "Z direction trajectory and error by time\n"
        f"N={config.sensor_count}, Pd={config.sensor_pd:.2f}, sigma={config.sensor_pos_std_m:.1f}m, "
        f"Pfa={config.sensor_fa_prob:.2f}, RMSE_Z={metrics.rmse_z_m:.2f}m"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8, loc="upper right")

    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Z error (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_representative_trace(
    output_dir: Path,
    config: ScenarioConfig,
    seed: int,
    prefix: str = "trajectory_compare",
) -> None:
    metrics, trace_frames = run_single_trial(config, seed, record_trace=True)
    if not trace_frames:
        return
    stale_z_plot = output_dir / f"{prefix}_z_error_timeseries.png"
    if stale_z_plot.exists():
        stale_z_plot.unlink()
    plot_trace_comparison_3d(trace_frames, config, metrics, output_dir / f"{prefix}_3d.png")
    plot_trace_comparison_timeseries(trace_frames, config, metrics, output_dir / f"{prefix}_timeseries.png")


def export_trace_case_from_row(
    output_dir: Path,
    row: Dict[str, object],
    base_seed: int,
    prefix: str,
) -> None:
    config = ScenarioConfig(
        sensor_count=int(row["sensor_count"]),
        sensor_pd=float(row["sensor_pd"]),
        sensor_pos_std_m=float(row["sensor_pos_std_m"]),
        sensor_fa_prob=float(row["sensor_fa_prob"]),
    )
    seed = (
        base_seed
        + config.sensor_count * 17
        + int(config.sensor_pd * 1000) * 19
        + int(config.sensor_pos_std_m * 10) * 23
        + int(config.sensor_fa_prob * 1000) * 29
    )
    export_representative_trace(output_dir, config, seed, prefix=prefix)


def format_metric(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def plot_sweep_metric(
    rows: Sequence[Dict[str, object]],
    metric: str,
    title: str,
    output_path: Path,
    sweep_field: str,
    fixed_config: ScenarioConfig,
    threshold: Optional[float] = None,
) -> None:
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda row: stable_sort_value(row, sweep_field))
    x_values = [stable_sort_value(row, sweep_field) for row in sorted_rows]
    y_values = [float("nan") if row.get(metric) is None else float(row[metric]) for row in sorted_rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.plot(x_values, y_values, marker="o", linewidth=2.0, color="tab:blue")
    if threshold is not None:
        ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_xlabel(SWEEP_FIELD_LABELS[sweep_field])
    ax.set_ylabel(title)
    ax.set_title(
        f"{title}\nsweep {sweep_field}, fixed N={fixed_config.sensor_count}, "
        f"Pd={fixed_config.sensor_pd:.2f}, sigma={fixed_config.sensor_pos_std_m:.1f}m, "
        f"Pfa_in={fixed_config.sensor_fa_prob:.2f}"
    )
    ax.grid(True, alpha=0.25)
    for x_value, y_value in zip(x_values, y_values):
        if np.isnan(y_value):
            continue
        label = f"{y_value:.2f}" if metric not in {"rmse_m", "rmse_z_m"} else f"{y_value:.1f}"
        ax.text(x_value, y_value, label, fontsize=8, ha="center", va="bottom")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_sweep_field(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None
    key = str(raw_value).strip()
    if key not in SWEEP_FIELD_TO_ROW_KEY:
        raise ValueError(
            f"unsupported --sweep-field: {key}. "
            "use one of sensor_count, sensor_pd, sensor_pos_std, sensor_pos_std_m, sensor_fa_prob"
        )
    return SWEEP_FIELD_TO_ROW_KEY[key]


def detect_default_sweep_field(args: argparse.Namespace) -> Optional[str]:
    # If user explicitly overrides any scan list from CLI, keep old behavior unless
    # they also explicitly request sweep mode.
    if any([
        args.sensor_count,
        args.sensor_pd,
        args.sensor_pos_std,
        args.sensor_fa_prob,
    ]):
        return None

    variable_fields: List[str] = []
    if len(expand_range_to_values(SENSOR_COUNT_RANGE, SENSOR_COUNT_STEPS, as_int=True)) > 1:
        variable_fields.append("sensor_count")
    if len(expand_range_to_values(SENSOR_PD_RANGE, SENSOR_PD_STEPS, as_int=False)) > 1:
        variable_fields.append("sensor_pd")
    if len(expand_range_to_values(SENSOR_POS_STD_RANGE_M, SENSOR_POS_STD_STEPS, as_int=False)) > 1:
        variable_fields.append("sensor_pos_std_m")
    if len(expand_range_to_values(SENSOR_FA_PROB_RANGE, SENSOR_FA_PROB_STEPS, as_int=False)) > 1:
        variable_fields.append("sensor_fa_prob")

    if len(variable_fields) == 1:
        return variable_fields[0]
    return None


def get_fixed_sweep_config(args: argparse.Namespace, sweep_field: str) -> ScenarioConfig:
    default_values = {
        "sensor_count": expand_range_to_values(SENSOR_COUNT_RANGE, SENSOR_COUNT_STEPS, as_int=True),
        "sensor_pd": expand_range_to_values(SENSOR_PD_RANGE, SENSOR_PD_STEPS, as_int=False),
        "sensor_pos_std": expand_range_to_values(SENSOR_POS_STD_RANGE_M, SENSOR_POS_STD_STEPS, as_int=False),
        "sensor_fa_prob": expand_range_to_values(SENSOR_FA_PROB_RANGE, SENSOR_FA_PROB_STEPS, as_int=False),
    }

    def _single_value(name: str, values: Optional[Sequence[float]], cast_fn):
        if SWEEP_FIELD_TO_ROW_KEY[name] == sweep_field:
            return None
        actual_values: Sequence[float] = values if values else default_values[name]
        if not actual_values or len(actual_values) != 1:
            raise ValueError(f"sweep mode requires non-sweep field {name} to be fixed to a single value")
        return cast_fn(actual_values[0])

    sensor_count = _single_value("sensor_count", args.sensor_count, int)
    sensor_pd = _single_value("sensor_pd", args.sensor_pd, float)
    sensor_pos_std_m = _single_value("sensor_pos_std", args.sensor_pos_std, float)
    sensor_fa_prob = _single_value("sensor_fa_prob", args.sensor_fa_prob, float)

    if sweep_field == "sensor_count":
        sensor_count = int(args.sweep_values[0])
    elif sweep_field == "sensor_pd":
        sensor_pd = float(args.sweep_values[0])
    elif sweep_field == "sensor_pos_std_m":
        sensor_pos_std_m = float(args.sweep_values[0])
    elif sweep_field == "sensor_fa_prob":
        sensor_fa_prob = float(args.sweep_values[0])

    return ScenarioConfig(
        sensor_count=int(sensor_count),
        sensor_pd=float(sensor_pd),
        sensor_pos_std_m=float(sensor_pos_std_m),
        sensor_fa_prob=float(sensor_fa_prob),
    )


def iter_sweep_configs(args: argparse.Namespace, sweep_field: str) -> List[ScenarioConfig]:
    if not args.sweep_values:
        default_values = {
            "sensor_count": expand_range_to_values(SENSOR_COUNT_RANGE, SENSOR_COUNT_STEPS, as_int=True),
            "sensor_pd": expand_range_to_values(SENSOR_PD_RANGE, SENSOR_PD_STEPS, as_int=False),
            "sensor_pos_std_m": expand_range_to_values(SENSOR_POS_STD_RANGE_M, SENSOR_POS_STD_STEPS, as_int=False),
            "sensor_fa_prob": expand_range_to_values(SENSOR_FA_PROB_RANGE, SENSOR_FA_PROB_STEPS, as_int=False),
        }
        values = default_values[sweep_field]
        args.sweep_values = [float(value) for value in values]
    fixed = get_fixed_sweep_config(args, sweep_field)
    configs: List[ScenarioConfig] = []
    for value in args.sweep_values:
        config = ScenarioConfig(
            sensor_count=fixed.sensor_count,
            sensor_pd=fixed.sensor_pd,
            sensor_pos_std_m=fixed.sensor_pos_std_m,
            sensor_fa_prob=fixed.sensor_fa_prob,
        )
        setattr(config, SWEEP_FIELD_TO_ATTR[sweep_field], int(value) if sweep_field == "sensor_count" else float(value))
        configs.append(config)
    return configs


def run_sweep_study(args: argparse.Namespace, sweep_field: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = iter_sweep_configs(args, sweep_field)
    fixed_config = get_fixed_sweep_config(args, sweep_field)
    mc_runs = int(args.mc_runs if args.mc_runs is not None else MONTE_CARLO_RUNS)
    base_seed = int(args.seed)

    print(f"[仿真] 输出目录: {output_dir}")
    print(f"[仿真] 纵向对比模式: sweep {sweep_field}, 取值数={len(configs)}, 蒙特卡洛次数={mc_runs}, 随机种子={base_seed}")
    rows: List[Dict[str, object]] = []
    started = time.time()
    for idx, config in enumerate(configs, start=1):
        sweep_value = getattr(config, SWEEP_FIELD_TO_ATTR[sweep_field])
        match_gate_m = compute_match_gate_m(config.sensor_pos_std_m)
        print(
            f"[仿真] {idx}/{len(configs)} sweep {sweep_field}={sweep_value} | "
            f"N={config.sensor_count}, Pd={config.sensor_pd:.2f}, sigma={config.sensor_pos_std_m:.1f}m, "
            f"Pfa_in={config.sensor_fa_prob:.2f}, gate={match_gate_m:.1f}m"
        )
        trial_metrics: List[RunMetrics] = []
        for run_idx in range(mc_runs):
            seed = (
                base_seed
                + idx * 100000
                + run_idx * 1009
                + config.sensor_count * 17
                + int(config.sensor_pd * 1000) * 19
                + int(config.sensor_pos_std_m * 10) * 23
                + int(config.sensor_fa_prob * 1000) * 29
            )
            metrics, _ = run_single_trial(config, seed)
            trial_metrics.append(metrics)
        row = summarize_trials(config, trial_metrics)
        row["sweep_field"] = sweep_field
        row["sweep_value"] = float(sweep_value)
        rows.append(row)
        print(
            f"[仿真]   -> Pd_out={format_metric(row['detection_rate'])}, "
            f"RMSE={format_metric(row['rmse_m'], 2)}m, "
            f"RMSE_Z={format_metric(row['rmse_z_m'], 2)}m, "
            f"Pfa_out={format_metric(row['false_alarm_rate'])}, "
            f"{'PASS' if bool(row['pass_requirement']) else 'FAIL'}"
        )

    sorted_rows = sorted(rows, key=lambda row: stable_sort_value(row, sweep_field))
    plot_sweep_metric(sorted_rows, "detection_rate", "Output detection rate", output_dir / "sweep_detection_rate.png", sweep_field, fixed_config, threshold=REQUIRED_DETECTION_RATE)
    plot_sweep_metric(sorted_rows, "rmse_m", "3D position RMSE (m)", output_dir / "sweep_rmse.png", sweep_field, fixed_config, threshold=REQUIRED_RMSE_M)
    plot_sweep_metric(sorted_rows, "false_alarm_rate", "Output false-alarm rate", output_dir / "sweep_false_alarm_rate.png", sweep_field, fixed_config, threshold=REQUIRED_FALSE_ALARM_RATE)

    elapsed = time.time() - started
    print(f"[仿真] 纵向对比完成，总耗时 {elapsed:.1f}s")
    return sorted_rows, []


def run_study(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_z_plot = output_dir / "trajectory_z_error_timeseries.png"
    if stale_z_plot.exists():
        stale_z_plot.unlink()
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
            f"RMSE_Z={format_metric(row['rmse_z_m'], 2)}m, "
            f"Pfa_out={format_metric(row['false_alarm_rate'])}, {status}"
        )

        # 每跑完一组参数就落盘一次，这样长时间仿真中断后也能保留部分结果。
    boundary_rows = build_requirement_boundary(rows)
    best_case, borderline_case = choose_trace_cases(rows)
    if best_case is not None:
        export_trace_case_from_row(output_dir, best_case, base_seed, prefix="trajectory_compare")
    if borderline_case is not None:
        export_trace_case_from_row(output_dir, borderline_case, base_seed, prefix="trajectory_borderline")

    write_summary_report(output_dir, rows, boundary_rows)
    write_json(
        output_dir / "simulation_summary.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": build_settings_dict(mc_runs, base_seed),
            "rows": rows,
            "boundary_rows": boundary_rows,
            "stability_summary": summarize_stability_intervals(rows),
            "failure_summary": build_failure_reason_summary(rows),
        },
    )

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
        "sensor_count_range": SENSOR_COUNT_RANGE,
        "sensor_count_steps": SENSOR_COUNT_STEPS,
        "sensor_pd_range": SENSOR_PD_RANGE,
        "sensor_pd_steps": SENSOR_PD_STEPS,
        "sensor_pos_std_range_m": SENSOR_POS_STD_RANGE_M,
        "sensor_pos_std_steps": SENSOR_POS_STD_STEPS,
        "sensor_fa_prob_range": SENSOR_FA_PROB_RANGE,
        "sensor_fa_prob_steps": SENSOR_FA_PROB_STEPS,
        "sensor_count_values": expand_range_to_values(SENSOR_COUNT_RANGE, SENSOR_COUNT_STEPS, as_int=True),
        "sensor_pd_values": expand_range_to_values(SENSOR_PD_RANGE, SENSOR_PD_STEPS, as_int=False),
        "sensor_pos_std_values_m": expand_range_to_values(SENSOR_POS_STD_RANGE_M, SENSOR_POS_STD_STEPS, as_int=False),
        "sensor_fa_prob_values": expand_range_to_values(SENSOR_FA_PROB_RANGE, SENSOR_FA_PROB_STEPS, as_int=False),
        "default_sensor_pos_std_m": DEFAULT_SENSOR_POS_STD_M,
        "required_detection_rate": REQUIRED_DETECTION_RATE,
        "required_rmse_m": REQUIRED_RMSE_M,
        "required_false_alarm_rate": REQUIRED_FALSE_ALARM_RATE,
        "match_gate_rule": "MATCH_GATE_M = 0.90 * sqrt(3) * sigma",
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
    # 支持两种传参方式：
    # 1. --sensor-pd 0.8,0.9
    # 2. --sensor-pd 0.8 --sensor-pd 0.9
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


def expand_range_to_values(value_range: Tuple[float, float], steps: int, as_int: bool) -> List[float]:
    low = float(value_range[0])
    high = float(value_range[1])
    count = max(int(steps), 1)
    if count == 1 or abs(high - low) < 1e-12:
        return [int(round(low)) if as_int else low]
    values = np.linspace(low, high, count)
    if as_int:
        return sorted({int(round(value)) for value in values})
    return [float(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="?? MHT ?????????????")
    parser.add_argument("--quick", action="store_true", help="??????????????")
    parser.add_argument("--mc-runs", type=int, default=None, help="????????????????")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="???????")
    parser.add_argument("--output-dir", default="", help="????????")
    parser.add_argument("--sensor-count", action="append", help="???????????????????")
    parser.add_argument("--sensor-pd", action="append", help="??????????")
    parser.add_argument("--sensor-pos-std", action="append", help="?????? XYZ ????????")
    parser.add_argument("--sensor-fa-prob", action="append", help="?????????")
    parser.add_argument("--sweep-field", default="", help="??????????sensor_count / sensor_pd / sensor_pos_std / sensor_pos_std_m / sensor_fa_prob")
    parser.add_argument("--sweep-values", action="append", help="?????????????? 10,15,20 ??")
    args = parser.parse_args()
    args.sensor_count = parse_int_list(args.sensor_count)
    args.sensor_pd = parse_float_list(args.sensor_pd)
    args.sensor_pos_std = parse_float_list(args.sensor_pos_std)
    args.sensor_fa_prob = parse_float_list(args.sensor_fa_prob)
    args.sweep_values = parse_float_list(args.sweep_values)
    return args


def main() -> int:
    args = parse_args()
    sweep_field = parse_sweep_field(args.sweep_field if args.sweep_field else None)
    if sweep_field is None:
        sweep_field = detect_default_sweep_field(args)
    if sweep_field is not None:
        run_sweep_study(args, sweep_field)
    else:
        run_study(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
