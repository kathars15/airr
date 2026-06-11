from copy import deepcopy, copy
from typing import Dict, List, Sequence, Type, Optional
import numpy as np
from common.Tracker import Get_H_k, Get_F_G_CV, Get_F_G_CP, Get_F_G_Bias, \
    Cal_Gate, Cal_Radar_Volume, Get_F_G_CV_Bias, KF_Prediction_CV, KF_Update, Cross_Covariance_CKF
import cvxpy as cp
from scipy.linalg import cholesky, block_diag
from common.Plots import plot_hyp
from common.utlis import Cal_Target_In_Sensor_Volume, merge_lists
from collections import deque
from itertools import combinations
import os

## 主要修改：
# 1. 死亡假设需要 漏检时长大于预设时长 才会生成，防止航迹断得厉害
# 2. 航迹起始的 M of N 序列 [Bool Bool Bool Bool]: 若一节点由传感器A初始化，只有A的漏检才会导致一个0，其他传感器的漏检会跳过；\
#    但是所有传感器的检测都会导致一个1，这样便于航迹起始。
# 3. 最优假设求解时候，第二个量测约束也用等号，不然Radar航迹会直接被删掉【因为很乱？】
# 6. 输出航迹的要求：同时满足 航迹确认 + 当前生命值 == 最大生命值 （即漏检之后就暂时不输出）
# 7. 增加一个直接根据预测位置和量测值位置的波门，因为漏检几次之后协方差很大，容易关联到很远的
# 8. 航迹融合的阈值，提升到50m

class Node():
    def __init__(
        self,
        id: int,
        label: int,
        score: float,
        x_k_k_1: np.ndarray, # 根据当前帧的情况，可能是目标状态，也可能是目标状态+传感器偏差
        P_k_k_1: np.ndarray,
        obs_id = [], # 方便后期拓展到非点目标场景
        last_detect_tmp: float = None, # 该节点上一次被探测的时间戳
        hyp_type: str = 'detect' # initial detect miss-detect death
    ) -> None:
        self.id = id
        self.label = label
        self.score = score
        self.x_k_k_1 = x_k_k_1
        self.P_k_k_1 = P_k_k_1
        self.obs_id = obs_id
        self.last_detect_tmp = last_detect_tmp
        self.hyp_type = hyp_type

        self.hyp_ids = [id]
        self.parent_id = None
        self.children_ids = []
        self.initial_sensor_name = None # 初始该节点所用的传感器名字，用于针对性航迹管理的
        self.detection_times = [1] # 记录该航迹最近是否被探测，用于判断航迹确认输出的
        # 新增：记录每个detection_times对应的时间戳
        self.detection_timestamps = [last_detect_tmp] 
        self.track_resolved = False # 初始航迹都未确认

        self.debug_info = {}


class POMHT_Bias():
    def __init__(
        self,
        Lambda_NT: float, # prior number of new target per scan.
        obs_k: List[np.ndarray], # a list of current observation. obs_k[i] is m*1 np.ndarray meaning a single measurement.
        timestamp: float, # 第一帧的时间戳
        sensor_config: Dict, # 第一帧的传感器配置
        Q_k: np.ndarray, # 目标状态演化的过程噪声协方差
        Max_Vel: float = 20.0, # 目标最大速度，用于初始化初始状态协方差
        N_Scan: int = 1, # 剪枝参数
        Pg: float = 0.99, # 波门概率
        P_death: float = 0.01, # 航迹死亡概率

        #********逐点输入版本修改 第四处 修改航迹起始方式********
        # 原Resoled_M_N=[8,10]（10帧内8帧检测）改为时间窗口参数（逐点输入场景）
        Resolved_Time_Window: float = 0.64, # 航迹确认的时间窗口（单位：秒）
        Resolved_Min_Detect: int = 8, # 时间窗口内需要的最小检测点数（原8帧→8个检测点）
        #Resoled_M_N: list = [8, 10], # 航迹确认参数，最近4次被检测3次即确认

        dim_d: int = 3, # 空间维度
        pool_1 = None, # 是否多进程
        Debug_Params = {'Debug': True},  # 日志保存路径
        extra_infos = [], # 额外的量测信息，与单帧每个量测一一对应
        sav_traj_max_len = 100, # Decided_Tree保存最近多少个时刻的确认轨迹数据
        max_detect_time = 20, # 最多可以漏检多久 10s
        Merge_Threshold = 50, # 航迹融合阈值
    ) -> None:
        self.Lambda_NT = Lambda_NT
        self.Q_k = Q_k
        self.Max_Vel = Max_Vel
        self.N_Scan = N_Scan
        self.Tree_maxlen = N_Scan + 2  # length of track tree.
        self.Tree = [{}]  # track tree, length is self.Tree_maxlen.
        self.Decided_Tree = deque(maxlen = sav_traj_max_len)  # save the decided track.
        self.Output_Nodes = deque(maxlen = sav_traj_max_len)  # 输出的结果，可以基于 Decided_Tree 进一步修正.
        self.Leaf_Nodes = {}  # the leaf of self.Tree: key is id, value is node. 
        self.obs_k = obs_k
        self.R_k = sensor_config['R'] # 默认单帧每个量测都一样
        self.Pg = Pg
        self.P_death = P_death
        self.dim_d = dim_d
        self.label_max = 0
        self.max_ids = [-1, -1] # logging the maximum id used by each layer of Tree, used for providing id for combination or split hyps.
        self.Timestamps = [timestamp, timestamp] # logging the timestamp of Tree.
        self.obs_s = [[], obs_k] # logging obs of each layer of Tree.
        self.info_s = [[], extra_infos] # 额外的量测信息
        self.Sensor_Config = [{}, sensor_config]
        self.sensor_config = sensor_config
        self.Bias = {} # 存下所有传感器的偏差估计结果 self.Bias['b', 'Pb', 'timestamp']
        ## 如果首次出现某个带偏差传感器，并且考虑偏差，则先初始化其偏差
        if self.sensor_config['Is_Biased'] and not self.sensor_config['Biased_Ignore']:
            if self.sensor_config['Name'] not in self.Bias.keys():
                self.Bias[self.sensor_config['Name']] = {}
                self.Bias[self.sensor_config['Name']]['b'] = np.zeros((self.dim_d, 1)) 
                self.Bias[self.sensor_config['Name']]['Pb'] = self.sensor_config['Bias_Guess']
                self.Bias[self.sensor_config['Name']]['timestamp'] = self.Timestamps[-1]

        self.time_k = 0
        self.Debug_Params = Debug_Params

        #********逐点输入版本修改 第四处 修改航迹起始方式********
        # 替换原Resoled_M_N为时间窗口参数
        self.Resolved_Time_Window = Resolved_Time_Window  # 航迹确认时间窗口（秒）
        self.Resolved_Min_Detect = Resolved_Min_Detect    # 时间窗口内最小检测点数
        #self.Resoled_M_N = Resoled_M_N

        self.max_detect_time = max_detect_time
        self.Merge_Threshold = Merge_Threshold

        self.generate_hypotheses(pool_1=pool_1)

    def forward(
        self,
        timestamp: float, # 该帧量测的时间戳
        obs_k: List[np.ndarray], # 量测
        sensor_config: Dict, # 量测对应的传感器配置
        pool_1 = None, # 是否多进程
        extra_infos = [], # 额外的量测信息
    ):
        self.Timestamps.append(timestamp)
        self.obs_s.append(obs_k)
        self.info_s.append(extra_infos)
        self.max_ids.append(-1)
        self.Sensor_Config.append(sensor_config)
        self.sensor_config = sensor_config
        self.obs_k = obs_k
        self.R_k = sensor_config['R'] # 默认单帧每个量测都一样
        ## 如果首次出现某个带偏差传感器，则先初始化其偏差
        if self.sensor_config['Is_Biased']:
            if self.sensor_config['Name'] not in self.Bias.keys():
                self.Bias[self.sensor_config['Name']] = {}
                self.Bias[self.sensor_config['Name']]['b'] = np.zeros((self.dim_d, 1)) 
                self.Bias[self.sensor_config['Name']]['Pb'] = self.sensor_config['Bias_Guess']
                self.Bias[self.sensor_config['Name']]['timestamp'] = self.Timestamps[-1]
        self.time_k += 1

        self.Predict()
        self.generate_hypotheses(pool_1=pool_1)
        self.Best_hypotheses()
        if self.Debug_Params['Debug']:
            for _, node in self.Tree[-1].items():
                if node.label == 352 and node.hyp_type == 'detect' and node.debug_info['timestamp'] > 1751006900:
                # if node.label == 352 and node.hyp_type == 'detect' and node.score>3490:
                    self.Debug()
                    a = 1
        self.Prune()
        self.Update()
        self.Merge()

    def Predict(
        self,
    ) -> None:
        sensor_config = self.Sensor_Config[-1]
        sensor_config_last = self.Sensor_Config[-2]
        self.Leaf_Nodes = self.Tree[-1]
        T = self.Timestamps[-1] - self.Timestamps[-2]
        self.T = T
        # 有系统偏差，需要分情况预测
        if sensor_config['Is_Biased'] and not sensor_config['Biased_Ignore']:
            F, G = Get_F_G_CV_Bias(T=T, dim_d=self.dim_d)
            Bias_Qk = sensor_config['Bias_Qk']
            Qk = block_diag(self.Q_k, Bias_Qk)
            # 连续两帧来自同一个传感器，且上一帧传感器也考虑了偏差（有可能上一帧还没出现基准传感器，还没考虑偏差），则直接预测
            if sensor_config['Name'] == sensor_config_last['Name'] and not sensor_config_last['Biased_Ignore']:
                for _, node in self.Leaf_Nodes.items():
                    if node.hyp_type == 'death':
                        continue
                    x_k_k, P_k_k = node.x_k_k, node.P_k_k
                    [x_k_k_1, P_k_k_1] = KF_Prediction_CV(x_k_1_k_1=x_k_k, P_k_1_k_1=P_k_k, F=F, G=G, Q=Qk)
                    node.x_k1_k, node.P_k1_k = x_k_k_1, P_k_k_1 # x_k1_k 表示 x_k+1_1,，表示为下一个节点去预测的
            # 连续两帧来自不同传感器，则偏差变了，需要重新拼接状态
            else:
                # 先将偏差预测到当前时刻
                b_k_1 = self.Bias[sensor_config['Name']]['b']
                Pb_k_1 = self.Bias[sensor_config['Name']]['Pb']
                timestamp_k_1 = self.Bias[sensor_config['Name']]['timestamp']
                dt = self.Timestamps[-1] - timestamp_k_1
                F_b, G_b = Get_F_G_Bias(dt, dim_d=self.dim_d)
                [b_k, Pb_k] = KF_Prediction_CV(x_k_1_k_1=b_k_1, P_k_1_k_1=Pb_k_1, F=F_b, G=G_b, Q=Bias_Qk)
                # 预测目标状态，再拼接上预测的偏差状态
                F_t, G_t = Get_F_G_CV(T, dim_d=self.dim_d)
                for _, node in self.Leaf_Nodes.items():
                    if node.hyp_type == 'death':
                        continue
                    x_k_k_target, P_k_k_target = node.x_k_k[:2*self.dim_d, :], node.P_k_k[:2*self.dim_d, :2*self.dim_d]
                    [x_k_k_1_target, P_k_k_1_target] = KF_Prediction_CV(x_k_1_k_1=x_k_k_target,\
                         P_k_1_k_1=P_k_k_target, F=F_t, G=G_t, Q=self.Q_k)
                    x_k_k_1 = np.concatenate([x_k_k_1_target, b_k], axis=0)
                    P_k_k_1 = block_diag(P_k_k_1_target, Pb_k)
                    Ptb = Cross_Covariance_CKF(x_k_k_1_target, P_k_k_1_target, b_k, Pb_k) # CKF采点考虑互协方差
                    P_k_k_1[:2*self.dim_d, 2*self.dim_d:] = Ptb
                    P_k_k_1[2*self.dim_d:, :2*self.dim_d] = Ptb.T
                    node.x_k1_k, node.P_k1_k = x_k_k_1, P_k_k_1
        # 无系统偏差，那就只需要预测更新目标状态即可
        else:
            F, G = Get_F_G_CV(T=T, dim_d=self.dim_d)
            for _, node in self.Leaf_Nodes.items():
                if node.hyp_type == 'death':
                    continue
                x_k_k, P_k_k = node.x_k_k[:2*self.dim_d, :], node.P_k_k[:2*self.dim_d, :2*self.dim_d]
                [x_k_k_1, P_k_k_1] = KF_Prediction_CV(x_k_1_k_1=x_k_k, P_k_1_k_1=P_k_k, F=F, G=G, Q=self.Q_k)
                node.x_k1_k, node.P_k1_k = x_k_k_1, P_k_k_1
            # 如果当前帧为基准，且上一帧有偏差却没考虑偏差，还需要给位置加上一点协方差，防止关联不上
            if not sensor_config['Is_Biased'] and sensor_config_last['Is_Biased'] and sensor_config_last['Biased_Ignore']:
                node.P_k1_k[:self.dim_d, :self.dim_d] = node.P_k1_k[:self.dim_d, :self.dim_d] + sensor_config_last['Bias_Guess']

    # def hypotheses_new_tracks(
    #     self,
    #     New_Leaf_Nodes, 
    # ) -> Dict[int, Type[Node]]:
    #     """
    #     generate new track using each measurement.
    #     【可以先故意不用带偏差的传感器去初始化航迹】
    #     """
    #     P_D = self.sensor_config['P_D']
    #     R_k = self.R_k
    #     for obs_id, obs_ in enumerate(self.obs_k):
    #         # 传感器有偏差并且考虑偏差，则状态包含偏差
            
    #         if (self.sensor_config['Name'] == 'ADS-B'):
    #             # 传感器是ads-b时，接受民航数据，最大速度很大
    #             maxVal = 230
    #         else:
    #             maxVal = self.Max_Vel

    #         if self.sensor_config['Is_Biased'] and not self.sensor_config['Biased_Ignore']:
    #             # 均值
    #             x_k_k_1 = np.zeros((3*self.dim_d, 1))
    #             x_k_k_1[:self.dim_d, :] = obs_ # 位置
    #             x_k_k_1[2*self.dim_d:, :] = self.Bias[self.sensor_config['Name']]['b'] # 偏差
    #             # 协方差
    #             P_k_k_1 = np.zeros((3*self.dim_d, 3*self.dim_d))
    #             # P_k_k_1[:self.dim_d, :self.dim_d] = R_k + self.Bias[self.sensor_config['Name']]['Pb'] # 位置
    #             P_k_k_1[:self.dim_d, :self.dim_d] = R_k + self.Bias[self.sensor_config['Name']]['Pb'] # 位置
    #             P_k_k_1[self.dim_d:2*self.dim_d, self.dim_d:2*self.dim_d] = np.identity(self.dim_d)*((maxVal/2.0)**2) # 速度
    #             P_k_k_1[2*self.dim_d:, 2*self.dim_d:] = self.Bias[self.sensor_config['Name']]['Pb'] # 偏差

    #             # break # 不用带偏差的传感器去初始化目标

    #         # 传感器无偏差，则状态只包含目标状态
    #         else:
    #             # 均值
    #             x_k_k_1 = np.zeros((2*self.dim_d, 1))
    #             x_k_k_1[:self.dim_d, :] = obs_ # 位置
    #             # 协方差
    #             P_k_k_1 = np.zeros((2*self.dim_d, 2*self.dim_d))
    #             # P_k_k_1[:self.dim_d, :self.dim_d] = R_k # 位置
    #             P_k_k_1[:self.dim_d, :self.dim_d] = R_k # 位置
    #             P_k_k_1[self.dim_d:2*self.dim_d, self.dim_d:2*self.dim_d] = np.identity(self.dim_d)*((maxVal/2.0)**2) # 速度

    #         score_k = np.log(self.Lambda_NT*P_D/self.sensor_config['FA_Num'])
    #         self.label_max += 1
    #         self.max_ids[-1] += 1
    #         leaf_node = Node(id=copy(self.max_ids[-1]), label=self.label_max, score=score_k, \
    #                         x_k_k_1=x_k_k_1, P_k_k_1=P_k_k_1, obs_id=[obs_id], \
    #                         last_detect_tmp=copy(self.Timestamps[-1]))
    #         leaf_node.x_k_k, leaf_node.P_k_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
    #         leaf_node.hyp_type = 'initial'
    #         leaf_node.initial_sensor_name = self.sensor_config['Name']
    #         # 新增：初始化detection_timestamps（与detection_times[1]对应）
    #         leaf_node.detection_timestamps = [copy(self.Timestamps[-1])]
    #         leaf_node.debug_info['obs'] = obs_
    #         leaf_node.debug_info['sensor_name'] = self.sensor_config['Name']
    #         leaf_node.debug_info['timestamp'] = copy(self.Timestamps[-1])

    #         # try:
    #         #     leaf_node.debug_info['droneName'] = self.info_s[-1][0].get('droneName', None)
    #         #     leaf_node.debug_info['direction'] = self.info_s[-1][0].get('direction', 0.0)
    #         #     leaf_node.debug_info['pilotLongitude'] = self.info_s[-1][0].get('pilotLongitude', 0.0)
    #         #     leaf_node.debug_info['pilotLatitude'] = self.info_s[-1][0].get('pilotLatitude', 0.0)
    #         #     leaf_node.debug_info['signalPowerDidch1'] = self.info_s[-1][0].get('signalPowerDidch1', None)
    #         #     leaf_node.debug_info['deviceCode'] = self.info_s[-1][0].get('deviceCode', 'RADAR')
    #         #     leaf_node.debug_info['sn'] = self.info_s[-1][0].get('sn', None)
    #         # except (KeyError, IndexError, TypeError, AttributeError):
    #         #     leaf_node.debug_info['droneName'] = None
    #         #     leaf_node.debug_info['direction'] = 0.0
    #         #     leaf_node.debug_info['pilotLongitude'] = 0.0
    #         #     leaf_node.debug_info['pilotLatitude'] = 0.0
    #         #     leaf_node.debug_info['signalPowerDidch1'] = None
    #         #     leaf_node.debug_info['deviceCode'] = 'RADAR'
    #         #     leaf_node.debug_info['sn'] = None
    #         leaf_node.debug_info['droneName'] = self.info_s[-1][0]['droneName']
    #         leaf_node.debug_info['direction'] = self.info_s[-1][0]['direction']
    #         leaf_node.debug_info['pilotLongitude'] = self.info_s[-1][0]['pilotLongitude']
    #         leaf_node.debug_info['pilotLatitude'] = self.info_s[-1][0]['pilotLatitude']
    #         leaf_node.debug_info['signalPowerDidch1'] = self.info_s[-1][0]['signalPowerDidch1']
    #         leaf_node.debug_info['deviceCode'] = self.info_s[-1][0]['deviceCode']
    #         leaf_node.debug_info['sn'] = self.info_s[-1][0]['sn']
    #         New_Leaf_Nodes[leaf_node.id] = leaf_node

    def hypotheses_new_tracks(
        self,
        New_Leaf_Nodes, 
    ) -> Dict[int, Type[Node]]:
        """
        generate new track using each measurement.
        【可以先故意不用带偏差的传感器去初始化航迹】
        """
        P_D = self.sensor_config['P_D']
        R_k = self.R_k
        
        # 安全获取 info_s 数据（提前获取，避免重复判断）
        info_data = {}
        try:
            if self.info_s and len(self.info_s) > 0:
                last_info = self.info_s[-1]
                if last_info and isinstance(last_info, (list, tuple)) and len(last_info) > 0:
                    info_data = last_info[0] if isinstance(last_info[0], dict) else {}
                elif isinstance(last_info, dict):
                    info_data = last_info
        except (KeyError, IndexError, TypeError, AttributeError):
            info_data = {}
        
        for obs_id, obs_ in enumerate(self.obs_k):
            # 传感器有偏差并且考虑偏差，则状态包含偏差
            if (self.sensor_config['Name'] == 'ADS-B'):
                # 传感器是ads-b时，接受民航数据，最大速度很大
                maxVal = 230
            else:
                maxVal = self.Max_Vel

            if self.sensor_config['Is_Biased'] and not self.sensor_config['Biased_Ignore']:
                # 均值
                x_k_k_1 = np.zeros((3*self.dim_d, 1))
                x_k_k_1[:self.dim_d, :] = obs_ # 位置
                x_k_k_1[2*self.dim_d:, :] = self.Bias[self.sensor_config['Name']]['b'] # 偏差
                # 协方差
                P_k_k_1 = np.zeros((3*self.dim_d, 3*self.dim_d))
                P_k_k_1[:self.dim_d, :self.dim_d] = R_k + self.Bias[self.sensor_config['Name']]['Pb'] # 位置
                P_k_k_1[self.dim_d:2*self.dim_d, self.dim_d:2*self.dim_d] = np.identity(self.dim_d)*((maxVal/2.0)**2) # 速度
                P_k_k_1[2*self.dim_d:, 2*self.dim_d:] = self.Bias[self.sensor_config['Name']]['Pb'] # 偏差
            else:
                # 传感器无偏差，则状态只包含目标状态
                # 均值
                x_k_k_1 = np.zeros((2*self.dim_d, 1))
                x_k_k_1[:self.dim_d, :] = obs_ # 位置
                # 协方差
                P_k_k_1 = np.zeros((2*self.dim_d, 2*self.dim_d))
                P_k_k_1[:self.dim_d, :self.dim_d] = R_k # 位置
                P_k_k_1[self.dim_d:2*self.dim_d, self.dim_d:2*self.dim_d] = np.identity(self.dim_d)*((maxVal/2.0)**2) # 速度

            score_k = np.log(self.Lambda_NT*P_D/self.sensor_config['FA_Num'])
            self.label_max += 1
            self.max_ids[-1] += 1
            leaf_node = Node(id=copy(self.max_ids[-1]), label=self.label_max, score=score_k, \
                            x_k_k_1=x_k_k_1, P_k_k_1=P_k_k_1, obs_id=[obs_id], \
                            last_detect_tmp=copy(self.Timestamps[-1]))
            leaf_node.x_k_k, leaf_node.P_k_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
            leaf_node.hyp_type = 'initial'
            leaf_node.initial_sensor_name = self.sensor_config['Name']
            # 新增：初始化detection_timestamps（与detection_times[1]对应）
            leaf_node.detection_timestamps = [copy(self.Timestamps[-1])]
            leaf_node.debug_info['obs'] = obs_
            leaf_node.debug_info['sensor_name'] = self.sensor_config['Name']
            leaf_node.debug_info['timestamp'] = copy(self.Timestamps[-1])
            
            # 安全设置 debug_info（使用前面获取的 info_data）
            leaf_node.debug_info['droneName'] = info_data.get('droneName', None) if info_data else None
            leaf_node.debug_info['direction'] = info_data.get('direction', 0.0) if info_data else 0.0
            leaf_node.debug_info['pilotLongitude'] = info_data.get('pilotLongitude', 0.0) if info_data else 0.0
            leaf_node.debug_info['pilotLatitude'] = info_data.get('pilotLatitude', 0.0) if info_data else 0.0
            leaf_node.debug_info['signalPowerDidch1'] = info_data.get('signalPowerDidch1', None) if info_data else None
            leaf_node.debug_info['deviceCode'] = info_data.get('deviceCode', 'RADAR') if info_data else 'RADAR'
            leaf_node.debug_info['sn'] = info_data.get('sn', None) if info_data else None
            
            New_Leaf_Nodes[leaf_node.id] = leaf_node

    def Generate_hyp_from_Leaf_nodes(
        self,
        node # 老节点
    ):  
        new_leaf_nodes = []
        
        # 安全获取 info_s 数据
        info_data = {}
        try:
            if self.info_s and len(self.info_s) > 0:
                last_info = self.info_s[-1]
                if last_info and isinstance(last_info, (list, tuple)) and len(last_info) > 0:
                    info_data = last_info[0] if isinstance(last_info[0], dict) else {}
                elif isinstance(last_info, dict):
                    info_data = last_info
        except (KeyError, IndexError, TypeError, AttributeError):
            info_data = {}
        
        # 死亡节点，不用波门处理，直接顺延死亡假设即可
        if node.hyp_type == 'death':
            ## hypotheses0 : old_track dies.
            score_k = 0 # 不用死两次
            new_leaf_node = Node(id = None, label=node.label, score = node.score + score_k, x_k_k_1=None, \
                P_k_k_1=None, obs_id=[], last_detect_tmp=node.last_detect_tmp)
            new_leaf_node.parent_id = node.id
            new_leaf_node.hyp_type = 'death'
            # 新增 死亡节点不更新detection_times和timestamps
            new_leaf_node.detection_times = copy(node.detection_times)
            new_leaf_node.detection_timestamps = copy(node.detection_timestamps)
            new_leaf_node.track_resolved = node.track_resolved
            new_leaf_node.initial_sensor_name = node.initial_sensor_name
            new_leaf_nodes.append(new_leaf_node)
        else:
            # 非死亡节点，波门后，往下生成多个假设
                ## hypotheses1 : old_track with obs
                R_k = self.R_k
                H_k = Get_H_k(sensor_config=self.sensor_config, dim_d=self.dim_d)
                S_k = H_k @ node.P_k1_k @ (H_k.T) + R_k
                S_k_inv = np.linalg.inv(S_k)
                gate = Cal_Gate(Pg = self.Pg, dim_d = self.dim_d)
                volume = self.sensor_config['Volume']
                x_k1_k_pre_pos = H_k@node.x_k1_k # 预测的目标位置

                #********逐点输入版本修改 第三处 P_death和P_D改成随时间变化********
                # 计算时间间隔Δt（当前时间 - 上一次检测时间）
                #Δt = self.Timestamps[-1] - node.last_detect_tmp
                Δt = self.Timestamps[-1] - self.Timestamps[-2]
                lambda_death = self.sensor_config['lambda_death']
                mu_detect = self.sensor_config['mu_detect']
                P_death_current = 1 - np.exp(-lambda_death * Δt)
                P_D_current =  1-np.exp(-mu_detect * Δt)

                for io, obs_ in enumerate(self.obs_k):
                    innovation = obs_ - x_k1_k_pre_pos
                    innovation_dis = np.linalg.norm(obs_ - x_k1_k_pre_pos)
                    ma_dis = innovation.T @ S_k_inv @ innovation
                    # 增加针对ads-b的启发式修改
                    if ma_dis <= gate and ((self.sensor_config['Name'] == 'ADS-B' and innovation_dis <= 1000) or (self.sensor_config['Name'] != 'ADS-B' and innovation_dis <= 200)):                    
                        score_k = np.log(1-P_death_current) - np.log(self.sensor_config['FA_Num']/volume) + \
                            np.log(np.clip(P_D_current, 1e-50, 1.0)) - 0.5*ma_dis - np.log(np.sqrt(np.linalg.det(2.0*np.pi*S_k)))
                        score = node.score + score_k
                        new_leaf_node = Node(id=None, label=node.label, score=score, \
                            x_k_k_1=deepcopy(node.x_k1_k), P_k_k_1=deepcopy(node.P_k1_k), obs_id=[io],\
                            last_detect_tmp=copy(self.Timestamps[-1]))
                        new_leaf_node.parent_id = node.id
                        new_leaf_node.hyp_type = 'detect'
                        new_leaf_node.detection_times = copy(node.detection_times)
                        new_leaf_node.detection_times.append(1)
                        # 新增：同步添加当前时间戳到detection_timestamps
                        new_leaf_node.detection_timestamps = copy(node.detection_timestamps)
                        new_leaf_node.detection_timestamps.append(copy(self.Timestamps[-1]))
                        new_leaf_node.track_resolved = node.track_resolved
                        new_leaf_node.initial_sensor_name = node.initial_sensor_name
                        new_leaf_node.debug_info['obs'] = obs_
                        new_leaf_node.debug_info['dt'] = self.T
                        new_leaf_node.debug_info['sensor_name'] = self.sensor_config['Name']
                        new_leaf_node.debug_info['timestamp'] = copy(self.Timestamps[-1])
                        
                        # 安全设置 debug_info（使用前面获取的 info_data）
                        new_leaf_node.debug_info['droneName'] = info_data.get('droneName', None) if info_data else None
                        new_leaf_node.debug_info['direction'] = info_data.get('direction', 0.0) if info_data else 0.0
                        new_leaf_node.debug_info['pilotLongitude'] = info_data.get('pilotLongitude', 0.0) if info_data else 0.0
                        new_leaf_node.debug_info['pilotLatitude'] = info_data.get('pilotLatitude', 0.0) if info_data else 0.0
                        new_leaf_node.debug_info['signalPowerDidch1'] = info_data.get('signalPowerDidch1', None) if info_data else None
                        new_leaf_node.debug_info['deviceCode'] = info_data.get('deviceCode', 'RADAR') if info_data else 'RADAR'
                        new_leaf_node.debug_info['sn'] = info_data.get('sn', None) if info_data else None
                        
                        new_leaf_nodes.append(new_leaf_node)

                ## hypotheses2 : old_track without obs
                delta_tmp = self.Timestamps[-1] - node.last_detect_tmp
                if delta_tmp <= self.max_detect_time:
                    score_k = np.log(1.0 - P_death_current) + np.log(np.clip(1.0 - P_D_current, 1e-50, 1.0))  # 限制最小值为1e-50
                    #score_k = np.log(1.0-self.P_death) + np.log(1.0-P_D)
                    score = node.score + score_k
                    new_leaf_node = Node(id=None, label=node.label, score=score, x_k_k_1=deepcopy(node.x_k1_k), P_k_k_1=deepcopy(node.P_k1_k),\
                                        last_detect_tmp=node.last_detect_tmp)
                    new_leaf_node.parent_id = node.id
                    new_leaf_node.hyp_type = 'miss-detect'
                    new_leaf_node.detection_times = copy(node.detection_times)
                    new_leaf_node.initial_sensor_name = node.initial_sensor_name
                    if new_leaf_node.initial_sensor_name == self.sensor_config['Name']:
                        new_leaf_node.detection_times.append(0)
                        # 新增：同步添加当前时间戳到detection_timestamps
                        new_leaf_node.detection_timestamps = copy(node.detection_timestamps)
                        new_leaf_node.detection_timestamps.append(copy(self.Timestamps[-1]))
                    else:
                        # 非初始传感器漏检，不添加0，也不添加时间戳（保持原列表）
                        new_leaf_node.detection_timestamps = copy(node.detection_timestamps)
                    new_leaf_node.track_resolved = node.track_resolved
                    new_leaf_node.debug_info['dt'] = self.T
                    new_leaf_node.debug_info['timestamp'] = copy(self.Timestamps[-1])
                    
                    # 安全设置 debug_info（使用前面获取的 info_data）
                    new_leaf_node.debug_info['droneName'] = info_data.get('droneName', None) if info_data else None
                    new_leaf_node.debug_info['direction'] = info_data.get('direction', 0.0) if info_data else 0.0
                    new_leaf_node.debug_info['pilotLongitude'] = info_data.get('pilotLongitude', 0.0) if info_data else 0.0
                    new_leaf_node.debug_info['pilotLatitude'] = info_data.get('pilotLatitude', 0.0) if info_data else 0.0
                    new_leaf_node.debug_info['signalPowerDidch1'] = info_data.get('signalPowerDidch1', None) if info_data else None
                    new_leaf_node.debug_info['deviceCode'] = info_data.get('deviceCode', 'RADAR') if info_data else 'RADAR'
                    new_leaf_node.debug_info['sn'] = info_data.get('sn', None) if info_data else None
                    
                    new_leaf_nodes.append(new_leaf_node)
                ## hypothesis3 : death
                else:
                    #score_k = np.log(self.P_death)
                    score_k = np.log(P_death_current)
                    score = node.score + score_k
                    new_leaf_node = Node(id=None, label=node.label, score=score, x_k_k_1=None, P_k_k_1=None, \
                                        last_detect_tmp=node.last_detect_tmp)
                    new_leaf_node.parent_id = node.id
                    new_leaf_node.hyp_type = 'death'
                    # 新增 死亡节点不更新detection_times和timestamps
                    new_leaf_node.detection_times = copy(node.detection_times)
                    new_leaf_node.detection_timestamps = copy(node.detection_timestamps)
                    new_leaf_node.track_resolved = node.track_resolved
                    new_leaf_node.initial_sensor_name = node.initial_sensor_name
                    new_leaf_nodes.append(new_leaf_node)

        return new_leaf_nodes

    def generate_hypotheses(
        self,
        pool_1 = None,
    ) -> None:
        """
        generate all hypotheses.
        """
        New_Leaf_Nodes = {}
        ## 老节点往下生成假设节点。
        if len(self.Leaf_Nodes) > 0:
            if pool_1 is None:
                result = []
                for node in self.Leaf_Nodes.values():
                    result.append(self.Generate_hyp_from_Leaf_nodes(node))
            else:
                result = pool_1.map(self.Generate_hyp_from_Leaf_nodes, [node for node in self.Leaf_Nodes.values()])
            for new_leaf_nodes in result:
                for new_leaf_node in new_leaf_nodes:
                    self.max_ids[-1] += 1
                    new_leaf_node.id = copy(self.max_ids[-1])
                    new_leaf_node.hyp_ids = [new_leaf_node.id]
                    self.Leaf_Nodes[new_leaf_node.parent_id].children_ids.append(new_leaf_node.id)
                    New_Leaf_Nodes[new_leaf_node.id] = new_leaf_node

        ## 新生航迹假设
        self.hypotheses_new_tracks(New_Leaf_Nodes)

        self.New_Leaf_Nodes = New_Leaf_Nodes
        self.Tree.append(self.New_Leaf_Nodes)

    def Best_hypotheses(
        self,
    ) -> None:
        """
        use cvxpy and cvxopt (pip install cvxopt) to solve 0-1 program.
        """
        ## work only when length satisfy the requirement
        if len(self.Tree) == self.Tree_maxlen:
            hyp_num = len(self.Tree[-1])
            if hyp_num > 0: 
                self.hyp_num = hyp_num
                ## prerequiste: update all hyps id for each node.
                for il in range(-2,-self.Tree_maxlen-1, -1):  # bottom-up calculate what hypotheses does each node have.
                    parent_nodes = self.Tree[il]
                    children_nodes = self.Tree[il+1]
                    for _, parent_node in parent_nodes.items():
                        parent_node.hyp_ids = []
                        for ic in parent_node.children_ids:
                            if ic in children_nodes.keys(): # may be deleted
                                parent_node.hyp_ids.extend(children_nodes[ic].hyp_ids)

                assign = cp.Variable(hyp_num, boolean=True)
                cp_constraints = []
                ## constraint part 1: decided track.
                track_num = len(self.Tree[0])
                if track_num > 0:
                    constraint_matrix = np.zeros((track_num, hyp_num), dtype=np.uint64)
                    it = 0
                    for _, root_node in self.Tree[0].items():
                        root_node_hpy_ids = root_node.hyp_ids
                        if len(root_node_hpy_ids) > 0:
                            constraint_matrix[it, root_node_hpy_ids] = 1
                        it = it + 1
                    Constraint_matrixs1 = constraint_matrix
                    cp_constraints.append(Constraint_matrixs1@assign == np.ones((Constraint_matrixs1.shape[0],))) # 约束用等号

                ## constraint part 2: observations whitin N_Scan.
                Constraint_matrixs = []
                for layer_id in range(1, self.Tree_maxlen): # begins from the second layer
                    obs_num = len(self.obs_s[layer_id])
                    branch_nodes = self.Tree[layer_id]
                    constraint_matrix = np.zeros((obs_num, hyp_num), dtype=np.uint64)
                    for _, node in branch_nodes.items():
                        if (len(node.obs_id) > 0) and (len(node.hyp_ids) > 0):
                            for obs_id_ in node.obs_id:
                                constraint_matrix[obs_id_, node.hyp_ids] = 1
                    Constraint_matrixs.append(constraint_matrix)
                Constraint_matrixs2 = np.concatenate(Constraint_matrixs)
                cp_constraints.append(Constraint_matrixs2@assign == np.ones((Constraint_matrixs2.shape[0],)))

                ## use cvxpy to solve 0-1.
                Score_list = []
                for _, leaf_node in self.Tree[-1].items():
                    Score_list.append(float(leaf_node.score))
                Score_list = np.array(Score_list).reshape(1,-1)
                Score_list = np.clip(Score_list, -1e10, 1e10)
                cp_obj = cp.Maximize(Score_list@assign)
                prob = cp.Problem(cp_obj, cp_constraints)
                prob.solve(solver='GLPK_MI')
                self.score_sum = prob.value
                self.assignment = assign.value  # (hyp_num,) 0-1.
            else:
                self.hyp_num = 0
        else:
            self.hyp_num = 0


    def Prune(
        self,
    ) -> None:
        ## 需要判断是否选中死亡的航迹, node.lives <= 0 or node.hyp_type == 'death'
        ## work only when length satisfy the requirement
        if len(self.Tree) == self.Tree_maxlen:
            ## prune the second layer's nodes that don't have the chosen hypotheses or death node.
            branch = self.Tree[1]
            children_del_ids = []
            parent_del_ids = []
            for node_id, node in branch.items():
                if len(node.hyp_ids) == 0:  # delete if doesn't have hypotheses originating from it.
                    parent_del_ids.append(node_id)
                else:
                    assign_num = np.sum(self.assignment[node.hyp_ids])
                    if (assign_num == 0) : # means all hypotheses orginating from it is bad.
                        children_del_ids.extend(node.children_ids)
                        parent_del_ids.append(node_id)
                    else:
                        assign_result = np.where(self.assignment[node.hyp_ids] > 0)[0]
                        assign_id = node.hyp_ids[assign_result[0]]
                        assign_node = self.Tree[-1][assign_id]
                        # 应该是对node进行判断，而不是assign_node，不然会删早了
                        # if assign_node.lives <= 0 or assign_node.score_cum < 0 or assign_node.hyp_type == 'death': # 可以在这里调整航迹管理
                        if node.hyp_type == 'death': # 可以在这里调整航迹管理
                            children_del_ids.extend(node.children_ids)
                            parent_del_ids.append(node_id)
                            self.assignment[assign_id] = 0
            for parent_del_id in parent_del_ids:
                del branch[parent_del_id]
            ## prune the branch nodes belonging to the second layer's nodes haven been deleted.
            parent_del_ids = children_del_ids
            for branch in self.Tree[2:]:
                children_del_ids = []
                for node_id in parent_del_ids:
                    children_del_ids.extend(branch[node_id].children_ids)
                for parent_del_id in parent_del_ids:
                    del branch[parent_del_id]

                parent_del_ids = children_del_ids
            
        else:
            pass

    def Update(
        self,
    ) -> Optional[List]:
        # 更新最新一层各假设状态
        for node_id, node in self.Tree[-1].items():
            if node.hyp_type == 'initial':
                pass
            elif node.hyp_type == 'detect':
                [x_k_k, P_k_k] = KF_Update(x_k_k_1=node.x_k_k_1, P_k_k_1=node.P_k_k_1, \
                    z_k = self.obs_k[node.obs_id[0]], sensor_config=self.sensor_config, dim_d=self.dim_d)
                node.x_k_k, node.P_k_k = x_k_k, P_k_k
            elif node.hyp_type == 'miss-detect':
                node.x_k_k, node.P_k_k = deepcopy(node.x_k_k_1), deepcopy(node.P_k_k_1)
            elif node.hyp_type == 'death':
                pass
            else:
                raise NotImplementedError
            # # 判断航迹是否确认
            # if (node.hyp_type == 'detect' or node.hyp_type == 'miss-detect') and not node.track_resolved:
            #     if len(node.detection_times) >= self.Resoled_M_N[1]: # 达到判定长度
            #         if sum(node.detection_times[-self.Resoled_M_N[1]:]) >= self.Resoled_M_N[0]-1e-5:
            #             node.track_resolved = True # 航迹确认

            #********逐点输入版本修改 第四处 修改航迹起始方式********
            #核心修改：航迹确认逻辑（帧数→时间窗口）
            if (node.hyp_type == 'detect' or node.hyp_type == 'miss-detect') and not node.track_resolved:
                current_timestamp = copy(self.Timestamps[-1])  # 当前时间戳
                time_window_start = current_timestamp - self.Resolved_Time_Window  # 时间窗口起始时间（当前-窗口长度）
                window_detect_count = 0
                # 遍历detection_timestamps，统计窗口内的检测次数（1的数量）
                for idx, ts in enumerate(node.detection_timestamps):
                    if ts >= time_window_start:  # 只统计时间窗口内的记录
                        if node.detection_times[idx] == 1:  # 检测到才计数
                            window_detect_count += 1

                # 确认条件：时间窗口内检测次数 >= 最小检测点数
                if window_detect_count >= self.Resolved_Min_Detect - 1e-5:  # 减1e-5避免浮点误差
                    node.track_resolved = True  # 航迹确认

        # 融合 resolved 层 （第二层）偏差
        if len(self.Tree) == self.Tree_maxlen: # 只有 resolved 后才开始
            sensor_config = self.Sensor_Config[1]
            branch_nodes = self.Tree[1]
            timestamp = self.Timestamps[1]
            node_ids_bias_fusion = [] # 用于偏差融合的第二层节点id【目前只有确认航迹的检测假设】
            node_ids_state_fusion = [] # 基于融合偏差进行状态修正的第二层节点id【目前有检测和漏检假设】
            if sensor_config['Is_Biased'] and not sensor_config['Biased_Ignore'] and  len(branch_nodes) >= 1:
                for node_id, node in branch_nodes.items():
                    # 偏差融合的条件
                    if node.hyp_type == 'detect' and node.track_resolved: # 检测假设 + 确认航迹 ######
                        node_ids_bias_fusion.append(node_id)
                # 至少得有一个检测假设，才会开始融合
                if len(node_ids_bias_fusion) >=1:
                    # 预测融合偏差
                    b_k_1_fusion = self.Bias[sensor_config['Name']]['b']
                    Pb_k_1_fusion = self.Bias[sensor_config['Name']]['Pb']
                    timestamp_k_1 = self.Bias[sensor_config['Name']]['timestamp']
                    dt = timestamp - timestamp_k_1
                    F_b, G_b = Get_F_G_Bias(T=dt, dim_d=self.dim_d)
                    Bias_Qk = sensor_config['Bias_Qk']
                    [b_k_k_1_fusion, Pb_k_k_1_fusion] = KF_Prediction_CV(x_k_1_k_1=b_k_1_fusion, \
                        P_k_1_k_1=Pb_k_1_fusion, F=F_b, G=G_b, Q=Bias_Qk)
                    Pb_k_k_1_fusion_inv = np.linalg.inv(Pb_k_k_1_fusion)
                    Pb_k_k_fusion_inv = np.zeros_like(Pb_k_k_1_fusion_inv)
                    b_k_k_fusion = np.zeros_like(b_k_k_1_fusion)
                    # 开始融合
                    for node_id in node_ids_bias_fusion:
                        node = branch_nodes[node_id]
                        Pb_k_k_1_inv = np.linalg.inv(node.P_k_k_1[2*self.dim_d:,2*self.dim_d:])
                        Pb_k_k_inv = np.linalg.inv(node.P_k_k[2*self.dim_d:,2*self.dim_d:])
                        Pb_k_k_fusion_inv = Pb_k_k_fusion_inv + (Pb_k_k_inv - Pb_k_k_1_inv)

                        b_k_k_1 = node.x_k_k_1[2*self.dim_d:, :]
                        b_k_k = node.x_k_k[2*self.dim_d:, :]
                        b_k_k_fusion = b_k_k_fusion + (Pb_k_k_inv@b_k_k - Pb_k_k_1_inv@b_k_k_1)
                    Pb_k_k_fusion_inv = Pb_k_k_fusion_inv + Pb_k_k_1_fusion_inv
                    Pb_k_k_fusion = np.linalg.inv(Pb_k_k_fusion_inv)
                    b_k_k_fusion = Pb_k_k_fusion @ (Pb_k_k_1_fusion_inv @ b_k_k_1_fusion + b_k_k_fusion)
                    # 偏差融合结果
                    self.Bias[sensor_config['Name']]['b'] =  b_k_k_fusion
                    self.Bias[sensor_config['Name']]['Pb'] =  Pb_k_k_fusion
                    self.Bias[sensor_config['Name']]['timestamp'] =  timestamp

                    ## 利用偏差，去修正目标状态
                    for node in branch_nodes.values():
                        if node.hyp_type == 'death':
                            continue # 死亡假设跳过
                        elif node.hyp_type == 'initial':
                            node.x_k_k[2*self.dim_d:, :] = b_k_k_fusion
                            node.P_k_k[2*self.dim_d:, 2*self.dim_d:] = Pb_k_k_fusion
                        else:
                            node_ids_state_fusion.append(node.id)
                            Pt = node.P_k_k[:2*self.dim_d, :2*self.dim_d]
                            Ptb = node.P_k_k[:2*self.dim_d, 2*self.dim_d:]
                            Pb = node.P_k_k[2*self.dim_d:, 2*self.dim_d:]
                            Pb_inv = np.linalg.inv(Pb)

                            node.x_k_k[:2*self.dim_d, :] = node.x_k_k[:2*self.dim_d, :] \
                                + Ptb@Pb_inv@(b_k_k_fusion - node.x_k_k[2*self.dim_d:, :])
                            node.x_k_k[2*self.dim_d:, :] = b_k_k_fusion

                            Pt_fusion = Pt - Ptb@Pb_inv@(Pb-Pb_k_k_fusion)@Pb_inv@(Ptb.T)
                            Ptb_fusion = Ptb@Pb_inv@Pb_k_k_fusion
                            node.P_k_k[:2*self.dim_d, :2*self.dim_d] = Pt_fusion
                            node.P_k_k[:2*self.dim_d, 2*self.dim_d:] = Ptb_fusion
                            node.P_k_k[2*self.dim_d:, :2*self.dim_d] = Ptb_fusion.T
                            node.P_k_k[2*self.dim_d:, 2*self.dim_d:] = Pb_k_k_fusion

                    ## 利用融合后的偏差，以及融合更新后的目标状态，对假设树从上到下更新一遍【偏差传递机制】
                    sensor_name_fusion = sensor_config['Name'] # 被更新的传感器名字
                    parent_ids = node_ids_state_fusion
                    children_ids = []
                    if len(parent_ids) >= 1:
                        for layer_id in range(1, self.Tree_maxlen-1):
                            sensor_config = self.Sensor_Config[layer_id]
                            branch_nodes = self.Tree[layer_id]
                            timestamp = self.Timestamps[layer_id]
                            for id in parent_ids:
                                children_ids.extend(branch_nodes[id].children_ids)
                            ## 挨个更新下一层 children_ids 的节点的目标状态，如果传感器相同，偏差也需要修改
                            sensor_config_next = self.Sensor_Config[layer_id+1]
                            branch_nodes_next = self.Tree[layer_id+1]
                            timestamp_next = self.Timestamps[layer_id+1]
                            obs_next = self.obs_s[layer_id+1]
                            dt = timestamp_next - timestamp
                            F_t, G_t = Get_F_G_CV(T=dt, dim_d=self.dim_d)
                            # 该时刻有传感器偏差，需要分情况讨论
                            if sensor_config_next['Is_Biased'] and not sensor_config_next['Biased_Ignore']:
                                F, G = Get_F_G_CV_Bias(T=dt, dim_d=self.dim_d)
                                Bias_Qk = sensor_config_next['Bias_Qk']
                                Qk = block_diag(self.Q_k, Bias_Qk)
                                # 两次传感器相同，直接预测目标+偏差状态
                                if sensor_config_next['Name'] == sensor_config['Name']:
                                    for id in children_ids:
                                        node = branch_nodes_next[id]
                                        if node.hyp_type == 'death':
                                            continue
                                        parent_node = branch_nodes[node.parent_id]
                                        [x_k_k_1, P_k_k_1] = KF_Prediction_CV(x_k_1_k_1=parent_node.x_k_k, \
                                            P_k_1_k_1=parent_node.P_k_k, F=F, G=G, Q=Qk)
                                        # 不同情况下的更新代码相同
                                        parent_node.x_k1_k, parent_node.P_k1_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
                                        node.x_k_k_1, node.P_k_k_1 = x_k_k_1, P_k_k_1
                                        if node.hyp_type == 'miss-detect':
                                            node.x_k_k, node.P_k_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
                                        elif node.hyp_type == 'detect':
                                            io = node.obs_id[0] # 假设点目标
                                            [x_k_k, P_k_k] = KF_Update(x_k_k_1=x_k_k_1, P_k_k_1=P_k_k_1, z_k=obs_next[io], \
                                                sensor_config=sensor_config_next, dim_d=self.dim_d)
                                            node.x_k_k, node.P_k_k = x_k_k, P_k_k

                                # 两次传感器不同，则预测目标状态，预测偏差状态，再拼接【是否考虑相关性】
                                else:
                                    b_k_1 = self.Bias[sensor_config_next['Name']]['b']
                                    Pb_k_1 = self.Bias[sensor_config_next['Name']]['Pb']
                                    timestamp_b = self.Bias[sensor_config_next['Name']]['timestamp']
                                    dt_b = timestamp_next-timestamp_b
                                    F_b, G_b = Get_F_G_Bias(T=dt_b, dim_d=self.dim_d)
                                    [b_k_k_1, Pb_k_k_1] = KF_Prediction_CV(x_k_1_k_1=b_k_1, \
                                            P_k_1_k_1=Pb_k_1, F=F_b, G=G_b, Q=Bias_Qk)
                                    for id in children_ids:
                                        node = branch_nodes_next[id]
                                        if node.hyp_type == 'death':
                                            continue
                                        parent_node = branch_nodes[node.parent_id]
                                        [x_k_k_1_t, P_k_k_1_t] = KF_Prediction_CV(x_k_1_k_1=parent_node.x_k_k[:2*self.dim_d,:], \
                                            P_k_1_k_1=parent_node.P_k_k[:2*self.dim_d,:2*self.dim_d], F=F_t, G=G_t, Q=self.Q_k)
                                        x_k_k_1 = np.concatenate([x_k_k_1_t, b_k_k_1], axis=0)
                                        P_k_k_1 = block_diag(P_k_k_1_t, Pb_k_k_1)
                                        Ptb = Cross_Covariance_CKF(x_k_k_1_t, P_k_k_1_t, b_k_k_1, Pb_k_k_1) # CKF采点考虑互协方差
                                        P_k_k_1[:2*self.dim_d, 2*self.dim_d:] = Ptb
                                        P_k_k_1[2*self.dim_d:, :2*self.dim_d] = Ptb.T
                                        # 不同情况下的更新代码相同
                                        parent_node.x_k1_k, parent_node.P_k1_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
                                        node.x_k_k_1, node.P_k_k_1 = x_k_k_1, P_k_k_1
                                        if node.hyp_type == 'miss-detect':
                                            node.x_k_k, node.P_k_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
                                        elif node.hyp_type == 'detect':
                                            io = node.obs_id[0] # 假设点目标
                                            [x_k_k, P_k_k] = KF_Update(x_k_k_1=x_k_k_1, P_k_k_1=P_k_k_1, z_k=obs_next[io], \
                                                sensor_config=sensor_config_next, dim_d=self.dim_d)
                                            node.x_k_k, node.P_k_k = x_k_k, P_k_k

                            # 该时刻无偏差，直接预测目标状态
                            else: 
                                for id in children_ids:
                                    node = branch_nodes_next[id]
                                    if node.hyp_type == 'death':
                                        continue
                                    parent_node = branch_nodes[node.parent_id]
                                    [x_k_k_1, P_k_k_1] = KF_Prediction_CV(x_k_1_k_1=parent_node.x_k_k[:2*self.dim_d,:], \
                                        P_k_1_k_1=parent_node.P_k_k[:2*self.dim_d,:2*self.dim_d], F=F_t, G=G_t, Q=self.Q_k)
                                    # 不同情况下的更新代码相同
                                    parent_node.x_k1_k, parent_node.P_k1_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
                                    node.x_k_k_1, node.P_k_k_1 = x_k_k_1, P_k_k_1
                                    if node.hyp_type == 'miss-detect':
                                        node.x_k_k, node.P_k_k = deepcopy(x_k_k_1), deepcopy(P_k_k_1)
                                    elif node.hyp_type == 'detect':
                                        io = node.obs_id[0] # 假设点目标
                                        [x_k_k, P_k_k] = KF_Update(x_k_k_1=x_k_k_1, P_k_k_1=P_k_k_1, z_k=obs_next[io], \
                                            sensor_config=sensor_config_next, dim_d=self.dim_d)
                                        node.x_k_k, node.P_k_k = x_k_k, P_k_k
                            
                            parent_ids = copy(children_ids)
                            children_ids = []

                            # 针对新生航迹，只需要把融合后的偏差预测替换一下即可
                            if sensor_config_next['Name'] == sensor_name_fusion:
                                b_k_1 = self.Bias[sensor_config_next['Name']]['b']
                                Pb_k_1 = self.Bias[sensor_config_next['Name']]['Pb']
                                timestamp_b = self.Bias[sensor_config_next['Name']]['timestamp']
                                dt_b = timestamp_next-timestamp_b
                                F_b, G_b = Get_F_G_Bias(T=dt_b, dim_d=self.dim_d)
                                [b_k_k_1, Pb_k_k_1] = KF_Prediction_CV(x_k_1_k_1=b_k_1, \
                                        P_k_1_k_1=Pb_k_1, F=F_b, G=G_b, Q=Bias_Qk)
                                for node in branch_nodes_next.values():
                                    if node.hyp_type == 'initial':
                                        node.x_k_k_1[2*self.dim_d:,:] = b_k_k_1
                                        node.x_k_k[2*self.dim_d:,:] = b_k_k_1
                                        node.P_k_k_1[2*self.dim_d:,2*self.dim_d:] = Pb_k_k_1
                                        node.P_k_k[2*self.dim_d:,2*self.dim_d:] = Pb_k_k_1
                                        parent_ids.append(node.id)

            ## 保存确认航迹【需要考虑是否只考虑确认航迹】
            decided_branch = {}
            for node_id, node in self.Tree[1].items():
                # if node.track_resolved and node.global_lives >= node.initial_global_lives: # 确认航迹+漏检要求
                if node.track_resolved: # 只存确认航迹
                # if node.hyp_type != 'death': # 存MHT选出来的根节点
                    decided_branch[node_id] = node
            self.Decided_Tree.append(decided_branch)

            ## 保存输出航迹【目前输出最底层】
            resolved_labels = [] # 当前已确认航迹的label
            for node in decided_branch.values():
                resolved_labels.append(node.label)
            output_branch = {}
            for node_id, node in self.Tree[-1].items():
                if self.assignment[node_id] > 0.5:
                    if node.label in resolved_labels and node.hyp_type == 'detect':
                    # if node.label in resolved_labels and node.hyp_type != 'death':
                        output_branch[node_id] = node
            self.Output_Nodes.append(output_branch)

            ## 删除第一层滑窗保存数据
            del self.Tree[0], self.max_ids[0], self.Timestamps[0], self.obs_s[0], self.Sensor_Config[0], self.info_s[0]

    def Merge(self):
        """
        航迹融合【在Update之后，对根节点（可以进一步增加判断条件，比如航迹确认的）进行判断】
        """
        root_nodes = self.Tree[0]
        if len(root_nodes) < 2:
            return # 节点太少，就不管了
        ## 两两对比根节点，把相近的节点id存在 merge_node_ids 元素中
        root_nodes_items = list(root_nodes.items())
        root_nodes_pairs = list(combinations(root_nodes_items, 2))
        merge_node_ids = []
        for (node_id_1, node_1), (node_id_2, node_2) in root_nodes_pairs:
            pos_1 = node_1.x_k_k[:self.dim_d, :]
            pos_2 = node_2.x_k_k[:self.dim_d, :]
            delta_pos = pos_1 - pos_2
            delta_dis = np.linalg.norm(delta_pos)
            if delta_dis <= self.Merge_Threshold: # 距离近，融合一下【粗暴融合，目前就选分数高的，一般分数高，证明活得久】
                new_merge_node_id = [node_id_1, node_id_2]
                whether_already_exist = False
                for existed_merge_id in merge_node_ids:
                    if (node_id_1 in existed_merge_id) or (node_id_2 in existed_merge_id):
                        existed_merge_id.extend(new_merge_node_id)
                        whether_already_exist = True
                        break
                if not whether_already_exist:
                    merge_node_ids.append(new_merge_node_id)
        
        if len(merge_node_ids) == 0:
            return
        ## 整理两两对比结果
        for id_ in range(len(merge_node_ids)):
            existed_merge_id = merge_node_ids[id_]
            existed_merge_id = list(set(existed_merge_id)) # 保证唯一性
            merge_node_ids[id_] = existed_merge_id
        if len(merge_node_ids) >= 2:
            merge_node_ids = merge_lists(merge_node_ids) # 再合并一下
        ## 挨个处理需要融合的结果
        for merge_ids in merge_node_ids:
            scores = []
            for id_ in merge_ids:
                score_ = float(root_nodes[id_].score)
                scores.append(score_)
            scores = np.array(scores)
            max_index = np.argmax(scores)
            max_id = merge_ids[max_index] # 分数最大的这个节点保留，其他删掉
            del_ids = [] # 需要删除的节点id
            for id_ in merge_ids:
                if id_ != max_id:
                    del_ids.append(id_)
            # 开始删掉节点
            parent_del_ids = del_ids
            for branch in self.Tree:
                children_del_ids = []
                for node_id in parent_del_ids:
                    children_del_ids.extend(branch[node_id].children_ids)
                for parent_del_id in parent_del_ids:
                    if parent_del_id in branch.keys():
                        del branch[parent_del_id]
                parent_del_ids = children_del_ids

    def Debug(self):
        if self.Debug_Params['Begin_Frame'] <= self.time_k:
            plot_hyp(self.Tree, self.assignment, self.time_k)
            print('show track tree')

    def Wether_In_Sensor_Volume(self, x_k1_k_pre_pos):
        if self.Consider_Target_Volume:
            return Cal_Target_In_Sensor_Volume(self.sensor_config, x_k1_k_pre_pos)
        else:
            return True
