from Classify.Models_z_Pzz_Prediction import Models_z_Pzz_Prediction
from Classify.VSIMM_Fusion import VSIMM_Fusion
from Classify.Weight_Fusion import Weight_Fusion
from Classify.VSIMM_Mix import VSIMM_Mix
from Classify.Models_Prediction import Models_Prediction
from Classify.VSIMM_update import VSIMM_update
from Classify.nonlinear_constrain_bird import nonlinear_constrain_bird
from Classify.nonlinear_constrain_uav import nonlinear_constrain_uav
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
import warnings
warnings.filterwarnings("ignore")
def Tracking(flag, k, z_k, Model_IMM, TPM, Qs, fitting_data, T, Result_k, R):
    # 初始化部分
    nonlinear_time = 100000
    if flag == 'bird':
        Vel_Acc_min1 = np.array([[-45], [-60], [-55], [-35], [-50], [-65]])
        Vel_Acc_max1 = np.array([[45], [60], [55], [35], [50], [65]])
        fitting_mu = np.zeros((6, 1))
        fitting_Sigma = fitting_data['New_Bird_Sigma']
    elif flag == 'UAV':
        Vel_Acc_min1 = np.array([[-22], [-20], [-6.5], [-7.5], [-10.5], [-4.5]])
        Vel_Acc_max1 = np.array([[22], [20], [6.5], [7.5], [10.5], [4.5]])
        fitting_mu = np.zeros((6, 1))
        fitting_Sigma = fitting_data['UAV_Sigma']
    else:
        raise ValueError('error in BMA_acc_select')

    Model_Num = len(Model_IMM)
    Hs = {'Flag': 'Pos', 'R': R}  # 量测模型
    Z_pre_result = {
        'Z_Pre': np.zeros((3, Model_Num)),
        'Pzz_Pre': np.zeros((3, 3, Model_Num)),
        'prob': np.zeros((Model_Num, 1))
    }

    if k == 1:  # 第一帧初始化
        for im in range(Model_Num):
            if Model_IMM[im]['Flag'] == 'CT':
                Model_IMM[im]['X_k'][0:3, :] = z_k.reshape(-1, 1)
                Model_IMM[im]['X_k'][6:9, :] = 0.001 * np.ones((3, 1))
            else:
             Model_IMM[im]['X_k'][0:3, :] = z_k.reshape(-1, 1)
        Result_k = VSIMM_Fusion(Model_IMM, Qs)
    else:
        # 构建G矩阵
        G_CT = np.vstack([np.zeros((6, 3)), np.eye(3) * T])
        G_CV = np.vstack([np.eye(3) * (T**2)/2, np.eye(3) * T])
        G_CP_u = np.vstack([
            np.hstack([np.eye(3) * T, np.eye(3) * (T**2)/2]),
            np.hstack([np.eye(3), np.zeros((3, 3))])
        ])
        G_CP = G_CP_u.copy()
        Gs = {
            'G_CT': G_CT,
            'G_CV': G_CV,
            'G_CP_u': G_CP_u,
            'G_CP': G_CP
        }

        # 构建F矩阵
        T = float(T)
        F_CV = np.array([
            [1, 0, 0, T, 0, 0],
            [0, 1, 0, 0, T, 0],
            [0, 0, 1, 0, 0, T],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        F_CP = block_diag(np.eye(3), np.zeros((3, 3)))
        Fs = {
            'F_CV': F_CV,
            'F_CP': F_CP
        }

        # VSIMM流程
        Model_IMM = VSIMM_Mix(Model_IMM, TPM)
        Model_IMM = Models_Prediction(Model_IMM, T, Fs, Gs, Qs)
        Model_IMM = Models_z_Pzz_Prediction(Model_IMM, Hs)

        # 准备数据
        x_k_k_1_Model_IMM = np.zeros((6, Model_Num))
        Pxx_Model_IMM = np.zeros((6, 6, Model_Num))
        mu = np.zeros((Model_Num, 1))
        acc_k_k_1Model_IMM = np.zeros((3, Model_Num))

        # 计算各模型加速度
        acc_k_k_1Model_IMM[:, [0]] = np.cross(
            Model_IMM[0]['X_k_k_1'][6:9, :],
            Model_IMM[0]['X_k_k_1'][3:6, :],
            axis=0
        )
        acc_k_k_1Model_IMM[:, [1]] = np.zeros((3, 1))
        acc_k_k_1Model_IMM[:, [2]] = Model_IMM[2]['input'][3:6, :]

        for im in range(Model_Num):
            x_k_k_1_Model_IMM[:, [im]] = Model_IMM[im]['X_k_k_1'][0:6, 0][:, None]
            Pxx_Model_IMM[:, :, im] = Model_IMM[im]['P_k_k_1'][0:6, 0:6]
            mu[im] = Model_IMM[im]['prob']

        # 融合
        z_fusion, Pzz_fusion = Weight_Fusion(x_k_k_1_Model_IMM, Pxx_Model_IMM, mu)
        acc_fusion_k_k_1 = acc_k_k_1Model_IMM @ mu

        # 优化部分
        G_u = Gs['G_CP_u']
        G = Gs['G_CP']
        F = Fs['F_CP']
        Q = Qs['Q_CP']

        X_k_1 = Result_k['X'][0:6].reshape(-1,1)
        P_k_1 = Result_k['P'][0:6, 0:6]

        def fun(x):
            x = x.reshape(-1, 1)
            term1 = F @ P_k_1 @ F.T + G @ ((x - fitting_mu) @ (x - fitting_mu).T + fitting_Sigma) @ G.T
            term2 = z_fusion - (F @ X_k_1 + G @ x)
            return float(np.log(np.linalg.det(term1)) - np.log(np.linalg.det(Pzz_fusion)) + \
                   np.trace(Pzz_fusion @ np.linalg.inv(term1)) + \
                   term2.T @ np.linalg.inv(term1) @ term2)

        X0 = np.zeros((6, 1))
        bounds = [(min_val[0], max_val[0]) for min_val, max_val in zip(Vel_Acc_min1, Vel_Acc_max1)]

        if k < nonlinear_time:
            res = minimize(fun, X0.flatten(), bounds=bounds, method='SLSQP', options={'disp': False})
            vel_acc_opt = res.x.reshape(-1, 1)
        else:
            if flag == 'bird':
                # 需要实现nonlinear_constrain_bird
                res = minimize(fun, X0.flatten(), bounds=bounds, constraints=nonlinear_constrain_bird, 
                              method='SLSQP', options={'disp': False})
            elif flag == 'UAV':
                # 需要实现nonlinear_constrain_uav
                res = minimize(fun, X0.flatten(), bounds=bounds, constraints=nonlinear_constrain_uav,
                              method='SLSQP', options={'disp': False})
            else:
                raise ValueError('优化约束选择错误')
            vel_acc_opt = res.x.reshape(-1, 1)

        # 更新模型
        Model_IMM[-1]['input'] = vel_acc_opt
        Qs['Q_CP'] = (vel_acc_opt - fitting_mu) @ (vel_acc_opt - fitting_mu).T + fitting_Sigma

        # 重新预测
        best_models = [Model_IMM[-1]]
        best_models = Models_Prediction(best_models, T, Fs, Gs, Qs)
        best_models = Models_z_Pzz_Prediction(best_models, Hs)
        Model_IMM[-1] = best_models[0]

        # 更新和融合
        Model_IMM = VSIMM_update(Model_IMM, z_k, Hs)

        # 存储预测结果
        pre_mea = {
            'Z_Pre': np.zeros((3, Model_Num)),
            'Pzz_Pre': np.zeros((3, 3, Model_Num)),
            'prob': np.zeros((Model_Num, 1))
        }
        for im in range(Model_Num):
            pre_mea['Z_Pre'][:, [im]] = Model_IMM[im]['z']
            pre_mea['Pzz_Pre'][:, :, im] = Model_IMM[im]['Pzz']
            pre_mea['prob'][im] = Model_IMM[im]['prob']
        Z_pre_result = pre_mea

        Result_k = VSIMM_Fusion(Model_IMM, Qs)

    return Model_IMM, Qs, Result_k, Z_pre_result