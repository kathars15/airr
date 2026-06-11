from Classify.CT_Pre import CT_Pre
import numpy as np
from scipy.linalg import block_diag

def Models_Prediction(Models, T, Fs, Gs, Qs):
    """
    预测，即修改Model_IMM每个模型的X和P
    :param Models: 模型列表
    :param T: 时间步长
    :param Fs: 状态转移矩阵集合
    :param Gs: 控制输入矩阵集合
    :param Qs: 过程噪声协方差矩阵集合
    :return: 更新后的模型列表
    """
    Model_Num = len(Models)
    for im in range(Model_Num):
        model = Models[im]
        X_k_1 = model['X_k']
        P_k_1 = model['P_k']
        if model['Flag'] == 'CT':
            X_k_k_1, P_k_k_1 = CT_Pre(X_k_1, P_k_1, T, Gs['G_CT'], Qs['Q_CT'])
        elif model['Flag'] == 'CV_input_0':
            acc_input = model['input']
            F = Fs['F_CV']
            G = Gs['G_CV']
            Q = Qs['Q_CV_0']
            X_k_k_1 = F @ (X_k_1.reshape(-1,1)) + G @ acc_input
            P_k_k_1 = F @ P_k_1 @ F.T + G @ Q @ G.T
        elif model['Flag'] == 'CP_input':
            input_ = model['input']
            F = Fs['F_CP']
            G_u = Gs['G_CP_u']
            G = Gs['G_CP']
            Q = Qs['Q_CP']
            X_k_k_1 = F @ (X_k_1.reshape(-1,1)) + G_u @ input_
            P_k_k_1 = F @ P_k_1 @ F.T + G @ Q @ G.T
        else:
            raise ValueError('error in Model_IMM_Prediction')
        
        Models[im]['X_k_k_1'] = X_k_k_1
        Models[im]['P_k_k_1'] = P_k_k_1
        # if (P_k_k_1.max() > 100):
        #      a=1
        # try:
        #         Sk_1 = np.linalg.cholesky(P_k_k_1).T  # 重新尝试Cholesky分解
        # except np.linalg.LinAlgError:
        #         print(P_k_k_1)
    
    return Models