import numpy as np

def Weight_Fusion(xk_1_IMM, Pk_1_IMM, mu):
    """
    加权融合函数
    :param xk_1_IMM: 状态向量，形状为 (nx, model_num)
    :param Pk_1_IMM: 协方差矩阵，形状为 (nx, nx, model_num)
    :param mu: 模型概率，形状为 (model_num,)
    :return: 融合后的状态向量 X_fusion 和协方差矩阵 P_fusion
    """
    nx, Model_Num = xk_1_IMM.shape
    X_fusion = xk_1_IMM @ mu
    P_fusion = np.zeros((nx, nx))
    
    for im in range(Model_Num):
        diff = xk_1_IMM[:, im].reshape(-1, 1) - X_fusion.reshape(-1, 1)
        P_fusion += mu[im] * (Pk_1_IMM[:, :, im] + diff @ diff.T)
    
    return X_fusion, P_fusion