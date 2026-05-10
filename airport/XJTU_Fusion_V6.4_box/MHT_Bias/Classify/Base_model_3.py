import numpy as np
from scipy.linalg import block_diag

def Base_model_3(R):
    """
    基础模型（1CT + 1CV_input）+ 最优模型（1个CV_input）
    :param R: 过程噪声协方差矩阵
    :return: model_num, Model_BASE, TPM
    """
    # 模型个数
    model_num = 3  # CT + 1*CV-input + 待选最优CV-input
    # 模型初始概率
    prob_CT = 1.0 / model_num
    prob_CV_input = (1 - prob_CT) / (model_num - 1)
    prob_CP_input = (1 - prob_CT) / (model_num - 1)
    
    # 基础模型集中的CT
    Model_CT = {
        'Flag': 'CT',
        'prob': prob_CT,
        'X_k': np.vstack((np.zeros((6, 1)), np.zeros((3, 1)))),
        'P_k': block_diag(R, np.eye(3) * 10**2, np.eye(3) * 10**2),
        'z': np.zeros((3, 1)),
        'Pzz': np.eye(3, 3),
        'X_k_k_1': np.zeros((9, 1)),
        'P_k_k_1': block_diag(R, np.eye(3) * 10**2, np.eye(3) * 10**2)
    }
    
    # 基础模型集中的[0 0 0]输入CV_input
    Model_CV0 = {
        'Flag': 'CV_input_0',
        'prob': prob_CV_input,
        'X_k': np.zeros((6, 1)),
        'P_k': block_diag(R, np.eye(3) * 10**2),
        'input': np.zeros((3, 1)),
        'z': np.zeros((3, 1)),
        'Pzz': np.eye(3, 3),
        'X_k_k_1': np.zeros((6, 1)),
        'P_k_k_1': block_diag(R, np.eye(3) * 10**2)
    }
    Model_BASE = [Model_CT, Model_CV0]  # 存储VSIMM的两个模型
    
    # 待选最优模型初始为[0 0 0]输入CV_input
    Model_Best = {
        'Flag': 'CP_input',
        'prob': prob_CP_input,
        'X_k': np.zeros((6, 1)),
        'P_k': block_diag(R, np.eye(3) * 10**2),
        'input': np.zeros((6, 1)),
        'z': np.zeros((3, 1)),
        'Pzz': np.eye(3, 3),
        'X_k_k_1': np.zeros((6, 1)),
        'P_k_k_1': block_diag(R, np.eye(3) * 10**2)
    }
    Model_BASE.append(Model_Best)
    
    # 状态转移矩阵
    prob = 0.95
    TPM = np.eye(model_num) * (prob - (1 - prob) / (model_num - 1)) + (1 - prob) / (model_num - 1) * np.ones((model_num, model_num))
    
    return model_num, Model_BASE, TPM