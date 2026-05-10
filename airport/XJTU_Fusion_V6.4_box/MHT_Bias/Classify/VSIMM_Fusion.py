import numpy as np
from Classify.Extract_Model import Extract_Model
from Classify.Weight_Fusion import Weight_Fusion
from Classify.CT_Acc import CT_Acc


def VSIMM_Fusion(Model_IMM, Qs):
    Model_Num = len(Model_IMM)
    xk_1_IMM = np.zeros((9, Model_Num))
    Pk_1_IMM = np.zeros((9, 9, Model_Num))
    mu = np.zeros(Model_Num)
    
    # 将Model_IMM信息提取出来，方便融合
    for im in range(Model_Num):
        model = Model_IMM[im]
        X, P, prob = Extract_Model(model, Model_IMM[0])  # 按照CT模型的w对CV进行扩维
        xk_1_IMM[:, im] = X[:, 0]  # 确保X是一维数组
        Pk_1_IMM[:, :, im] = P
        mu[im] = prob
    
    # 进行加权融合
    X_fusion, P_fusion = Weight_Fusion(xk_1_IMM, Pk_1_IMM, mu)
    
    # 输出
    X_CT = xk_1_IMM[:, 0]
    P_CT = Pk_1_IMM[:, :, 0]
    # Acc_CT = np.cross(X_CT[6:9], X_CT[3:6])
    Acc_CT, Acc_CT_Cov = CT_Acc(X_CT[3:9], P_CT[3:9, 3:9])
    
    Acc_Fusion = np.zeros(3)
    for im in range(Model_Num):
        model = Model_IMM[im]
        prob = model['prob']
        if model['Flag'] == 'CT':
            acc = Acc_CT
        elif model['Flag'] == 'CV_input_0':
            acc = model['input']
        elif model['Flag'] == 'CP_input':
            acc = model['input'][3:6]
        else:
            raise ValueError('error in VSIMM_Fusion')
        
        # 确保acc是一维数组
        if acc.ndim == 2 and acc.shape[1] == 1:
            acc = acc[:, 0]
        
        Acc_Fusion += prob * acc
    
    X_P_k = {
        'X': X_fusion,
        'P': P_fusion,
        'probs': mu,
        'Acc_Fusion': Acc_Fusion,
        'input': Model_IMM[-1]['input'],
        'Acc_CT': Acc_CT,
        'CT_est': xk_1_IMM[0:6, 0],
        'CV_est': xk_1_IMM[0:6, 1],
        'CP_est': xk_1_IMM[0:6, 2],
        'CP_PK': Pk_1_IMM[0:6, 0:6, 2]
    }
    
    return X_P_k