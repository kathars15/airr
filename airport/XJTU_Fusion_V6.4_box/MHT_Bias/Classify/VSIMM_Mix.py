from Classify.Extract_Model import Extract_Model
from Classify.IMM_Mix import IMM_Mix
import numpy as np

def VSIMM_Mix(Model_IMM, TPM):
    Model_Num = len(Model_IMM)
    xk_1_IMM = np.zeros((9, Model_Num))  # x v w
    Pk_1_IMM = np.zeros((9, 9, Model_Num))
    mu = np.zeros(Model_Num)
    
    # 将Model_IMM信息提取出来，方便调用IMM_Mix
    for im in range(Model_Num):
        model = Model_IMM[im]
        X, P, prob = Extract_Model(model, Model_IMM[0])  # 交互时谁多出来补谁
        xk_1_IMM[:, im] = X[:, 0]  # 修改这里
        Pk_1_IMM[:, :, im] = P
        mu[im] = prob
    
    # 调用IMM_Mix进行交互
    xk_1_mix, Pk_1_mix, ck_1 = IMM_Mix(xk_1_IMM, Pk_1_IMM, mu, TPM)
    
    # 交互后的值赋值给Model_IMM
    for im in range(Model_Num):
        Model_IMM[im]['prob'] = ck_1[im]
        if Model_IMM[im]['Flag'] == 'CT':
            Model_IMM[im]['X_k'] = xk_1_mix[0:9, im]
            Model_IMM[im]['P_k'] = Pk_1_mix[0:9, 0:9, im]
        elif Model_IMM[im]['Flag'] == 'CV_input_0':
            Model_IMM[im]['X_k'] = xk_1_mix[0:6, im]
            Model_IMM[im]['P_k'] = Pk_1_mix[0:6, 0:6, im]
            # Model_IMM[im]['input'] = xk_1_mix[9:12, im]  # input改成交互后的结果
        elif Model_IMM[im]['Flag'] == 'CP_input':
            Model_IMM[im]['X_k'] = xk_1_mix[0:6, im]
            Model_IMM[im]['P_k'] = Pk_1_mix[0:6, 0:6, im]
        else:
            raise ValueError('error in mix')
    
    return Model_IMM