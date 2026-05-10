import numpy as np

def IMM_Mix(xk_1_IMM, Pk_1_IMM, mu, TPM):
    Model_Num = mu.shape[0]
    x_dim = xk_1_IMM.shape[0]
    ck_1 = (mu.T @ TPM).T  # TPM(i,j)意味着模型i转移为模型j的概率
    muij = np.zeros((Model_Num, Model_Num))
    
    for j in range(Model_Num):
        for i in range(Model_Num):
            muij[i, j] = mu[i] * TPM[i, j] / ck_1[j]
    
    xk_1_mix = np.zeros((x_dim, Model_Num))
    for j in range(Model_Num):
        for i in range(Model_Num):
            xk_1_mix[:, j] += xk_1_IMM[:, i] * muij[i, j]
    
    Pk_1_mix = np.zeros((x_dim, x_dim, Model_Num))
    for j in range(Model_Num):
        for i in range(Model_Num):
            diff = xk_1_mix[:, j] - xk_1_IMM[:, i]
            Pk_1_mix[:, :, j] += muij[i, j] * (Pk_1_IMM[:, :, i] + np.outer(diff, diff))
    
    return xk_1_mix, Pk_1_mix, ck_1