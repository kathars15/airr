import numpy as np

def CT_Acc(X_V_W, P_V_W):
    nx = X_V_W.shape[0]  # x维度
    try:
        Sk_1 = np.linalg.cholesky(P_V_W).T  # 重新尝试Cholesky分解
    except np.linalg.LinAlgError:
        print(P_V_W)
        # P_V_W = (P_V_W + P_V_W.T) / 2 
        # eps = 1e-6
        # P_V_W = P_V_W + eps * np.eye(nx)
        # Sk_1 = np.linalg.cholesky(P_V_W).T
        # print("P_V_W非正定已修正：强制矩阵对称 + 微小正则化")
    Sk_1 = np.linalg.cholesky(P_V_W).T
    n1 = nx  # 状态维度
    m1 = 2 * n1  # CKF采点个数
    kesi = np.sqrt(m1 / 2) * np.hstack((np.eye(n1), -np.eye(n1)))
    weight1 = 1.0 / m1 * np.ones(m1)
    xk_1_sigma = np.zeros((n1, m1))  # k-1时刻采样点
    acc_sigma = np.zeros((3, m1))  # 加速度采样点

    for i in range(m1):
        xk_1_sigma[:, i] = Sk_1 @ kesi[:, i] + X_V_W
        acc_sigma[:, i] = np.cross(xk_1_sigma[3:6, i], xk_1_sigma[0:3, i])

    Acc_Mean = np.mean(acc_sigma, axis=1)
    Acc_Cov = np.zeros((3, 3))

    for j in range(m1):
        Acc_Cov += weight1[j] * np.outer(acc_sigma[:, j] - Acc_Mean, acc_sigma[:, j] - Acc_Mean)

    return Acc_Mean, Acc_Cov