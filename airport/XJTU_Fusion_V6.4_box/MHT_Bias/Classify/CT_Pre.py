import numpy as np
from scipy.linalg import block_diag, solve, solve_continuous_lyapunov

def CT_Pre(xk_1, Pk_1, T, Gk_CT, Qk_CT):
    nx = xk_1.shape[0]  # x维度
    nw = Qk_CT.shape[0]  # 噪声维度
    w_mu = np.zeros((nw, 1))  # 噪声均值
    xk_1 = xk_1.reshape(-1, 1)  # 将 xk_1 转换为二维数组
    XK_1 = np.concatenate((xk_1, w_mu), axis=0)  # 扩维采点
    PK_1 = block_diag(Pk_1, Qk_CT)
    try:
        Sk_1 = np.linalg.cholesky(PK_1).T  # 重新尝试Cholesky分解
    except np.linalg.LinAlgError:
        print(PK_1)
        # PK_1 = (PK_1 + PK_1.T) / 2 
        # eps = 1e-6
        # PK_1 = PK_1 + eps * np.eye(nx)
        # Sk_1 = np.linalg.cholesky(PK_1).T
        print("PK_1非正定")

    # Sk_1 = np.linalg.cholesky(PK_1).T
    n1 = nx + nw  # 状态+噪声 维度
    m1 = 2 * n1  # CKF采点个数
    kesi = np.sqrt(m1 / 2) * np.hstack((np.eye(n1), -np.eye(n1)))
    weight1 = 1.0 / m1 * np.ones(m1)
    xk_1_sigma = np.zeros((n1, m1))  # k-1时刻采样点
    xkk_1_sigma = np.zeros((nx, m1))  # k时刻预测采样点

    for i in range(m1):
        xk_1_sigma[:, i] = Sk_1 @ kesi[:, i] + XK_1.flatten()
        wx = xk_1_sigma[6, i]
        wy = xk_1_sigma[7, i]
        wz = xk_1_sigma[8, i]
        d1 = wy**2 + wz**2
        d2 = wx**2 + wz**2
        d3 = wx**2 + wy**2
        omega = np.sqrt(wx**2 + wy**2 + wz**2)
        if omega == 0:
            Fk_CT = np.block([
                [np.eye(3), np.eye(3) * T, np.zeros((3, 3))],
                [np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]
            ])
        else:
            c1 = (np.cos(omega * T) - 1) / omega**2
            c2 = np.sin(omega * T) / omega
            c3 = 1 / omega**2 * (np.sin(omega * T) / omega - T)
            A = np.array([
                [c1 * d1, -c2 * wz - c1 * wx * wy, c2 * wy - c1 * wx * wz],
                [c2 * wz - c1 * wx * wy, c1 * d2, -c2 * wx - c1 * wy * wz],
                [-c2 * wy - c1 * wx * wz, c2 * wx - c1 * wy * wz, c1 * d3]
            ])
            B = np.array([
                [c3 * d1, c1 * wz - c3 * wx * wy, -c1 * wy - c3 * wx * wz],
                [-c1 * wz - c3 * wx * wy, c3 * d2, c1 * wx - c3 * wy * wz],
                [c1 * wy - c3 * wx * wz, -c1 * wx - c3 * wy * wz, c3 * d3]
            ])
            Fk_CT = np.block([
                [np.eye(3), B + np.eye(3) * T, np.zeros((3, 3))],
                [np.zeros((3, 3)), np.eye(3) + A, np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]
            ])
        xkk_1_sigma[:, i] = Fk_CT @ xk_1_sigma[:9, i] + Gk_CT @ xk_1_sigma[nx:n1, i]

    XKK_1 = np.zeros((nx, 1))
    for j in range(m1):
        XKK_1 += weight1[j] * xkk_1_sigma[:, j].reshape(nx, 1)

    PKK_1 = np.zeros((nx, nx))
    for j in range(m1):
        delta = xkk_1_sigma[:, j].reshape(nx, 1) - XKK_1
        PKK_1 += weight1[j] * delta @ delta.T
    if (PKK_1.max() > 100):
             a=1
    return XKK_1, PKK_1