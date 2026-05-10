from copy import deepcopy
from typing import Dict, List, Type, Optional, Tuple
import numpy as np
import cvxpy as cp
from scipy.linalg import cholesky, block_diag, eigh
   

def Get_H_k(
    sensor_config: Dict,
    dim_d: int = 3,
) -> np.ndarray:
    """
    根据目标扩维状态 x_k_k_1 和传感器配置 sensor_config，得到量测矩阵H_k
    前 2*dim_d 维是位置速度，后 dim_d 维是传感器偏差
    """
    if sensor_config['Meas_Type']=="Position":
        if sensor_config['Is_Biased'] and not sensor_config['Biased_Ignore']:
            H_k = np.zeros((dim_d, 3*dim_d))
            H_k[0,0], H_k[1,1], H_k[2,2] = 1.0, 1.0, 1.0
            H_k[0,-3], H_k[1,-2], H_k[2,-1] = 1.0, 1.0, 1.0
        else:
            H_k = np.zeros((dim_d, 2*dim_d))
            H_k[0,0], H_k[1,1], H_k[2,2] = 1.0, 1.0, 1.0
    else:
        raise NotImplementedError

    return H_k

def Get_F_G_CV(
    T,
    dim_d: int = 3,
) -> Tuple:
    """
    CV模型的F和G
    """
    F = np.array([[1.0, T], [0.0, 1.0]])
    F = np.kron(F, np.identity(dim_d))
    G = np.array([[T*T/2.0], [T]])
    G = np.kron(G, np.identity(dim_d))
    return F, G

def Get_F_G_CP(
    T,
    dim_d: int = 3,
) -> Tuple:
    """
    CP模型的F和G
    """
    F = np.identity(dim_d)
    G = np.identity(dim_d)*T
    return F, G

def Get_F_G_Bias(
    T,
    dim_d: int = 3,
    lam_bias: float = 1e-3 # 1e-3 1e2
) -> Tuple:
    """
    b_k = e^(-\lam*T)b_{k-1}
    """
    F = np.identity(dim_d)*np.exp(-lam_bias*T)
    G = np.identity(dim_d)*T
    return F, G

def Get_F_G_CV_Bias(
    T,
    dim_d: int = 3,
    lam_bias: float = 1e-3 # 1e-3 1e2
) -> Tuple:
    """
    CV 模型的F和G，对角阵加上偏差的F和G
    """
    F_t, G_t = Get_F_G_CV(T, dim_d=dim_d)
    F_b, G_b = Get_F_G_Bias(T, dim_d=dim_d, lam_bias=lam_bias)
    G_b = np.identity(dim_d)*T
    F = block_diag(F_t, F_b)
    G = block_diag(G_t, G_b)
    return F, G

def KF_Prediction_CV(x_k_1_k_1, P_k_1_k_1, F, G, Q):
    x_k_k_1 = F @ x_k_1_k_1
    P_k_k_1 = F @ P_k_1_k_1 @ (F.T) + G @ Q @ (G.T)
    return x_k_k_1, P_k_k_1

def KF_Update(x_k_k_1, P_k_k_1, z_k, sensor_config, dim_d: int = 3):
    H_k = Get_H_k(sensor_config, dim_d)
    R_k = sensor_config['R']
    Pxz = P_k_k_1 @ (H_k.T)
    Pzz = H_k @ P_k_k_1 @ (H_k.T) + R_k
    K = np.matmul(Pxz, np.linalg.inv(Pzz))
    x_k_k = x_k_k_1 + K@(z_k - H_k@x_k_k_1)
    P_k_k = P_k_k_1 - K@Pzz@(K.T)
    return x_k_k, P_k_k

def Cal_Gate(Pg, dim_d):
    gate = None
    if np.abs(Pg-1) <= 1e-6: # 接近于不考虑波门
        gate = 1e2
    
    if np.abs(Pg-0.99) <= 1e-6:
        if dim_d == 3:
            gate = 11.34
        elif dim_d == 2:
            gate = 9.21
    elif np.abs(Pg-0.999) <= 1e-6:
        if dim_d == 3:
            gate = 16.266
        elif dim_d == 2:
            gate = 13.816
    elif np.abs(Pg - 0.95) <= 1e-6:  # 新增：Pg=0.95
        if dim_d == 3:
            gate = 7.81  # 自由度3的卡方分布95%分位数
        elif dim_d == 2:
            gate = 5.99  # 自由度2的卡方分布95%分位数

    elif np.abs(Pg - 0.92) <= 1e-6:  # 新增：Pg=0.92
        if dim_d == 3:
            gate = 6.7492  # 自由度3的卡方分布92%分位数
        elif dim_d == 2:
            gate = 5.0232  # 自由度2的卡方分布92%分位数
    elif np.abs(Pg - 0.91) <= 1e-6:  # 新增：Pg=0.92
        if dim_d == 3:
            gate = 6.4924  # 自由度3的卡方分布92%分位数
        elif dim_d == 2:
            gate = 4.8057  # 自由度2的卡方分布92%分位数
    elif np.abs(Pg-0.9) <= 1e-6:
        if dim_d == 3:
            gate = 6.25
        elif dim_d == 2:
            gate = 4.61
    elif np.abs(Pg-0.85) <= 1e-6:
        if dim_d == 3:
            gate = 5.39
        elif dim_d == 2:
            gate = 3.94
    elif np.abs(Pg-0.8) <= 1e-6:
        if dim_d == 3:
            gate = 4.64
        elif dim_d == 2:
            gate = 3.22
    return gate

def Cal_Radar_Volume(range, pitch, yaw):
    """
    计算三维雷达探测范围的体积
    距离(m) 俯仰角(中心到边界的角度,rad) 方位角(中心到边界的角度,rad)
    """
    pitch = pitch / 180.0*np.pi
    yaw = yaw / 180.0*np.pi
    volume = 2.0/3.0*np.power(range,3)*np.sin(pitch/2.0)*yaw
    return volume

def nearest_pd_robust(A, max_attempts=5, min_eig_threshold=1e-12):
    """强化版最近正定矩阵计算，处理多负特征值情况"""
    # 输入校验
    if A.ndim != 2 or A.shape  [0] != A.shape  [1]:
        raise ValueError("输入必须为方阵")
    
    # 预处理：强制对称并保存原始迹
    A_sym = 0.5 * (A + A.T)
    original_trace = np.trace(A_sym)
    
    for attempt in range(max_attempts):
        # 特征值分解（使用加速的对称矩阵分解）
        eigvals, eigvecs = eigh(A_sym, check_finite=False)
        
        # 检测负特征值
        neg_mask = eigvals < min_eig_threshold
        if not np.any(neg_mask):
            break  # 已经是正定矩阵
        
        # 动态计算修正量：基于最大负特征值的绝对值
        max_neg = np.abs(np.min(eigvals[neg_mask])) if np.any(neg_mask) else 0
        epsilon = max_neg + min_eig_threshold
        
        # 构造修正矩阵（非对角元素保持，仅修正特征值）
        eigvals_corrected = np.where(neg_mask, eigvals + epsilon, eigvals)
        
        # 重建矩阵
        A_sym = eigvecs @ np.diag(eigvals_corrected) @ eigvecs.T
        
        # 保持迹守恒（关键步骤！）
        current_trace = np.trace(A_sym)
        trace_diff = original_trace - current_trace
        if trace_diff > 0:
            A_sym += np.eye(A_sym.shape  [0]) * (trace_diff / A_sym.shape  [0])
        
        # 再次强制对称
        A_sym = 0.5 * (A_sym + A_sym.T)
    else:
        raise RuntimeError(f"经过{max_attempts}次修正仍未获得正定矩阵")
    
    return A_sym

def robust_scipy_cholesky(A, lower=True, diagnostics=False):
    """工业级鲁棒Cholesky分解"""
    # 初始检查
    if not (A.ndim == 2 and A.shape  [0] == A.shape  [1]):
        raise ValueError("输入必须是方阵")
    
    # 预处理：强制对称
    A_sym = 0.5 * (A + A.T)
    
    # 分解尝试
    try:
        L = cholesky(A_sym, lower=lower, check_finite=False)
        if diagnostics:
            # print("首次分解成功")
            pass
        return L
    except np.linalg.LinAlgError:
        if diagnostics:
            print(f"初始分解失败，执行修正...")
        
        # 执行强化修正
        A_corrected = nearest_pd_robust(A_sym)
        
        # 验证修正结果
        min_eig = np.linalg.eigvalsh(A_corrected).min()
        if min_eig < -1e-14:
            raise RuntimeError("修正失败：矩阵仍非正定")
        
        # 最终分解
        try:
            L = cholesky(A_corrected, lower=lower, check_finite=False)
            if diagnostics:
                print(f"修正后分解成功，最小特征值：{min_eig:.2e}")
            return L
        except np.linalg.LinAlgError:
            raise RuntimeError("无法分解修正后的矩阵")

def Cross_Covariance_CKF(x1, P1, x2, P2):
    """
    通过CKF采点，计算两个变量之间的互协方差矩阵
    """
    ## calculate X1_sigma
    n1 = x1.shape[0]
    m1 = 2*n1
    X1_sigma = np.zeros((n1, m1)) # 每一列对应一个采样点
    # P1_sqrt = cholesky(P1, lower=True)
    P1_sqrt = robust_scipy_cholesky(P1, lower=True, diagnostics=True)
    w1 = 1.0/(m1*1.0)
    for i in range(m1):
        epi = np.zeros((n1, 1))
        if i >= n1:
            epi[i-n1,0] = -1.0
        else:
            epi[i,0] = 1.0
        X1_sigma[:,i] = x1[:,0] + (P1_sqrt@(epi)*np.sqrt(n1*1.0))[:,0]

    ## calculate X1_sigma
    n2 = x2.shape[0]
    m2 = 2*n2
    X2_sigma = np.zeros((n2, m2)) # 每一列对应一个采样点
    # P2_sqrt = cholesky(P2, lower=True)
    P2_sqrt = robust_scipy_cholesky(P2, lower=True, diagnostics=True)
    w2 = 1.0/(m2*1.0)
    for i in range(m2):
        epi = np.zeros((n2, 1))
        if i >= n2:
            epi[i-n2,0] = -1.0
        else:
            epi[i,0] = 1.0
        X2_sigma[:,i] = x2[:,0] + (P2_sqrt@(epi)*np.sqrt(n2*1.0))[:,0]

    ## 计算互协方差
    Px1x2 = np.zeros((n1, n2))
    for i in range(m1):
        for j in range(m2):
            Px1x2 = Px1x2 + w1*w2*(X1_sigma[:,i].reshape(-1,1)-x1)@((X2_sigma[:,j].reshape(-1,1)-x2).T)
    
    return Px1x2