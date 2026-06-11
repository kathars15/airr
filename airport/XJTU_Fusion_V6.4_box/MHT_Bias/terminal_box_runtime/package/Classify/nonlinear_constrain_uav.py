import numpy as np

def nonlinear_constrain_uav(x):
    """
    计算无人机的非线性约束条件
    :param x: 输入向量，形状为 (6, 1)
    :return: f_inequality 和 f_equ
    """
    vel_norm_min = 0.1
    vel_norm_max = 21.7
    acc_norm_min = 0
    acc_norm_max = 13

    v = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    a = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
    
    f_inequality = np.array([
        vel_norm_min - v,
        a - 1.4074 * v - 5.859,
        a + 1.1029 * v - 19.666,
        a + 0.4536 * v - 11.679,
        v - vel_norm_max,
        -1.0 * a
    ])
    
    f_equ = np.array([])
    
    return f_inequality, f_equ