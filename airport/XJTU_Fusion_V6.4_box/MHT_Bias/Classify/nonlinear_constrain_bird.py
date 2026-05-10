import numpy as np

def nonlinear_constrain_bird(x):
    """
    计算非线性约束条件
    :param x: 输入向量，形状为 (6, 1)
    :return: f_inequality 和 f_equ
    """

    v = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    a = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
    
    f_inequality = np.array([
        6.75 - v,
        a - 1.707 * v + 7.524,
        a - 2.613 * v + 15.493,
        a - 23.7,
        v - 21.6,
        3.375 * v - a - 67.5,
        -1.0 * a
    ])
    
    f_equ = np.array([])
    
    return f_inequality, f_equ