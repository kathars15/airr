import numpy as np

def log_norm(log_ori):
    # log_ori：待归一化的对数列向量
    # log_norm：归一化后的对数列向量
    num = log_ori.shape[0]
    log_norm = np.zeros(num)
    max_log = np.max(log_ori)
    exp_sum = 0
    for i in range(num):
        exp_sum += np.exp(log_ori[i] - max_log)
    for i in range(num):
        log_norm[i] = log_ori[i] - max_log - np.log(exp_sum)
    
    return log_norm