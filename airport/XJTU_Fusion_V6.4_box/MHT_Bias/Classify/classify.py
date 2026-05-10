import numpy as np
import math

def classify(z_k, Z_Pre_bird, Z_Pre_UAV, Log_Likelihood_Ratio):
    Target_type = 0  # 0 为没检验出  1为bird  -1为UAV

    alpha = 0.01
    beta = 0.01
    log_A = np.log((1 - beta) / alpha)
    log_B = np.log(beta / (1 - alpha))

    # UAV
    z_pre_UAV = Z_Pre_UAV['Z_Pre']
    Pzz_pre_UAV = Z_Pre_UAV['Pzz_Pre']
    prob_UAV = Z_Pre_UAV['prob']
    # Bird
    z_pre_bird = Z_Pre_bird['Z_Pre']
    Pzz_pre_bird = Z_Pre_bird['Pzz_Pre']
    prob_bird = Z_Pre_bird['prob']
  

    # 计算似然
    z_bird = z_pre_bird[:, 2]
    z_bird = z_bird.reshape(-1,1)
    P_bird = Pzz_pre_bird[:, :, 2]
    z_bird_det=(np.linalg.det(P_bird) ** (-0.5)) 
    z_bird_exp=-0.5 *((z_bird - z_k).T @ np.linalg.inv(P_bird) @ (z_bird - z_k))
    Bird_Likelihood = ((2 * np.pi) ** (-0.5 * 3)) * z_bird_det * np.exp(z_bird_exp)

    z_UAV = z_pre_UAV[:, 2]
    z_UAV = z_UAV.reshape(-1,1)
    P_UAV = Pzz_pre_UAV[:, :, 2]
    UAV_Likelihood = ((2 * np.pi) ** (-0.5 * 3)) * (np.linalg.det(P_UAV) ** (-0.5)) * np.exp(
        -0.5 * ((z_UAV - z_k).T @ np.linalg.inv(P_UAV) @ (z_UAV - z_k)))

    Log_Likelihood_Ratio_add = np.log(Bird_Likelihood) - np.log(UAV_Likelihood)
    if not math.isfinite(Log_Likelihood_Ratio_add):
        Log_Likelihood_Ratio_add=0
    Log_Likelihood_Ratio = Log_Likelihood_Ratio + Log_Likelihood_Ratio_add

    if Log_Likelihood_Ratio > log_A:
        Target_type = 1
    elif Log_Likelihood_Ratio < log_B:
        Target_type = -1
    else:
        Target_type = 0

    return Target_type, Log_Likelihood_Ratio