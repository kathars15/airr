from Classify.Tracking import Tracking
from Classify.classify import classify

def TrackingClassify(Time_N, z_k, T, Bird_Model_IMM_11111, Bird_Qs, Bird_Result_k,
                     UAV_Model_IMM_1111111222, UAV_Qs, UAV_Result_k, Log_Likelihood_Ratio, ConstValue):
    """
    跟踪和分类函数
    :param Time_N: 当前帧
    :param z_k: 量测
    :param T: 时间间隔
    :param Bird_Model_IMM_11111_11111: 存储假设为鸟的多模型
    :param Bird_Qs: 存储假设为鸟的过程噪声
    :param Bird_Result_k: 存储假设为鸟的融合结果
    :param UAV_Model_IMM_1111111222: 存储假设为无人机的多模型
    :param UAV_Qs: 存储假设为无人机的过程噪声
    :param UAV_Result_k: 存储假设为无人机的融合结果
    :param Log_Likelihood_Ratio: 累积似然比
    :param ConstValue: 算法中用的常值，包括R, TPM, fitting_datas
    :return: Target_type, Log_Likelihood_Ratio, Bird_Model_IMM_11111, Bird_Qs, Bird_Result_k, UAV_Model_IMM_1111111222, UAV_Qs, UAV_Result_k
    """
    TPM = ConstValue['TPM']
    fitting_data = ConstValue['fitting_data']
    R = ConstValue['R']
    Target_type = 0

    Bird_Model_IMM_11111, Bird_Qs, Bird_Result_k, Z_pre_result_bird = Tracking('bird', Time_N, z_k, Bird_Model_IMM_11111, TPM, Bird_Qs, fitting_data, T, Bird_Result_k, R)
    UAV_Model_IMM_1111111222, UAV_Qs, UAV_Result_k, Z_pre_result_uav = Tracking('UAV', Time_N, z_k, UAV_Model_IMM_1111111222, TPM, UAV_Qs, fitting_data, T, UAV_Result_k, R)
    
    if Time_N >= 2:
        Target_type, Log_Likelihood_Ratio = classify(z_k, Z_pre_result_bird, Z_pre_result_uav, Log_Likelihood_Ratio)
    
    return Target_type, Log_Likelihood_Ratio, Bird_Model_IMM_11111, Bird_Qs, Bird_Result_k, UAV_Model_IMM_1111111222, UAV_Qs, UAV_Result_k
    