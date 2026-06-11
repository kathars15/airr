import numpy as np

def Models_z_Pzz_Prediction(Models, Hs):
    Model_Num = len(Models)  # 获取模型数量
    for im in range(Model_Num):  # 遍历每个模型
        model = Models[im]  # 获取当前模型
        X_k_k_1 = model['X_k_k_1']  # 获取状态预测矩阵
        P_k_k_1 = model['P_k_k_1']  # 获取协方差预测矩阵


        if Hs['Flag'] == 'Pos':  # 检查Hs的标志是否为'Pos'
            R = Hs['R']  # 获取噪声矩阵
            if model['Flag'] == 'CT':  # 根据模型标志构造观测矩阵H
                H = np.block([[np.eye(3), np.zeros((3, 6))]])
            elif model['Flag'] == 'CV_input_0':
                H = np.block([[np.eye(3), np.zeros((3, 3))]])
            elif model['Flag'] == 'CP_input':
                H = np.block([[np.eye(3), np.zeros((3, 3))]])  # 调整H的形状为(3, 6)
            else:
                raise ValueError('error in Model_z_Pzz_Prediction')  # 如果标志不匹配，抛出错误

            z_k_k_1 = H @ X_k_k_1  # 计算观测预测
            Pzz = H @ P_k_k_1 @ H.T + R  # 计算观测协方差
        else:
            raise ValueError('error in Model_z_Pzz_Prediction')  # 如果Hs的标志不为'Pos'，抛出错误

        Models[im]['z'] = z_k_k_1  # 更新模型的观测预测
        Models[im]['Pzz'] = Pzz  # 更新模型的观测协方差

    return Models  # 返回更新后的模型列表