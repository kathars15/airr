import numpy as np
from Classify.log_norm import log_norm

def VSIMM_update(Model_IMM, z_k, Hs):
    Model_Num = len(Model_IMM)
    log_likeli = np.zeros(Model_Num)  # 对数似然
    log_prior = np.zeros(Model_Num)  # 对数概率先验

    for im in range(Model_Num):
        model = Model_IMM[im]
        X_k_k_1 = model['X_k_k_1']
        P_k_k_1 = model['P_k_k_1']
        # try:
        #         Sk_1 = np.linalg.cholesky(P_k_k_1).T  # 重新尝试Cholesky分解
        # except np.linalg.LinAlgError:
        #         print(P_k_k_1)
        z_k_k_1 = model['z']
        Pzz = model['Pzz']

        if Hs['Flag'] == 'Pos':
            R = Hs['R']
            if model['Flag'] == 'CT':
                H = np.block([[np.eye(3), np.zeros((3, 6))]])
                Id = np.eye(9)

            elif model['Flag'] == 'CV_input_0':
                H = np.block([[np.eye(3), np.zeros((3, 3))]])
                Id = np.eye(6)
            elif model['Flag'] == 'CP_input':
                H = np.block([[np.eye(3), np.zeros((3, 3))]])
                Id = np.eye(6)
            else:
                raise ValueError('error in Model_z_Pzz_Prediction')
            
            Kk = P_k_k_1 @ H.T @ np.linalg.inv(Pzz)
            X_k_k = X_k_k_1 + Kk @ (z_k - z_k_k_1)
            P_k_k = (Id - Kk @ H) @ P_k_k_1 @ (Id - Kk @ H).T + Kk @ R @ Kk.T
            # try:
            #     Sk_1 = np.linalg.cholesky(P_k_k).T  # 重新尝试Cholesky分解
            # except np.linalg.LinAlgError:
            #     print(P_k_k)
        else:
            raise ValueError('error in Model_z_Pzz_Prediction')

        Model_IMM[im]['X_k'] = X_k_k
        Model_IMM[im]['P_k'] = P_k_k 
        ll1 = -0.5 * (z_k - z_k_k_1).T @ np.linalg.inv(Pzz) @ (z_k - z_k_k_1) - 1.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Pzz))
        log_likeli[im]= float(-0.5 * (z_k - z_k_k_1).T @ np.linalg.inv(Pzz) @ (z_k - z_k_k_1) - 1.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Pzz)))
        log_prior[im] = np.log(Model_IMM[im]['prob'])

    # 概率进行对数归一化
    log_posterior = log_likeli + log_prior
    log_posterior_norm = log_norm(log_posterior)  # 对数归一化
    if not np.isreal(log_posterior_norm).all():
        raise ValueError('error in VSIMM_update')

    for im in range(Model_Num):
        Model_IMM[im]['prob'] = np.exp(log_posterior_norm[im])

    return Model_IMM