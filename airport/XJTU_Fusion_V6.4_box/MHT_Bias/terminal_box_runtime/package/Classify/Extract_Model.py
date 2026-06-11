import numpy as np
from scipy.linalg import block_diag

def Extract_Model(model, CT_model):
    prob = model['prob']
    if model['Flag'] == 'CT':
        X_aug = model['X_k']
        P_aug = model['P_k']
    elif model['Flag'] == 'CV_input_0':
        X_aug = np.vstack((model['X_k'], CT_model['X_k'][6:9, :]))
        P_aug = block_diag(model['P_k'], CT_model['P_k'][6:9, 6:9])
    elif model['Flag'] == 'CP_input':
        X_aug = np.vstack((model['X_k'], CT_model['X_k'][6:9, :]))
        P_aug = block_diag(model['P_k'], CT_model['P_k'][6:9, 6:9])
    else:
        raise ValueError('error in Extract_Model')
    
    return X_aug, P_aug, prob