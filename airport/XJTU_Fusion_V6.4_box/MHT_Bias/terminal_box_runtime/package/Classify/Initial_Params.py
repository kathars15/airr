import numpy as np
import scipy.io as sio
from Classify.Base_model_3 import Base_model_3
from copy import deepcopy
import os

Initial_Classify_Params = {} # 目标分类算法参数

Initial_Classify_Params['R'] = np.eye(3) * 10**2
# Initial_Classify_Params['Data_fitting'] = sio.loadmat('/data/XJTU_Fusion_V5_box/MHT_Bias/Classify/Data_Gussian_fitting.mat')
# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 构建 mat 文件的完整路径
mat_file_path = os.path.join(CURRENT_DIR, 'Data_Gussian_fitting.mat')

# 加载
Initial_Classify_Params['Data_fitting'] = sio.loadmat(mat_file_path)

# Initial_Classify_Params['Data_fitting'] = sio.loadmat('D:\\desk\\airr\\airport\\XJTU_Fusion_V6.4_box\\MHT_Bias\\Classify\\Data_Gussian_fitting.mat')
# 各过程噪声初始化
Bird_Q_CP = Initial_Classify_Params['Data_fitting']['New_Bird_Sigma']
UAV_Q_CP = Initial_Classify_Params['Data_fitting']['UAV_Sigma']
Q_CT = np.array([[6**2, 0, 0], [0, 6**2, 0], [0, 0, 6**2]])  # CT模型过程噪声协方差，建模为角加速度rad/s^2   V1 8
Q_CV_0 = np.eye(3) * 0.1**2  # 基础模型CV_inout模型过程噪声协方差，建模为加速度m/s^2 0.1
Bird_Qs = {'Q_CT': Q_CT, 'Q_CV_0': Q_CV_0, 'Q_CP': Bird_Q_CP}  # 所有Q存在一起
UAV_Qs = {'Q_CT': Q_CT, 'Q_CV_0': Q_CV_0, 'Q_CP': UAV_Q_CP}  # 所有Q存在一起
Initial_Classify_Params['Bird_Qs'] = Bird_Qs
Initial_Classify_Params['UAV_Qs'] = UAV_Qs
# 记录每部融合状态
Bird_Result_k = {'X': np.zeros((9, 1)), 'P': np.zeros((9, 9))}  # 记录每个时刻的模型融合结果 [x v w]
UAV_Result_k = {'X': np.zeros((9, 1)), 'P': np.zeros((9, 9))}  # 记录每个时刻的模型融合结果 [x v w]
Initial_Classify_Params['Bird_Result_k'] = Bird_Result_k
Initial_Classify_Params['UAV_Result_k'] = UAV_Result_k
# 记录每步各模型状态
Model_Num, Bird_Model_IMM, TPM = Base_model_3(Initial_Classify_Params['R'])
UAV_Model_IMM =deepcopy(Bird_Model_IMM)
Initial_Classify_Params['Model_Num'] = Model_Num
Initial_Classify_Params['TPM'] = TPM
Initial_Classify_Params['Bird_Model_IMM'] = Bird_Model_IMM
Initial_Classify_Params['UAV_Model_IMM'] = UAV_Model_IMM
# 记录每步累计的似然比
Initial_Classify_Params['Log_Likelihood_Ratio'] = 0
# 识别标志和识别结果
Initial_Classify_Params['classify_flag'] = 0
Initial_Classify_Params['final_classify_target'] = 0
Initial_Classify_Params['final_classify_time'] = 0
# 算法中使用的常量
Initial_Classify_Params['ConstValue'] = {'TPM': TPM, 'R': Initial_Classify_Params['R'], 'fitting_data': Initial_Classify_Params['Data_fitting']}