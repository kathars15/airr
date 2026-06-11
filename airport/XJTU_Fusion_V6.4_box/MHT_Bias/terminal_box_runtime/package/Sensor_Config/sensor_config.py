import numpy as np

def Cal_Radar_Volume(range, pitch, yaw):
    """
    计算三维雷达探测范围的体积
    距离(m) 俯仰角(中心到边界的角度,rad) 方位角(中心到边界的角度,rad)
    """
    pitch = pitch / 180.0*np.pi
    yaw = yaw / 180.0*np.pi
    volume = 2.0/3.0*np.power(range,3)*np.sin(pitch/2.0)*yaw
    return volume

## 东北天原点
lla_original = [30.274770, 122.143520, 65.0]
## 所有传感器配置存储到一个字典
Sensor_Config = {}
SignalType2Name = {1:'Lateral', 2:'Protocol', 3: 'RID', 4:'Radar', 5:'AOA',\
    6:'Lateral_Position', 7:'TDOA', 8:'5G-A',9:'RTK',10:'ADS-B'} # SignalType 与 sensor_name 的映射关系
# 创建反向映射字典
Name2SignalType = {'Lateral':1, 'Protocol':2, 'RID':3, 'Radar':4, 'AOA':5,\
    'Lateral_Position':6, 'TDOA':7, '5G-A':8, 'RTK':9, 'ADS-B':10}
SingleFrameDt = {'Lateral': 0.1, 'Protocol': 0.5, 'RID': 0.5, 'Radar': 2.0, 'AOA':0.1, \
                'Lateral_Position':0.1, 'TDOA':0.1, '5G-A': 0.32, 'RTK':0.5, 'ADS-B':0.5} # 每个传感器半帧时间

# 传感器1：Lateral 数据
sensor = {}
sensor['Name'] = 'Lateral'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([10.0, 10.0, 10.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = True
sensor['Biased_Ignore'] = False # 选择是否考虑偏差，可以选择开局先不考虑偏差，直到有基准传感器出现，才开始考虑偏差
sensor['Bias'] = None
sensor['Bias_Guess'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['Bias_Qk'] = np.power(np.diag([0.1, 0.1, 0.1]), 2)
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor

## 传感器2：Protocol 数据
sensor = {}
sensor['Name'] = 'Protocol'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([5.0, 5.0, 5.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = False
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor

## 传感器3：RID数据
sensor = {}
sensor['Name'] = 'RID'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([5.0, 5.0, 5.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = False
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor

## 传感器4：Radar数据
sensor = {}
sensor['Name'] = 'Radar'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = True
sensor['Biased_Ignore'] = False # 选择是否考虑偏差，可以选择开局先不考虑偏差，直到有基准传感器出现，才开始考虑偏差
sensor['Bias'] = None
sensor['Bias_Guess'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['Bias_Qk'] = np.power(np.diag([0.1, 0.1, 0.1]), 2)
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0

sensor['Max_Range'] = 7000.0             # 7km
sensor['Max_Pitch'] = 25.0               # ±25°
sensor['Max_Yaw'] = 360.0                # 全方位

sensor['Volume'] =4000 * 4000 * 100.0
sensor['FA_Num'] = 1
Sensor_Config[sensor['Name']] = sensor

## 传感器5：AOA 数据
sensor = {}
sensor['Name'] = 'AOA'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = True
sensor['Biased_Ignore'] = False # 选择是否考虑偏差，可以选择开局先不考虑偏差，直到有基准传感器出现，才开始考虑偏差
sensor['Bias'] = None
sensor['Bias_Guess'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['Bias_Qk'] = np.power(np.diag([0.1, 0.1, 0.1]), 2)
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor

## 传感器6：Lateral_Position 数据
sensor = {}
sensor['Name'] = 'Lateral_Position'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([10.0, 10.0, 10.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = True
sensor['Biased_Ignore'] = False # 选择是否考虑偏差，可以选择开局先不考虑偏差，直到有基准传感器出现，才开始考虑偏差
sensor['Bias'] = None
sensor['Bias_Guess'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['Bias_Qk'] = np.power(np.diag([0.1, 0.1, 0.1]), 2)
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor

## 传感器7：TDOA 数据
sensor = {}
sensor['Name'] = 'TDOA'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([10.0, 10.0, 10.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = True
sensor['Biased_Ignore'] = False # 选择是否考虑偏差，可以选择开局先不考虑偏差，直到有基准传感器出现，才开始考虑偏差
sensor['Bias'] = None
sensor['Bias_Guess'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['Bias_Qk'] = np.power(np.diag([0.1, 0.1, 0.1]), 2)
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor

## 传感器8：5G-A数据
sensor = {}
sensor['Name'] = '5G-A'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([10.0, 10.0, 10.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = True
sensor['Biased_Ignore'] = False # 选择是否考虑偏差，可以选择开局先不考虑偏差，直到有基准传感器出现，才开始考虑偏差
sensor['Bias'] = None
sensor['Bias_Guess'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
sensor['Bias_Qk'] = np.power(np.diag([0.1, 0.1, 0.1]), 2)
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 1.0
Sensor_Config[sensor['Name']] = sensor

## RTK 数据
sensor = {}
sensor['Name'] = 'RTK'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([10.0, 10.0, 10.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = False
# sensor['Biased_Ignore'] = False # 选择是否考虑偏差，可以选择开局先不考虑偏差，直到有基准传感器出现，才开始考虑偏差
# sensor['Bias'] = None
# sensor['Bias_Guess'] = np.power(np.diag([20.0, 20.0, 20.0]), 2)
# sensor['Bias_Qk'] = np.power(np.diag([0.1, 0.1, 0.1]), 2)
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor

## 传感器10：ADS-B数据
sensor = {}
sensor['Name'] = 'ADS-B'
sensor['Position'] =  np.array([[0.0, 0.0, 0.0]]).T 
sensor['Meas_Type'] = 'Position'
sensor['R'] = np.power(np.diag([5.0, 5.0, 5.0]), 2)
sensor['P_D'] = 0.8
sensor['Is_Biased'] = False
sensor['Sensor_Yaw'] = 0
sensor['Sensor_Pitch'] = 0
sensor['Sensor_Roll'] = 0
sensor['Max_Range'] = 175.0 # 错的
sensor['Max_Pitch'] = 22.5 # 错的
sensor['Max_Yaw'] = 22.5 # 错的
sensor['Volume'] = 1000.0*1000.0*100.0
sensor['FA_Num'] = 0.001
Sensor_Config[sensor['Name']] = sensor


#********逐点输入版本修改 第二处 增加传感器参数计算********
# 原有SingleFrameDt定义后，为每个传感器计算λ和μ
for sensor_name, T in SingleFrameDt.items():
    lambda_death = -np.log(1-0.01) / (2*T)  # 由P_death=0.01反推λ
    mu_detect = -np.log(1-0.8) / (2*T)  # 由P_D=0.8反推μ
    # 为每个传感器配置添加这两个参数
    Sensor_Config[sensor_name]['lambda_death'] = lambda_death
    Sensor_Config[sensor_name]['mu_detect'] = mu_detect
    