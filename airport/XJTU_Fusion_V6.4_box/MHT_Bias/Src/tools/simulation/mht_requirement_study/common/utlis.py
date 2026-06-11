from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import sys
import rasterio
from pyproj import Transformer
# sys.path.append('/root/桌面/Fusion/python/MHT_Bias')
import cvxpy as cp
import math
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from collections import defaultdict

def Angle_to_Rotation_2D(angle: float = 0) -> Type[np.ndarray]:
    """angle is in degrees by default"""
    angle = angle*np.pi/180.0
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),np.cos(angle)]])

def Angle_to_Rotation_3D(yaw, pitch, roll):
    """
    局部（雷达）坐标系相对于全局坐标系的角度。
    将偏航角、俯仰角和滚转角转换为旋转矩阵。
    参数:
        yaw (float): 偏航角（绕 Z 轴旋转，单位：度）。
        pitch (float): 俯仰角（绕 Y 轴旋转，单位：度）。
        roll (float): 滚转角（绕 X 轴旋转，单位：度）。
    返回:
        np.ndarray: 3x3 旋转矩阵。将全局坐标系位置转到局部坐标系位置
    """
    yaw = yaw/180.0*np.pi
    pitch = pitch/180.0*np.pi
    roll = roll/180.0*np.pi
    # 绕 Z 轴的旋转矩阵（偏航角）
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # 绕 Y 轴的旋转矩阵（俯仰角）
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # 绕 X 轴的旋转矩阵（滚转角）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 组合旋转矩阵
    R = Rz @ Ry @ Rx
    return R.T # 全局坐标系位置转到局部坐标系位置

def OSPA(truth_tracks, est_tracks, p=2, c=1, dim_d=2):
    """
    Calculate the ospa distance.
    Args:
        truth_tracks: a list of truth tracks at current timestep.
        est_tracks: a list of estimated tracks at current timestep.
    """
    m, n = len(truth_tracks), len(est_tracks)
    assert m>=1 and n>=1
    Dis_matrix = np.zeros((1, m*n))
    for i, truth_track in enumerate(truth_tracks):
        for j, est_track in enumerate(est_tracks):
            Dis_matrix[0, i*n + j] = min([np.linalg.norm((truth_track[:dim_d, 0]-est_track[:dim_d, 0])), c])
    assign = cp.Variable(m*n, boolean=True) # 每n个元素对应某个真实目标分配给谁
    Constraint_matrixs1 = np.zeros((m, m*n), dtype=np.uint8) # truth can only be assigned to one est.
    for i in range(m):
        Constraint_matrixs1[i, n*i:n*(i+1)] = np.ones((n, ), dtype=np.uint8)
    Constraint_matrixs2 = np.zeros((n, m*n), dtype=np.uint8) # est can only be assigned to one truth.
    for i in range(n):
        for j in range(m):
            Constraint_matrixs2[i, i+n*j] = 1
    Constraint_matrixs_12 = np.concatenate([Constraint_matrixs1, Constraint_matrixs2])
    prob = cp.Problem(cp.Maximize(-Dis_matrix@assign),\
            [ Constraint_matrixs_12@assign <= np.ones((Constraint_matrixs_12.shape[0],)),\
            np.ones((m*n), dtype=np.uint8)@assign == np.min([m, n]) ])
    prob.solve(solver='GLPK_MI')
    assignment = assign.value
    dis_all = Dis_matrix[0, np.where(assignment>0.5)].squeeze()
    dis_all_p = np.power(dis_all, p)
    dis_all_p_sum = np.sum(dis_all_p)
    dis_ospa = dis_all_p_sum + np.power(c, p)*np.abs(n-m)
    dis_ospa = dis_ospa / np.max([n, m])
    dis_ospa = np.power(dis_ospa, 1.0/(p*1.0))
    return dis_ospa
    
def Cal_Target_In_Sensor_Volume(sensor_config, target_pos):
    sensor_pos = np.array(sensor_config['Position']).reshape(-1,1)
    target_pos = target_pos.reshape(-1,1)
    delta_pos_global = target_pos - sensor_pos # 全局坐标系下目标相对雷达的位置
    #### 旋转矩阵和雷达坐标系旋转的定义有关
    Rotation_3D = Angle_to_Rotation_3D(sensor_config['Sensor_Yaw'], \
                    sensor_config['Sensor_Pitch'], sensor_config['Sensor_Roll'])
    delta_pos_local = Rotation_3D @ delta_pos_global # 雷达坐标系下目标相对雷达的位置
    x, y, z = delta_pos_local[0,0], delta_pos_local[1,0], delta_pos_local[2,0]

    # 判断目标是否超出距离
    target_range = np.linalg.norm(delta_pos_local)
    if target_range >= sensor_config['Max_Range']:
        return False
    
    # 判断目标是否超出俯仰范围
    r_xy = np.sqrt(x**2 + y**2)
    target_pitch = np.degrees(np.arccos(r_xy/target_range)) # 目标俯仰角
    if target_pitch >= sensor_config['Max_Pitch']:
        return False
    
    # 判断目标是否超出方位范围【注意和X轴夹角还是Y轴夹角】此处为X轴夹角
    target_yaw = np.abs(np.degrees(np.arctan2(y, z))) # 目标俯仰角
    if target_yaw >= sensor_config['Max_Yaw']:
        return False
    
    return True
def timestamp_ms_to_s(time_input):
     """
    将时间戳从毫秒级转为秒级。

    参数:
    time_input : float
        毫秒级时间戳

    返回:
    timestamp : float
        对应的时间戳（Unix 时间戳，单位为秒）。
    """
     timestamp = time_input / 1000.0
     return timestamp
    
     
def time_string_to_timestamp(time_input):
    """
    将时间字符串或 numpy.datetime64 转换为时间戳。

    参数:
    time_input : str 或 numpy.datetime64
        时间字符串或 numpy.datetime64 对象。
        支持的字符串格式：
        - 'YYYY-MM-DD HH:MM:SS.sss'
        - 'YYYY-MM-DD HH:MM:SS'
        - 'YYYY/MM/DD HH:MM:SS'
        - 'YYYY/M/DD HH:MM:SS'
        - 'YYYY/M/D HH:MM:SS'

    返回:
    timestamp : float
        对应的时间戳（Unix 时间戳，单位为秒）。
    """
    # 如果输入是 numpy.datetime64 类型
    if isinstance(time_input, np.datetime64):
        # 将 numpy.datetime64 转换为时间戳 【貌似与datetime.strptime对不上，还是统一转换成datetime吧】
        dt_obj = time_input.astype('datetime64[us]').item()
        return dt_obj.timestamp()

    # 如果输入是字符串类型
    elif isinstance(time_input, str):
        # 定义支持的时间格式列表
        formats = [
            '%Y-%m-%d %H:%M:%S.%f',  # 带毫秒的格式
            '%Y-%m-%d %H:%M:%S',     # 不带毫秒的格式
            '%Y/%m/%d %H:%M:%S',     # 斜杠分隔的格式
            '%Y-%m-%d %H:%M:%S:%f',  # 带毫秒的格式
            '%Y-%m-%d %H:%M:%S %f',  # 带毫秒的格式
        ]

        # 尝试每种格式进行解析
        for fmt in formats:
            try:
                dt = datetime.strptime(time_input, fmt)
                return dt.timestamp()
            except ValueError:
                continue

    # 如果输入类型不支持，返回错误信息
    print(f"无法解析输入: {time_input}")
    return None

def group_similar_indices(arr, tolerance=0.01):
    """
    将数值相近的元素的索引分组存储【通常输入arr=timestamps】
    参数:
    arr (np.ndarray): 一维数值数组
    tolerance (float): 判定相近的阈值，默认为0.01
    返回:
    list: 包含索引组的列表，每个索引组是一个列表
    """
    # 初始化结果列表和已处理索引的集合
    groups = []
    processed = set()
    for i in range(len(arr)):
        if i not in processed:
            # 找到与当前元素相近的所有索引
            close_indices = np.where(np.abs(arr - arr[i]) <= tolerance)[0]
            # 将索引组添加到结果中
            groups.append(close_indices.tolist())
            # 将这些索引标记为已处理
            processed.update(close_indices)
    return groups

def Split_Measurements_Into_Frames(Real_Data_Meas, Real_Data_timestamp, Real_Data_frame_index):
    """
    Real_Data_Meas：2行/3行/4行 n列的所有数据，每一列对应一个时间戳
    Real_Data_timestamp：对应所有时间戳
    转为一帧一帧的数据
    """
    Measurements = []
    Timestamps = []
    for index in Real_Data_frame_index:
        Measurements.append(Real_Data_Meas[:, index]) # (4, n)
        Timestamps.append(Real_Data_timestamp[index[0]])

    return Measurements, Timestamps

def geodetic_to_ecef(lat, lon, alt):
    """批量转换经纬高到ECEF坐标系"""
    # WGS84椭球体参数
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e_sq = (a**2 - b**2) / a**2

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    alt = np.asarray(alt)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    sin_phi = np.sin(lat_rad)
    cos_phi = np.cos(lat_rad)
    sin_lambda = np.sin(lon_rad)
    cos_lambda = np.cos(lon_rad)
    
    N = a / np.sqrt(1 - e_sq * sin_phi**2)
    x = (N + alt) * cos_phi * cos_lambda
    y = (N + alt) * cos_phi * sin_lambda
    z = (N * (1 - e_sq) + alt) * sin_phi
    
    return np.vstack((x, y, z))

def ecef_to_geodetic_batch(ecef_points):
    """批量转换ECEF到经纬高"""
    # WGS84椭球体参数
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e_sq = (a**2 - b**2) / a**2

    def single_point(point):
        x, y, z = point
        p = math.hypot(x, y)
        if p < 1e-9:
            return (np.copysign(90, z), 0.0, z - np.copysign(b, z))
        
        lon = math.degrees(math.atan2(y, x))
        tan_phi = z / p / (1 - e_sq)
        phi = math.atan(tan_phi)
        
        for _ in range(100):
            sin_phi = math.sin(phi)
            N = a / math.sqrt(1 - e_sq * sin_phi**2)
            h = p / math.cos(phi) - N
            phi_new = math.atan(z / (p * (1 - e_sq * N / (N + h))))
            if abs(phi_new - phi) < 1e-11:
                phi = phi_new
                break
            phi = phi_new
        
        return (math.degrees(phi), lon, h)
    
    return np.apply_along_axis(single_point, 0, ecef_points)

def geodetic_to_enu(lat_A, lon_A, alt_A, B_points):
    """批量转换到ENU坐标系"""
    # WGS84椭球体参数
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e_sq = (a**2 - b**2) / a**2

    # 转换A点到ECEF
    ecef_A = geodetic_to_ecef([lat_A], [lon_A], [alt_A])
    
    # 提取B点并转换
    B_points = np.asarray(B_points)
    ecef_B = geodetic_to_ecef(B_points  [0], B_points  [1], B_points  [2])
    
    # 计算相对坐标
    delta = ecef_B - ecef_A
    
    # 构建旋转矩阵
    lat_rad = math.radians(lat_A)
    lon_rad = math.radians(lon_A)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    # 执行坐标旋转
    east  = -sin_lon * delta  [0] + cos_lon * delta  [1]
    north = -sin_lat * cos_lon * delta  [0] - sin_lat * sin_lon * delta  [1] + cos_lat * delta  [2]
    up    =  cos_lat * cos_lon * delta  [0] + cos_lat * sin_lon * delta  [1] + sin_lat * delta  [2]
    
    return np.vstack((east, north, up))

def enu_to_geodetic(ref_lat, ref_lon, ref_alt, enu_points):
    """
    东北天（ENU）坐标转经纬高（WGS84）
    
    参数:
        ref_lat: 参考原点纬度（度）
        ref_lon: 参考原点经度（度）
        ref_alt: 参考原点高度（米）
        enu_points: 东北天坐标，形状为(3, N)的数组，每行对应E、N、U分量（米）
        
    返回:
        np.ndarray: 经纬高坐标，形状为(3, N)，每行对应纬度（度）、经度（度）、高度（米）
    """
    # 1. 将参考原点转换为ECEF坐标
    ecef_ref = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)  # 形状(3, 1)
    
    # 2. 计算ENU到ECEF的旋转矩阵
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    sin_lat = math.sin(ref_lat_rad)
    cos_lat = math.cos(ref_lat_rad)
    sin_lon = math.sin(ref_lon_rad)
    cos_lon = math.cos(ref_lon_rad)
    
    # 旋转矩阵（ENU->ECEF）
    R = np.array([
        [-sin_lon,          -sin_lat * cos_lon,   cos_lat * cos_lon],
        [cos_lon,           -sin_lat * sin_lon,   cos_lat * sin_lon],
        [0,                 cos_lat,              sin_lat          ]
    ])
    
    # 3. 将ENU坐标转换为ECEF偏移量（delta）
    delta_ecef = R @ enu_points  # 矩阵乘法，形状(3, N)
    
    # 4. 计算目标点的ECEF坐标（原点ECEF + 偏移量）
    ecef_target = ecef_ref + delta_ecef  # 形状(3, N)
    
    # 5. 将ECEF坐标转换为经纬高
    geodetic = ecef_to_geodetic_batch(ecef_target)
    
    return geodetic

# def ellipsoidal_to_orthometric(lat_deg, lon_deg, h_ell, geoid_model="EGM96"):
#     """
#     将 WGS84 椭球高转换为海拔高（MSL）
#     支持: "EGM96"（推荐，pyproj 内置）；"EGM2008" 需额外数据，暂不支持
#     """
#     if geoid_model.upper() == "EGM96":
#         # EPSG:4326+5773 = WGS84 + EGM96 (MSL)
#         transformer = Transformer.from_crs("EPSG:4979", "EPSG:4326+5773", always_xy=True)
#     else:
#         raise ValueError("当前仅支持 EGM96（因 EGM2008 需额外安装数据）")
    
#     try:
#         _, _, h_msl = transformer.transform(lon_deg, lat_deg, h_ell)
#         return h_msl
#     except Exception as e:
#         raise RuntimeError(f"大地水准面校正失败: {e}")

# def get_ground_elevation_from_dem(dem_path, lon, lat):
#     """从本地 DEM 查询地面高程（MSL）"""
#     with rasterio.open(dem_path) as src:
#         if not (src.bounds.left <= lon <= src.bounds.right and
#                 src.bounds.bottom <= lat <= src.bounds.top):
#             raise ValueError(f"点 ({lat}, {lon}) 超出 DEM 范围: {src.bounds}")
#         row, col = src.index(lon, lat)
#         elev = src.read(1)[row, col]
#         nodata = src.nodata
#         if nodata is not None and elev == nodata:
#             return np.nan
#         if np.isnan(elev) or elev < -1000:
#             return np.nan
#         return float(elev)

# def get_agl_from_geodetic(lat_deg, lon_deg, h_ell, dem_path):
#     """
#     输入单点 WGS84 经纬度 + 椭球高，输出离地高度（AGL）
#     """
#     # 1. 椭球高 → 海拔高（MSL，使用 EGM96）
#     h_msl = ellipsoidal_to_orthometric(lat_deg, lon_deg, h_ell, "EGM96")
    
#     # 2. 查询 DEM 地面高程（MSL）
#     ground_msl = get_ground_elevation_from_dem(dem_path, lon_deg, lat_deg)
#     if np.isnan(ground_msl):
#         raise ValueError("DEM 返回无效高程值")
    
#     # 3. 计算 AGL
#     agl = h_msl - ground_msl
#     return max(agl, 0.0)

def estimate_time_delay(t_A, x_A, t_B, x_B, bounds=(-3, 3)):
    """
    估计传感器B相对于A的固定时间延迟Δτ
    
    参数：
    t_A : np.array, shape (N,)
        传感器A的时间戳数组（单位：秒）
    x_A : np.array, shape (N, 3)
        传感器A的三维位置测量值，每行对应t_A相应时间的坐标(x, y)
    t_B : np.array, shape (M,)
        传感器B的时间戳数组（单位：秒）
    x_B : np.array, shape (M, 3)
        传感器B的三维维位置测量值，每行对应t_B相应时间的坐标(x, y)
    bounds : tuple, optional
        时间延迟Δτ的搜索范围，默认(-5, 5)
    
    返回：
    delta_tau : float
        传感器B相对于A的时间延迟（B时间 = A时间 + delta_tau）
    """
   
    # 创建B的线性插值器（允许外推）
    interp_B = interp1d(t_B, x_B, axis=0, 
                        bounds_error=False, fill_value="extrapolate", kind='cubic')
    
    # 定义损失函数：对齐后的坐标差异平方和
    def loss(delta_tau):
        total_loss = 0.0
        for t_i, x_i in zip(t_A, x_A):
            t_B_aligned = t_i - delta_tau
            x_B_interp = interp_B(t_B_aligned)
            # 忽略无法插值的时间点（通常由外推导致的不稳定区域）
            if np.isnan(x_B_interp).any():
                continue
            d = x_i - x_B_interp
            total_loss += np.sum(d**2)
        return total_loss
    
    # 优化时间延迟参数
    result = minimize_scalar(loss, bounds=bounds, method='bounded')
    
    return result.x, result.fun

class DSU:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False
        # 按大小合并，小树挂到大树上
        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]
        return True

def merge_lists(lists):
    dsu = DSU()
    # 将每个整数映射到其所在的小列表的索引
    num_to_index = defaultdict(list)
    for i, lst in enumerate(lists):
        for num in lst:
            num_to_index[num].append(i)
    # 合并有交集的小列表
    for num in num_to_index:
        indices = num_to_index[num]
        for i in range(1, len(indices)):
            dsu.union(indices  [0], indices[i])
    # 提取合并后的小列表
    merged_lists = defaultdict(list)
    for i, lst in enumerate(lists):
        root = dsu.find(i)
        merged_lists[root].extend(lst)
    # 去重并返回结果
    result = [list(set(lst)) for lst in merged_lists.values()]
    return result
