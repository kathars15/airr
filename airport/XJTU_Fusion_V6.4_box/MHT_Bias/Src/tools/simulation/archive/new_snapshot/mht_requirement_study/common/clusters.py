import numpy as np
from typing import List, Type, Union, Optional, Sequence, Tuple
from sklearn.cluster import DBSCAN, SpectralClustering, OPTICS, KMeans
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from scipy.cluster.hierarchy import fcluster

def Clustering_Obs(
        obs_k: Sequence,
        eps: float = 0.75, # 0.75
        min_samples: int = 5,
        Sigma: Type[np.ndarray] = None,
        n_clusters = 2,
        max_eps = 5,
        n_components = 2,
        init_params = 'k-means++',
        init = 'k-means++',
        n_init='auto',
        cluster_method = 'dbscan',
        Clustering_Type: str = 'SpectralClustering'
) -> Tuple[List]:
    """
    clustering the measurement by DBSCAN.
    Args:
        obs_k: measurement, List[np.array(m,1)].
        eps and min_samples: parameters of DBSCAN.
        Sigma: sigma matrice used for calculating mahalanobis dis.
    Returns:
        obs_clusters: List[List[np.ndarray(m, 1)]].
        obs_indexs: List[List[int]].
    """
    if len(obs_k) == 0:
        return [], []
    if Sigma is None: # 协方差矩阵的逆矩阵【精度矩阵】
        Sigma = np.identity(obs_k[0].shape[0])
        # Sigma[-1, -1] = 1.0 / 4.0
    obs_k_np = np.concatenate(obs_k, 1).T # (n, m)
    if Clustering_Type == 'DBSCAN':
        obs_label = DBSCAN(eps=eps, min_samples=min_samples, metric='mahalanobis', metric_params={'VI': Sigma}).fit_predict(obs_k_np)
    elif Clustering_Type == 'SpectralClustering':
        obs_label = SpectralClustering(n_clusters=n_clusters).fit_predict(obs_k_np)
    elif Clustering_Type == 'OPTICS':
        obs_label = OPTICS(eps=eps, min_samples=min_samples, metric='mahalanobis', metric_params={'VI': Sigma},\
            max_eps = max_eps, cluster_method = cluster_method).fit_predict(obs_k_np)
    elif Clustering_Type == 'GaussianMixture':
        obs_label = GaussianMixture(n_components=n_components, init_params=init_params, n_init=n_init).fit_predict(obs_k_np)
    elif Clustering_Type == 'KMeans':
        obs_label = KMeans(n_clusters=n_clusters, init=init, n_init=n_init).fit_predict(obs_k_np)
    obs_clusters = []  # clustering result
    obs_indexs = []  # clustering index result
    label_class = np.unique(obs_label).squeeze()
    if label_class.shape == (): # only one number (-1)
        label_class_num = 1
    else:
        label_class_num = label_class.shape[0]
    for i in range(label_class_num):
        if label_class_num == 1:
            label_ = label_class
        else:
            label_ = label_class[i]
        if label_ != -1:
            obs_index = np.array(np.where(obs_label == label_)).squeeze().tolist()
            if isinstance(obs_index, int):
                obs_index = [obs_index]
            obs_indexs.append(obs_index)
            obs_cluster = [obs_k[io] for io in obs_index]
            obs_clusters.append(obs_cluster)
    
    # # 按数量从大到小排列
    # if len(obs_indexs) >= 2:
    #     n_ks = [len(obs_index) for obs_index in obs_indexs]
    #     index_n_ks = sorted(enumerate(n_ks), key = lambda x: x[1], reverse=True)
    #     indexs = [index_n_k[0] for index_n_k in index_n_ks]
    #     obs_clusters_sorted, obs_indexs_sorted = [], []
    #     for index in indexs:
    #         obs_clusters_sorted.append(obs_clusters[index])
    #         obs_indexs_sorted.append(obs_indexs[index])
    #     obs_clusters, obs_indexs = obs_clusters_sorted, obs_indexs_sorted
    return obs_clusters, obs_indexs

def Clustering_Obs_PHD(
        obs_k: Sequence,
        eps: float = 0.75, # 0.75
        min_samples: int = 5,
        Sigma: Type[np.ndarray] = None,
        n_clusters = 2,
        max_eps = 5,
        n_components = 2,
        init_params = 'k-means++',
        init = 'k-means++',
        cluster_method = 'dbscan',
        Clustering_Type: str = 'SpectralClustering',
        weights_init = None,
        means_init = None,
        precisions_init = None
) -> Tuple[List]:
    "与上述区别是，此处把认为杂波的序列也单独列为一簇"
    """
    clustering the measurement by DBSCAN.
    Args:
        obs_k: measurement, List[np.array(m,1)].
        eps and min_samples: parameters of DBSCAN.
        Sigma: sigma matrice used for calculating mahalanobis dis.
    Returns:
        obs_clusters: List[List[np.ndarray(m, 1)]].
        obs_indexs: List[List[int]].
    """
    if len(obs_k) == 0:
        return [], []
    if Sigma is None:
        Sigma = np.identity(obs_k[0].shape[0])
        # Sigma[-1, -1] = 1.0 / 4.0
    obs_k_np = np.concatenate(obs_k, 1).T # (n, m)
    if Clustering_Type == 'DBSCAN':
        obs_label = DBSCAN(eps=eps, min_samples=min_samples, metric='mahalanobis', metric_params={'VI': Sigma}).fit_predict(obs_k_np)
    elif Clustering_Type == 'SpectralClustering':
        obs_label = SpectralClustering(n_clusters=n_clusters).fit_predict(obs_k_np)
    elif Clustering_Type == 'OPTICS':
        obs_label = OPTICS(eps=eps, min_samples=min_samples, metric='mahalanobis', metric_params={'VI': Sigma},\
            max_eps = max_eps, cluster_method = cluster_method).fit_predict(obs_k_np)
    elif Clustering_Type == 'GaussianMixture':
        if weights_init is None:
            obs_label = GaussianMixture(n_components=n_components, init_params=init_params).fit_predict(obs_k_np)
        else:
            obs_label = GaussianMixture(n_components=n_components, weights_init = weights_init,\
                means_init = means_init, precisions_init = precisions_init).fit_predict(obs_k_np)
    elif Clustering_Type == 'BisectingKMeans':
        obs_label = BisectingKMeans(n_clusters=n_clusters, init=init).fit_predict(obs_k_np)
    elif Clustering_Type == 'KMeans':
        obs_label = KMeans(n_clusters=n_clusters, init=init, n_init='auto').fit_predict(obs_k_np)
    obs_clusters = []  # clustering result
    obs_indexs = []  # clustering index result
    label_class = np.unique(obs_label).squeeze()
    if label_class.shape == (): # only one number (-1)
        label_class_num = 1
    else:
        label_class_num = label_class.shape[0]
    for i in range(label_class_num):
        if label_class_num == 1:
            label_ = label_class
        else:
            label_ = label_class[i]
        if label_ != -1:
            obs_index = np.array(np.where(obs_label == label_)).squeeze().tolist()
            if isinstance(obs_index, int):
                obs_index = [obs_index]
            obs_indexs.append(obs_index)
        else:
            obs_index = np.array(np.where(obs_label == label_)).squeeze().tolist()
            if isinstance(obs_index, int):
                obs_index = [obs_index]
            for obs_id in obs_index:
                obs_indexs.append([obs_id])
        obs_cluster = [obs_k[io] for io in obs_index]
        obs_clusters.append(obs_cluster)
    
    # # 按数量从大到小排列
    # if len(obs_indexs) >= 2:
    #     n_ks = [len(obs_index) for obs_index in obs_indexs]
    #     index_n_ks = sorted(enumerate(n_ks), key = lambda x: x[1], reverse=True)
    #     indexs = [index_n_k[0] for index_n_k in index_n_ks]
    #     obs_clusters_sorted, obs_indexs_sorted = [], []
    #     for index in indexs:
    #         obs_clusters_sorted.append(obs_clusters[index])
    #         obs_indexs_sorted.append(obs_indexs[index])
    #     obs_clusters, obs_indexs = obs_clusters_sorted, obs_indexs_sorted
    return obs_clusters, obs_indexs

def Hierarchy_Clustering_Obs(
        Z, t, criterion='distance', min_samples = 2
) -> Tuple[List]:
    """
    clustering the measurement by DBSCAN.
    Args:
        obs_k: measurement, List[np.array(m,1)].
        eps and min_samples: parameters of DBSCAN.
        Sigma: sigma matrice used for calculating mahalanobis dis.
    Returns:
        obs_clusters: List[List[np.ndarray(m, 1)]].
        obs_indexs: List[List[int]].
    """
    obs_label = fcluster(Z, t, criterion=criterion)
    obs_indexs = []  # clustering index result
    label_class = np.unique(obs_label).squeeze()
    if label_class.shape == (): # only one number (-1)
        label_class_num = 1
    else:
        label_class_num = label_class.shape[0]
    for i in range(label_class_num):
        if label_class_num == 1:
            label_ = label_class
        else:
            label_ = label_class[i]
        if label_ != -1:
            obs_index = np.array(np.where(obs_label == label_)).squeeze().tolist()
            if isinstance(obs_index, int):
                obs_index = [obs_index]
            if len(obs_index) >= min_samples:
                obs_indexs.append(obs_index)
    return obs_indexs

def Hierarchy_Clustering_Obs_PHD(
        Z, t, criterion='distance'
) -> Tuple[List]:
    "与上述区别是，此处把认为杂波的序列也单独列为一簇"
    """
    clustering the measurement by DBSCAN.
    Args:
        obs_k: measurement, List[np.array(m,1)].
        eps and min_samples: parameters of DBSCAN.
        Sigma: sigma matrice used for calculating mahalanobis dis.
    Returns:
        obs_clusters: List[List[np.ndarray(m, 1)]].
        obs_indexs: List[List[int]].
    """
    obs_label = fcluster(Z, t, criterion=criterion)
    obs_indexs = []  # clustering index result
    label_class = np.unique(obs_label).squeeze()
    if label_class.shape == (): # only one number (-1)
        label_class_num = 1
    else:
        label_class_num = label_class.shape[0]
    for i in range(label_class_num):
        if label_class_num == 1:
            label_ = label_class
        else:
            label_ = label_class[i]
        if label_ != -1:
            obs_index = np.array(np.where(obs_label == label_)).squeeze().tolist()
            if isinstance(obs_index, int):
                obs_index = [obs_index]
            obs_indexs.append(obs_index)
        else:
            obs_index = np.array(np.where(obs_label == label_)).squeeze().tolist()
            if isinstance(obs_index, int):
                obs_index = [obs_index]
            for obs_id in obs_index:
                obs_indexs.append([obs_id])
    return obs_indexs