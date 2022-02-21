import numpy as np
from scipy.spatial import KDTree

def find_driver_nn(timestamp, for_estim_param, drivers, k):
    """
    return k nearest neighbors considering drivers (list) of price at time timestamp
    """
    kdtree = KDTree(for_estim_param[drivers].to_numpy())
    x = for_estim_param.query('@timestamp == TIMESTAMP')[drivers]
    nn = kdtree.query(x, k)
    try:
        return for_estim_param.iloc[list(nn[1].reshape(-1))], nn[1] # return index
    except:
        return np.array([]), np.array([])