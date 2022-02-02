import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def find_driver_nn(timestamp, for_estim_param, drivers):
    kdtree = KDTree(for_estim_param[drivers].to_numpy())
    x = for_estim_param.query('@timestamp == TIMESTAMP')[drivers]
    k = 1
    nn = kdtree.query(x, k)
    try:
        return for_estim_param.iloc[list(nn[1].reshape(-1))], nn[1] # return index
    except:
        return np.array([]), np.array([])

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# estim = pd.read_csv("C:/Users/cotil/Desktop/COURS/Mines/2A/Sophia/Projet/main/data_elecprices/estim_param/for_estim_param_df.csv")
# estim = estim[ (estim['AREAS'] == 'FR') & (estim['TECHNOLOGIES'] == 'Nuclear') ]
# print(estim)
# nn = find_driver_nn(7, estim, ['margin'])
# print(nn)