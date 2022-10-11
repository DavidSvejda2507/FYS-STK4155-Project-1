import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv
from BootstrapFunc import SendToSolve


def CrossVal(x, y, F, N, n_folds, method, poly_min=1, poly_max=gv.poly_max, print_=True, L=None):

    MSE = np.zeros_like(range(poly_min, poly_max+1))
    if print_:print('20 | ', end='')
    for p in range(poly_min, poly_max):
        if print_:
            if (p%5 == 0):print('%', end='', flush=True)
            else:print('#', end='', flush=True)
        y_pred_folds = np.empty((int(N/n_folds), n_folds))
        F_test_multi = np.empty((int(N/n_folds), n_folds))
        for k in range(n_folds):
            mask = np.arange(0, N)%n_folds!=k
            nmask = np.logical_not(mask)
            x_ = x[mask]
            y_ = y[mask]
            F_ = F[mask]
            X = Solve.FeatureMatrix(p, x_, y_)
            X, m_x, s_x = Dataset.Scaling(X)
            F_train_sc, m_F, s_F = Dataset.Scaling(F_)
            B = SendToSolve(X, F_train_sc, L, method)
            B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
            F_test_multi[:, k] = F[nmask]
            y_pred_folds[:, k] = Solve.PredictFunctionInter(p, B_)(x[nmask], y[nmask])
        MSE[p-1] = np.mean(np.mean((F_test_multi - y_pred_folds)**2, axis=1))
    if print_:print(' |')
    return MSE
