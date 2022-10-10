import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv

def SendToSolve(X, F_train_sc, L, method):
    if method == 'OLS':
        B = Solve.SolveOLS(X, F_train_sc)
    elif method == 'Ridge':
        if L == None:
            raise ValueError(f'Need a lambda for {method} method')
        B = Solve.RidgeMethod(X, F_train_sc, L)
    elif method == 'Lasso':
        if L == None:
            raise ValueError(f'Need a lambda for {method} method')
        LassoModel = Lasso(L)
        LassoModel.fit(X, F_train_sc)
        B = LassoModel.coef_
    else:
        raise ValueError(f'Method {method} not accepted')
    return B


def CrossVal(x, y, F, N, n_folds, method, L=None):

    MSE = np.zeros(gv.poly_max)
    Bias = np.zeros(gv.poly_max)
    Variance = np.zeros(gv.poly_max)
    print('20 | ', end='')
    for p in gv.poly_range:
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
            B = SendToSolve(X, F_train_sc, L)
            B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
            F_test_multi[:, k] = F[nmask]
            y_pred_folds[:, k] = Solve.PredictFunctionInter(p, B_)(x[nmask], y[nmask])
        MSE[p-1] = np.mean(np.mean((F_test_multi - y_pred_folds)**2, axis=1))
    print(' |')
    return MSE
