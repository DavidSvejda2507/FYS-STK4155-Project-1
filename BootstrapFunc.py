import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.linear_model import Lasso
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

def Bootstrap(N, method, ns, n_bootstraps, poly_max, L=None):

    x, y, F = Dataset.GenerateData(N)
    x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)

    poly_range = np.arange(1, poly_max + 1)

    n_bootstraps = gv.bootstraps
    MSE = np.zeros(poly_max)
    Bias = np.zeros(poly_max)
    Variance = np.zeros(poly_max)
    MSEVarB = np.zeros(poly_max)
    VarSizeAn = np.zeros(poly_max)
    VarSizeNum = np.zeros(poly_max)

    F_test_multi = np.array([list(F_test)]*n_bootstraps).T
    print('20 | ', end='')
    for p in range(1, poly_max+1):
        if (p%5 == 0):print('%', end='', flush=True)
        else:print('#', end='', flush=True)
        y_pred_bootstraps = np.empty((F_test.shape[0], n_bootstraps))
        B_bootstraps = np.empty((int(0.5*(p+1)*(p+2)), n_bootstraps))
        for n in range(n_bootstraps):
            x_, y_, F_ = resample(x_train, y_train, F_train)
            X = Solve.FeatureMatrix(p, x_, y_)
            X, m_x, s_x = Dataset.Scaling(X)
            F_train_sc, m_F, s_F = Dataset.Scaling(F_)
            B = SendToSolve(X, F_train_sc, L, method)
            B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
            B_bootstraps[:, n] = B_
            y_pred_bootstraps[:, n] = Solve.PredictFunctionInter(p, B_)(x_test, y_test)

        X_p = np.ones((X.shape[0], X.shape[1] + 1))
        X_p[:, 1:]= Solve.FeatureMatrix(p, x_train, y_train)
        B_var_analytic = ns*ns*np.linalg.pinv(X_p.T@X_p)
        B_var_analytic = B_var_analytic.diagonal()
        B_var = np.var(B_bootstraps, axis=1)
        VarSizeAn[p-1] = np.linalg.norm(B_var_analytic)
        VarSizeNum[p-1] = np.linalg.norm(B_var)
        MSEVarB[p-1] = np.linalg.norm(B_var_analytic-B_var)
        MSE[p-1] = np.mean(np.mean((F_test_multi - y_pred_bootstraps)**2, axis=1))
        Bias[p-1] = np.mean((F_test - np.mean(y_pred_bootstraps, axis=1))**2)
        Variance[p-1] = np.mean(np.var(y_pred_bootstraps, axis=1) )
    print(' |')
    return MSE, Bias, Variance, MSEVarB, VarSizeNum, VarSizeAn

def printpoly(p, MSE, Bias, Variance, MSEVarB, VarSizeNum, VarSizeAn)
    print(f'------------ Poly degree {p} -------------')
    print(f'MSE  = {MSE[p-1]}')
    print(f'Bias = {Bias[p-1]}')
    print(f'Variance = {Variance[p-1]}')
    print(f'MSE Var(B) = {MSEVarB[p-1]}')
    print(f'|Var(B)| Numerical = {VarSizeNum[p-1]}')
    print(f'|Var(B)| Analytical = {VarSizeAn[p-1]}')
