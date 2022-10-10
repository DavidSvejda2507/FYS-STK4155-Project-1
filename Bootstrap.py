import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv
from BootstrapFunc import Bootstrap, printpoly

plot = True
save = True
print_ = True
plotvar = True

if print_:
    print('--------- Run Info ----------')
    print(f'Noise = {gv.ns}')
    print(f'Poly max = {gv.poly_max}')
    print(f'Bootstraps = {gv.bootstraps}')

for N in gv.N_list:
    #N = 100
    #poly_max = 20
    # poly_range = np.arange(1, poly_max + 1)
    #
    # n_bootstraps = gv.bootstraps
    # MSE = np.zeros(poly_max)
    # Bias = np.zeros(poly_max)
    # Variance = np.zeros(poly_max)
    #
    # F_test_multi = np.array([list(F_test)]*n_bootstraps).T
    #
    # for p in range(1, poly_max+1):
    #     y_pred_bootstraps = np.empty((F_test.shape[0], n_bootstraps))
    #     for n in range(n_bootstraps):
    #         x_, y_, F_ = resample(x_train, y_train, F_train)
    #         X = Solve.FeatureMatrix(p, x_, y_)
    #         X, m_x, s_x = Dataset.Scaling(X)
    #         F_train_sc, m_F, s_F = Dataset.Scaling(F_)
    #         B = Solve.SolveOLS(X, F_train_sc)
    #         B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
    #         y_pred_bootstraps[:, n] = Solve.PredictFunctionInter(p, B_)(x_test, y_test)
    #     MSE[p-1] = np.mean(np.mean((F_test_multi - y_pred_bootstraps)**2, axis=1))
    #     Bias[p-1] = np.mean((F_test - np.mean(y_pred_bootstraps, axis=1))**2)
    #     Variance[p-1] = np.mean(np.var(y_pred_bootstraps, axis=1) )

    MSE, Bias, Variance, MSEVarB, VarSizeNum, VarSizeAn = Bootstrap(N, 'OLS', gv.ns, gv.bootstraps, gv.poly_max)

    if print_:
        print('---------- Run Info ---------------')
        print(f' Datapoints = {N}')

    for p in gv.poly_list:
        prange = np.arange(1, p+1)
        if plot:
            Analysis.plotfunc(f'BiaVarTradeBootN{N}P{p}ns{gv.ns}.pdf',
                            {
                            'MSE' : (prange, MSE[:p]),
                            'Bias' : (prange, Bias[:p]),
                            'Variance' : (prange, Variance[:p]),
                            })
        if print_:
            Bootstrap.printpoly(p, MSE, Bias, Variance, MSEVarB, VarSizeNum, VarSizeAn)

    if save:
        np.save(f'ArrayData/MSEBV{N}', np.stack((MSE, Bias, Variance, MSEVarB, VarSizeNum, VarSizeAn)))
    if plotvar:
        Analysis.plotfunc(f'MSESizeVarBN{N}.pdf',
                {
                'MSEVarB' : (np.arange(1, gv.poly_max+1), MSEVarB),
                'VarSizeNum' : (np.arange(1, gv.poly_max+1), VarSizeNum),
                'VarSizeAn' : (np.arange(1, gv.poly_max+1), VarSizeAn)
                })
