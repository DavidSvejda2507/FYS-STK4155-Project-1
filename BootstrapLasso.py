import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv

poly_list = gv.poly_list
poly_max = gv.poly_max

for N in gv.N_list:
    for L in gv.lambda_list:
        x, y, F = Dataset.GenerateData(N)
        x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)

        poly_range = np.arange(1, poly_max + 1)

        n_bootstraps = gv.bootstraps
        MSE = np.zeros(poly_max)
        Bias = np.zeros(poly_max)
        Variance = np.zeros(poly_max)

        F_test_multi = np.array([list(F_test)]*n_bootstraps).T

        for p in range(1, poly_max+1):
            y_pred_bootstraps = np.empty((F_test.shape[0], n_bootstraps))
            for n in range(n_bootstraps):
                x_, y_, F_ = resample(x_train, y_train, F_train)
                X = Solve.FeatureMatrix(p, x_, y_)
                X, m_x, s_x = Dataset.Scaling(X)
                F_train_sc, m_F, s_F = Dataset.Scaling(F_)
                LassoModel = Lasso()
                LassoModel.fit(X, F_)
                B = LassoModel.coef_
                B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
                y_pred_bootstraps[:, n] = Solve.PredictFunctionInter(p, B_)(x_test, y_test)
            MSE[p-1] = np.mean(np.mean((F_test_multi - y_pred_bootstraps)**2, axis=1))
            Bias[p-1] = np.mean((F_test - np.mean(y_pred_bootstraps, axis=1))**2)
            Variance[p-1] = np.mean(np.var(y_pred_bootstraps, axis=1) )

        for p in poly_list:

            prange = np.arange(1, p+1)
            Analysis.plotfunc(f'LassoBiaVarTradeBootN{N}P{p}L{L}.pdf',
                            {
                            'MSE' : (prange, MSE[:p]),
                            'Bias' : (prange, Bias[:p]),
                            'Variance' : (prange, Variance[:p]),
                            })

            np.save(f'ArrayData/LassoMSEBV{N}P{p}L{L}', np.stack((MSE[:p], Bias[:p], Variance[:p])))
