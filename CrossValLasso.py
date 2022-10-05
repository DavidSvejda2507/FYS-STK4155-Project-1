import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv
#N = 100
poly_list = gv.poly_list
poly_max = gv.poly_max
poly_range = np.arange(1, poly_max + 1)

for N in gv.N_list:
    x, y, F = Dataset.GenerateData(N)
    for n_folds in gv.k_list:
        for L in gv.lambda_list:

            MSE = np.zeros(poly_max)
            Bias = np.zeros(poly_max)
            Variance = np.zeros(poly_max)
            for p in poly_range:
                y_pred_folds = np.empty((int(N/n_folds), n_folds))
                F_test_multi = np.empty((int(N/n_folds), n_folds))
                for k in range(n_folds):
                    mask = np.arange(0, N)%n_folds!=k
                    nmask = np.logical_not(mask)
                    x_ = x[mask]
                    y_ = y[mask]
                    F_ = F[mask]
                    X = Solve.FeatureMatrix(p, x_, y_)
                    X_test = Solve.FeatureMatrix(x[nmask], y[nmask])
                    X, m_x, s_x = Dataset.Scaling(X)
                    F_train_sc, m_F, s_F = Dataset.Scaling(F_)
                    LassoModel = Lasso()
                    LassoModel.fit(X, F_)
                    B = LassoModel.coef_
                    B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
                    F_test_multi[:, k] = F[nmask]
                    y_pred_folds[:, k] = Solve.PredictFunctionInter(p, B_)(x[nmask], y[nmask])
                MSE[p-1] = np.mean(np.mean((F_test_multi - y_pred_folds)**2, axis=1))
                # Bias[p-1] = np.mean((F_test_multi - np.mean(y_pred_folds, axis=1))**2)
                # Variance[p-1] = np.mean(np.var(y_pred_folds, axis=1) )
            for p in poly_list:

                b_MSE, b_Bias, b_Variance = np.load(f'ArrayData/LassoMSEBV{N}P{p}L{L}.npy')
                Analysis.plotfunc(f'RidgeCrosValN{N}P{p}k{n_folds}L{L}.pdf',
                                {
                                'Cross Validation' : (poly_range[:p], MSE[:p]),
                                'Bootstrap' : (poly_range[:p], b_MSE)
                                })

                np.save(f'ArrayData/LassoCrosValN{N}P{p}k{n_folds}', np.stack((MSE[:p], b_MSE)))
