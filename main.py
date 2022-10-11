import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import GlobVar as gv

for N in gv.N_list:
    for ns in [0, 0.1, 0.3]:
        x, y, F = Dataset.GenerateData(1000, True, ns)
        x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)

        poly_max = 20
        poly_range = np.arange(1, poly_max + 1)
        MSE_List = np.zeros(poly_max)
        R2_List = np.zeros(poly_max)
        for p in poly_range:
            X = Solve.FeatureMatrix(p, x_train, y_train)
            X, m_x, s_x = Dataset.Scaling(X)
            F_train_sc, m_F, s_F = Dataset.Scaling(F_train)
            B = Solve.SolveOLS(X, F_train_sc)
            B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
            F_predict_B = Solve.PredictFunctionInter(p, B_)(x_test, y_test)
            MSE_List[p-1] = Analysis.MSE(F_test, F_predict_B)
            R2_List[p-1] = Analysis.R2(F_test, F_predict_B)
        Analysis.plotfunc(f'Updated/MSE OLS/MSEN{N}P{p}ns{ns}.pdf',
                        {
                        'σ^2' : (poly_range, np.repeat(ns**2, len(poly_range))),
                        'MSE' : (poly_range, MSE_List)
                        }, yscale='linear')
