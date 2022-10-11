import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import GlobVar as gv
from BootstrapFunc import SendToSolve

def MSELAmbadaTest():
    L_i = 0
    p = 11
    x, y, F = Dataset.GenerateData(500, True, 0.5)
    x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)
    MSE_Ridge = np.zeros(100)
    MSE_Lasso = np.zeros(100)
    X = Solve.FeatureMatrix(p, x_train, y_train)
    X, m_x, s_x = Dataset.Scaling(X)
    F_train_sc, m_F, s_F = Dataset.Scaling(F_train)
    #B_OLS = SendToSolve(X, F_train_sc, None, 'OLS')
    #B_OLS_U = Dataset.Unscale(B_OLS, m_x, s_x, m_F, s_F)
    #F_predict_B_OLS = Solve.PredictFunctionInter(p, B_OLS_U)(x_test, y_test)
    #print(Analysis.MSE(F_test, F_predict_B_OLS))
    B_Ridge = SendToSolve(X, F_train_sc, L_i, 'Lasso')
    B_Ridge_U = Dataset.Unscale(B_Ridge, m_x, s_x, m_F, s_F)
    F_predict_B_Ridge = Solve.PredictFunctionInter(p, B_Ridge_U)(x_test, y_test)
    print(Analysis.MSE(F_test, F_predict_B_Ridge))

MSELAmbadaTest()
