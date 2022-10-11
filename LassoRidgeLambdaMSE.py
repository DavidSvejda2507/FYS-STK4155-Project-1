import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import GlobVar as gv
from BootstrapFunc import SendToSolve


def MSELambda(x, y, F, p, name):
    L_range = np.logspace(-8, 0, 100)
    x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)
    MSE_Ridge = np.zeros(100)
    MSE_Lasso = np.zeros(100)
    X = Solve.FeatureMatrix(p, x_train, y_train)
    X, m_x, s_x = Dataset.Scaling(X)
    F_train_sc, m_F, s_F = Dataset.Scaling(F_train)
    B_OLS = SendToSolve(X, F_train_sc, None, 'OLS')
    B_OLS_U = Dataset.Unscale(B_OLS, m_x, s_x, m_F, s_F)
    F_predict_B_OLS = Solve.PredictFunctionInter(p, B_OLS_U)(x_test, y_test)
    MSE_OLS = Analysis.MSE(F_test, F_predict_B_OLS)
    for i, L_i in enumerate(L_range):
        B_Ridge = SendToSolve(X, F_train_sc, L_i, 'Ridge')
        B_Lasso = SendToSolve(X, F_train_sc, L_i, 'Lasso')
        B_Ridge_U = Dataset.Unscale(B_Ridge, m_x, s_x, m_F, s_F)
        B_Lasso_U = Dataset.Unscale(B_Lasso, m_x, s_x, m_F, s_F)
        F_predict_B_Ridge = Solve.PredictFunctionInter(p, B_Ridge_U)(x_test, y_test)
        F_predict_B_Lasso = Solve.PredictFunctionInter(p, B_Lasso_U)(x_test, y_test)
        MSE_Ridge[i] = Analysis.MSE(F_test, F_predict_B_Ridge)
        MSE_Lasso[i] = Analysis.MSE(F_test, F_predict_B_Lasso)
    Analysis.plotfunc(name,
                {
                'MSE Ridge' : (np.log10(L_range), MSE_Ridge),
                'MSE Lasso' : (np.log10(L_range), MSE_Lasso),
                'MSE OLS' : (np.log10(L_range), np.repeat(MSE_OLS, len(L_range)))
                }, xtickset=False, ylim=(0, max(np.max(MSE_Lasso), np.max(MSE_Ridge))*1.1), yscale='linear', xlabel='log(λ)')
    return MSE_Ridge, MSE_Lasso, MSE_OLS


if __name__=='__main__':
    for N in [20, 100, 500]:
        for ns in [0.1]:
            x, y, F = Dataset.GenerateData(N, True, ns)
            plist = [8, 11, 14, 17, 20]
            for p in plist:
                MSE_Ridge, MSE_Lasso, MSE_OLS = MSELambda(x, y, F, p, f'Lambda/ShiftedDomain/LassoRidgeMSEPN{N}P{p}ns{ns}.pdf')
