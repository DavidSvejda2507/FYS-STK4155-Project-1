import numpy as np
import Analysis
import GlobVar as gv
import Dataset
from BootstrapFunc import Bootstrap, RealBootstrap, SendToSolve
from sklearn.model_selection import train_test_split
import Solve
from LassoRidgeLambdaMSE import MSELambda

def PlotLambda(x, y, terrain):
    plist = [13, 14, 15]
    for p in plist:
        MSELambda(x, y, terrain, p, f'Real/Lambda/LassoRidgeMSEP{p}.pdf')

def OLS(x, y, terrain):
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

            # plt.figure(figsize=(10,6))
            # plt.subplot(121)
            # plt.plot(x_train, F_train_sc, '.')
            # plt.plot(x_train, X@B, '.')
            # plt.subplot(122)
            # plt.plot(x_test, F_test, '.')
            # plt.plot(x_test, F_predict_B, '.')
            # #plt.show()
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(121, title='MSE', xlabel='Polynomial Degree')
        plt.plot(poly_range, MSE_List)
        plt.subplot(122, title='R2', xlabel='Polynomial Degree')
        plt.plot(poly_range, R2_List)
        plt.savefig(f'Figures/Real/MSER2deg{poly_max}.pdf')
        plt.close(fig)

def RunBootstrap(x, y, terrain, L, print_=True):
    #Bootsrap bias variance tradeoff
    p = 20
    #MSE, Bias, Variance = RealBootstrap(x, y, terrain, 'OLS', gv.bootstraps, p)
    #, MSEVarB, VarSizeNum, VarSizeAn
    MSE_Ridge, Bias_Ridge, Variance_Ridge = RealBootstrap(x, y, terrain, 'Ridge', gv.bootstraps, p, print_=False, L=L)
    MSE_Lasso, Bias_Lasso, Variance_Lasso = RealBootstrap(x, y, terrain, 'Lasso', gv.bootstraps, p, print_=False, L=L)
    #np.save(f'ArrayData/RealBootstrapOLSBVariance', np.stack((MSEVarB, VarSizeNum, VarSizeAn)))
    Analysis.plotfunc(f'Real/MSEBiasVarOLSRidge.pdf',
                    {
                    'MSE' : (range(1, p+1), MSE_Ridge),
                    'Bias' : (range(1, p+1), Bias_Ridge),
                    'Variance' : (range(1, p+1), Variance_Ridge),
                    })
    Analysis.plotfunc(f'Real/MSEBiasVarOLSLasso.pdf',
                    {
                    'MSE' : (range(1, p+1), MSE_Lasso),
                    'Bias' : (range(1, p+1), Bias_Lasso),
                    'Variance' : (range(1, p+1), Variance_Lasso),
                    })
    return
def RealCrossVal(x, y, terrain, p, L, print_=True):

    MSE = np.zeros_like(range(1, gv.poly_max+1))
    MSE_Ridge = np.zeros_like(range(1, gv.poly_max+1))
    MSE_Lasso = np.zeros_like(range(1, gv.poly_max+1))

    x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(x, y, terrain, test_size=0.2)
    F_test_multi = np.empty((int(N/n_folds), n_folds))

    y_pred_folds = np.empty((int(N/n_folds), n_folds))
    y_pred_folds_Ridge = np.empty((int(N/n_folds), n_folds))
    y_pred_folds_Lasso = np.empty((int(N/n_folds), n_folds))

    for k in range(n_folds):
        mask = np.arange(0, N)%n_folds!=k
        nmask = np.logical_not(mask)
        x_ = x[mask]
        y_ = y[mask]
        F_ = F[mask]
        X = Solve.FeatureMatrix(p, x_, y_)
        X, m_x, s_x = Dataset.Scaling(X)
        F_train_sc, m_F, s_F = Dataset.Scaling(F_)
        B = SendToSolve(X, F_train_sc, L, 'OLS')
        B_U = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
        B_Ridge = SendToSolve(X, F_train_sc, L, 'Ridge')
        B_Ridge_U = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
        B_Lasso = SendToSolve(X, F_train_sc, L, 'Lasso')
        B_Lasso_U = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
        F_test_multi[:, k] = F[nmask]
        y_pred_folds[:, k] = Solve.PredictFunctionInter(p, B_U)(x[nmask], y[nmask])
        y_pred_folds_Ridge[:, k] = Solve.PredictFunctionInter(p, B_Ridge_U)(x[nmask], y[nmask])
        y_pred_folds_Lasso[:, k] = Solve.PredictFunctionInter(p, B_Lasso_U)(x[nmask], y[nmask])
    MSE[p-1] = np.mean(np.mean((F_test_multi - y_pred_folds)**2, axis=1))
    MSE_Ridge[p-1] = np.mean(np.mean((F_test_multi - y_pred_folds_Ridge)**2, axis=1))
    MSE_Lasso[p-1] = np.mean(np.mean((F_test_multi - y_pred_folds_Lasso)**2, axis=1))
    return MSE

x, y, t = Dataset.GetRealData(1)
RunBootstrap(x, y, t, 1e-8)
