import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

N = 100
DatNum = 2*np.arange(5, N)
N_true = len(DatNum)
R2_List = np.zeros(N_true)
MSE_List = np.zeros(N_true)

for k, Num in enumerate(DatNum):
    x, y, F = Dataset.GenerateData(Num)
    x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)
    p = 10
    X = Solve.FeatureMatrix(p, x_train, y_train)
    X, m_x, s_x = Dataset.Scaling(X)
    F_train_sc, m_F, s_F = Dataset.Scaling(F_train)
    B = Solve.SolveOLS(X, F_train_sc)
    B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
    F_predict_B = Solve.PredictFunctionInter(p, B_)(x_test, y_test)
    MSE_List[k] = Analysis.MSE(F_test, F_predict_B)
    R2_List[k] = Analysis.R2(F_test, F_predict_B)
    # plt.figure(figsize=(10,6))
    # plt.subplot(121)
    # plt.plot(x_train, F_train_sc, '.')
    # plt.plot(x_train, X@B, '.')
    # plt.subplot(122)
    # plt.plot(x_test, F_test, '.')
    # plt.plot(x_test, F_predict_B, '.')
    # #plt.show()
plt.figure(figsize=(10, 6))
plt.subplot(121, title='MSE', xlabel='Total Data Points')
plt.plot(DatNum, MSE_List)
plt.subplot(122, title='R2', xlabel='Total Data Points')
plt.plot(DatNum, R2_List)
plt.savefig(f'Figures/MSER2DatNum.pdf')
#plt.show()
