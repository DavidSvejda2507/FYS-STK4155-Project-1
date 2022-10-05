import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y, F = Dataset.GenerateData(1000)
x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)

poly_max = 5
poly_range = np.arange(1, poly_max + 1)
MSE_List = np.zeros(poly_max)
R2_List = np.zeros(poly_max)
for p in poly_range:
    X = Solve.FeatureMatrix(p, x_train, y_train)
    X, m_x, σ_x = Dataset.Scaling(X)
    F_train, m_F, σ_F = Dataset.Scaling(F_train)
    B_ = Solve.SolveOLS(X, F_train)
    #B_ = Dataset.Unscale(B, m_x, σ_x, m_F, σ_F)
    X_test = Solve.FeatureMatrix(p, x_test, y_test)
    F_predict_B = X_test@B_
    F_predict_B = Dataset.UnscaleOutput(F_predict_B, m_F, σ_F)
    MSE_List[p-1] = Analysis.MSE(F_test, F_predict_B)
    R2_List[p-1] = Analysis.R2(F_test, F_predict_B)

    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(x_train, F_train, '.')
    plt.plot(x_train, X@B_, '.')
    plt.subplot(122)
    plt.plot(x_test, F_test, '.')
    plt.plot(x_test, F_predict_B, '.')
    plt.show()
plt.figure(figsize=(10, 6))
plt.subplot(121, title='MSE', xlabel='Polynomial Degree')
plt.plot(poly_range, MSE_List)
plt.subplot(122, title='R2', xlabel='Polynomial Degree')
plt.plot(poly_range, R2_List)
plt.savefig('Figures/MSER2polydeg.pdf')
plt.show()
