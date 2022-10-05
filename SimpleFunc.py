import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

n = 1000
x = np.random.rand(n)
y = Dataset.SimpleFunc(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

poly_max = 5
poly_range = np.arange(1, poly_max + 1)
MSE_List = np.zeros(poly_max)
R2_List = np.zeros(poly_max)
for p in range(1, poly_max):
    X_sk = Solve.SimpleFeature(p, x)
    lin_reg = LinearRegression()
    lin_reg.fit(X_sk, y)
    X = Solve.SimpleFeature(p, x_train)
    X, m_x, s_x = Dataset.Scaling(X)
    Y_train, m_F, s_F = Dataset.Scaling(y_train)
    B = Solve.SolveOLS(X, Y_train)
    B_ = Dataset.Unscale(B, m_x, s_x, m_F, s_F)
    print('--------------------------')
    print(lin_reg.coef_)
    print(B)
    print(B_)
    X_test = Solve.SimpleFeature(p, x_test, inter=True)
    print(np.mean(X, axis=0))
    print(np.mean(X_test, axis=0))
    y_predict_B = X_test@B_
    MSE_List[p-1] = Analysis.MSE(y_test, y_predict_B)
    R2_List[p-1] = Analysis.R2(y_test, y_predict_B)

    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(x_train, Y_train, '.')
    plt.plot(x_train, X@B, '.')
    X_train = Solve.SimpleFeature(p, x_train)
    plt.plot(x_train, lin_reg.predict(X_train), '.')
    plt.subplot(122)
    plt.plot(x_test, y_test, '.')
    plt.plot(x_test, y_predict_B, '.')
    X_test_sk = Solve.SimpleFeature(p, x_test)
    plt.plot(x_test, lin_reg.predict(X_test_sk), '.')
    plt.show()
plt.figure(figsize=(10, 6))
plt.subplot(121, title='MSE', xlabel='Polynomial Degree')
plt.plot(poly_range, MSE_List)
plt.subplot(122, title='R2', xlabel='Polynomial Degree')
plt.plot(poly_range, R2_List)
plt.savefig('Figures/MSER2polydeg.pdf')
plt.show()
