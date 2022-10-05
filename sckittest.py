import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed()
nmax = 15
r2_train_scaled_list = np.zeros(nmax-1)
r2_test_scaled_list = np.zeros(nmax-1)
mse_train_scaled_list = np.zeros(nmax-1)
mse_test_scaled_list = np.zeros(nmax-1)

import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y, F = Dataset.GenerateData(1000)
x_train, x_test, y_train, y_test, F_train, F_test = train_test_split(x, y, F, test_size=0.2)

for n in range(1, nmax):
    # Make data set.

    poly_features = PolynomialFeatures(degree=n, include_bias=False)
    X = Solve.FeatureMatrix(n, x_train, y_train)
    X_test = Solve.FeatureMatrix(n, x_test, y_test)
    OLSbeta = np.linalg.inv(X.T@X)@X.T@F_train
    ytildeOLS = X@OLSbeta
    ypredictOLS = X@OLSbeta

    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train_scaled = scaler.transform(X)
    X_test_scaled = scaler.transform(X_test)
    OLSbeta_scaled = np.linalg.inv(X_train_scaled.T@X_train_scaled)@X_train_scaled.T@F_train
    ytildeOLS_scaled = x_train_scaled@OLSbeta_scaled
    ypredictOLS_scaled = x_test_scaled@OLSbeta_scaled



    r2_train, r2_test = r2_score(y_train, ytildeOLS), r2_score(y_test, ypredictOLS)
    mse_train, mse_test = mean_squared_error(y_train, ytildeOLS), mean_squared_error(y_test, ypredictOLS)


    r2_train_scaled, r2_test_scaled = r2_score(y_train, ytildeOLS_scaled), \
    r2_score(y_test, ypredictOLS_scaled)
    mse_train_scaled, mse_test_scaled = mean_squared_error(y_train, \
    ytildeOLS_scaled), mean_squared_error(y_test, ypredictOLS_scaled)
    for i in ('r2', 'mse'):
        for k in ('train', 'test'):
            command = i + '_' + k
            exec('ans = ' + command)
            exec('ans_scaled = ' + command + '_scaled')
            exec("print(" + "'"+command+"'" + ', ":", ' + f'{ans})')
            exec(command+'_scaled_list' + f'[{n-1}]' f'= {ans_scaled}')

n_range = np.arange(1,n+1)
plt.plot(n_range, np.array(r2_test_scaled_list), label='r2_test')
plt.plot(n_range, np.array(r2_train_scaled_list), label='r2_train')
plt.plot(n_range, np.array(mse_test_scaled_list), label='mse_test')
plt.plot(n_range, np.array(mse_train_scaled_list), label='mse_train')
plt.legend()
plt.show()
