import numpy as np
import sympy as sp

# To-Do
# Feature Matrix Function
# OLS Solver
# Ridge Method


def FeatureMatrix(poly_degree, x_in, y_in):
    n = len(x_in)
    tot_comb = int(0.5*(poly_degree+1)*(poly_degree+2)-1)
    X = np.zeros((n, tot_comb))
    X[:, 0] = x_in
    X[:, 1] = y_in
    X[:, 2] = x_in*x_in
    X[:, 3] = y_in*y_in
    X[:, 4] = y_in*x_in
    lower = 2
    upper = 4
    for i in range(2, poly_degree):
        print(upper)
        prev_l = lower
        prev_u = upper
        lower += i + 1
        upper += i + 2
        X[:, lower] = X[:, prev_l]*x_in
        X[:, lower+1] = X[:, prev_l+1]*y_in
        comb = upper - lower
        for k in range(2, comb):
            X[:, lower + k] = X[:, prev_l+k]*x_in
        X[:, upper] = X[:, lower-1]*y_in
    return X

def SolveOLS(X, y):
    B = np.liealg.inv(X@X.T)@X.T@y
    return B

#λ
def RidgeMethod(X, y, λ):
    n = len(y)
    I = np.identity(n)
    B = np.linalg.inv(X.T@X + n*λ*I)@X.T@y
    return B



x_in = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
y_in = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
print(FeatureMatrix(5, x_in, y_in))
