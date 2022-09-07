import numpy as np
import sympy as sp

# To-Do
# Feature Matrix Function
# OLS Solver
# Ridge Method


def FeatureMatrix(poly_degree, x_in, y_in):
    X = np.ones(len(y), poly_degree*poly_degree/2)
    for p in range(1,poly_degree):
        for a in range(p+1):
            X[:,] = x_in**p
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
