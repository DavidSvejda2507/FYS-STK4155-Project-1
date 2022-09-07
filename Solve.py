import numpy as np
import sympy as sp

# To-Do
# Feature Matrix Function
# OLS Solver
# Ridge Method


def Feature_Matrix(poly_degree, x_in):
    X = np.zeros(len(y), poly_degree)
    for p in range(poly_degree):
        X[:,p] = x_in**p
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
