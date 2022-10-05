import numpy as np
import sympy
from sympy import lambdify
# To-Do
# Feature Matrix Function
# OLS Solver
# Ridge Method

def SimpleFeature(poly_degree, x_in, inter=False):
    if not inter:
        n = len(x_in)
        X = np.zeros((n, poly_degree))
        X[:, 0] = x_in
        for i in range(1, poly_degree):
            X[:, i] = x_in**(i+1)#X[:, i-1]*x_in
    else:
        n = len(x_in)
        X = np.zeros((n, poly_degree+1))
        X[:, 0] = 1
        X[:, 1] = x_in
        for i in range(2, poly_degree+1):
            X[:, i] = x_in**i#X[:, i-1]*x_in
    return X

def FeatureMatrix(poly_degree, x_in, y_in):
    n = len(x_in)
    tot_comb = int(0.5*(poly_degree+1)*(poly_degree+2)-1)
    X = np.zeros((n, tot_comb))
    X[:, 0] = x_in
    X[:, 1] = y_in
    if poly_degree > 1:
        X[:, 2] = x_in*x_in
        X[:, 3] = y_in*y_in
        X[:, 4] = y_in*x_in
    else:
        return X
    lower = 2
    upper = 4
    for i in range(2, poly_degree):
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

def PredictFunction(poly_degree, B):
    tot_comb = int(0.5*(poly_degree+1)*(poly_degree+2)-1)
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    X = [x]*tot_comb
    X[0] = x
    X[1] = y
    if poly_degree > 1:
        X[2] = x*x
        X[3] = y*y
        X[4] = y*x
    else:
        X = np.array(X)
        F_predict = lambdify([x, y], X@B)
        return F_predict
    lower = 2
    upper = 4
    for i in range(2, poly_degree):
        prev_l = lower
        prev_u = upper
        lower += i + 1
        upper += i + 2
        X[lower] = X[prev_l]*x
        X[lower+1] = X[prev_l+1]*y
        comb = upper - lower
        for k in range(2, comb):
            X[lower + k] = X[prev_l+k]*x
        X[upper] = X[lower-1]*y
    X = np.array(X)
    F_predict = lambdify([x, y], X@B)
    return F_predict

def PredictFunctionInter(poly_degree, B):
    tot_comb = int(0.5*(poly_degree+1)*(poly_degree+2)-1)
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    X = [x]*tot_comb
    X[0] = x
    X[1] = y
    if poly_degree > 1:
        X[2] = x*x
        X[3] = y*y
        X[4] = y*x
    else:
        X.insert(0, 1)
        X = np.array(X)
        F_predict = lambdify([x, y], X@B)
        return F_predict
    lower = 2
    upper = 4
    for i in range(2, poly_degree):
        prev_l = lower
        prev_u = upper
        lower += i + 1
        upper += i + 2
        X[lower] = X[prev_l]*x
        X[lower+1] = X[prev_l+1]*y
        comb = upper - lower
        for k in range(2, comb):
            X[lower + k] = X[prev_l+k]*x
        X[upper] = X[lower-1]*y
    X.insert(0, 1)
    X = np.array(X)
    F_predict = lambdify([x, y], X@B)
    return F_predict


def SolveOLS(X, y):
    B = np.linalg.pinv(X.T @ X) @ X.T @ y
    return B

#λ
def RidgeMethod(X, y, λ):
    n = np.shape(X.T @ X)[0]
    I = np.identity(n)
    B = np.linalg.pinv(X.T @ X + λ*I) @ X.T @ y
    return B
