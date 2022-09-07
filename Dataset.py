import hashlib
import numpy as np

loss = '|I|Ii|II|I_|'
seed = int(hashlib.sha1(loss.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
np.random.seed(seed)

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#σ
def Scaling(X):
    n = len(X[0])
    for i in range(n):
        x_m = np.sum(X[i])/n
        σ = np.sqrt((1/n)(X[i]-x_m)@(X[i]-x_m))
        ones = np.ones(n)
        X[i] = (X[i]-x_m*ones)/(σ*ones)
    return X

def GenerateData(n, noise=False, noise_strength=0.1):
    xy = np.random.rand(2*n)
    x = xy[:n]
    y = xy[n:2*n]
    F = FrankeFunction(x, y)
    if noise=True:
        return F + noise_strength*np.random.randn(len(y), len(x))
    else:
        return F
