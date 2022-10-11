import hashlib
import numpy as np
from imageio import imread

loss = '|I|Ii|II|I_|'
seed = int(hashlib.sha1(loss.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
np.random.seed(seed)

def SimpleFunc(x):
    return np.exp(2*x)

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#σ
def Scaling(X):
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    if isinstance(s, np.ndarray):
        for i, ss in enumerate(s):
            if ss<1:
                s[i] = 1
    else:
        if s < 1: s = 1
    X = (X-m)/s
    return X, m, s

def Unscale(B, m_x, s_x, m_y, s_y):
    B_ = np.zeros(len(B)+1)
    B_[1:] = B*s_y/s_x
    B_[0]= m_y - np.sum(m_x*B*s_y/s_x)
    return B_

def UnscaleOutput(Y, m, σ):
    return Y*σ + m


def GenerateData(n, noise=False, noise_strength=0.1):
    x = np.random.rand(n)
    y = np.random.rand(n)
    F = FrankeFunction(x, y)
    if noise:
        return x, y, F + noise_strength*np.random.randn(len(F))
    else:
        return x, y, F

def GetRealData(n):
    assert n in [1, 2]
    t = imread(f'Data/SRTM_data_Norway_{n}.tif')[:200:10, :200:10]
    x = np.linspace(0, 1, t.shape[0])
    y = np.linspace(0, 1, t.shape[1])
    x, y = np.meshgrid(x, y)
    return x.ravel(), y.ravel(), t.ravel()
# def Bootstrap(X):
