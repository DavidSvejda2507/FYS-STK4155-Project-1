import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv
from BootstrapFunc import RealBootstrap, printpoly

plot = True
save = True
print_ = True
plotvar = True

if print_:
    print('--------- Run Info ----------')
    print(f'Noise = {gv.ns}')
    print(f'Poly max = {gv.poly_max}')
    print(f'Bootstraps = {gv.bootstraps}')

for N in gv.N_list:

    x, y, F = Dataset.GenerateData(N, True, 0.1)
    MSE, Bias, Variance = RealBootstrap(x, y, F, 'OLS', gv.bootstraps, gv.poly_max)

    if print_:
        print('---------- Run Info ---------------')
        print(f' Datapoints = {N}')

    for p in gv.poly_list:
        prange = np.arange(1, p+1)
        if plot:
            Analysis.plotfunc(f'Updated/BiaVarTradeOLS/BiaVarTradeBootN{N}P{p}ns{gv.ns}.pdf',
                            {
                            'MSE' : (prange, MSE[:p]),
                            'Bias' : (prange, Bias[:p]),
                            'Variance' : (prange, Variance[:p]),
                            })
        if print_:
            printpoly(p, MSE, Bias, Variance)

    if save:
        np.save(f'ArrayData/MSEBV{N}', np.stack((MSE, Bias, Variance)))
