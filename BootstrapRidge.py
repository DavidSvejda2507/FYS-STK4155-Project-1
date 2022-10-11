import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv
from BootstrapFunc import Bootstrap, printpoly

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
    for L in [1e-8]:
        MSE, Bias, Variance = Bootstrap(N, 'Ridge', gv.ns, gv.bootstraps, gv.poly_max, L=L)

        for p in gv.poly_list:
            prange = np.arange(1, p+1)
            if plot:
                Analysis.plotfunc(f'Updated/BiaVarTradeRidge/RidgeBiaVarTradeBootN{N}P{p}ns{gv.ns}L{L}.pdf',
                                {
                                'MSE' : (prange, MSE[:p]),
                                'Bias' : (prange, Bias[:p]),
                                'Variance' : (prange, Variance[:p]),
                                })
            if print_:
                printpoly(p, MSE, Bias, Variance, MSEVarB, VarSizeNum, VarSizeAn)

        if save:
            np.save(f'ArrayData/RidgeMSEBV{N}L{L}', np.stack((MSE, Bias, Variance, MSEVarB, VarSizeNum, VarSizeAn)))
        if plotvar:
            Analysis.plotfunc(f'Updated/RidgeMSE/RidgeMSESizeVarBN{N}L{L}.pdf',
                    {
                    'MSEVarB' : (np.arange(1, gv.poly_max+1), MSEVarB),
                    'VarSizeNum' : (np.arange(1, gv.poly_max+1), VarSizeNum),
                    'VarSizeAn' : (np.arange(1, gv.poly_max+1), VarSizeAn)
                    })
