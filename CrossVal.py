import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv
from CrossValFUnc import CrossVal

plot = True
save = True
print_ = True
plotvar = True

if print_:
    print('--------- Run Info ----------')
    print(f'Noise = {gv.ns}')
    print(f'Poly max = {gv.poly_max}')

for N in gv.N_list:
    x, y, F = Dataset.GenerateData(N)
    for n_folds in gv.k_list:
        MSE = CrossVal(x, y, F, N, n_folds, 'OLS')

        if print_:
            print('---------- Run Info ---------------')
            print(f' Datapoints = {N}')
            print(f'Folds = {n_folds}')

        b_MSE, b_Bias, b_Variance, b_MSEVarB, b_VarSizeNum, b_VarSizeAn = np.load(f'ArrayData/MSEBV{N}.npy')

        for p in gv.poly_list:
            if plot:
                Analysis.plotfunc(f'CrosValN{N}P{p}k{n_folds}.pdf',
                                {
                                'Cross Validation' : (gv.poly_range[:p], MSE[:p]),
                                'Bootstrap' : (gv.poly_range[:p], b_MSE[:p])
                                })
            if print_:
                print(f'------------ Poly degree {p} -------------')
                print(f'CV MSE  = {MSE[p-1]}')
                print(f'B MSE = {b_MSE[p-1]}')

        if save:
            np.save(f'ArrayData/CrosValN{N}k{n_folds}', np.stack((MSE, b_MSE)))
