import Solve
import Dataset
import Analysis
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import resample
import GlobVar as gv
from CrossValFunc import CrossVal

plot = True
save = True
print_ = True

if print_:
    print('--------- Run Info ----------')
    print(f'Noise = {gv.ns}')
    print(f'Poly max = {gv.poly_max}')
    print(f'Folds = {gv.k_list}')

for N in [20, 100]:
    x, y, F = Dataset.GenerateData(N)
    for n_folds in gv.k_list:
        for L in [0.001]:

            MSE = CrossVal(x, y, F, N, n_folds, 'Lasso', L=L)

            b_MSE, b_Bias, b_Variance, aa, bb, cc = np.load(f'ArrayData/LassoMSEBV{N}L{L}.npy')

            if print_:
                print('---------- Run Info ---------------')
                print(f' Datapoints = {N}')
                print(f'Folds = {n_folds}')
                print(f'Lambda = {L}')

            for p in [20]:

                if plot:
                    Analysis.plotfunc(f'Updated/CrossValLasso/LassoCrosValN{N}P{p}k{n_folds}L{L}.pdf',
                                    {
                                    'Cross Validation' : (gv.poly_range, MSE),
                                    'Bootstrap' : (gv.poly_range, b_MSE)
                                    })
                if print_:
                    if print_:
                        print(f'------------ Poly degree {p} -------------')
                        print(f'CV MSE  = {MSE[p-1]}')
                        print(f'B MSE = {b_MSE[p-1]}')

            if save:
                np.save(f'ArrayData/LassoCrosValN{N}k{n_folds}L{L}', np.stack((MSE, b_MSE)))
