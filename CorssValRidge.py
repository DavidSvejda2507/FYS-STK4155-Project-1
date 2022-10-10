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


if print_:
    print('--------- Run Info ----------')
    print(f'Noise = {gv.ns}')
    print(f'Poly max = {gv.poly_max}')
    print(f'Folds = {gv.k_list}')

for N in gv.N_list:
    x, y, F = Dataset.GenerateData(N)
    for n_folds in gv.k_list:
        for L in gv.lambda_list:

            MSE = CrossVal(x, y, F, N, n_folds, 'Ridge', L=L)

            b_MSE, b_Bias, b_Variance = np.load(f'ArrayData/RidgeMSEBV{N}L{L}.npy')

            if print_:
                print('---------- Run Info ---------------')
                print(f' Datapoints = {N}')
                print(f'Folds = {n_folds}')
                print(f'Lambda = {L}')

            for p in poly_list:

                if plot:
                    Analysis.plotfunc(f'RidgeCrosValN{N}P{p}k{n_folds}L{L}.pdf',
                                    {
                                    'Cross Validation' : (poly_range[:p], MSE[:p]),
                                    'Bootstrap' : (poly_range[:p], b_MSE[p:])
                                    })
                if print_:
                    if print_:
                        print(f'------------ Poly degree {p} -------------')
                        print(f'CV MSE  = {MSE[p-1]}')
                        print(f'B MSE = {b_MSE[p-1]}')

            if save:
                np.save(f'ArrayData/RidgeCrosValN{N}k{n_folds}L{L}', np.stack((MSE, b_MSE)))
