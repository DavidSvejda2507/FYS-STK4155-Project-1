from BootstrapFunc import Bootstrap, printpoly
from CrossValFunc import CrossVal
import numpy as np
import GlobVar as gv
import Dataset
import Analysis


L_array = np.logspace(-4, 4, 100)
polynomials = [5, 7, 9, 11, 13, 15, 17, 19]



for N in gv.N_list:
    x, y, F = Dataset.GenerateData(N)
    for p in gv.poly_list:
        RidgeCrossMSE = np.zeros_like(L_array)
        RidgeBootstrapData = np.zeros((6, len(L_array)))
        LassoCrossMSE = np.zeros_like(L_array)
        LassoBootstrapData = np.zeros((6, len(L_array)))
        for i, L in enumerate(L_array):
            RidgeCrossMSE[i] = CrossVal(x, y, F, N, 10, 'Ridge', poly_max=p, poly_min=p, print_=False, L=L)[0]
            RidgeBootstrapData[:, i] = Bootstrap(N, 'Ridge', gv.ns, gv.bootstraps, p, poly_min=p, print_=False, L=L)[:][0]
            LassoCrossMSE[i] = CrossVal(x, y, F, N, 10, 'Lasso', poly_max=p, poly_min=p, print_=False, L=L)[0]
            LassoBootstrapData[:, i] = Bootstrap(N, 'Lasso', gv.ns, gv.bootstraps, p, poly_min=p, print_=False, L=L)[:][0]

        Analysis.plotfunc(f'Lambda/LassoRidgeCrossValLAxisN{N}P{p}.pdf',
                        {
                        'Lasso' : (L_array, LassoCrossMSE),
                        'Ridge' : (L_array, RidgeCrossMSE)
                        }, yscale='linear', xlabel = 'λ')
        Analysis.plotfunc(f'Lambda/LRBootstrapMSEBiasVarLAxisN{N}P{p}.pdf',
                        {
                        'LassoMSE' : (L_array, LassoBootstrapData[0]),
                        'LassoBias' : (L_array, LassoBootstrapData[1]),
                        'LassoVarience' : (L_array, LassoBootstrapData[2]),
                        'RidgeMSE' : (L_array, RidgeBootstrapData[0]),
                        'RidgeBias' : (L_array, RidgeBootstrapData[1]),
                        'RidgeVariance' : (L_array, RidgeBootstrapData[2])
                        }, yscale='linear', xlabel = 'λ')
        Analysis.plotfunc(f'Lambda/RidgeMSEBiasVarLAxisN{N}P{p}.pdf',
                        {
                        'RidgeMSE' : (L_array, RidgeBootstrapData[0]),
                        'RidgeBias' : (L_array, RidgeBootstrapData[1]),
                        'RidgeVariance' : (L_array, RidgeBootstrapData[2])
                        }, yscale='linear', xlabel = 'λ')
        Analysis.plotfunc(f'Lambda/LassoMSEBiasVarLAxisN{N}P{p}.pdf',
                        {
                        'LassoMSE' : (L_array, LassoBootstrapData[0]),
                        'LassoBias' : (L_array, LassoBootstrapData[1]),
                        'LassoVarience' : (L_array, LassoBootstrapData[2]),
                        }, yscale='linear', xlabel = 'λ')
