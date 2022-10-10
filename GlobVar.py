import numpy as np

N_list = [20, 100, 500]
poly_list = [5, 10, 20]
poly_max = max(poly_list)
poly_range = np.arange(1, poly_max+1)
lambda_list = np.logspace(-3,3,7)
k_list = [5, 10]
bootstraps = 10
ns = 0.1
