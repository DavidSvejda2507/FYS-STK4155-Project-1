import os
import numpy as np
import matplotlib.pyplot as plt
import GlobVar as gv


if not os.path.exists('figures'):
    os.mkdir('figures')

def MSE(y, y_predict):
    n = len(y) + 1
    error = (y - y_predict).T@(y-y_predict)
    return (1/n)*error

def R2(y, y_predict):
    n = len(y) + 1
    S_res = (y - y_predict).T@(y-y_predict)
    y_mean = np.mean(y)
    S_tot = (y-y_mean).T@(y-y_mean)
    return 1 - S_res/S_tot

def plotfunc(filename, data, xtickset=True, xtick=gv.poly_range, yset=True, ylim=[0, None], xlabel = 'Poly degree', ylabel = 'MSE',
        yscale = 'log', legend = True, figs = (10, 6), ret = False):
    fig = plt.figure(figsize=figs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xtickset:
        plt.xticks(xtick)
    max_value = 0
    for key, value in data.items():
        plt.plot(value[0], value[1], label=key)
        max_value = max(max_value, np.max(value[1]))
    #if yset:
        #ylim[1] = max_value*1.1
        #plt.ylim(bottom=ylim[0])
    plt.yscale(yscale)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/{filename}')
    if ret:
        return fig
    plt.close(fig)
