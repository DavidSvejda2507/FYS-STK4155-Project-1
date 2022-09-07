import numpy as np

# To-Do
# MSE, R2
# Bias Variance trade off
# Bootstrap analysis

if not os.path.exists(figures):
    os.mkdir(figures)

def MSE(y, y_predict):
    n = len(y) + 1
    error = (y - y_predict)@(y-y_predict)
    return (1/n)*error

def R2(y, y_predict):
    n = len(y) + 1
    S_res = (y - y_predict)@(y-y_predict)
    y_mean = np.sum(y)/n
    S_tot = (y-y_mean)@(y-y_mean)
    return 1 - S_res/S_tot

def PolyPlot(MSE, R2, figure_id):
    fig = plt.figure()
    fig.plot()
    plt.legend()
    plt.savefig(f'figures/{figure_id}.pdf')
