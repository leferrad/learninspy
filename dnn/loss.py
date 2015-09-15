__author__ = 'leferrad'

# Se define la funcion de error 'fun(x)' y su derivada respecto a x 'fun_d(x)'

import numpy as np

def mse(y, t):
    # Supongo o, t: np.array
    err = y - t
    N = err.size
    return np.sum(np.square(err)) / (1.0 * N)


def mse_d(y, t):
    # Supongo o, t: np.array
    if t.shape != y.shape:
        y = y.T
    err = y - t
    N = err.size
    return 2 * err / (1.0 * N)


def cross_entropy(y, t):
    return -sum(t * np.log(y))

def cross_entropy_d(y, t):
    return y - t


fun_loss = {'MSE': mse, 'CrossEntropy': cross_entropy}
fun_loss_d = {'MSE': mse_d, 'CrossEntropy': cross_entropy_d}