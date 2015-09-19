__author__ = 'leferrad'

# Se define la funcion de error 'fun(x)' y su derivada respecto a x 'fun_d(x)'

import numpy as np
from utils.util import label_to_vector

def mse(y, t):
    num_classes = len(y)
    #y = label_to_vector(np.argmax(y), num_classes)
    t = label_to_vector(t, num_classes)
    err = y - t
    N = err.size
    return np.sum(np.square(err)) / (1.0 * N)


def mse_d(y, t):
    num_classes = len(y)
    #y = label_to_vector(np.argmax(y), num_classes)
    t = label_to_vector(t, num_classes)
    err = y - t
    N = err.size
    return 2 * err / (1.0 * N)


def cross_entropy(y, t):
    num_classes = len(y)
    t = label_to_vector(t, num_classes)
    return -sum(t * np.log(y))[0]


def cross_entropy_d(y, t):
    num_classes = len(y)
    t = label_to_vector(t, num_classes)
    return y - t


fun_loss = {'MSE': mse, 'CrossEntropy': cross_entropy}
fun_loss_d = {'MSE': mse_d, 'CrossEntropy': cross_entropy_d}