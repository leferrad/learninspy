#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np

# Dependencias internas
from learninspy.utils.data import label_to_vector

""" Se define la funcion de error 'fun(y, t)', y su derivada 'fun_d(y, t)' respecto a la entrada x,
    siendo 'y(x)' la activación de dicha entrada (llamada salida real) y 't' la salida esperada """


def mse(value, target):
    err = np.array(map(lambda(y, t): y - t, zip(value, target)))
    n = err.size
    return np.sum(np.square(err)) / (1.0 * n)


def mse_d(value, target):
    err = np.array(map(lambda (y, t): y - t, zip(value, target)))
    n = err.size
    return 2 * err / (1.0 * n)


def cross_entropy(y, t):
    num_classes = len(y)
    t = label_to_vector(t, num_classes)
    return -sum(t * np.log(y))[0]


def cross_entropy_d(y, t):
    # Por regla de la cadena, se puede demostrar que la derivada del error respecto a la entrada es y - t
    num_classes = len(y)
    t = label_to_vector(t, num_classes)
    return y - t


fun_loss = {'MSE': mse, 'CrossEntropy': cross_entropy}
fun_loss_d = {'MSE': mse_d, 'CrossEntropy': cross_entropy_d}
