#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'leferrad'

# Dependencias externas
import numpy as np

# Dependencias internas
from learninspy.utils.data import label_to_vector

"""
Se define la funcion de error 'fun(o, t)', y su derivada 'fun_d(o, t)' respecto a la entrada x,
siendo 'o(x)' la activación de dicha entrada (llamada salida real) y 't' la salida esperada
"""


def mse(o, t):
    """
    Función de error cuadrático medio.
    Ver más info en `Mean squared error <https://en.wikipedia.org/wiki/Mean_squared_error>`_

    :param o: list, correspondiente a la salida real.
    :param t: list, correspondiente a la salida esperada.
    :return: float, error de clasificación.
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    n = err.size
    return np.sum(np.square(err)) / (1.0 * n)


def mse_d(o, t):
    """
    Derivada de la función MSE.

    :param o: list, correspondiente a la salida real.
    :param t: list, correspondiente a la salida esperada.
    :return: list, derivada de la función de error.
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    n = err.size
    return 2 * err / (1.0 * n)


def cross_entropy(o, t):
    """
    Función de entropía cruzada.

    :param o: list, correspondiente a la salida real.
    :param t: list, correspondiente a la salida esperada.
    :return: float, error de clasificación.

    .. note:: el vector *o* debe ser la salida de la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`
    """
    num_classes = len(o)
    t = label_to_vector(t, num_classes)
    return -sum(t * np.log(o))[0]


def cross_entropy_d(o, t):
    """
    Derivada de la función CE.

    :param o: list, correspondiente a la salida real.
    :param t: list, correspondiente a la salida esperada.
    :return: list, derivada de la función de error.

    .. note:: el vector *o* debe ser la salida de la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`
    """
    # Por regla de la cadena, se puede demostrar que la derivada del error respecto a la entrada es o - t
    num_classes = len(o)
    t = label_to_vector(t, num_classes)
    return o - t


fun_loss = {'MSE': mse, 'CrossEntropy': cross_entropy}
fun_loss_d = {'MSE': mse_d, 'CrossEntropy': cross_entropy_d}
