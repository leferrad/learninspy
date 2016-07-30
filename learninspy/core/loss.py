#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
En este módulo se proveen dos funciones de costo populares, cuyo uso
se corresponde a la tarea designada para el modelo:

    * **Clasificación**: Entropía Cruzada (en inglés, *Cross Entropy* o *CE*),
    * **Regresión**: Error Cuadrático Medio (en inglés, *Mean Squared Error* o *MSE*).

Se define la funcion de error ‘fun(o, t)’, y su derivada ‘_fun_d(o, t)’ respecto a la entrada x,
siendo ‘o(x)’ la activación de dicha entrada (llamada salida real) y ‘t’ la salida esperada.

.. note:: Las derivadas deben tener un underscore '_' de prefijo en su nombre, de forma que no sean parte de la API.
"""

__author__ = 'leferrad'

from learninspy.utils.data import label_to_vector
import numpy as np


def mse(o, t):
    """
    Función de error cuadrático medio.
    Ver más info en `Mean squared error <https://en.wikipedia.org/wiki/Mean_squared_error>`_

    :math:`J=\\frac{1}{2}\sum_i (o^{(i)}-t^{(i)})^2`

    :param o: list, correspondiente a la salida real.
    :param t: list, correspondiente a la salida esperada.
    :return: float, costo asociado a la predicción.
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    n = err.size
    return np.sum(np.square(err)) / float(n)


def _mse_d(o, t):
    """
    Derivada de la función MSE.

    :param o: list, correspondiente a la salida real.
    :param t: list, correspondiente a la salida esperada.
    :return: list, derivada de la función de error.
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    n = err.size
    return 2 * err / float(n)


def cross_entropy(o, t):
    """
    Función de entropía cruzada.

    :math:`J=-\sum_i \sum_k \left(t^{(i)}_k \cdot \log(o^{(i)}) \\right)`

    :param o: list, correspondiente a la salida real.
    :param t: float, correspondiente a la salida esperada.
    :return: float, costo asociado a la predicción.

    .. note:: el vector *o* debe ser la salida de la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`
    """
    num_classes = len(o)
    t = label_to_vector(t, num_classes)
    return -sum(t * np.log(o))[0]


def _cross_entropy_d(o, t):
    """
    Derivada de la función CE.

    :param o: list, correspondiente a la salida real.
    :param t: float, correspondiente a la salida esperada.
    :return: list, derivada de la función de error.

    .. note:: el vector *o* debe ser la salida de la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`
    """
    # Por regla de la cadena, se puede demostrar que la derivada del error respecto a la entrada es o - t
    num_classes = len(o)
    t = label_to_vector(t, num_classes)
    return o - t


fun_loss = {'MSE': mse, 'CrossEntropy': cross_entropy}
fun_loss_d = {'MSE': _mse_d, 'CrossEntropy': _cross_entropy_d}
