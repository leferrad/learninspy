#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
En este módulo se proveen dos funciones de costo populares, cuyo uso
se corresponde a la tarea designada para el modelo:

    * **Clasificación**: Entropía Cruzada (en inglés, *Cross Entropy* o *CE*),
    * **Regresión**: Error Cuadrático Medio (en inglés, *Mean Squared Error* o *MSE*).

Para agregar más a este módulo, se debe definir la funcion de error ‘fun(o, t)’,
y su derivada ‘_fun_d(o, t)’ respecto a la entrada x,
siendo ‘o(x)’ la activación de dicha entrada (llamada salida real) y ‘t’ la salida esperada.

.. note:: Las derivadas deben tener un underscore '_' de prefijo en su nombre, de forma que no sean parte de la API.
"""

__author__ = 'leferrad'

import numpy as np


def mse(o, t):
    """
    Función de error cuadrático medio.
    Ver más info en Wikipedia: `Mean squared error <https://en.wikipedia.org/wiki/Mean_squared_error>`_

    Las entradas *o* y *t* son arreglos de N x 1, y corresponden respectivamente a la salida real
    y la esperada de predicciones realizadas sobre un batch de N ejemplos. La función devuelve el
    error cuadrático medio asociado a dichas predicciones.

    :math:`J=\\dfrac{1}{N}\displaystyle\sum\limits_{i}^N (o^{(i)}-t^{(i)})^2`

    :param o: np.array
    :param t: np.array
    :return: float
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    n = err.size
    return np.sum(np.square(err)) / float(n)


def _mse_d(o, t):
    """
    Derivada de la función MSE.

    :param o: np.array
    :param t: np.array
    :return: np.array
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    n = err.size
    return 2 * err / float(n)


def cross_entropy(o, t):
    """
    Función de entropía cruzada, usada para medir el error de clasificación sobre una regresión Softmax.

    La entrada *o* es un arreglo de N x K que representa la salida real de una clasificación realizada por la
    función softmax sobre un batch de N ejemplos, y *t* es la salida esperada en dicha clasificación.
    Dicho parámetro *t* corresponde a un vector binario de dimensión K (obtenido por
    :func:`~learninspy.utils.data.label_to_vector`), por lo cual se aplica en forma directa
    la función de CE que resulta en el costo asociado a las predicciones hechas sobre el batch.

    :math:`J=-\displaystyle\sum\limits_{i}^N \displaystyle\sum\limits_{k}^K \left(t^{(i)}_k \cdot \log(o^{(i)}) \\right)`

    :param o: np.array
    :param t: np.array
    :return: float

    .. note:: el arreglo *o* debe ser generado por la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`.
    """
    return -sum(t * np.log(o))[0]  # sum(np.array) devuelve un np.array de un elemento, por lo cual accedo a él con [0]


def _cross_entropy_d(o, t):
    """
    Derivada de la función CE.

    Por regla de la cadena, se puede demostrar que la derivada del error respecto a la entrada (generada por
    la función softmax) es *o - t*.

    :param o: np.array
    :param t: float
    :return: np.array

    .. note:: el arreglo *o* debe ser generado por la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`.
    """
    return o - t


fun_loss = {'MSE': mse, 'CrossEntropy': cross_entropy}
fun_loss_d = {'MSE': _mse_d, 'CrossEntropy': _cross_entropy_d}
