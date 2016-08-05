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

    :param o: numpy.ndarray
    :param t: numpy.ndarray
    :return: float
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    n = err.size
    return np.sum(np.square(err)) / float(n)


def _mse_d(o, t):
    """
    Derivada de la función MSE.

    :param o: numpy.ndarray
    :param t: numpy.ndarray
    :return: numpy.ndarray
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

    :math:`J=-\displaystyle\sum\limits_{i}^N \displaystyle\sum\limits_{k}^K \left(t^{(i)}_k  \log(o^{(i)}_k) \\right)`

    :param o: numpy.ndarray
    :param t: numpy.ndarray
    :return: float

    .. note:: el arreglo *o* debe ser generado por la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`.
    """
    # return np.sum(np.log(o[range(len(o)), t]))/float(len(o))  # extraido de http://cs231n.github.io/neural-networks-case-study/
    loss = -sum(t * np.log(o))
    if type(loss) is list:
        loss = loss[0]
    return loss  # sum(np.ndarray) devuelve un np.ndarray de un elemento, por lo cual accedo a él con [0]


def _cross_entropy_d(o, t):
    """
    Derivada de la función CE.

    Por regla de la cadena, se puede demostrar que la derivada del error respecto a la entrada (generada por
    la función softmax) es *o - t*.

    :param o: numpy.ndarray
    :param t: numpy.ndarray
    :return: numpy.ndarray

    .. note:: el arreglo *o* debe ser generado por la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`.
    """
    return o - t


fun_loss = {'MSE': mse, 'CrossEntropy': cross_entropy}
fun_loss_d = {'MSE': _mse_d, 'CrossEntropy': _cross_entropy_d}
