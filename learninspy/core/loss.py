#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
En este módulo se proveen dos funciones de costo populares, cuyo uso
se corresponde a la tarea designada para el modelo:

    * **Clasificación**: Entropía Cruzada (en inglés, *Cross Entropy* o *CE*),
    * **Regresión**: Error Cuadrático Medio (en inglés, *Mean Squared Error* o *MSE*).

Para agregar otra a este módulo se debe definir la funcion de error ‘fun(o, t)’
y su derivada ‘_fun_d(o, t)’ respecto a la entrada x,
siendo ‘o(x)’ la activación de dicha entrada (llamada salida real) y ‘t’ la salida esperada.

.. note:: Las derivadas deben tener un underscore '_' de prefijo en su nombre, de forma que no sean parte de la API.
"""

__author__ = 'leferrad'

import numpy as np


def mse(o, t):
    """
    Función de Error Cuadrático Medio.
    Ver más info en Wikipedia: `Mean squared error <https://en.wikipedia.org/wiki/Mean_squared_error>`_

    Las entradas *o* y *t* son arreglos que corresponden respectivamente a la salida real
    y la esperada de una predicción realizada sobre un ejemplo. La función devuelve el
    error cuadrático medio asociado a dicha predicción. Notar que la constante 1/2 es incluida para
    que se cancele con el exponente en la función derivada.

    :math:`J=\\dfrac{1}{2}\displaystyle\sum\limits_{j} (t_j - y_j)^2`

    :param o: numpy.ndarray
    :param t: numpy.ndarray
    :return: float
    """
    err = np.array(map(lambda(output, target): target - output, zip(o, t)))
    return 0.5 * np.sum(np.square(err))


def _mse_d(o, t):
    """
    Derivada de la función MSE.

    :param o: numpy.ndarray
    :param t: numpy.ndarray
    :return: numpy.ndarray
    """
    err = np.array(map(lambda(output, target): output - target, zip(o, t)))
    return err


def cross_entropy(o, t):
    """
    Función de Entropía Cruzada, usada para medir el error de clasificación en una regresión Softmax.

    La entrada *o* es un arreglo de K x 1 que representa la salida real de una clasificación realizada por la
    función Softmax sobre un ejemplo dado, y *t* es la salida esperada en dicha clasificación. Siendo
    K la cantidad de clases posibles a predecir, el arreglo *t* corresponde a un vector binario de dimensión K
    (obtenido por :func:`~learninspy.utils.data.label_to_vector`), por lo cual se aplica en forma directa
    la función de CE que resulta en el costo asociado a la predicción.

    :math:`J=-\displaystyle\sum\limits_{k}^K \left(t_k  \log(o_k) \\right)`

    :param o: numpy.ndarray
    :param t: numpy.ndarray
    :return: float

    .. note:: el arreglo *o* debe ser generado por la función :func:`~learninspy.core.neurons.LocalNeurons.softmax`.
    """
    # Extraido de http://cs231n.github.io/neural-networks-case-study/ :
    # return np.sum(np.log(o[range(len(o)), t]))/float(len(o))
    loss = -sum(t * np.log(o))
    return loss


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
