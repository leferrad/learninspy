#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
En este módulo se pueden configurar las funciones de activación que se deseen.
Para ello, simplemente se codifica tanto la función como su derivada analítica
(o aproximación, como en el caso de la ReLU), y luego se insertan en los diccionarios
de funciones correspondientes, que se encuentran al final del script,
con una key común que identifique la activación.

.. note:: Las derivadas deben tener un underscore '_' de prefijo en su nombre, de forma que no sean parte de la API.
"""

__author__ = 'leferrad'

import numpy as np


# NOTA: las derivadas deben tener un underscore '_' de prefijo en su nombre, de forma que no sean parte de la API.

def tanh(z):
    r"""
    Tangente Hiperbólica

    :math:`f(z)=\dfrac{e^z - e^{-z}}{e^z + e^{-z}}`

    """
    return np.tanh(z)


def _tanh_d(z):
    r"""
    Derivada de Tangente Hiperbólica

    :math:`f(z)=1-tanh^2(z)`

    """
    return 1.0 - np.tanh(z) ** 2


def sigmoid(z):
    r"""
    Sigmoidea

    :math:`f(z)=\dfrac{1}{1 + e^{-z}}`

    """
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_d(z):
    r"""
    Derivada de Sigmoidea

    :math:`f(z)=\dfrac{e^z - e^{-z}}{e^z + e^{-z}}`

    """
    return sigmoid(z) * (1.0 - sigmoid(z))


def relu(z):
    r"""
    Rectifier Linear Unit (ReLU)

    :math:`f(z)=max(0,z)`

    """
    if isinstance(z, np.ndarray) or type(z) == list:
        z = z[0]
    return max(0.0, z)


def _relu_d(z):
    r"""
    Derivada de ReLU

    :math:`f(z) = \begin{cases}1 & z > 0 \\ 0 & z \leq 0\end{cases}`

    """
    if z > 0.0:
        ret = 1.0
    else:
        ret = 0.0
    return ret


def leaky_relu(z):
    r"""
    Leaky ReLU

    :math:`f(z) = \begin{cases}z & z > 0 \\ 0.01z & z \leq 0\end{cases}`

    """
    if z > 0.0:
        ret = z
    else:
        ret = 0.01 * z
    return ret

def _lrelu_d(z):
    r"""
    Derivada de Leaky ReLU

    :math:`f(z) = \begin{cases}1 & z > 0 \\ 0.01 & z \leq 0\end{cases}`

    """
    if z > 0.0:
        ret = 1
    else:
        ret = 0.01
    return ret


def softplus(z):
    r"""
    Softplus

    :math:`f(x)=\log{(1+e^z)}`

    """
    return np.log(1.0 + np.exp(z))


def _softplus_d(z):
    r"""
    Derivada de Softplus

    :math:`f(z)=sigmoid(z)=\dfrac{1}{1 + e^{-z}}`

    """
    return sigmoid(z)


def identity(z):
    r"""
    Lineal o identidad

    :math:`f(z)=z`

    """
    return z


def _identity_d(z):
    r"""
    Derivada de Identidad

    :math:`f(z)=1`

    """
    return 1


def lecunn_sigmoid(z):
    r"""
    Sigmoidea propuesta por LeCunn en [lecun2012efficient]_.

    :math:`f(z)=1.7159 tanh(\dfrac{2z}{3})`

    En dicha definicion, se escala una función Tanh de forma que se obtenga
    máxima derivada segunda en valor absoluto para z=1 y z=-1, lo cual mejora
    la convergencia del entrenamiento, y una efectiva ganancia de dicha
    transformación cerca de 1.


    .. [lecun2012efficient] LeCun, Y. A. et. al (2012).
     Efficient backprop. In Neural networks: Tricks of the trade (pp. 9-48).
     Springer Berlin Heidelberg.
     http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf


    """
    return 1.7159 * np.tanh(z * 2.0/3.0)


def _lecunn_sigmoid_d(z):
    r"""
    Derivada de Sigmoidea propuesta por LeCunn

    :math:`f(z)=1.14393 (1 - tanh^2(\dfrac{2z}{3}))`

    """
    return 1.7159 * (2.0 / 3.0) * (1.0 - np.tanh(z * 2.0/3.0) ** 2)


fun_activation = {'Tanh': tanh, 'Sigmoid': sigmoid, 'ReLU': relu, 'Softplus': softplus,
                                'Identity': identity, 'LeakyReLU': leaky_relu, 'LeCunnSigm': lecunn_sigmoid}
fun_activation_d = {'Tanh': _tanh_d, 'Sigmoid': _sigmoid_d, 'ReLU': _relu_d, 'Softplus': _softplus_d,
                                    'Identity': _identity_d, 'LeakyReLU': _lrelu_d, 'LeCunnSigm': _lecunn_sigmoid_d}

