__author__ = 'leferrad'

# Dependencias externas
import numpy as np


def tanh(z):
    r"""
    Tangente Hiperbolica

    :math:`f(z)=\dfrac{e^z - e^{-z}}{e^z + e^{-z}}`

    """
    return np.tanh(z)


def tanh_d(z):
    r"""
    Derivada de Tangente Hiperbolica

    :math:`f(z)=1-tanh^2(z)`

    """
    return 1.0 - np.tanh(z) ** 2


def sigmoid(z):
    r"""
    Sigmoidea

    :math:`f(z)=\dfrac{1}{1 + e^{-z}}`

    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_d(z):
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


def relu_d(z):
    r"""
    Derivada de ReLU

    :math:`f(z) = \begin{cases}1 & z > 0 \\ 0 & z \leq 0\end{cases}`

    """
    if z > 0.0:
        ret = 1.0
    else:
        ret = 0.0
    return ret


def lrelu(z):
    r"""
    Leaky ReLU

    :math:`f(z) = \begin{cases}z & z > 0 \\ 0.01z & z \leq 0\end{cases}`

    """
    if z > 0.0:
        ret = z
    else:
        ret = 0.01 * z
    return ret

def lrelu_d(z):
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


def softplus_d(z):
    r"""
    Derivada de Softplus

    :math:`f(z)=sigmoid(z)=\dfrac{1}{1 + e^{-z}}`

    """
    return sigmoid(z)


def identity(z):
    r"""
    Identidad

    :math:`f(z)=z`

    """
    return z


def identity_d(z):
    r"""
    Derivada de Identidad

    :math:`f(z)=1`

    """
    return 1


def lecunn_sigmoid(z):
    r"""
    Sigmoidea recomendada por LeCunn

    http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf

    :math:`f(z)=1.7159 tanh(\dfrac{2z}{3})`
    """
    return 1.7159 * np.tanh(z * 2.0/3.0)


def lecunn_sigmoid_d(z):
    r"""
    Derivada de Sigmoidea recomendada por LeCunn

    :math:`f(z)=1.14393 (1 - tanh^2(\dfrac{2z}{3}))`

    """
    return 1.7159 * (2.0 / 3.0) * (1.0 - np.tanh(z * 2.0/3.0) ** 2)

# TODO definirlas mejor
# def sin(z):
#     return np.sin(z)
#
#
# def sin_d(z):
#     return np.cos(z)


fun_activation = {'Tanh': tanh, 'Sigmoid': sigmoid, 'ReLU': relu, 'Softplus': softplus,
                                'Identity': identity, 'LeakyReLU': lrelu, 'LeCunnSigm': lecunn_sigmoid}
fun_activation_d = {'Tanh': tanh_d, 'Sigmoid': sigmoid_d, 'ReLU': relu_d, 'Softplus': softplus_d,
                                    'Identity': identity_d, 'LeakyReLU': lrelu_d, 'LeCunnSigm': lecunn_sigmoid_d}

