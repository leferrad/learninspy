__author__ = 'leferrad'

# Dependencias externas
import numpy as np


# TODO cambiar nombre de ReLU a rectifier ? ya que me refiero a la funcion, no a la unidad

def tanh(x):
    r"""
    Tangente Hiperbolica

    :math:`f(x)=\dfrac{e^x - e^{-x}}{e^x + e^{-x}}`

    """
    return np.tanh(x)


def tanh_d(x):
    r"""
    Derivada de Tangente Hiperbolica

    :math:`f(x)=1-tanh^2(x)`

    """
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x):
    r"""
    Sigmoidea

    :math:`f(x)=\dfrac{1}{1 + e^{-x}}`

    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_d(x):
    r"""
    Derivada de Sigmoidea

    :math:`f(x)=\dfrac{e^x - e^{-x}}{e^x + e^{-x}}`

    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    r"""
    Rectifier Linear Unit (ReLU)

    :math:`f(x)=max(0,x)`

    """
    if isinstance(x, np.ndarray) or type(x) == list:
        x = x[0]
    return max(0.0, x)


def relu_d(x):
    r"""
    Derivada de ReLU

    :math:`f(x) = \begin{cases}1 & x > 0 \\ 0 & x \leq 0\end{cases}`

    """
    if x > 0.0:
        ret = 1.0
    else:
        ret = 0.0
    return ret


def lrelu(x):
    r"""
    Leaky ReLU

    :math:`f(x) = \begin{cases}x & x > 0 \\ 0.01x & x \leq 0\end{cases}`

    """
    if x > 0.0:
        ret = x
    else:
        ret = 0.01 * x
    return ret

def lrelu_d(x):
    r"""
    Derivada de Leaky ReLU

    :math:`f(x) = \begin{cases}1 & x > 0 \\ 0.01 & x \leq 0\end{cases}`

    """
    if x > 0.0:
        ret = 1
    else:
        ret = 0.01
    return ret


def softplus(x):
    r"""
    Softplus

    :math:`f(x)=\log{(1+e^x)}`

    """
    return np.log(1.0 + np.exp(x))


def softplus_d(x):
    r"""
    Derivada de Softplus

    :math:`f(x)=sigmoid(x)=\dfrac{1}{1 + e^{-x}}`

    """
    return sigmoid(x)


def identity(x):
    r"""
    Identidad

    :math:`f(x)=x`

    """
    return x


def identity_d(x):
    r"""
    Derivada de Identidad

    :math:`f(x)=1`

    """
    return 1


def lecunn_sigmoid(x):
    r"""
    Sigmoid recomendada por LeCunn

    http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf

    :math:`f(x)=1.7159 tanh(\dfrac{2x}{3})`
    """
    return 1.7159 * np.tanh(x * 2.0/3.0)


def lecunn_sigmoid_d(x):
    r"""
    Derivada de Sigmoid recomendada por LeCunn

    :math:`f(x)=1.14393 (1 - tanh^2(\dfrac{2x}{3}))`

    """
    return 1.7159 * (2.0 / 3.0) * (1.0 - np.tanh(x * 2.0/3.0) ** 2)

# TODO definirlas mejor
# def sin(x):
#     return np.sin(x)
#
#
# def sin_d(x):
#     return np.cos(x)


fun_activation = {'Tanh': tanh, 'Sigmoid': sigmoid, 'ReLU': relu, 'Softplus': softplus,
                                'Identity': identity, 'LeakyReLU': lrelu, 'LeCunnSigm': lecunn_sigmoid}
fun_activation_d = {'Tanh': tanh_d, 'Sigmoid': sigmoid_d, 'ReLU': relu_d, 'Softplus': softplus_d,
                                    'Identity': identity_d, 'LeakyReLU': lrelu_d, 'LeCunnSigm': lecunn_sigmoid_d}





